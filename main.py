import os
from assessment import assessment, assessment_for_multitime,extract_score_and_feedback
from openai import OpenAI
from adaptive_RAG import Eval_RAG
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import accuracy_score
import copy
from advanced_text_inference import TextInference
from difflib import SequenceMatcher
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats import norm
import json
import random
from collections import defaultdict

def calculate_metrics(ground_truth, predict):
    
    # Accuracy using sklearn
    accuracy = accuracy_score(ground_truth, predict)
    print(f"accuracy: {accuracy}")
    
    # Spearman Rank Order Correlation Coefficient (SROCC)
    srocc, p = spearmanr(ground_truth, predict)
    print(f"SROCC: {srocc:.4f}, p-value: {p:.4f}")

    # Pearson Linear Correlation Coefficient (PLCC)
    plcc, p = pearsonr(ground_truth, predict)
    print(f"PLCC: {plcc:.4f}, p-value: {p:.4f}")

    return accuracy,srocc,plcc

def discrete_histogram_matching(source, reference):
    """
    对离散打分数据进行分布匹配。
    
    参数:
    source (np.array): 源数据（0, 1, 2, 3, 4, 5）
    reference (np.array): 参考数据（0, 1, 2, 3, 4, 5）
    
    返回:
    matched_data (np.array): 匹配后的数据
    """
    # 计算源数据和参考数据的概率分布
    source_bins = np.bincount(source, minlength=6) / len(source)
    reference_bins = np.bincount(reference, minlength=6) / len(reference)
    
    # 计算累积分布函数（CDF）
    source_cdf = np.cumsum(source_bins)
    reference_cdf = np.cumsum(reference_bins)
    
    # 映射类别：使用searchsorted找到第一个≥源CDF的参考索引
    mapping = {}
    for i in range(6):
        source_value = source_cdf[i]
        # 找到参考CDF中第一个≥源CDF的索引
        mapped_index = np.searchsorted(reference_cdf, source_value, side='left')
        # 确保索引不超过5（最高类别为5）
        mapping[i] = min(mapped_index, 5)
    
    # 对源数据进行映射
    matched_data = np.array([mapping[x] for x in source])
    
    return matched_data.tolist()

def rank_aggregate(method_scores: dict):
    """
    对多个方法的多个评估指标进行逐项排序，返回总名次最小的方法。
    
    参数：
        method_scores (dict): 
            结构如 {
                'method1': (acc, srocc, plcc),
                'method2': (acc, srocc, plcc),
                ...
            }

    返回：
        best_method (str): 总名次最小的最优方法名
        rank_scores (dict): 每个方法对应的总名次分数
    """

    # 初始化每个方法的总名次
    rank_scores = {method: 0 for method in method_scores}
    
    # 每个指标的索引
    num_metrics = len(next(iter(method_scores.values())))
    
    # 遍历每个指标，按指标值排序，累加名次分数
    for metric_idx in range(num_metrics):
        # 按当前指标排序（从大到小）
        ranking = sorted(method_scores.items(), key=lambda x: x[1][metric_idx], reverse=True)
        for rank, (method, _) in enumerate(ranking):
            rank_scores[method] += rank

    # 选出名次分数最小的方法
    best_method = min(rank_scores, key=rank_scores.get)
    return best_method

def hyperparameter_search(eval_rag, query, student_answers, rubric, ground_truth, few_shot_examples, few_shot=True, sa=True, k=5):
    best_predictions = None  # <-- 保存最优预测分数列表
    # Step 1: Query vs Keyword
    queryRAG_scores = [res['score'] for res in assessment_for_multitime(query=query, student_answers=student_answers, rubrics=rubric, eval_RAG=eval_rag, types='Normal', sa=False)]
    keywordRAG_scores = [res['score'] for res in assessment_for_multitime(query=query, student_answers=student_answers, rubrics=rubric, eval_RAG=eval_rag, types='Normal', sa=False, keyword=True)]

    acc_query, srocc_query, plcc_query = calculate_metrics(ground_truth, discrete_histogram_matching(queryRAG_scores, ground_truth))
    acc_keyword, srocc_keyword, plcc_keyword = calculate_metrics(ground_truth, discrete_histogram_matching(keywordRAG_scores, ground_truth))

    method_scores = {
        'query': (acc_query, srocc_query, plcc_query),
        'keyword': (acc_keyword, srocc_keyword, plcc_keyword)
    }
    best_method = rank_aggregate(method_scores)
    keyword = (best_method == 'keyword')

    # 保存当前 best
    best_performance = method_scores[best_method]
    best_predictions = queryRAG_scores if best_method == 'query' else keywordRAG_scores
    best_config = {
        'method': best_method,
        'sa': False,
        'k': 0,
        'few_shot': False,
        'num_few_shot': 0
    }

    # Step 2: SA Search
    if sa:
        method_scores = {0: best_performance}
        prediction_map = {0: best_predictions}
        for ki in range(1, k + 1):
            scores = [res['score'] for res in assessment_for_multitime(
                query=query, student_answers=student_answers, rubrics=rubric,
                eval_RAG=eval_rag, types='Normal', sa=True, k=ki, keyword=keyword)]
            acc, srocc, plcc = calculate_metrics(ground_truth, discrete_histogram_matching(scores, ground_truth))
            method_scores[ki] = (acc, srocc, plcc)
            prediction_map[ki] = scores
        
        best_sa = rank_aggregate(method_scores)
        if best_sa != 0:
            sa = True
            k = best_sa
            best_performance = method_scores[best_sa]
            best_predictions = prediction_map[best_sa]  # <== 更新最优预测
            best_config.update({'sa': True, 'k': best_sa})
        else:
            sa = False

    # Step 3: Few-Shot Search
    if few_shot:
        sa_scores = {0: best_performance}
        prediction_map = {0: best_predictions}
        for ni in range(1, len(few_shot_examples) + 1):
            scores = [res['score'] for res in assessment_for_multitime(
                query=query, student_answers=student_answers, rubrics=rubric,
                eval_RAG=eval_rag, types='Normal', sa=sa, k=k if sa else 0,
                keyword=keyword, few_shot=True, few_shot_examples=few_shot_examples[:ni])]
            acc, srocc, plcc = calculate_metrics(ground_truth, discrete_histogram_matching(scores, ground_truth))
            sa_scores[ni] = (acc, srocc, plcc)
            prediction_map[ni] = scores
        
        best_ex = rank_aggregate(sa_scores)
        if best_ex != 0:
            few_shot = True
            best_performance = sa_scores[best_ex]
            best_predictions = prediction_map[best_ex]  # <== 更新最优预测
            best_config.update({'few_shot': True, 'num_few_shot': best_ex})
        else:
            few_shot = False

    acc_final, srocc_final, plcc_final = best_performance
    print(f"✅ Best parameter configuration:")
    print(f"→ Method: {best_config['method']}")
    print(f"→ Use Sample Answers (SA): {best_config['sa']}, k = {best_config['k']}")
    print(f"→ Use Few-Shot: {best_config['few_shot']}, num = {best_config['num_few_shot']}")
    print(f"🎯 Best Performance:")
    print(f"→ Accuracy: {acc_final:.4f}, SROCC: {srocc_final:.4f}, PLCC: {plcc_final:.4f}")

    return best_config, best_performance, best_predictions


    


if __name__ =='__main__':
    q1 = (
        "You are restoring a Class IV lesion on tooth 11. What would be the best approach in placing different shades composite resin to mimic the natural appearance - palatally, centrally, labially and incisally? (5 marks) "
    )
    # q1_image = "Q1.jpg"
    q1_query = {'question': q1, 'image':''}
    rubric = ['''
0 points: No relevant content is provided. The response does not address the question.			
1 point: Minimal relevance; mentions one or more concepts related to the question but with significant inaccuracies or misinterpretations. Lacks sufficient explanation and appropriate terminology.			
2 points: Partial explanation with some relevance; identifies basic concepts but lacks depth and clarity. Contains significant errors or omissions that impede a correct understanding. Limited use of professional terminology.			
3 points: Adequate explanation that identifies and discusses key concepts. Some details are correct, but the response may miss  minor elements or contain some incorrect statements that do not affect the overall understanding or use slightly inaccurate terminology. 			
4 points: Strong explanation that correctly identifies and elaborates on key concepts. The response includes related concepts and is  accurate, with minor inaccuracies in  professional terminology.			
5 points: Comprehensive and accurate explanation of all key concepts related to the question. The response is well-articulated, aligns closely with the required keywords, and contains no incorrect information. Excellent use of professional terminology.			
''']
    rubric_text = rubric[0]  # 原始字符串

    rubric_dict = {}
    for line in rubric_text.strip().splitlines():
        line = line.strip()
        if not line or not line[0].isdigit():
            continue
        score_str, desc = line.split(":", 1)
        score = int(score_str.strip().split()[0])
        rubric_dict[score] = desc.strip()

    folder_path = r'D:\USYD\LLM project\Dataset\Dental_shotanswer_dataset\student answers'

    ground_truth = []
    level_list = []
    student_answers = []

    # 存储按得分分组的样本索引
    score_to_indices = defaultdict(list)

    # 遍历所有 JSON 文件，提取 Q1 数据并建立 score ➝ indices 映射
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith('.json'):
            continue

        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for q in data.get('questions', []):
            if q.get("question_id") == "Q1":
                answer = q.get("student_answer", "").strip()
                score = q.get("score", None)
                level = q.get("level", None)

                if score is not None and answer:
                    idx = len(student_answers)
                    student_answers.append(answer)
                    ground_truth.append(score)
                    level_list.append(level if level is not None else None)
                    score_to_indices[score].append(idx)

    # 遍历分数 0–5，分别选一个样本作为 few-shot example
    few_shot_examples = []
    example_indices = []

    for score in range(6):
        indices = score_to_indices.get(score, [])
        if not indices:
            print(f"⚠️ 无法为得分 {score} 找到样本")
            continue
        chosen_idx = random.choice(indices)
        example_indices.append(chosen_idx)

        example = f'''
            Student Answer: {student_answers[chosen_idx]} \nScore: {score} \nFeedback: {rubric_dict[score]}
        '''
        few_shot_examples.append(example)

    # 根据 example_indices 反向排序并从原始列表中删除（保持索引同步）
    for idx in sorted(example_indices, reverse=True):
        del student_answers[idx]
        del ground_truth[idx]
        del level_list[idx]
    
    output_path = "few_shot_examples_q1.txt"

    # 写入操作
    with open(output_path, "w", encoding="utf-8") as f:
        for i, example in enumerate(few_shot_examples):
            f.write(f"Example {i + 1}:\n")
            f.write(example.strip())  # 去除开头缩进
            f.write("\n" + "-" * 80 + "\n")
    
    inference = TextInference()
    eval_rag = Eval_RAG(inference)
    best_config, best_score, best_predictions = hyperparameter_search(eval_rag=eval_rag,query=q1_query,student_answers=student_answers,rubric=rubric,
                                                                      ground_truth=ground_truth,few_shot_examples=few_shot_examples,k=4)
    output_data = {
    "best_config": best_config,
    "best_score": {
        "accuracy": best_score[0],
        "srocc": best_score[1],
        "plcc": best_score[2]
    },
    "results": []
    }

    # 打包每个样本信息（prediction, ground truth, level）
    for i in range(len(best_predictions)):
        output_data["results"].append({
            "sample_id": i + 1,
            "predicted_score": best_predictions[i],
            "ground_truth": ground_truth[i],
            "level": level_list[i]
        })

    # 写入 JSON 文件
    output_path = "best_results_summary_q1.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)