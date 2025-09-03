import random
import re
import time
import numpy as np
import copy
from collections import Counter

from batch_processor import BatchProcessor

def extract_score_and_feedback(text):
    score_match = re.search(r"'score':\s*(\d+)", text)
    feedback_match = re.search(r"'feedback':\s*'(.*?)'", text)
    if score_match and feedback_match:
        return {
            "score": int(score_match.group(1)),
            "feedback": feedback_match.group(1)
        }
    return {"score": None, "feedback": None}

def assessment(query, student_answers: list, rubrics, eval_RAG, examples:list = [], few_shot:bool = False, sa: bool = False , keyword: bool = False, types: str = 'Normal', k: int = 1,temperature : float = 0.0):
    init_prompt =  [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": """Now you are a dental professor and you need to assess the student's answer in a dental assignment.
                            Assess the student's answer based on the provided rubric. 
                            Your output should always follow this JSON format: {'score': <numeric value>, 'feedback': '<your comments>'}.
                            Do not include any additional words outside this format.""",
                        },
                    ]
                },
        ]
    prompt = eval_RAG.add_question(query,init_prompt)
    final_prompt = eval_RAG.add_rubric(rubrics,prompt)
    RAG_result =  eval_RAG.Active_RAG(query=query, types=types, sa=sa, keyword=keyword, k =k)

    if RAG_result is not None:
        if sa:
            # Append reference answers if available
            for i, result in enumerate(RAG_result):
                msg = {
                    "type": "text",
                    "text": f"""Reference Answer {i}: {result}"""
                }
                final_prompt[1]['content'].append(msg)
        else:
            # Append prior knowledge (text) and images
            msg = {
                "type": "text",
                "text": f"""Reference Information: {RAG_result['prior_knowledge']}"""
            }
            final_prompt[1]['content'].append(msg)

    results = []

    # For each student's answer, append to the prompt and evaluate
    for student_answer in student_answers:
        answer = {
            "type": "text",
            "text": f"""Student Answer: {student_answer}"""
        }
        # Append the student's answer to the prompt
        prompt_with_answer = copy.deepcopy(final_prompt)
        prompt_with_answer[1]['content'].append(answer)

        if few_shot and examples:
            few_msg = {
                "type": "text",
                "text": "Here are some examples for you:"
            }
            prompt_with_answer[1]['content'].append(few_msg)

            for example in examples:
                example_msg = {
                    "type": "text",
                    "text": f"Example: {example}"
                }
                prompt_with_answer[1]["content"].append(example_msg)

        # Prepare for the result collection

        # Call the model for assessment after appending all necessary context
        completion = eval_RAG.inference.model.chat.completions.create(
            model="gpt-4o-mini",  # Adjust as per the actual model name you're using
            messages=prompt_with_answer,
            temperature=temperature,
            seed = 42,
            top_p =1,
            max_tokens=200,
        )

        # Process the result (e.g., extract score and feedback)
        result = extract_score_and_feedback(completion.choices[0].message.content)
        results.append(result)

    return results



def assessment_for_multitime(query, student_answers: list, rubrics, eval_RAG, examples:list = [], few_shot:bool = False, sa: bool = False , keyword: bool = False, types: str = 'Normal', k: int = 1,temperature : float = 0.5, round=5):
    init_prompt =  [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": """You are a senior professor in restorative dentistry with extensive experience teaching and grading dental students. 
                                Your task is to rigorously evaluate the student's written clinical response based on the rubric provided below.
                                Please assess primarily based on the scientific accuracy, completeness, and relevance of content. Do not penalize for bullet-point or list-style formatting unless it impedes comprehension.
                                Your response must always follow this JSON format:
                                {'score': <numeric value>, 'feedback': '<your feedback in 2–4 sentences>'}
                                Only return this JSON. Do not include explanations outside this format.""",
                        },
                    ]
                },
        ]
    prompt = eval_RAG.add_question(query,init_prompt)
    final_prompt = eval_RAG.add_rubric(rubrics,prompt)
    RAG_result =  eval_RAG.Active_RAG(query=query, types=types, sa=sa, keyword=keyword, k =k)

    if RAG_result is not None:
        if sa:
            # Append reference answers if available
            for i, result in enumerate(RAG_result):
                msg = {
                    "type": "text",
                    "text": f"""Reference Answer {i}: {result}"""
                }
                final_prompt[1]['content'].append(msg)
        else:
            # Append prior knowledge (text) and images
            msg = {
                "type": "text",
                "text": f"""Reference Information: {RAG_result['prior_knowledge']}"""
            }
            final_prompt[1]['content'].append(msg)

    results = []

    # For each student's answer, append to the prompt and evaluate
    for student_answer in student_answers:
        answer = {
            "type": "text",
            "text": f"""
            Student Answer: {student_answer}"""
        }
        # Append the student's answer to the prompt
        prompt_with_answer = copy.deepcopy(final_prompt)
        prompt_with_answer[1]['content'].append(answer)

        if few_shot and examples:
            few_msg = {
                "type": "text",
                "text": "Here are some examples for you:"
            }
            prompt_with_answer[1]['content'].append(few_msg)

            for example in examples:
                example_msg = {
                    "type": "text",
                    "text": f"{example}"
                }
                prompt_with_answer[1]["content"].append(example_msg)

        # Call the model for assessment after appending all necessary context

        completion = eval_RAG.inference.model.chat.completions.create(
            model="gpt-4o-mini",  # Adjust as per the actual model name you're using
            messages=prompt_with_answer,
            seed = 42,
            temperature=temperature,
            max_tokens=200,
            n = round
        )
        time.sleep(2)
            # Process the result (e.g., extract score and feedback)
        temp_results =[extract_score_and_feedback(choice.message.content) for choice in completion.choices]
        scores = []
        feedbacks = []
        for result in temp_results:
            if result["score"] is not None:
                scores.append(result["score"])
                feedbacks.append(result["feedback"])
        counter = Counter(scores)

        if not counter:
            raise ValueError("Counter is empty. No scores available. Please run it again!")

        # 找到最大出现次数
        max_count = max(counter.values())

        # 获取所有众数
        modes = [key for key, count in counter.items() if count == max_count]
        final_score = random.choice(modes)
        indexes = [i for i, score in enumerate(scores) if score == final_score]

        # 如果存在匹配的索引，取第一个索引的反馈
        if indexes:
            feedback = feedbacks[indexes[0]]
        else:
            feedback = feedbacks[0]  # 默认取第一个反馈
        results.append({"score": final_score, "feedback": feedback})

    return results

def assessment_for_multireqiest(query, student_answers: list, rubrics, eval_RAG, examples:list = [], few_shot:bool = False, sa: bool = False , keyword: bool = False, types: str = 'Normal', k: int = 1,temperature : float = 0.5, round=5):
    init_prompt =  [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": """Now you are a dental professor and you need to assess the student's answer in a dental assignment.
                            Assess the student's answer based on the provided rubric. 
                            Your output should always follow this JSON format: {'score': <numeric value>, 'feedback': '<your comments>'}.
                            Do not include any additional words outside this format.""",
                        },
                    ]
                },
        ]
    prompt = eval_RAG.add_question(query,init_prompt)
    final_prompt = eval_RAG.add_rubric(rubrics,prompt)
    RAG_result =  eval_RAG.Active_RAG(query=query, types=types, sa=sa, keyword=keyword, k =k)

    if RAG_result is not None:
        if sa:
            # Append reference answers if available
            for i, result in enumerate(RAG_result):
                msg = {
                    "type": "text",
                    "text": f"""Reference Answer {i}: {result}"""
                }
                final_prompt[1]['content'].append(msg)
        else:
            # Append prior knowledge (text) and images
            msg = {
                "type": "text",
                "text": f"""Reference Information: {RAG_result['prior_knowledge']}"""
            }
            final_prompt[1]['content'].append(msg)

    results = []

    # For each student's answer, append to the prompt and evaluate
    for student_answer in student_answers:
        answer = {
            "type": "text",
            "text": f"""Student Answer: {student_answer}"""
        }
        # Append the student's answer to the prompt
        prompt_with_answer = copy.deepcopy(final_prompt)
        prompt_with_answer[1]['content'].append(answer)

        if few_shot and examples:
            few_msg = {
                "type": "text",
                "text": "Here are some examples for you:"
            }
            prompt_with_answer[1]['content'].append(few_msg)

            for example in examples:
                example_msg = {
                    "type": "text",
                    "text": f"Example: {example}"
                }
                prompt_with_answer[1]["content"].append(example_msg)

        # Call the model for assessment after appending all necessary context
        temp_results = []
        for _ in range(round):
            completion = eval_RAG.inference.model.chat.completions.create(
                model="gpt-4o-mini",  # Adjust as per the actual model name you're using
                messages=prompt_with_answer,
                temperature=temperature,
                max_tokens=200,
            )

            # Process the result (e.g., extract score and feedback)
            temp_results.append(extract_score_and_feedback(completion.choices[0].message.content))
        scores = []
        feedbacks = []
        for result in temp_results:
            if result["score"] is not None:
                scores.append(result["score"])
                feedbacks.append(result["feedback"])
        counter = Counter(scores)

        # 找到最大出现次数
        max_count = max(counter.values())

        # 获取所有众数
        modes = [key for key, count in counter.items() if count == max_count]
        final_score = max(modes)
        indexes = [i for i, score in enumerate(scores) if score == final_score]

        # 如果存在匹配的索引，取第一个索引的反馈
        if indexes:
            feedback = feedbacks[indexes[0]]
        else:
            feedback = feedbacks[0]  # 默认取第一个反馈
        results.append({"score": final_score, "feedback": feedback})

    return results

