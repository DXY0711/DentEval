import copy
import os
import time
import numpy as np
from openai import OpenAI
from advanced_text_inference import TextInference
from transformers import CLIPProcessor, CLIPModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from assessment import assessment, assessment_for_multitime,extract_score_and_feedback

class Eval_RAG:
    
    def __init__(self, inference):
        self.inference = inference
        self.prompt = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""Now you are a dental professor and you need to access the student’s answer in dental assignment.
                            you should assess the student' answer based on the rubric.""",
                        },
                    ]
                },
        ]

    def Active_RAG(self, query: dict, sa: bool = False , keyword: bool = False, types: str = 'Normal', k: int = 1):
        if types == 'Normal':
            if sa:
                answers = self.inference.inference(question=query['question'], image= query['image'], sample_answer= sa, keyword_search=keyword, temperature=1.0, k= k)
                return answers
            else:
                retrive_result = self.inference.inference(question=query['question'], image= query['image'], sample_answer= sa, keyword_search=keyword)
                return retrive_result
        # elif types == 'Adaptive_CLIP':
        #     answers = []
        #     init_prompt =  [
        #         {
        #             "role": "user",
        #             "content": [
        #                 {
        #                     "type": "text",
        #                     "text": f"""You are an excellent dental college student.
        #                     You will be given a mix of text, tables, and images (usually related to dental knowledge).
        #                     Use this information to answer the provided short-answer question."""
        #                 }
        #             ]
        #         }
        #     ]
        #     temperatures = [0.0,0.3,0.5,0.7,0.9]
        #     for i in range(len(temperatures)):
        #         prompt = copy.deepcopy(init_prompt)
        #         prompt = self.add_question(query, prompt)
        #         # result = self.inference.inference(query['question'], query['image'], sample_answer= True, temperature=temperatures[i])
        #         result = self.zero_shot(prompt, temperatures[i])
        #         answers.append(result)

        #     model_name = "openai/clip-vit-base-patch32"
        #     model = CLIPModel.from_pretrained(model_name)
        #     processor = CLIPProcessor.from_pretrained(model_name)

        #     inputs = processor(text=answers, return_tensors="pt", padding=True, truncation=True)
        #     with torch.no_grad():
        #         text_embeddings = model.get_text_features(**inputs)
        #     # 将嵌入转换为 NumPy 数组
        #     text_embeddings_np = text_embeddings.cpu().numpy()

        #     # 计算余弦相似度
        #     similarity_matrix = cosine_similarity(text_embeddings_np)

        #     n = similarity_matrix.shape[0]
        #     off_diagonal_indices = np.triu_indices(n, k=1)  # 获取上三角非对角线索引
        #     similarities = similarity_matrix[off_diagonal_indices]
        #     mean_similarity = np.mean(similarities)
        #     if mean_similarity >= 0.8:
        #         return answers
        #     else:
        #         # retrive_result = self.inference.inference(query['question'], query['image'], sample_answer= False)
        #         # return retrive_result
        #         return None
        elif types == 'Adaptive_LLM':
            model = OpenAI(
            api_key= os.getenv('OPENAI_API_KEY'),
            # base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
            if query['image'] == '':
                massage = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""Classify the following question based on whether it has a single correct answer or multiple correct answers. Your output should be strictly one of the following:
                                Single Answer or Multiple Answers""",
                            }, 
                            {
                                "type":"text",
                                "text": f'''question: {query['question']}'''
                            },
                        ]
                    }
                ]
            else:
                base64_image = self.inference.encode_image(query['image'])
                massage = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f""" Classify the following question based on whether it has a single correct answer or multiple correct answers. Your output should only content:
                                Single Answer or Multiple Answers"""
                            }, 
                            {
                                "type":"text",
                                "text": f'''Short-answer question: {query['question']}'''
                            },
                            {
                                "type": "image_url",
                                "image_url":{
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                }
                            },
                        ]
                    }
                ]
            completion = model.chat.completions.create(
                model="gpt-4o-mini",
                messages=massage,
                temperature=0.1,
                max_tokens=100,
            )
            result_LLM = completion.choices[0].message.content.lower()

            try:
                if 'single answer' in result_LLM and 'multiple answers' not in result_LLM:
                    if sa:
                        answers = []
                        for _ in range(k):
                            result = self.inference.inference(question=query['question'], image= query['image'], sample_answer= sa, temperature=0.7, keyword_search=keyword)
                            answers.append(result)
                        return answers
                    else:
                        result = self.inference.inference(question=query['question'], image= query['image'], sample_answer= sa, keyword_search=keyword)
                        return result
                elif 'multiple answers' in result_LLM and 'single answer' not in result_LLM:
                    return None
            except Exception as e: 
                print(f"An error occurred: {e}")
                return None
            
    
    def add_rubric(self, rubrics, init_prompt):
        for i, rubric in enumerate(rubrics):
            massgae = {
                            "type": "text",
                            "text": f"""Rubric {i}: {rubric},
                            """,
                        }
            init_prompt[1]['content'].append(massgae)
        return init_prompt
    
    def add_question(self, query, init_prompt):
        if not isinstance(init_prompt, list):
            raise ValueError("init_prompt should be a list.")
        
        if query['image'] == '':
            msg = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Short-answer question: {query['question']},
                        """,
                    },
                ]
            }
        else:
            base64_image = self.inference.encode_image(query['image'])
            msg = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Short-answer question: {query['question']},
                        """,
                    },
                    {
                            "type": "image_url",
                            "image_url":{
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            }
                    },
                ]
            }
        init_prompt.append(msg)
        
        return init_prompt

    def few_shot(self, examples:list, init_prompt,k):
        try:
            few_msg = {
            "type": "text",
            "text": "Here are some examples for you:"
        }
            init_prompt[1]['content'].append(few_msg)

            k = min(k, len(examples))
            for i in range(k):
                p = {
                        "type": "text",
                        "text": f"""Example: {examples[i]},""",
                }
                init_prompt[1]['content'].append(p)
            return init_prompt
        
        except Exception as e: 
            print(f"An error occurred: {e}")
            return None
        
    def zero_shot(self, prompt, temp):
        model = OpenAI(
            api_key= os.getenv('OPENAI_API_KEY'),
            # base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        completion = model.chat.completions.create(
                model="gpt-4o-mini",
                messages=prompt,
                temperature=temp,
                seed= 42,
                top_p= 1,
                max_tokens=200,
            )
        return completion.choices[0].message.content



    





            
