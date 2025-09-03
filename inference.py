import base64
from openai import OpenAI
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import os
import pickle
from save import RAGDatasetSaver
from langchain_community.llms import Tongyi


class Inferencer:

    def __init__(self):
        self.model = OpenAI(
            api_key= os.getenv('OPENAI_API_KEY'),
            # base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.inference_prompt = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.img_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(
            self.device
        )
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # self.image_paths, self.image_features = self.load_knowledge_base_images(
        #     knowledge_base_img_folder
        # )
        # self.faiss_index = self.build_faiss_index(self.image_features)
        self.retriver = RAGDatasetSaver()
        # self.loading(self.retriver, knowledge_base_text)

    def loading(self, retriver, file):
        with open(file, "rb") as f:  # text, text_summaries, table, table_summaries
            data = pickle.load(f)
            tables = data["tables"]
            table_summaries = data["table_summaries"]
            texts = data["texts_4k_token"]
            text_summaries = data["text_summaries"]
        retriver.save_documents(texts, text_summaries)
        retriver.save_documents(tables, table_summaries)

    def inference(self, question: str, sample_answer: bool = False, image: str = "", temperature: float = 0.0, keyword_search: bool = False):
        if keyword_search:
            keywords = self.extract_keyword(question)
            if image == "":
                priori_knowledge = self.get_priori_knowledge(keywords)
            else:
            #    img_base64 = self.encode_image(image) 
               priori_knowledge = self.get_priori_knowledge(keywords)
        else:
            if image == "":
                priori_knowledge = self.get_priori_knowledge(question)
            else:
            #    img_base64 = self.encode_image(image) 
               priori_knowledge = self.get_priori_knowledge(question)
        # if image == "":
        #     priori_imgs = []
        # else:
        #     priori_imgs = self.get_priori_img(
        #         self.read_image(image), self.image_paths, self.image_features
        #     )
        if priori_knowledge['doc']:
            external_text =priori_knowledge['doc']
        else:
            return None
        if priori_knowledge['image']:
            external_images = priori_knowledge['image'][0].metadata["content"]
        else:
            external_images = None
        if sample_answer:
            self.get_inference_prompt(question, external_text, external_images, image)
            completion = self.model.chat.completions.create(
                model="gpt-4o-mini",
                messages=self.inference_prompt,
                temperature=temperature,
                max_tokens=300,
            )
            return completion.choices[0].message.content
        else: 
            return {'prior_knowledge': external_text[0].metadata["content"], 'prior_imgs': external_images}

    def get_priori_knowledge(self, question, image=None):
        return self.retriver.retrive_image_and_doc(question)

    def get_inference_prompt(self, question, priori_knowledge, priori_imgs, image):
        texts = priori_knowledge[0].metadata["content"]
        if image == "":
            massage = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""You are an excellent dental college student.
                            You will be given a mix of text, tables, and images (usually related to dental knowledge).
                            Use this information to answer the provided short-answer question.
                            Short-answer question: {question}""",
                        },
                        {
                            "type": "text",
                            "text": f"""Retriving information:
                            Text and/or tables:{texts} """,
                        },
                    ],
                },
            ]
            self.inference_prompt = massage
        else:
            base64_image = self.encode_image(image)
            massage = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""You are an excellent dental college student.
                            You will be given a mix of text, tables, and images (usually related to dental knowledge).
                            Use retriving information to answer the provided short-answer question.
                            Short-answer question: {question}""",
                        },
                        {
                            "type": "image_url",
                            "image_url":{
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            }
                        },
                        {
                            "type": "text",
                            "text": f"""Retriving Information:
                         Text and/or tables:{texts} """,
                        },
                    ],
                }
            ]
            self.inference_prompt = massage
        if priori_imgs:
            for img_path in [priori_imgs]:
                img = self.encode_image(img_path)
                add_dict = {
                    "type": "image_url",
                    "image_url":{ 
                    "url": f"data:image/jpeg;base64,{img}",
                    }
                }
                self.inference_prompt[0]["content"].append(add_dict)

    def extract_keyword(self, query):  
        messages = [
        {
            "role": "user",
            "content": f"Please extract the most relevant and specific keywords for document retrieval. Provide them as a comma-separated list. Question: {query}"
        }
    ]
        try:
            completion = self.model.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.1,
                max_tokens=50,
            )
            # 提取关键词并格式化
            keywords = completion.choices[0].message.content.strip()
            keywords_list = [kw.strip() for kw in keywords.split(",") if kw.strip()]
            
            # 控制关键词数量
            if len(keywords_list) > 5:
                keywords_list = keywords_list[:5]  # 限制前 5 个关键词
            
            return keywords_list
        except Exception as e:
            print(f"Error during keyword extraction: {e}")
            return []

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def read_image(self, image_path):
        return Image.open(image_path).convert("RGB")

    # def extract_image_features(self, image):
    #     inputs = self.processor(images=image, return_tensors="pt").to(self.device)
    #     with torch.no_grad():
    #         features = self.img_model.get_image_features(**inputs)
    #     return features.cpu().numpy()

    # def load_knowledge_base_images(self, prior_image_folder):
    #     image_paths = [
    #         os.path.join(prior_image_folder, fname)
    #         for fname in os.listdir(prior_image_folder)
    #         if fname.endswith((".png", ".jpg", ".jpeg"))
    #     ]
    #     features = []
    #     for image_path in image_paths:
    #         image = self.read_image(image_path)
    #         features.append(self.extract_image_features(image))
    #     return image_paths, np.vstack(features)

    # def build_cosine_index(self, image_features):
    #     """构建用于相似度搜索的索引（基于余弦相似度）"""
    #     normalized_features = normalize(image_features)
    #     return normalized_features

    # def get_priori_img(
    #     self, query_image, image_paths, knowledge_base_features, top_n=3, threshold=0.8
    # ):
    #     # 提取查询图像的特征
    #     query_features = self.extract_image_features(query_image)
    #     query_features = normalize(query_features)
        
    #     # 构建余弦相似度索引
    #     knowledge_base_features = self.build_cosine_index(knowledge_base_features)

    #     # 计算余弦相似度
    #     similarities = cosine_similarity(query_features, knowledge_base_features)[0]

    #     # 根据相似度和阈值筛选出符合条件的图像
    #     filtered_images = [
    #         (image_paths[i], similarities[i])
    #         for i in range(len(similarities))
    #         if similarities[i] >= threshold
    #     ]

    #     # 按相似度降序排序
    #     filtered_images.sort(key=lambda x: x[1], reverse=True)

    #     # 返回相似度最高的 top_n 个图像
    #     return [img[0] for img in filtered_images[:top_n]]
