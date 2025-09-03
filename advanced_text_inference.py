import time
from inference import Inferencer


class TextInference(Inferencer):
    def inference(self, question: str, sample_answer: bool = False, image: str = "", temperature: float = 0.0, keyword_search: bool = False, k: int = 5) -> str:
        RAG_limit = 3
        is_sufficient = 'no'
        summarized_knowledge = None
        if keyword_search:
            keywords = self.extract_keyword(question)
            combined_query = " ".join(keywords)
            priori_knowledge = self.get_priori_knowledge(combined_query)
        else:
            priori_knowledge = self.get_priori_knowledge(question)

        while is_sufficient != 'yes' and RAG_limit < 15:
            summarized_knowledge = self.multi_round_refinement(question=question,rag_results=priori_knowledge,image=image,top_n=RAG_limit)
            is_sufficient = self.evaluate_rag_sufficiency(question = question, rag_results= summarized_knowledge, image = image)
            RAG_limit += 2
        if sample_answer:
            self.get_inference_prompt(question, summarized_knowledge, image)
            completion = self.model.chat.completions.create(
                model="gpt-4o-mini",
                messages=self.inference_prompt,
                temperature=temperature,
                max_tokens=300,
                n=k,
                seed=int(time.time() * 1000)  # Use current time in milliseconds as seed
            )
            return [choice.message.content for choice in completion.choices]
        else: 
            text = "\n".join([doc for doc in summarized_knowledge])
            return {'prior_knowledge': text}
        
    def get_priori_knowledge(self, question):
        return self.retriver.retrive_doc(question)
    
    def multi_round_refinement(self, question: str, rag_results: list, top_n:int = 5, image:str = "") -> str:
        sort_rag_results = sorted(rag_results, key=lambda x: x[1], reverse=True)
        relevant_content = sort_rag_results[:top_n]
        refinement = []

        if image == "":
            for content in relevant_content:
                messages = [
                    {
                        "role": "user",
                        "content": [
                        {
                                "type": "text",
                                "text":f'''Refine and summarize the following content to answer the question.
                                question: {question}
                        '''},
                        {
                            "type": "text",
                            "text":f'''content: {content[0].metadata["content"]}'''
                    },
                    ]
                }
            ]
                completion = self.model.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=1.0,
                    max_tokens=300,
                )
                refinement.append(completion.choices[0].message.content)
        else:
            base64_image = self.encode_image(image)
            for content in relevant_content:
                messages = [
                    {
                        "role": "user",
                        "content": [
                        {
                                "type": "text",
                                "text":f'''Refine and summarize the following content to answer the question.
                                question: {question}
                        '''},
                        {
                            "type": "image_url",
                            "image_url":{
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                }
                        },
                        {
                            "type": "text",
                            "text":f'''content: {content[0].metadata["content"]}'''
                    },
                    ]
                }
            ]
                completion = self.model.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=1.0,
                    max_tokens=300,
                )
                refinement.append(completion.choices[0].message.content)
        return refinement

    
    def evaluate_rag_sufficiency(self, question: str, rag_results: list, image: str = "") -> bool:
        # Combine RAG results into a single string
        rag_content = "\n".join([doc for doc in rag_results])
        if image == "":
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""Based on the question '{question}', evaluate whether the following information is sufficient to give a full mark answer:{rag_content}
                            Respond only with 'yes' or 'no'.""",
                        },
                    ]
                }
            ]
            completion = self.model.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=1.0,
                max_tokens=10,
            )
            response = completion.choices[0].message.content.lower()
        else:
            base64_image = self.encode_image(image)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""Based on the question, evaluate whether the following information is sufficient to answer the question
                            Respond only with 'yes' or 'no'.""",
                        },
                        {
                            "type": "text",
                            "text": f"""question: {question}"""
                        },
                        {
                            "type": "image_url",
                            "image_url":{
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                }
                        },
                        {
                            "type": "text",
                            "text": f"""information: {rag_content}"""
                        },
                    ]
                }
            ]
            completion = self.model.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=1.0,
                max_tokens=10,
            )
            response = completion.choices[0].message.content.lower()
        if "yes" in response:
            result = "yes"
        elif "no" in response:
            result = "no"
        return result
    
    def get_inference_prompt(self, question, rag_results, image=""):
        texts = "\n".join([doc for doc in rag_results])
        if image == "":
            massage = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""You are an exceptional professor at a dental college.
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
                            "text": f"""You are an exceptional professor at a dental college.
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