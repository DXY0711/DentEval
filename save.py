import pickle
from typing import List
from uuid import uuid4

from langchain_milvus import Milvus
from embedding_model.tongyi_embedding import TongyiEmbeddings
from langchain_core.documents import Document

URI = "http://127.0.0.1:19530"


class RAGDatasetSaver:
    def __init__(
        self,
    ):
        self.vectorstore = Milvus(
            TongyiEmbeddings(),
            connection_args={"uri": URI},
            collection_name="multimodal_rag_table",
        )
        self.image_vectorstore = Milvus(
            TongyiEmbeddings(),
            connection_args={"uri": URI},
            collection_name="multimodal_rag_image",
        )
        self.image_vectorstore.auto_id = True
        self.table_vectorstore = Milvus(
            TongyiEmbeddings(),
            connection_args={"uri": URI},
            collection_name="multimodal_rag_table",
        )  
        self.vectorstore.auto_id = True
        self.retriever = self.vectorstore.as_retriever()

    def summarize_pdf(self):
        pass

    def summarize_image(self):
        pass

    def summarize_text(self):
        pass

    def save_documents(self, texts, summaries):
        """保存文本和摘要

        Args:
            texts (list): 文本
            summaries (list): 摘要
        """
        documents = [
            Document(page_content=summary, metadata={"content": text})
            for text, summary in zip(texts, summaries)
        ]
        uuids = [str(uuid4()) for _ in range(len(documents))]
        self.vectorstore.add_documents(documents, ids=uuids)

    def save_images(self, images, summaries):
        """
        保存图片和摘要
        Args:
            images (list): 图片的base64编码
            summaries (list): 图片的摘要
        """
        documents = [
            Document(page_content=summary, metadata={"content": image})
            for image, summary in zip(images, summaries)
        ]
        print(documents[0])
        uuids = [str(uuid4()) for _ in range(len(documents))]
        self.image_vectorstore.add_documents(documents, ids=uuids)

    def save_tables(self, tables, summaries):
        """保存表格和摘要

        Args:
            tables (list): 表格
            summaries (list): 摘要
        """
        documents = [
            Document(page_content=summary, metadata={"content": table})
            for table, summary in zip(tables, summaries)
        ]
        uuids = [str(uuid4()) for _ in range(len(documents))]
        self.table_vectorstore.add_documents(documents, ids=uuids)

    def retrive_doc(self, query_text=None, limit=15) -> List[Document]:
        """
        Search for documents using a text query, an image query, or both, 
        and return the corresponding summaries and metadata.

        Args:
            query_text (str, optional): The text query. Defaults to None.
            query_image (str, optional): The image query in base64-encoded format. Defaults to None.
            limit (int, optional): The number of results to return. Defaults to 1.

        Returns:
            List[Document]: List of Document objects with summaries and metadata.
        """
        # Ensure at least one query input is provided
        if not query_text:
            raise ValueError("At least one of query_text or query_image must be provided.")

        # Perform similarity search based on the metadata
        return  self.vectorstore.similarity_search_with_score(
            query=query_text,
            k=limit,
        )
    
    def retrive_image_and_doc(self, query=None, limit=1) -> List[Document]:
        # Ensure at least one query input is provided
        result = {
            "doc":self.vectorstore.similarity_search(
            query,
            k=limit,
        ),
        "image":self.image_vectorstore.similarity_search(
            query,
            k=limit
        )
        }
        return result 

    def retrive_table(self, query, limit=1) -> List[Document]:
        """搜索query对应的摘要并返回对应的表格

        Args:
            query (str):  搜索的关键词
            limit (int, optional): 返回的结果个数. Defaults to 1.

        Returns:
            List[Document]: Document(page_content=summary, metadata={"content": table})
        """
        return self.table_vectorstore.similarity_search(
            query,
            k=limit,
        )

