# Create Milvus vector store and store the embeddings inside it
from langchain_milvus import Milvus
from langchain_core.documents import Document
from embedding_model.tongyi_embedding import TongyiEmbeddings

URI = "http://127.0.0.1:19530"
# 初始化仓库
# 可更换embedding模型
vectorstore = Milvus(
    TongyiEmbeddings(),
    connection_args={"uri": URI},
    collection_name="multimodal_rag_demo",
)
vectorstore.auto_id = True

# Create a retriever from the vector store
# 检索器
retriever = vectorstore.as_retriever()

if __name__ == "__main__":

    document_1 = Document(
        page_content="I had chocalate chip pancakes and scrambled eggs for breakfast this morning.",
    )
    document_2 = Document(
        page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    )

    vectorstore.add_documents([document_1, document_2])

    query = "How's the wheather like tomorrow?"

    retrieved_docs = retriever.invoke(query, limit=1)

    print(retrieved_docs)
