from typing import Generator, List
from loguru import logger
from langchain.embeddings.base import Embeddings

import dashscope
from http import HTTPStatus

"""
需要设置环境变量DASHSCOPE_API_KEY
"""

DASHSCOPE_MAX_BATCH_SIZE = 4
dashscope.api_key_file_path = "./dashscope_api_key"


def batched(
    inputs: List, batch_size: int = DASHSCOPE_MAX_BATCH_SIZE
) -> Generator[List, None, None]:
    for i in range(0, len(inputs), batch_size):
        yield inputs[i : i + batch_size]


class TongyiEmbeddings(Embeddings):

    def __init__(self) -> None:
        self.model = dashscope.TextEmbedding.Models.text_embedding_v3

    def embed_documents(self, texts: list):
        embeddings = self.embed_with_list_of_str(texts)
        return [i["embedding"] for i in embeddings["output"]["embeddings"]]

    def embed_query(self, text):
        embedding = dashscope.TextEmbedding.call(model=self.model, input=text)
        return embedding["output"]["embeddings"][0]["embedding"]

    def embed_with_list_of_str(self, inputs: List):
        result = None  # merge the results.
        batch_counter = 0
        for batch in batched(inputs):
            resp = dashscope.TextEmbedding.call(model=self.model, input=batch)
            if resp.status_code == HTTPStatus.OK:
                if result is None:
                    result = resp
                else:
                    for emb in resp.output["embeddings"]:
                        emb["text_index"] += batch_counter
                        result.output["embeddings"].append(emb)
                    result.usage["total_tokens"] += resp.usage["total_tokens"]
            else:
                logger.error("embedding error: {}", resp)
            batch_counter += len(batch)
        logger.info("embbedding spend total tokens: {}", result.usage["total_tokens"])
        return result


# Instantiate embedding model and LLM
if __name__ == "__main__":

    embedding_model = TongyiEmbeddings()
    # embedding = embedding_model.embed_query(
    #     "衣服的质量杠杠的，很漂亮，不枉我等了这么久啊，喜欢，以后还来这里买"
    # )
    embedding = embedding_model.embed_documents(
        ["风急天高猿啸哀", "渚清沙白鸟飞回", "无边落木萧萧下", "不尽长江滚滚来"]
    )
    print("true")
