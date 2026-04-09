from typing import List

from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en"):
        self.model = SentenceTransformer(model_name)

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        批量编码文本
        """
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """
        编码查询
        """
        embedding = self.model.encode([query], normalize_embeddings=True)
        return embedding[0]