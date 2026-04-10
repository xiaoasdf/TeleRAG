from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from src.runtime import get_compute_device


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None):
        self.model_name = model_name
        self.device = device or get_compute_device()
        self.model = None

    def _ensure_model_loaded(self) -> None:
        if self.model is None:
            self.model = SentenceTransformer(self.model_name, device=self.device)

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        批量编码文本
        """
        self._ensure_model_loaded()
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """
        编码查询
        """
        self._ensure_model_loaded()
        embedding = self.model.encode([query], normalize_embeddings=True)
        return embedding[0]
