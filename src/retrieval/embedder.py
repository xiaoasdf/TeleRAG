from typing import List

import numpy as np

from src.runtime import get_compute_device

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None):
        self.model_name = model_name
        self.device = device or get_compute_device()
        self.model = None

    def _ensure_model_loaded(self) -> None:
        if self.model is None:
            if SentenceTransformer is None:  # pragma: no cover
                raise RuntimeError(
                    "The 'sentence-transformers' package is required for retrieval embeddings. "
                    "Install retrieval dependencies before querying the local knowledge base."
                )

            self.model = SentenceTransformer(self.model_name, device=self.device)

    def encode_texts(self, texts: List[str], batch_size: int | None = None) -> np.ndarray:
        """
        批量编码文本
        """
        self._ensure_model_loaded()
        encode_kwargs = {"normalize_embeddings": True}
        if batch_size is not None:
            encode_kwargs["batch_size"] = batch_size
        embeddings = self.model.encode(texts, **encode_kwargs)
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """
        编码查询
        """
        self._ensure_model_loaded()
        embedding = self.model.encode([query], normalize_embeddings=True)
        return embedding[0]
