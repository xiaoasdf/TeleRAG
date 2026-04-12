from src.runtime import get_compute_device


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str | None = None):
        self.model_name = model_name
        self.device = device or get_compute_device()
        self.model = None

    def _ensure_model_loaded(self) -> None:
        if self.model is None:
            try:
                from sentence_transformers import CrossEncoder
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "The 'sentence-transformers' package is required for reranking. "
                    "Install retrieval dependencies or disable rerank."
                ) from exc

            self.model = CrossEncoder(self.model_name, device=self.device)

    def rerank(self, query, docs):
        self._ensure_model_loaded()
        pairs = [(query, doc["text"]) for doc in docs]
        scores = self.model.predict(pairs)

        for doc, score in zip(docs, scores):
            doc["rerank_score"] = float(score)

        docs = sorted(docs, key=lambda x: x["rerank_score"], reverse=True)
        return docs
