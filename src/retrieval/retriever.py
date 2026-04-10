from typing import Dict, List

from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore


class Retriever:
    def __init__(self, model_name: str = "BAAI/bge-large-en"):
        self.embedder = Embedder(model_name=model_name)
        self.vector_store = None
        self.chunks: List[Dict] = []

    def build_index(self, chunks: List[Dict]) -> None:
        if not chunks:
            raise ValueError("Chunks cannot be empty")

        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedder.encode_texts(texts)

        self.vector_store = VectorStore(dim=embeddings.shape[1])
        self.vector_store.add(embeddings, chunks)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        if self.vector_store is None:
            raise ValueError("Index has not been built yet")

        query_vec = self.embedder.encode_query(query)
        results = self.vector_store.search(query_vec, top_k=top_k)

        formatted_results = []
        for item, score in results:
            result = item.copy()
            result["score"] = score
            formatted_results.append(result)

        return formatted_results