from pathlib import Path
from typing import Dict, List

from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.runtime import get_compute_device


class Retriever:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None):
        self.device = device or get_compute_device()
        self.embedder = Embedder(model_name=model_name, device=self.device)
        self.vector_store = None
        self.chunks: List[Dict] = []

    def build_index(self, chunks: List[Dict]) -> None:
        if not chunks:
            raise ValueError("Chunks cannot be empty")

        self.start_index()
        self.add_chunks(chunks)

    def start_index(self) -> None:
        self.vector_store = None
        self.chunks = []

    def add_chunks(self, chunks: List[Dict], embedding_batch_size: int | None = None) -> None:
        if not chunks:
            return

        texts = [chunk["text"] for chunk in chunks]
        if embedding_batch_size is None:
            embeddings = self.embedder.encode_texts(texts)
        else:
            embeddings = self.embedder.encode_texts(texts, batch_size=embedding_batch_size)

        if self.vector_store is None:
            self.vector_store = VectorStore(dim=embeddings.shape[1], device=self.device)

        self.vector_store.add(embeddings, chunks)
        self.chunks.extend(chunks)

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

    def save_index(self, output_dir: str | Path) -> None:
        if self.vector_store is None:
            raise ValueError("Index has not been built yet")
        self.vector_store.save(output_dir)

    def load_index(self, input_dir: str | Path) -> None:
        self.vector_store = VectorStore.load(input_dir, device=self.device)
        self.chunks = list(self.vector_store.metadata)
