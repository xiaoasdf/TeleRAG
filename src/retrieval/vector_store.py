import json
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np

from src.runtime import faiss_supports_gpu, get_compute_device


class VectorStore:
    def __init__(self, dim: int, device: str | None = None):
        self.dim = dim
        self.device = device or get_compute_device()
        self.index_backend = "cpu"
        self.gpu_resources = None
        self.index = self._build_index(dim)
        self.metadata: List[Dict] = []

    def _build_index(self, dim: int):
        cpu_index = faiss.IndexFlatIP(dim)

        if self.device != "cuda" or not faiss_supports_gpu():
            return cpu_index

        try:
            self.gpu_resources = faiss.StandardGpuResources()
            self.index_backend = "gpu"
            return faiss.index_cpu_to_gpu(self.gpu_resources, 0, cpu_index)
        except Exception:
            self.gpu_resources = None
            self.index_backend = "cpu"
            return cpu_index

    def add(self, embeddings: np.ndarray, metadata: List[Dict]) -> None:
        if len(embeddings) != len(metadata):
            raise ValueError("Embeddings and metadata must have the same length")

        embeddings = np.asarray(embeddings, dtype="float32")

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")

        if embeddings.shape[1] != self.dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dim}, got {embeddings.shape[1]}"
            )

        self.index.add(embeddings)
        self.metadata.extend(metadata)

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[Tuple[Dict, float]]:
        query_vector = np.asarray(query_vector, dtype="float32").reshape(1, -1)

        if query_vector.shape[1] != self.dim:
            raise ValueError(
                f"Query dimension mismatch: expected {self.dim}, got {query_vector.shape[1]}"
            )

        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append((self.metadata[idx], float(score)))

        return results

    def save(self, output_dir: str | Path) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        cpu_index = self.index
        if self.index_backend == "gpu" and hasattr(faiss, "index_gpu_to_cpu"):
            cpu_index = faiss.index_gpu_to_cpu(self.index)

        faiss.write_index(cpu_index, str(path / "index.faiss"))
        metadata = {
            "dim": self.dim,
            "device": self.device,
            "index_backend": self.index_backend,
            "metadata": self.metadata,
        }
        (path / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, input_dir: str | Path, device: str | None = None) -> "VectorStore":
        path = Path(input_dir)
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        store = cls(dim=int(metadata["dim"]), device=device or metadata.get("device"))
        store.metadata = list(metadata.get("metadata", []))

        cpu_index = faiss.read_index(str(path / "index.faiss"))
        if store.device == "cuda" and faiss_supports_gpu():
            try:
                store.gpu_resources = faiss.StandardGpuResources()
                store.index = faiss.index_cpu_to_gpu(store.gpu_resources, 0, cpu_index)
                store.index_backend = "gpu"
                return store
            except Exception:
                pass

        store.index = cpu_index
        store.index_backend = "cpu"
        return store
