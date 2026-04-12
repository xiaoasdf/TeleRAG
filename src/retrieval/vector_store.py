import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.runtime import faiss_supports_gpu, get_compute_device, get_faiss_gpu_status


class VectorStore:
    def __init__(self, dim: int, device: str | None = None):
        self.dim = dim
        self.device = device or get_compute_device()
        self.index_backend = "cpu"
        self.index_backend_reason = "cpu_requested"
        self.gpu_resources = None
        self.index = self._build_index(dim)
        self.metadata: List[Dict] = []

    def _build_index(self, dim: int):
        faiss = _import_faiss()
        cpu_index = faiss.IndexFlatIP(dim)

        if self.device != "cuda":
            self.index_backend_reason = "device_not_cuda"
            return cpu_index

        gpu_supported = faiss_supports_gpu()
        gpu_reason = get_faiss_gpu_status()[1]
        if not gpu_supported:
            self.index_backend_reason = gpu_reason
            return cpu_index

        try:
            self.gpu_resources = faiss.StandardGpuResources()
            self.index_backend = "gpu"
            self.index_backend_reason = "gpu_enabled"
            return faiss.index_cpu_to_gpu(self.gpu_resources, 0, cpu_index)
        except Exception as exc:
            self.gpu_resources = None
            self.index_backend = "cpu"
            self.index_backend_reason = f"gpu_init_failed:{type(exc).__name__}"
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
        faiss = _import_faiss()
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
            "index_backend_reason": self.index_backend_reason,
            "metadata": self.metadata,
        }
        (path / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, input_dir: str | Path, device: str | None = None) -> "VectorStore":
        faiss = _import_faiss()
        path = Path(input_dir)
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        store = cls(dim=int(metadata["dim"]), device=device or metadata.get("device"))
        store.metadata = list(metadata.get("metadata", []))
        store.index_backend_reason = metadata.get("index_backend_reason", store.index_backend_reason)

        cpu_index = faiss.read_index(str(path / "index.faiss"))
        gpu_supported = faiss_supports_gpu()
        gpu_reason = get_faiss_gpu_status()[1]
        if store.device == "cuda" and gpu_supported:
            try:
                store.gpu_resources = faiss.StandardGpuResources()
                store.index = faiss.index_cpu_to_gpu(store.gpu_resources, 0, cpu_index)
                store.index_backend = "gpu"
                store.index_backend_reason = "gpu_enabled"
                return store
            except Exception as exc:
                store.index_backend_reason = f"gpu_load_failed:{type(exc).__name__}"
                pass

        store.index = cpu_index
        store.index_backend = "cpu"
        if store.device == "cuda" and not gpu_supported:
            store.index_backend_reason = gpu_reason
        return store


def _import_faiss():
    try:
        import faiss
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "The 'faiss' package is required for the local vector store. "
            "Install FAISS before loading or searching the persisted knowledge base."
        ) from exc

    return faiss
