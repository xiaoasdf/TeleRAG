from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def get_compute_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"

    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def faiss_supports_gpu() -> bool:
    if get_compute_device() != "cuda":
        return False

    try:
        import faiss
    except ImportError:
        return False

    return hasattr(faiss, "StandardGpuResources") and hasattr(faiss, "index_cpu_to_gpu")
