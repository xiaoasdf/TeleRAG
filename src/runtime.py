from __future__ import annotations

from functools import lru_cache
from typing import Tuple


@lru_cache(maxsize=1)
def get_compute_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"

    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def faiss_supports_gpu() -> bool:
    supported, _ = get_faiss_gpu_status()
    return supported


@lru_cache(maxsize=1)
def get_faiss_gpu_status() -> Tuple[bool, str]:
    if get_compute_device() != "cuda":
        return False, "cuda_unavailable"

    try:
        import faiss
    except ImportError:
        return False, "faiss_not_installed"

    has_gpu_resources = hasattr(faiss, "StandardGpuResources")
    has_gpu_transfer = hasattr(faiss, "index_cpu_to_gpu")
    if has_gpu_resources and has_gpu_transfer:
        return True, "gpu_enabled"
    return False, "faiss_gpu_symbols_missing"
