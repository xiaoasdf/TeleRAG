from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.generation.llm_client import BALANCED_LLM_MODEL, DEFAULT_MAX_NEW_TOKENS, FAST_LLM_MODEL


@dataclass(slots=True)
class AppConfig:
    project_name: str = "TeleRAG"
    data_dir: str = "data"
    raw_data_dir: str = "data/raw"
    vector_store_dir: str = "data/vector_store/default"
    retriever_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    fast_llm_model: str = FAST_LLM_MODEL
    balanced_llm_model: str = BALANCED_LLM_MODEL
    default_llm_model: str = FAST_LLM_MODEL
    chunk_size: int = 300
    overlap: int = 50
    top_k: int = 3
    candidate_k: int = 5
    rerank_top_n: int = 3
    prompt_char_budget: int = 1200
    enable_rerank: bool = True
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS

    @property
    def vector_store_path(self) -> Path:
        return Path(self.vector_store_dir)

    @property
    def raw_data_path(self) -> Path:
        return Path(self.raw_data_dir)


def load_config(config_path: str | Path = "config/settings.yaml") -> AppConfig:
    path = Path(config_path)
    if not path.exists():
        return AppConfig()

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    flattened = _flatten_settings(data)
    return AppConfig(**flattened)


def _flatten_settings(data: dict[str, Any]) -> dict[str, Any]:
    pipeline = data.get("pipeline", {})
    models = data.get("models", {})
    paths = data.get("paths", {})
    app = data.get("app", {})

    return {
        "project_name": app.get("project_name", "TeleRAG"),
        "data_dir": paths.get("data_dir", "data"),
        "raw_data_dir": paths.get("raw_data_dir", "data/raw"),
        "vector_store_dir": paths.get("vector_store_dir", "data/vector_store/default"),
        "retriever_model": models.get("retriever", "BAAI/bge-m3"),
        "reranker_model": models.get("reranker", "BAAI/bge-reranker-v2-m3"),
        "fast_llm_model": models.get("fast_llm", FAST_LLM_MODEL),
        "balanced_llm_model": models.get("balanced_llm", BALANCED_LLM_MODEL),
        "default_llm_model": models.get("default_llm", FAST_LLM_MODEL),
        "chunk_size": pipeline.get("chunk_size", 300),
        "overlap": pipeline.get("overlap", 50),
        "top_k": pipeline.get("top_k", 3),
        "candidate_k": pipeline.get("candidate_k", 5),
        "rerank_top_n": pipeline.get("rerank_top_n", 3),
        "prompt_char_budget": pipeline.get("prompt_char_budget", 1200),
        "enable_rerank": pipeline.get("enable_rerank", True),
        "max_new_tokens": pipeline.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS),
    }
