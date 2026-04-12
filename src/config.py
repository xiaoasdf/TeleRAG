from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.generation.llm_client import BALANCED_LLM_MODEL, DEFAULT_MAX_NEW_TOKENS, FAST_LLM_MODEL


@dataclass(slots=True)
class AppConfig:
    project_name: str = "TeleRAG Communications Knowledge Assistant"
    rag_provider: str = "dashscope_compatible"
    data_dir: str = "data"
    raw_data_dir: str = "data/raw"
    vector_store_dir: str = "data/vector_store/default"
    standards_source_dir: str = "D:/project/pythoncrawler/downloads"
    retriever_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    fast_llm_model: str = FAST_LLM_MODEL
    balanced_llm_model: str = BALANCED_LLM_MODEL
    default_llm_model: str = FAST_LLM_MODEL
    dashscope_api_key_env: str = "DASHSCOPE_API_KEY"
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_model: str = "qwen-plus"
    chunk_size: int = 300
    overlap: int = 50
    top_k: int = 3
    candidate_k: int = 5
    rerank_top_n: int = 3
    prompt_char_budget: int = 1200
    enable_rerank: bool = True
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    embedding_batch_size: int = 16
    standards_batch_files: int = 10
    standards_save_every_batches: int = 2
    standards_resume_enabled: bool = True
    prefer_faiss_gpu: bool = True

    @property
    def vector_store_path(self) -> Path:
        return Path(self.vector_store_dir)

    @property
    def raw_data_path(self) -> Path:
        return Path(self.raw_data_dir)

    @property
    def standards_source_path(self) -> Path:
        return _resolve_local_path(self.standards_source_dir)

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
        "project_name": app.get("project_name", "TeleRAG Communications Knowledge Assistant"),
        "rag_provider": app.get("rag_provider", "dashscope_compatible"),
        "data_dir": paths.get("data_dir", "data"),
        "raw_data_dir": paths.get("raw_data_dir", "data/raw"),
        "vector_store_dir": paths.get("vector_store_dir", "data/vector_store/default"),
        "standards_source_dir": paths.get("standards_source_dir", "D:/project/pythoncrawler/downloads"),
        "retriever_model": models.get("retriever", "BAAI/bge-m3"),
        "reranker_model": models.get("reranker", "BAAI/bge-reranker-v2-m3"),
        "fast_llm_model": models.get("fast_llm", FAST_LLM_MODEL),
        "balanced_llm_model": models.get("balanced_llm", BALANCED_LLM_MODEL),
        "default_llm_model": models.get("default_llm", FAST_LLM_MODEL),
        "dashscope_api_key_env": models.get("dashscope_api_key_env", "DASHSCOPE_API_KEY"),
        "dashscope_base_url": models.get("dashscope_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "dashscope_model": models.get("dashscope_model", "qwen-plus"),
        "chunk_size": pipeline.get("chunk_size", 300),
        "overlap": pipeline.get("overlap", 50),
        "top_k": pipeline.get("top_k", 3),
        "candidate_k": pipeline.get("candidate_k", 5),
        "rerank_top_n": pipeline.get("rerank_top_n", 3),
        "prompt_char_budget": pipeline.get("prompt_char_budget", 1200),
        "enable_rerank": pipeline.get("enable_rerank", False),
        "max_new_tokens": pipeline.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS),
        "embedding_batch_size": pipeline.get("embedding_batch_size", 16),
        "standards_batch_files": pipeline.get("standards_batch_files", 10),
        "standards_save_every_batches": pipeline.get("standards_save_every_batches", 2),
        "standards_resume_enabled": pipeline.get("standards_resume_enabled", True),
        "prefer_faiss_gpu": pipeline.get("prefer_faiss_gpu", True),
    }


def _resolve_local_path(value: str | Path) -> Path:
    path = Path(value)
    if path.exists():
        return path

    raw = str(value).replace("\\", "/")
    match = re.match(r"^([A-Za-z]):/(.+)$", raw)
    if match and os.name != "nt":
        drive = match.group(1).lower()
        remainder = match.group(2)
        wsl_path = Path("/mnt") / drive / remainder
        if wsl_path.exists():
            return wsl_path

    return path
