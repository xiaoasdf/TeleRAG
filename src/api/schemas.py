from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IndexRequest(BaseModel):
    file_path: str = Field(..., description="Path to a local document file")
    chunk_size: int | None = None
    overlap: int | None = None
    persist: bool = True


class QueryRequest(BaseModel):
    query: str
    top_k: int | None = None
    enable_rerank: bool | None = None
    max_new_tokens: int | None = None


class HealthResponse(BaseModel):
    project_name: str
    device: str
    is_ready: bool
    vector_backend: str
    llm_model_name: str
    has_persisted_index: bool


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[dict[str, Any]]
    retrieved_contexts: list[dict[str, Any]]
    timings: dict[str, float]
    config: dict[str, Any]


class IndexResponse(BaseModel):
    source_file: str
    chunk_count: int
    persisted: bool
    vector_store_dir: str
