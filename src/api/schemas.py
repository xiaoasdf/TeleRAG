from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IndexRequest(BaseModel):
    file_path: str | None = Field(default=None, description="Path to a local document file")
    file_paths: list[str] | None = Field(default=None, description="Paths to multiple local document files")
    chunk_size: int | None = None
    overlap: int | None = None
    persist: bool = True


class StandardsIndexRequest(BaseModel):
    download_first: bool = False
    source_orgs: list[str] | None = None
    limit: int | None = None
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
    rag_provider: str
    generation_backend: str
    api_key_detected: bool
    standards_root: str | None = None
    source_org_counts: dict[str, int] | None = None
    doc_failures: int | None = None
    faiss_gpu_status: str | None = None
    index_backend_reason: str | None = None
    embedding_batch_size: int | None = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[dict[str, Any]]
    retrieved_contexts: list[dict[str, Any]]
    timings: dict[str, float]
    config: dict[str, Any]


class IndexResponse(BaseModel):
    source_file: str
    source_count: int | None = None
    chunk_count: int
    persisted: bool
    vector_store_dir: str | None = None
    backend: str = "local"
    vector_store_id: str | None = None
    cloud_state_path: str | None = None
    uploaded_files: int | None = None
    cloud_index_status: str | None = None
    local_cleanup_applied: bool = False


class StandardsIndexResponse(BaseModel):
    source_file: str
    chunk_count: int
    persisted: bool
    vector_store_dir: str | None = None
    standards_root: str | None = None
    backend: str = "local"
    vector_store_id: str | None = None
    cloud_state_path: str | None = None
    uploaded_files: int | None = None
    cloud_index_status: str | None = None
    local_cleanup_applied: bool = False
    processed_sources: int
    new_or_updated_sources: int
    skipped_unchanged_sources: int
    failed_sources: int
    removed_sources: int
    extracted_documents: int
    index_ready_root: str
    state_path: str
    download_manifest_path: str | None = None
    processed_index_ready_files: int | None = None
    total_index_ready_files: int | None = None
    processed_batches: int | None = None
    current_stage: str | None = None
    progress_state_path: str | None = None
    index_backend: str | None = None
    index_backend_reason: str | None = None
    requires_rebuild: bool = False
    source_org_counts: dict[str, int] | None = None
    doc_failures: int | None = None
    device: str | None = None
    faiss_gpu_status: str | None = None
    embedding_batch_size: int | None = None
