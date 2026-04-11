from __future__ import annotations

import logging
from pathlib import Path

from src.config import AppConfig, load_config
from src.pipeline.index_pipeline import build_chunks_from_file
from src.pipeline.qa_pipeline import QAPipeline
from src.runtime import get_compute_device


logger = logging.getLogger(__name__)


class TeleRAGService:
    def __init__(self, config: AppConfig | None = None):
        self.config = config or load_config()
        self.pipeline = QAPipeline(
            model_name=self.config.retriever_model,
            llm_model_name=self.config.default_llm_model,
            reranker_model_name=self.config.reranker_model,
            enable_rerank=self.config.enable_rerank,
            candidate_k=self.config.candidate_k,
            rerank_top_n=self.config.rerank_top_n,
            prompt_char_budget=self.config.prompt_char_budget,
            max_new_tokens=self.config.max_new_tokens,
        )
        self.source_file: str | None = None
        self.chunk_count = 0
        self._try_restore_index()

    def index_document(
        self,
        file_path: str,
        chunk_size: int | None = None,
        overlap: int | None = None,
        persist: bool = True,
    ) -> dict:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        logger.info("index_started file=%s", path)
        chunks = build_chunks_from_file(
            str(path),
            chunk_size=chunk_size or self.config.chunk_size,
            overlap=overlap or self.config.overlap,
        )
        self.pipeline.build_knowledge_base(chunks)
        self.source_file = path.name
        self.chunk_count = len(chunks)

        if persist:
            self.pipeline.save_knowledge_base(self.config.vector_store_path)

        logger.info("index_completed file=%s chunks=%s persisted=%s", path, self.chunk_count, persist)
        return {
            "source_file": self.source_file,
            "chunk_count": self.chunk_count,
            "persisted": persist,
            "vector_store_dir": str(self.config.vector_store_path),
        }

    def ask(
        self,
        query: str,
        top_k: int | None = None,
        enable_rerank: bool | None = None,
        max_new_tokens: int | None = None,
    ) -> dict:
        if not self.pipeline.is_ready:
            raise ValueError("Knowledge base has not been built yet")

        result = self.pipeline.ask(
            query=query,
            top_k=top_k or self.config.top_k,
            enable_rerank=enable_rerank,
            max_new_tokens=max_new_tokens,
        )
        logger.info(
            "query_completed total_ms=%s retrieve_ms=%s rerank_ms=%s generate_ms=%s",
            result["timings"]["total_ms"],
            result["timings"]["retrieve_ms"],
            result["timings"]["rerank_ms"],
            result["timings"]["generate_ms"],
        )
        return result

    def health(self) -> dict:
        return {
            "project_name": self.config.project_name,
            "device": get_compute_device(),
            "is_ready": self.pipeline.is_ready,
            "vector_backend": self.pipeline.retriever.vector_store.index_backend if self.pipeline.retriever.vector_store else "uninitialized",
            "llm_model_name": self.pipeline.llm_client.model_name,
            "has_persisted_index": (self.config.vector_store_path / "index.faiss").exists(),
        }

    def _try_restore_index(self) -> None:
        index_file = self.config.vector_store_path / "index.faiss"
        metadata_file = self.config.vector_store_path / "metadata.json"
        if not index_file.exists() or not metadata_file.exists():
            return

        try:
            self.pipeline.load_knowledge_base(self.config.vector_store_path)
            self.chunk_count = len(self.pipeline.retriever.chunks)
            logger.info("restored_index path=%s chunks=%s", self.config.vector_store_path, self.chunk_count)
        except Exception as exc:
            logger.warning("restore_index_failed path=%s error=%s", self.config.vector_store_path, exc)
