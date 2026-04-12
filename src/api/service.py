from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Iterable

from src.config import AppConfig, load_config
from src.pipeline.index_pipeline import build_chunks_from_file
from src.pipeline.qa_pipeline import QAPipeline
from src.runtime import get_compute_device, get_faiss_gpu_status
from src.standards.build_state import init_build_state, save_build_state
from src.standards.downloader import build_download_jobs, download_jobs, load_targets
from src.standards.ingest import stage_standard_sources


logger = logging.getLogger(__name__)


def _vector_store_exists(path: Path) -> bool:
    return (path / "index.faiss").exists() and (path / "metadata.json").exists()


class TeleRAGService:
    def __init__(self, config: AppConfig | None = None):
        self.config = config or load_config()
        self.pipeline = self._build_pipeline()
        self.source_file: str | None = None
        self.chunk_count = 0
        self._try_restore_index()

    def index_document(
        self,
        file_paths: list[str],
        chunk_size: int | None = None,
        overlap: int | None = None,
        persist: bool = True,
    ) -> dict:
        if not file_paths:
            raise ValueError("file_paths cannot be empty")

        paths = [Path(file_path) for file_path in file_paths]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"File not found: {missing[0]}")

        logger.info("index_started files=%s", [str(path) for path in paths])
        if _vector_store_exists(self.config.vector_store_path):
            self.pipeline.load_knowledge_base(self.config.vector_store_path)
        chunks = []
        for path in paths:
            chunks.extend(
                build_chunks_from_file(
                    str(path),
                    chunk_size=chunk_size or self.config.chunk_size,
                    overlap=overlap or self.config.overlap,
                )
            )
        if self.pipeline.is_ready:
            self.pipeline.add_knowledge_chunks(chunks)
        else:
            self.pipeline.build_knowledge_base(chunks)
        sources = []
        seen = set()
        for chunk in self.pipeline.retriever.chunks:
            source = chunk.get("source")
            if source and source not in seen:
                seen.add(source)
                sources.append(source)
        self.source_file = sources[0] if len(sources) == 1 else f"{len(sources)} sources"
        self.chunk_count = len(self.pipeline.retriever.chunks)

        if persist:
            self.pipeline.save_knowledge_base(self.config.vector_store_path)

        logger.info(
            "index_completed files=%s added_chunks=%s total_chunks=%s persisted=%s",
            [str(path) for path in paths],
            len(chunks),
            self.chunk_count,
            persist,
        )
        return {
            "source_file": self.source_file,
            "source_count": len(sources),
            "chunk_count": self.chunk_count,
            "persisted": persist,
            "vector_store_dir": str(self.config.vector_store_path),
        }

    def index_standards(
        self,
        *,
        download_first: bool = False,
        source_orgs: Iterable[str] | None = None,
        limit: int | None = None,
        persist: bool = True,
    ) -> dict:
        download_manifest_path = None
        if download_first and self.config.standards_source_path == (self.config.raw_data_path / "standards"):
            download_manifest_path = self._download_standards(source_orgs=source_orgs, limit=limit)

        standards_root = self.config.standards_source_path
        index_ready_root = self.config.raw_data_path / "index_ready" / "standards"
        state_path = self.config.raw_data_path / "index_ready" / "standards_ingest_state.json"
        build_state_path = self.config.raw_data_path / "index_ready" / "standards_build_state.json"
        self._reset_standards_artifacts(
            index_ready_root=index_ready_root,
            state_path=state_path,
            build_state_path=build_state_path,
        )
        staging_result = stage_standard_sources(
            standards_root,
            index_ready_root,
            state_path,
            source_orgs=source_orgs,
            limit=limit,
        )
        index_ready_files = [Path(path) for path in staging_result["index_ready_files"]]
        if not index_ready_files:
            raise ValueError("No standards documents are available for indexing")

        current_rel_paths = [str(path.relative_to(index_ready_root)) for path in index_ready_files]
        batch_size = max(self.config.standards_batch_files, 1)
        save_every_batches = max(self.config.standards_save_every_batches, 1)
        self.pipeline = self._build_pipeline()
        self.pipeline.start_knowledge_base()
        self.chunk_count = 0

        build_state = init_build_state(
            total_files=len(current_rel_paths),
            batch_size=batch_size,
            vector_store_dir=str(self.config.vector_store_path),
            resume_enabled=False,
        )
        build_state["requires_rebuild"] = True
        build_state["rebuild_reason"] = "external_source_full_rebuild"
        build_state["status"] = "running"
        build_state["source_root"] = str(standards_root)
        save_build_state(build_state_path, build_state)

        for start in range(0, len(index_ready_files), batch_size):
            batch_paths = index_ready_files[start : start + batch_size]
            batch_rel_paths = [str(path.relative_to(index_ready_root)) for path in batch_paths]

            build_state["current_stage"] = "chunking"
            save_build_state(build_state_path, build_state)
            batch_chunks = []
            for path in batch_paths:
                batch_chunks.extend(
                    build_chunks_from_file(
                        str(path),
                        chunk_size=self.config.chunk_size,
                        overlap=self.config.overlap,
                    )
                )

            build_state["current_stage"] = "embedding"
            save_build_state(build_state_path, build_state)
            self.pipeline.add_knowledge_chunks(
                batch_chunks,
                embedding_batch_size=self.config.embedding_batch_size,
            )
            self.chunk_count = len(self.pipeline.retriever.chunks)

            build_state["processed_files"] += len(batch_rel_paths)
            build_state["processed_batches"] += 1
            build_state["accumulated_chunks"] = self.chunk_count
            build_state["completed_files"].extend(batch_rel_paths)
            build_state["last_completed_file"] = batch_rel_paths[-1]
            build_state["device"] = self.pipeline.device
            build_state["embedding_batch_size"] = self.config.embedding_batch_size
            build_state["faiss_gpu_status"] = get_faiss_gpu_status()[1]
            if self.pipeline.retriever.vector_store:
                build_state["index_backend"] = self.pipeline.retriever.vector_store.index_backend
                build_state["index_backend_reason"] = getattr(
                    self.pipeline.retriever.vector_store,
                    "index_backend_reason",
                    build_state.get("index_backend_reason"),
                )

            if persist and (
                build_state["processed_batches"] % save_every_batches == 0
                or build_state["processed_files"] == len(current_rel_paths)
            ):
                build_state["current_stage"] = "saving"
                save_build_state(build_state_path, build_state)
                self.pipeline.save_knowledge_base(self.config.vector_store_path)

            build_state["current_stage"] = "running"
            save_build_state(build_state_path, build_state)

        if self.pipeline.retriever.vector_store and persist and build_state["current_stage"] != "saving":
            self.pipeline.save_knowledge_base(self.config.vector_store_path)

        if self.pipeline.retriever.vector_store:
            build_state["index_backend"] = self.pipeline.retriever.vector_store.index_backend
            build_state["index_backend_reason"] = getattr(
                self.pipeline.retriever.vector_store,
                "index_backend_reason",
                build_state.get("index_backend_reason"),
            )
        build_state["status"] = "completed"
        build_state["current_stage"] = "completed"
        build_state["processed_files"] = len(build_state.get("completed_files", []))
        build_state["total_files"] = len(current_rel_paths)
        build_state["accumulated_chunks"] = self.chunk_count
        build_state["device"] = self.pipeline.device
        build_state["embedding_batch_size"] = self.config.embedding_batch_size
        build_state["faiss_gpu_status"] = get_faiss_gpu_status()[1]
        save_build_state(build_state_path, build_state)
        self.source_file = "communications_standards_corpus"
        source_org_counts = self._build_source_org_counts(state_path)
        doc_failures = self._count_doc_failures(state_path)

        return {
            "source_file": self.source_file or "communications_standards_corpus",
            "chunk_count": self.chunk_count,
            "persisted": persist,
            "vector_store_dir": str(self.config.vector_store_path),
            "standards_root": str(standards_root),
            "processed_sources": staging_result["processed_sources"],
            "new_or_updated_sources": staging_result["new_or_updated_sources"],
            "skipped_unchanged_sources": staging_result["skipped_unchanged_sources"],
            "failed_sources": staging_result["failed_sources"],
            "removed_sources": staging_result["removed_sources"],
            "extracted_documents": staging_result["extracted_documents"],
            "index_ready_root": str(index_ready_root),
            "state_path": staging_result["state_path"],
            "download_manifest_path": download_manifest_path,
            "processed_index_ready_files": build_state["processed_files"],
            "total_index_ready_files": build_state["total_files"],
            "processed_batches": build_state["processed_batches"],
            "current_stage": build_state["current_stage"],
            "progress_state_path": str(build_state_path),
            "index_backend": build_state.get("index_backend"),
            "index_backend_reason": build_state.get("index_backend_reason"),
            "requires_rebuild": build_state.get("requires_rebuild", False),
            "source_org_counts": source_org_counts,
            "doc_failures": doc_failures,
            "device": self.pipeline.device,
            "faiss_gpu_status": get_faiss_gpu_status()[1],
            "embedding_batch_size": self.config.embedding_batch_size,
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
        faiss_supported, faiss_reason = get_faiss_gpu_status()
        index_backend_reason = None
        if self.pipeline.is_ready and self.pipeline.retriever.vector_store:
            index_backend_reason = self.pipeline.retriever.vector_store.index_backend_reason
        elif not faiss_supported:
            index_backend_reason = faiss_reason

        return {
            "project_name": self.config.project_name,
            "device": get_compute_device(),
            "is_ready": self.pipeline.is_ready,
            "vector_backend": self.pipeline.vector_backend if self.pipeline.is_ready else "uninitialized",
            "llm_model_name": self.pipeline.llm_client.model_name,
            "has_persisted_index": (self.config.vector_store_path / "index.faiss").exists(),
            "rag_provider": self.config.rag_provider,
            "generation_backend": self.pipeline.generation_backend,
            "api_key_detected": self._api_key_detected(),
            "standards_root": str(self.config.standards_source_path),
            "source_org_counts": self._build_source_org_counts(self.config.raw_data_path / "index_ready" / "standards_ingest_state.json"),
            "doc_failures": self._count_doc_failures(self.config.raw_data_path / "index_ready" / "standards_ingest_state.json"),
            "faiss_gpu_status": faiss_reason,
            "index_backend_reason": index_backend_reason,
            "embedding_batch_size": self.config.embedding_batch_size,
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

    def _build_pipeline(self) -> QAPipeline:
        return QAPipeline(
            model_name=self.config.retriever_model,
            llm_model_name=self.config.dashscope_model if self.config.rag_provider == "dashscope_compatible" else self.config.default_llm_model,
            reranker_model_name=self.config.reranker_model,
            enable_rerank=self.config.enable_rerank,
            candidate_k=self.config.candidate_k,
            rerank_top_n=self.config.rerank_top_n,
            prompt_char_budget=self.config.prompt_char_budget,
            max_new_tokens=self.config.max_new_tokens,
            generation_backend=self.config.rag_provider,
            compatible_api_key_env=self.config.dashscope_api_key_env,
            compatible_base_url=self.config.dashscope_base_url,
        )

    def _reset_standards_artifacts(
        self,
        *,
        index_ready_root: Path,
        state_path: Path,
        build_state_path: Path,
    ) -> None:
        self.pipeline = self._build_pipeline()
        self.source_file = None
        self.chunk_count = 0

        targets = [
            self.config.vector_store_path,
            self.config.vector_store_path.parent / "uploads",
            index_ready_root,
            state_path,
            build_state_path,
        ]
        if self.config.standards_source_path != (self.config.raw_data_path / "standards"):
            targets.append(self.config.raw_data_path / "standards")

        for target in targets:
            self._delete_path(target)

    def _delete_path(self, path: Path) -> None:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=False)
        elif path.exists():
            path.unlink()

    def _build_source_org_counts(self, state_path: Path) -> dict[str, int]:
        state = self._load_json(state_path)
        counts: dict[str, int] = {}
        for record in state.get("records", []):
            source_org = record.get("source_org", "unknown")
            counts[source_org] = counts.get(source_org, 0) + 1
        return dict(sorted(counts.items()))

    def _count_doc_failures(self, state_path: Path) -> int:
        state = self._load_json(state_path)
        failures = 0
        for record in state.get("records", []):
            rel_path = str(record.get("source_rel_path", "")).lower()
            if rel_path.endswith(".doc") and record.get("status") == "failed":
                failures += 1
        return failures

    def _load_json(self, path: Path) -> dict:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _download_standards(self, *, source_orgs: Iterable[str] | None, limit: int | None) -> str:
        project_root = Path(__file__).resolve().parents[2]
        targets_path = project_root / "config" / "standards_targets.json"
        manifest_path = self.config.raw_data_path / "standards" / "download_manifest.json"
        targets = load_targets(targets_path)
        if source_orgs:
            allowed = {item.lower() for item in source_orgs}
            filtered_targets = []
            for item in targets:
                source_org = item["source_org"].lower()
                if source_org in allowed:
                    filtered_targets.append(item)
                    continue
                if "itu" in allowed and source_org.startswith("itu-"):
                    filtered_targets.append(item)
            targets = filtered_targets

        jobs = build_download_jobs(targets)
        download_jobs(
            jobs,
            output_root=self.config.raw_data_path,
            manifest_path=manifest_path,
            limit=limit,
        )
        return str(manifest_path)

    def _api_key_detected(self) -> bool:
        import os

        if self.config.rag_provider != "dashscope_compatible":
            return True
        return bool(os.getenv(self.config.dashscope_api_key_env))
