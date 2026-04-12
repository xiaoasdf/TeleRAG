from pathlib import Path

from src.api.service import TeleRAGService
from src.config import AppConfig


def test_service_health_reports_uninitialized_state(tmp_path):
    config = AppConfig(vector_store_dir=str(tmp_path / "vs"))
    service = TeleRAGService(config=config)

    health = service.health()

    assert health["project_name"] == "TeleRAG Communications Knowledge Assistant"
    assert health["is_ready"] is False
    assert health["vector_backend"] == "uninitialized"
    assert health["has_persisted_index"] is False


def test_service_restores_existing_index(monkeypatch, tmp_path):
    restored = {"called": False}

    def fake_load(self, path):
        restored["called"] = True
        self.is_ready = True
        self.retriever.chunks = [{"chunk_id": "c1", "text": "beamforming", "source": "doc1.txt"}]
        self.retriever.vector_store = type("Store", (), {"index_backend": "cpu"})()

    monkeypatch.setattr("src.api.service.QAPipeline.load_knowledge_base", fake_load)

    vector_dir = tmp_path / "vs"
    vector_dir.mkdir(parents=True)
    (vector_dir / "index.faiss").write_bytes(b"placeholder")
    (vector_dir / "metadata.json").write_text("{}", encoding="utf-8")

    service = TeleRAGService(config=AppConfig(vector_store_dir=str(vector_dir)))

    assert restored["called"] is True
    assert service.pipeline.is_ready is True
    assert service.chunk_count == 1


def test_service_index_standards_builds_corpus(monkeypatch, tmp_path):
    class FakePipeline:
        def __init__(self, *args, **kwargs):
            self.is_ready = False
            self.saved_to = None
            self.retriever = type("Retriever", (), {"chunks": [], "vector_store": None})()
            self.llm_client = type("LLM", (), {"model_name": "fake-model"})()

        def build_knowledge_base(self, chunks):
            self.is_ready = True
            self.retriever.chunks = chunks
            self.retriever.vector_store = type("Store", (), {"index_backend": "cpu"})()

        def start_knowledge_base(self):
            self.is_ready = False
            self.retriever.chunks = []
            self.retriever.vector_store = None

        def add_knowledge_chunks(self, chunks, embedding_batch_size=None):
            self.is_ready = True
            self.retriever.chunks.extend(chunks)
            self.retriever.vector_store = type("Store", (), {"index_backend": "cpu"})()

        def save_knowledge_base(self, path):
            self.saved_to = path

        def load_knowledge_base(self, path):
            self.is_ready = True
            self.retriever.chunks = [{"chunk_id": "loaded"}]
            self.retriever.vector_store = type("Store", (), {"index_backend": "cpu"})()

    monkeypatch.setattr("src.api.service.QAPipeline", FakePipeline)
    monkeypatch.setattr(
        "src.api.service.stage_standard_sources",
        lambda *args, **kwargs: {
            "processed_sources": 2,
            "new_or_updated_sources": 1,
            "skipped_unchanged_sources": 1,
            "failed_sources": 0,
            "removed_sources": 0,
            "extracted_documents": 2,
            "index_ready_files": [
                str(tmp_path / "data" / "raw" / "index_ready" / "standards" / "doc1.txt"),
                str(tmp_path / "data" / "raw" / "index_ready" / "standards" / "doc2.txt"),
            ],
            "state_path": str(tmp_path / "data" / "raw" / "index_ready" / "standards_ingest_state.json"),
        },
    )
    monkeypatch.setattr("src.api.service.compute_sha256", lambda path: f"hash-{Path(path).name}")
    monkeypatch.setattr(
        "src.api.service.build_chunks_from_file",
        lambda path, chunk_size, overlap: [{"chunk_id": Path(path).stem, "text": "wireless", "source": Path(path).name}],
    )

    config = AppConfig(
        raw_data_dir=str(tmp_path / "data" / "raw"),
        vector_store_dir=str(tmp_path / "data" / "vector_store" / "default"),
    )
    service = TeleRAGService(config=config)

    result = service.index_standards(persist=True)

    assert result["source_file"] == "communications_standards_corpus"
    assert result["processed_sources"] == 2
    assert result["chunk_count"] == 2
    assert result["persisted"] is True
    assert result["processed_index_ready_files"] == 2
    assert result["total_index_ready_files"] == 2
    assert result["processed_batches"] == 1
    assert result["index_backend"] == "cpu"
