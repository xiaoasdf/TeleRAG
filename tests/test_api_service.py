from pathlib import Path

from src.api.service import TeleRAGService
from src.config import AppConfig


def test_service_health_reports_uninitialized_state(tmp_path):
    config = AppConfig(vector_store_dir=str(tmp_path / "vs"))
    service = TeleRAGService(config=config)

    health = service.health()

    assert health["project_name"] == "TeleRAG"
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
