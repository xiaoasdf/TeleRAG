def test_index_standards_endpoint_returns_summary(monkeypatch):
    import importlib

    from fastapi.testclient import TestClient

    def fake_index_standards(**kwargs):
        assert kwargs["download_first"] is True
        assert kwargs["source_orgs"] == ["3GPP"]
        return {
            "source_file": "communications_standards_corpus",
            "chunk_count": 42,
            "persisted": True,
            "vector_store_dir": "data/vector_store/default",
            "processed_sources": 4,
            "new_or_updated_sources": 2,
            "skipped_unchanged_sources": 2,
            "failed_sources": 0,
            "removed_sources": 0,
            "extracted_documents": 4,
            "index_ready_root": "data/raw/index_ready/standards",
            "state_path": "data/raw/index_ready/standards_ingest_state.json",
            "download_manifest_path": "data/raw/standards/download_manifest.json",
            "processed_index_ready_files": 4,
            "total_index_ready_files": 4,
            "processed_batches": 1,
            "current_stage": "completed",
            "progress_state_path": "data/raw/index_ready/standards_build_state.json",
            "index_backend": "gpu",
            "requires_rebuild": False,
        }

    class FakeService:
        def index_standards(self, **kwargs):
            return fake_index_standards(**kwargs)

        def health(self):
            return {
                "project_name": "TeleRAG Communications Knowledge Assistant",
                "device": "cpu",
                "is_ready": False,
                "vector_backend": "uninitialized",
                "llm_model_name": "fake-model",
                "has_persisted_index": False,
            }

    monkeypatch.setattr("src.api.service.TeleRAGService", lambda: FakeService())

    import src.api.app as api_app_module

    api_app_module = importlib.reload(api_app_module)
    client = TestClient(api_app_module.app)

    response = client.post(
        "/index/standards",
        json={"download_first": True, "source_orgs": ["3GPP"], "persist": True},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["chunk_count"] == 42
    assert body["processed_sources"] == 4
    assert body["download_manifest_path"] == "data/raw/standards/download_manifest.json"
    assert body["index_backend"] == "gpu"
