import numpy as np

from src.retrieval.vector_store import VectorStore


def test_vector_store_falls_back_to_cpu_when_gpu_unavailable(monkeypatch):
    monkeypatch.setattr("src.retrieval.vector_store.faiss_supports_gpu", lambda: False)

    store = VectorStore(dim=8, device="cuda")

    assert store.index_backend == "cpu"


def test_vector_store_uses_gpu_backend_when_supported(monkeypatch):
    created = {"resources": 0, "moved": 0}

    class FakeGpuResources:
        def __init__(self):
            created["resources"] += 1

    def fake_index_cpu_to_gpu(resources, device_id, index):
        created["moved"] += 1
        return index

    monkeypatch.setattr("src.retrieval.vector_store.faiss_supports_gpu", lambda: True)
    monkeypatch.setattr(
        "src.retrieval.vector_store.faiss.StandardGpuResources",
        FakeGpuResources,
        raising=False,
    )
    monkeypatch.setattr(
        "src.retrieval.vector_store.faiss.index_cpu_to_gpu",
        fake_index_cpu_to_gpu,
        raising=False,
    )

    store = VectorStore(dim=8, device="cuda")

    assert store.index_backend == "gpu"
    assert created == {"resources": 1, "moved": 1}


def test_vector_store_search():
    metadata = [
        {"chunk_id": "c1", "text": "Beamforming improves signal quality.", "source": "doc1.txt"},
        {"chunk_id": "c2", "text": "DVB-S2 is used in satellite communication.", "source": "doc2.txt"},
        {"chunk_id": "c3", "text": "Dynamic TDD allows flexible allocation.", "source": "doc3.txt"},
    ]
    embeddings = np.asarray(
        [
            [1.0, 0.1, 0.0, 0.0],
            [0.1, 1.0, 0.0, 0.0],
            [0.2, 0.1, 0.9, 0.0],
        ],
        dtype="float32",
    )
    store = VectorStore(dim=4)
    store.add(embeddings, metadata)
    query_vec = np.asarray([1.0, 0.0, 0.0, 0.0], dtype="float32")

    results = store.search(query_vec, top_k=2)

    assert len(results) > 0
    assert results[0][0]["chunk_id"] == "c1"


def test_vector_store_save_and_load(tmp_path):
    metadata = [
        {"chunk_id": "c1", "text": "Beamforming improves signal quality.", "source": "doc1.txt"},
        {"chunk_id": "c2", "text": "DVB-S2 is used in satellite communication.", "source": "doc2.txt"},
    ]
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype="float32",
    )
    store = VectorStore(dim=embeddings.shape[1], device="cpu")
    store.add(embeddings, metadata)
    store.save(tmp_path)

    restored = VectorStore.load(tmp_path, device="cpu")
    assert restored.dim == store.dim
    assert len(restored.metadata) == 2
    assert restored.index_backend == "cpu"


def test_vector_store_supports_multiple_adds():
    store = VectorStore(dim=2, device="cpu")

    store.add(np.asarray([[1.0, 0.0]], dtype="float32"), [{"chunk_id": "c1", "text": "one", "source": "a.txt"}])
    store.add(np.asarray([[0.0, 1.0]], dtype="float32"), [{"chunk_id": "c2", "text": "two", "source": "b.txt"}])

    assert len(store.metadata) == 2
