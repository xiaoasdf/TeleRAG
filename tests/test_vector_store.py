from src.retrieval.embedder import Embedder
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
    texts = [
        "Beamforming improves signal quality in wireless systems.",
        "DVB-S2 is used in satellite communication.",
        "Dynamic TDD allows flexible uplink and downlink allocation.",
    ]

    metadata = [
        {"chunk_id": "c1", "text": texts[0], "source": "doc1.txt"},
        {"chunk_id": "c2", "text": texts[1], "source": "doc2.txt"},
        {"chunk_id": "c3", "text": texts[2], "source": "doc3.txt"},
    ]

    embedder = Embedder()
    embeddings = embedder.encode_texts(texts)

    store = VectorStore(dim=embeddings.shape[1])
    store.add(embeddings, metadata)

    query = "What is beamforming in wireless communication?"
    query_vec = embedder.encode_query(query)

    results = store.search(query_vec, top_k=2)

    assert len(results) > 0
    assert results[0][0]["chunk_id"] in {"c1", "c2", "c3"}
