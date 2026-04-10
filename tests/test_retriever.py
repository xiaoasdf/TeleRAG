from src.retrieval.retriever import Retriever


def test_retriever_default_model_name():
    assert Retriever.__init__.__defaults__[0] == "BAAI/bge-m3"


def test_retriever_passes_device_to_dependencies(monkeypatch):
    captured = {"embedder": None, "vector_store": None}

    class FakeEmbedder:
        def __init__(self, model_name, device=None):
            captured["embedder"] = (model_name, device)

        def encode_texts(self, texts):
            import numpy as np

            return np.ones((len(texts), 4))

        def encode_query(self, query):
            import numpy as np

            return np.ones(4)

    class FakeVectorStore:
        def __init__(self, dim, device=None):
            captured["vector_store"] = (dim, device)
            self.index_backend = "gpu"

        def add(self, embeddings, metadata):
            self.items = metadata

        def search(self, query_vec, top_k=3):
            return [(self.items[0], 1.0)]

    monkeypatch.setattr("src.retrieval.retriever.Embedder", FakeEmbedder)
    monkeypatch.setattr("src.retrieval.retriever.VectorStore", FakeVectorStore)

    retriever = Retriever(device="cuda")
    retriever.build_index([{"chunk_id": "c1", "text": "beamforming", "source": "doc"}])
    retriever.retrieve("beamforming")

    assert captured["embedder"] == ("BAAI/bge-m3", "cuda")
    assert captured["vector_store"] == (4, "cuda")


def test_retriever():
    chunks = [
        {
            "chunk_id": "c1",
            "text": "Beamforming improves signal quality in wireless systems.",
            "source": "doc1.txt",
            "start_idx": 0,
            "end_idx": 60,
        },
        {
            "chunk_id": "c2",
            "text": "DVB-S2 is used in satellite communication.",
            "source": "doc2.txt",
            "start_idx": 0,
            "end_idx": 50,
        },
        {
            "chunk_id": "c3",
            "text": "Dynamic TDD allows flexible uplink and downlink allocation.",
            "source": "doc3.txt",
            "start_idx": 0,
            "end_idx": 70,
        },
    ]

    retriever = Retriever()
    retriever.build_index(chunks)

    results = retriever.retrieve("Explain beamforming in wireless communication", top_k=2)

    assert len(results) > 0
    assert "score" in results[0]
    assert "text" in results[0]
