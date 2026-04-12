import numpy as np

from src.retrieval.retriever import Retriever


def test_retriever_add_chunks_appends_to_existing_index(monkeypatch):
    class FakeEmbedder:
        def __init__(self, model_name, device=None):
            self.calls = []

        def encode_texts(self, texts, batch_size=None):
            self.calls.append((list(texts), batch_size))
            return np.asarray([[float(index + 1), 0.0] for index, _ in enumerate(texts)], dtype="float32")

    monkeypatch.setattr("src.retrieval.retriever.Embedder", FakeEmbedder)

    retriever = Retriever(device="cpu")
    retriever.start_index()
    retriever.add_chunks([{"chunk_id": "c1", "text": "alpha", "source": "a.txt"}], embedding_batch_size=8)
    retriever.add_chunks([{"chunk_id": "c2", "text": "beta", "source": "b.txt"}], embedding_batch_size=8)

    assert retriever.vector_store is not None
    assert len(retriever.chunks) == 2
    assert len(retriever.vector_store.metadata) == 2
