from src.retrieval.embedder import Embedder


def test_embedder_default_model_name():
    assert Embedder.__init__.__defaults__[0] == "BAAI/bge-m3"


def test_embedder_uses_device_and_lazy_loads(monkeypatch):
    calls = []

    class FakeModel:
        def encode(self, texts, normalize_embeddings=True):
            import numpy as np

            return np.ones((len(texts), 3))

    def fake_sentence_transformer(model_name, device=None):
        calls.append((model_name, device))
        return FakeModel()

    monkeypatch.setattr("src.retrieval.embedder.SentenceTransformer", fake_sentence_transformer)

    embedder = Embedder(device="cuda")
    assert embedder.model is None

    embeddings = embedder.encode_texts(["a", "b"])

    assert embeddings.shape == (2, 3)
    assert calls == [("BAAI/bge-m3", "cuda")]


def test_embedder():
    embedder = Embedder()

    texts = [
        "Beamforming is important",
        "Wireless communication system",
    ]

    embeddings = embedder.encode_texts(texts)

    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] > 0
