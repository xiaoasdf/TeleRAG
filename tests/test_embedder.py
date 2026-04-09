from src.retrieval.embedder import Embedder


def test_embedder():
    embedder = Embedder()

    texts = [
        "Beamforming is important",
        "Wireless communication system",
    ]

    embeddings = embedder.encode_texts(texts)

    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] > 0