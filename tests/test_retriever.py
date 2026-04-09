from src.retrieval.retriever import Retriever


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