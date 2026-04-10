from src.pipeline.qa_pipeline import QAPipeline


def test_qa_pipeline():
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

    pipeline = QAPipeline()
    pipeline.build_knowledge_base(chunks)

    result = pipeline.ask("What is beamforming?", top_k=2)

    assert "answer" in result
    assert "sources" in result
    assert "retrieved_contexts" in result
    assert len(result["sources"]) > 0