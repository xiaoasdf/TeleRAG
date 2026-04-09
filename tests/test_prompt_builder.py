from src.generation.prompt_builder import build_prompt


def test_build_prompt():
    query = "What is beamforming?"

    contexts = [
        {
            "chunk_id": "c1",
            "text": "Beamforming improves signal quality in wireless systems.",
            "source": "doc1.txt",
        },
        {
            "chunk_id": "c2",
            "text": "DVB-S2 is used in satellite communication.",
            "source": "doc2.txt",
        },
    ]

    prompt = build_prompt(query, contexts)

    assert "What is beamforming?" in prompt
    assert "Beamforming improves signal quality" in prompt
    assert "doc1.txt" in prompt
    assert "[Answer]" in prompt