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
    assert "Answer the user's question using only the context below." in prompt
    assert "Ignore context that is unrelated to the user's question." in prompt
    assert "do not add unrelated topics or examples" in prompt
    assert "Write in complete sentences with natural sentence boundaries and proper punctuation." in prompt
    assert "Do not output sentence fragments" in prompt
    assert "Do not output special tokens" in prompt
    assert prompt.endswith("Answer:")
