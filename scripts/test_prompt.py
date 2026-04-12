from src.generation.prompt_builder import build_prompt


query = "What is beamforming?"

contexts = [
    {
        "chunk_id": "c1",
        "text": "Beamforming improves signal quality in wireless systems.",
        "source": "doc1.txt",
    },
    {
        "chunk_id": "c2",
        "text": "Dynamic TDD allows flexible uplink and downlink allocation.",
        "source": "doc2.txt",
    },
]

prompt = build_prompt(query, contexts)
print(prompt)