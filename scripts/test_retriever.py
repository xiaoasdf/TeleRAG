from src.retrieval.retriever import Retriever


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

results = retriever.retrieve("What is beamforming?", top_k=3)

for item in results:
    print("=" * 50)
    print("score:", round(item["score"], 4))
    print("chunk_id:", item["chunk_id"])
    print("source:", item["source"])
    print("text:", item["text"])