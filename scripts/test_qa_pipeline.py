from src.pipeline.qa_pipeline import QAPipeline


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

print("=" * 60)
print("QUERY:")
print(result["query"])

print("=" * 60)
print("ANSWER:")
print(result["answer"])

print("=" * 60)
print("SOURCES:")
for item in result["sources"]:
    print(item)

print("=" * 60)
print("TOP CONTEXTS:")
for ctx in result["retrieved_contexts"]:
    print(f"{ctx['chunk_id']} | {ctx['source']} | score={ctx['score']:.4f}")
    print(ctx["text"])
    print("-" * 40)