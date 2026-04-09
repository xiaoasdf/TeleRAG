from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore


texts = [
    "Beamforming improves signal quality in wireless systems.",
    "DVB-S2 is used in satellite communication.",
    "Dynamic TDD allows flexible uplink and downlink allocation.",
]

metadata = [
    {"chunk_id": "c1", "text": texts[0], "source": "doc1.txt"},
    {"chunk_id": "c2", "text": texts[1], "source": "doc2.txt"},
    {"chunk_id": "c3", "text": texts[2], "source": "doc3.txt"},
]

embedder = Embedder()
embeddings = embedder.encode_texts(texts)

store = VectorStore(dim=embeddings.shape[1])
store.add(embeddings, metadata)

query = "Explain beamforming"
query_vec = embedder.encode_query(query)

results = store.search(query_vec, top_k=3)

for item, score in results:
    print("=" * 50)
    print("score:", round(score, 4))
    print("chunk_id:", item["chunk_id"])
    print("source:", item["source"])
    print("text:", item["text"])