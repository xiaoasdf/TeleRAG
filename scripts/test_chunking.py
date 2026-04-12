from src.ingestion.loaders import load_document
from src.ingestion.chunker import split_text

doc = load_document("data/raw/wireless_systems_overview.md")
chunks = split_text(doc["text"], doc["source"], chunk_size=30, overlap=10)

for chunk in chunks:
    print("=" * 40)
    print(chunk["chunk_id"])
    print(chunk["text"])
