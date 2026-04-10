from typing import List

from src.ingestion.loaders import load_document
from src.ingestion.chunker import split_text


def build_chunks_from_file(file_path: str, chunk_size=300, overlap=50):
    doc = load_document(file_path)

    chunks = split_text(
        text=doc["text"],
        source=doc["source"],
        chunk_size=chunk_size,
        overlap=overlap,
    )

    return chunks