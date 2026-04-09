from typing import Dict, List


def split_text(
    text: str,
    source: str,
    chunk_size: int = 300,
    overlap: int = 50,
) -> List[Dict]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(
                {
                    "chunk_id": f"{source}_chunk_{chunk_id}",
                    "text": chunk_text,
                    "source": source,
                    "start_idx": start,
                    "end_idx": end,
                }
            )
            chunk_id += 1

        if end == len(text):
            break

        start = end - overlap

    return chunks