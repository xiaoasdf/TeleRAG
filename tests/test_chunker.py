from src.ingestion.chunker import split_text


def test_split_text_basic():
    text = "a" * 1000
    chunks = split_text(text=text, source="test.txt", chunk_size=300, overlap=50)

    assert len(chunks) > 0
    assert chunks[0]["source"] == "test.txt"
    assert "chunk_id" in chunks[0]
    assert "text" in chunks[0]


def test_split_text_empty():
    chunks = split_text(text="   ", source="empty.txt")
    assert chunks == []