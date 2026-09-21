from src.data.loader import chunk_text


def test_chunk_text_produces_unique_stable_ids():
    text = " ".join(f"word{i}" for i in range(20))
    chunks = chunk_text("doc-1", text, chunk_size=10, overlap=2)

    assert len(chunks) >= 2
    assert len(chunks) == len({c.chunk_id for c in chunks})
    assert all(c.doc_id == "doc-1" for c in chunks)


def test_chunk_text_ids_are_deterministic():
    text = "alpha beta gamma delta"
    first = chunk_text("doc-1", text, chunk_size=2, overlap=0)
    second = chunk_text("doc-1", text, chunk_size=2, overlap=0)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
