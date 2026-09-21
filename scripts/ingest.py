"""One-off ingestion: chunk sample_notes/ and upsert into Qdrant.

Usage:
    python scripts/ingest.py
"""
from src.data.loader import load_notes
from src.retrieval.vector_store import VectorStore


def main() -> None:
    chunks = load_notes()
    store = VectorStore()
    store.upsert_chunks(chunks)
    print(f"Ingested {len(chunks)} chunks into collection '{store.collection}'.")


if __name__ == "__main__":
    main()
