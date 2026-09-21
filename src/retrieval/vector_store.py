"""Qdrant-backed dense vector store for clinical note chunks."""
from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.data.loader import Chunk


@dataclass
class ScoredChunk:
    chunk_id: str
    doc_id: str
    text: str
    score: float


class VectorStore:
    def __init__(self, url: str | None = None, collection: str | None = None) -> None:
        self.client = QdrantClient(url=url or settings.qdrant_url)
        self.collection = collection or settings.qdrant_collection
        self.embedder = SentenceTransformer(settings.embedding_model)

    def ensure_collection(self) -> None:
        vector_size = self.embedder.get_sentence_embedding_dimension()
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        self.ensure_collection()
        vectors = self.embedder.encode([c.text for c in chunks], show_progress_bar=False)
        points = [
            PointStruct(
                id=i,
                vector=vector.tolist(),
                payload={"chunk_id": c.chunk_id, "doc_id": c.doc_id, "text": c.text},
            )
            for i, (c, vector) in enumerate(zip(chunks, vectors))
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query: str, top_k: int | None = None) -> list[ScoredChunk]:
        top_k = top_k or settings.top_k_dense
        query_vector = self.embedder.encode(query).tolist()
        hits = self.client.query_points(
            collection_name=self.collection, query=query_vector, limit=top_k
        ).points
        return [
            ScoredChunk(
                chunk_id=hit.payload["chunk_id"],
                doc_id=hit.payload["doc_id"],
                text=hit.payload["text"],
                score=hit.score,
            )
            for hit in hits
        ]
