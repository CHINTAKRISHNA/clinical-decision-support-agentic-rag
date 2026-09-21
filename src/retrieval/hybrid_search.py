"""Hybrid retrieval: combines dense (Qdrant) and sparse (BM25) search via reciprocal rank fusion."""
from __future__ import annotations

from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from src.config import settings
from src.data.loader import Chunk
from src.retrieval.vector_store import ScoredChunk, VectorStore


@dataclass
class HybridSearcher:
    vector_store: VectorStore
    chunks: list[Chunk]

    def __post_init__(self) -> None:
        tokenized = [c.text.lower().split() for c in self.chunks]
        self._bm25 = BM25Okapi(tokenized)
        self._chunk_by_id = {c.chunk_id: c for c in self.chunks}

    def _bm25_search(self, query: str, top_k: int) -> list[ScoredChunk]:
        scores = self._bm25.get_scores(query.lower().split())
        ranked = sorted(zip(self.chunks, scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            ScoredChunk(chunk_id=c.chunk_id, doc_id=c.doc_id, text=c.text, score=float(s))
            for c, s in ranked
            if s > 0
        ]

    def search(self, query: str, top_k: int | None = None) -> list[ScoredChunk]:
        top_k = top_k or settings.top_k_rerank
        dense_hits = self.vector_store.search(query, top_k=settings.top_k_dense)
        sparse_hits = self._bm25_search(query, top_k=settings.top_k_bm25)

        # Reciprocal rank fusion (k=60 is the standard RRF constant).
        rrf_scores: dict[str, float] = {}
        for rank, hit in enumerate(dense_hits):
            rrf_scores[hit.chunk_id] = rrf_scores.get(hit.chunk_id, 0.0) + 1.0 / (60 + rank)
        for rank, hit in enumerate(sparse_hits):
            rrf_scores[hit.chunk_id] = rrf_scores.get(hit.chunk_id, 0.0) + 1.0 / (60 + rank)

        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for chunk_id, score in fused:
            chunk = self._chunk_by_id.get(chunk_id)
            if chunk:
                results.append(ScoredChunk(chunk_id=chunk.chunk_id, doc_id=chunk.doc_id, text=chunk.text, score=score))
        return results
