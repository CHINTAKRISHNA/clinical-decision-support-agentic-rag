"""Cross-encoder reranking of hybrid retrieval candidates (bge-reranker-large by default)."""
from __future__ import annotations

from sentence_transformers import CrossEncoder

from src.config import settings
from src.retrieval.vector_store import ScoredChunk


class Reranker:
    def __init__(self, model_name: str | None = None) -> None:
        self.model = CrossEncoder(model_name or settings.reranker_model)

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int | None = None) -> list[ScoredChunk]:
        top_k = top_k or settings.top_k_rerank
        if not candidates:
            return []
        pairs = [(query, c.text) for c in candidates]
        scores = self.model.predict(pairs)
        reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            ScoredChunk(chunk_id=c.chunk_id, doc_id=c.doc_id, text=c.text, score=float(s))
            for c, s in reranked
        ]
