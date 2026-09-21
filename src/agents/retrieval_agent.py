"""Retrieval agent: hybrid search + rerank + prompt-injection filtering,
then generation grounded in the surviving chunks.
"""
from __future__ import annotations

from typing import Any

from src.guardrails.prompt_injection import scan
from src.llm import get_llm
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.reranker import Reranker

_reranker = Reranker()
_llm = get_llm()

PROMPT_TEMPLATE = """You are a clinical decision support assistant. Answer the question using ONLY \
the numbered context below. Every factual claim MUST end with the exact chunk id it came from, in \
the form [chunk_id], copied verbatim from the context. If the context does not answer the question, \
say so explicitly instead of guessing.

Context:
{context}

Question: {question}

Answer:"""


def build_retrieval_node(searcher: HybridSearcher):
    def retrieve(state: dict[str, Any]) -> dict[str, Any]:
        query = state.get("redacted_query") or state["query"]

        guard = scan(query)
        if guard.is_flagged:
            return {
                "flagged": True,
                "flag_reason": f"query matched injection patterns: {guard.matched_patterns}",
                "retrieved_chunks": [],
                "context_by_chunk_id": {},
                "answer": "This query could not be processed due to a safety check.",
            }

        candidates = searcher.search(query)
        reranked = _reranker.rerank(query, candidates)

        safe_chunks = [c for c in reranked if not scan(c.text).is_flagged]
        context_by_chunk_id = {c.chunk_id: c.text for c in safe_chunks}
        context_block = "\n".join(f"- [{c.chunk_id}] {c.text}" for c in safe_chunks)

        prompt = PROMPT_TEMPLATE.format(context=context_block, question=query)
        answer = _llm.generate(prompt)

        return {
            "retrieved_chunks": [c.chunk_id for c in safe_chunks],
            "context_by_chunk_id": context_by_chunk_id,
            "answer": answer,
        }

    return retrieve
