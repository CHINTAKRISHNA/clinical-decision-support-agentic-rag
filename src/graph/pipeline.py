"""LangGraph orchestration wiring: Redaction -> Retrieval/Generation -> Verification -> Redaction.

    query
      |
      v
  [redact_input]  (Presidio strips PHI before it reaches the LLM)
      |
      v
  [retrieve]      (hybrid search -> rerank -> guardrail filter -> grounded generation)
      |
      v
  [verify]        (every citation checked against the retrieved context)
      |
      v
  [redact_output] (Presidio strips PHI from the final answer)
      |
      v
    result
"""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.agents.redaction_agent import redact_input, redact_output
from src.agents.retrieval_agent import build_retrieval_node
from src.agents.verification_agent import verify
from src.data.loader import load_notes
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.vector_store import VectorStore


class PipelineState(TypedDict, total=False):
    query: str
    redacted_query: str
    retrieved_chunks: list[str]
    context_by_chunk_id: dict[str, str]
    answer: str
    citations: list[dict[str, Any]]
    flagged: bool
    flag_reason: str
    redacted_answer: str


def build_pipeline(searcher: HybridSearcher):
    graph = StateGraph(PipelineState)
    graph.add_node("redact_input", redact_input)
    graph.add_node("retrieve", build_retrieval_node(searcher))
    graph.add_node("verify", verify)
    graph.add_node("redact_output", redact_output)

    graph.set_entry_point("redact_input")
    graph.add_edge("redact_input", "retrieve")
    graph.add_edge("retrieve", "verify")
    graph.add_edge("verify", "redact_output")
    graph.add_edge("redact_output", END)

    return graph.compile()


def default_pipeline():
    """Builds a pipeline over the bundled synthetic sample notes, ingesting
    them into Qdrant on first use."""
    chunks = load_notes()
    store = VectorStore()
    store.upsert_chunks(chunks)
    searcher = HybridSearcher(vector_store=store, chunks=chunks)
    return build_pipeline(searcher)
