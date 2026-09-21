"""Verification/citation agent: checks every claim in the draft answer against
the retrieved context and flags the response if any citation is fabricated
or unsupported, rather than letting it reach the clinician silently.
"""
from __future__ import annotations

from typing import Any

from src.citation.citation_engine import has_unverifiable_claims, verify_citations


def verify(state: dict[str, Any]) -> dict[str, Any]:
    if state.get("flagged"):
        return {}

    answer = state.get("answer", "")
    context_by_chunk_id = state.get("context_by_chunk_id", {})
    records = verify_citations(answer, context_by_chunk_id)

    if not records:
        return {
            "citations": [],
            "flagged": True,
            "flag_reason": "answer contained no verifiable chunk-id citations",
        }

    if has_unverifiable_claims(records):
        return {
            "citations": [r.__dict__ for r in records],
            "flagged": True,
            "flag_reason": "one or more claims are unsupported by or absent from the retrieved context",
        }

    return {"citations": [r.__dict__ for r in records], "flagged": False}
