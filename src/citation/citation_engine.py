"""Verifiable citation engine.

The generation prompt requires the LLM to tag every factual claim with the
exact chunk_id it drew from, e.g. "Patient reports chest pain [note-04::chunk-2::9f1a2b3c]."
This module parses those tags, confirms the chunk_id was actually part of the
retrieved context (catches fabricated references), and scores lexical support
between the claim and the source chunk as a cheap hallucination check ahead
of the full Ragas faithfulness evaluation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

CITATION_PATTERN = re.compile(r"([^.\[\]]+?)\s*\[([\w\-:]+::chunk-\d+::[0-9a-f]{8})\]")


@dataclass
class CitationRecord:
    claim: str
    chunk_id: str
    exists_in_context: bool
    lexically_supported: bool
    source_snippet: str | None


def parse_citations(answer: str) -> list[tuple[str, str]]:
    return [(claim.strip(), chunk_id) for claim, chunk_id in CITATION_PATTERN.findall(answer)]


def _lexical_support(claim: str, source_text: str, threshold: float = 0.3) -> bool:
    claim_tokens = {w.lower() for w in re.findall(r"\w+", claim) if len(w) > 3}
    source_tokens = {w.lower() for w in re.findall(r"\w+", source_text) if len(w) > 3}
    if not claim_tokens:
        return False
    overlap = len(claim_tokens & source_tokens) / len(claim_tokens)
    return overlap >= threshold


def verify_citations(answer: str, context_by_chunk_id: dict[str, str]) -> list[CitationRecord]:
    records = []
    for claim, chunk_id in parse_citations(answer):
        source_text = context_by_chunk_id.get(chunk_id)
        exists = source_text is not None
        supported = _lexical_support(claim, source_text) if exists else False
        records.append(
            CitationRecord(
                claim=claim,
                chunk_id=chunk_id,
                exists_in_context=exists,
                lexically_supported=supported,
                source_snippet=(source_text[:200] if source_text else None),
            )
        )
    return records


def has_unverifiable_claims(records: list[CitationRecord]) -> bool:
    return any(not (r.exists_in_context and r.lexically_supported) for r in records)
