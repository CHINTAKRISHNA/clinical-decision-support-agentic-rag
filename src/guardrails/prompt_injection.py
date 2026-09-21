"""Heuristic prompt-injection guardrails.

Applied to (a) user queries before they reach the retrieval agent and
(b) retrieved chunks before they're placed in the LLM context, since
injected instructions can hide inside stored documents, not just user input.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior|above) instructions",
    r"disregard (all )?(previous|prior|above)",
    r"you are now",
    r"system prompt",
    r"act as (an?|the) (unrestricted|unfiltered|jailbroken)",
    r"reveal (your|the) (system|hidden) prompt",
    r"do not (cite|verify|redact)",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


@dataclass
class GuardrailResult:
    is_flagged: bool
    matched_patterns: list[str]


def scan(text: str) -> GuardrailResult:
    matched = [p.pattern for p in _COMPILED if p.search(text)]
    return GuardrailResult(is_flagged=bool(matched), matched_patterns=matched)


def sanitize_context_chunks(chunks: list[str]) -> list[str]:
    """Drops retrieved chunks that appear to contain injected instructions."""
    return [c for c in chunks if not scan(c).is_flagged]
