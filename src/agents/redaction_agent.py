"""Redaction agent: strips PHI from clinician input and from the final answer
before either touches the LLM context window or leaves the system.
"""
from __future__ import annotations

from typing import Any

from src.redaction.presidio_redactor import PresidioRedactor

_redactor = PresidioRedactor()


def redact_input(state: dict[str, Any]) -> dict[str, Any]:
    return {"redacted_query": _redactor.redact(state["query"])}


def redact_output(state: dict[str, Any]) -> dict[str, Any]:
    answer = state.get("answer", "")
    return {"redacted_answer": _redactor.redact(answer)}
