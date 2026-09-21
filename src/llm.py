"""Pluggable LLM backend. Defaults to a deterministic mock so the pipeline
and tests run offline without an API key; set LLM_PROVIDER=anthropic and
ANTHROPIC_API_KEY to generate with Claude instead.
"""
from __future__ import annotations

from typing import Protocol

from src.config import settings


class LLM(Protocol):
    def generate(self, prompt: str) -> str: ...


class MockLLM:
    """Produces a citation-tagged answer by quoting the first retrieved chunks verbatim.

    Useful for exercising the full agent graph and citation verification
    without any external API calls.
    """

    def generate(self, prompt: str) -> str:
        context_lines = [line for line in prompt.splitlines() if "::chunk-" in line]
        if not context_lines:
            return "No relevant context was retrieved for this query."
        sentences = []
        for line in context_lines[:3]:
            if "]" in line and "[" in line:
                chunk_id = line[line.rfind("[") + 1 : line.rfind("]")]
                snippet = line.split("]", 1)[-1].strip()[:120] or "Relevant finding noted in the record"
                sentences.append(f"{snippet} [{chunk_id}]")
        return " ".join(sentences)


class AnthropicLLM:
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        from langchain_anthropic import ChatAnthropic

        self._client = ChatAnthropic(
            model=model or settings.anthropic_model,
            api_key=api_key or settings.anthropic_api_key,
        )

    def generate(self, prompt: str) -> str:
        content = self._client.invoke(prompt).content
        if isinstance(content, list):  # content blocks -> plain text
            return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
        return content


def get_llm() -> LLM:
    if settings.llm_provider == "anthropic":
        return AnthropicLLM()
    return MockLLM()
