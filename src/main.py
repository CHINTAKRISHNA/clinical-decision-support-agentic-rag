"""CLI entry point: ask a clinical question against the synthetic note corpus.

Example:
    python -m src.main "What is the patient's home medication for hypertension?"
"""
from __future__ import annotations

import sys

from src.graph.pipeline import default_pipeline


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m src.main '<question>'")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    pipeline = default_pipeline()
    result = pipeline.invoke({"query": query})

    print("\n--- Answer ---")
    print(result.get("redacted_answer") or result.get("answer"))

    if result.get("flagged"):
        print(f"\n[FLAGGED] {result.get('flag_reason')}")

    print("\n--- Citations ---")
    for citation in result.get("citations", []):
        status = "OK" if citation["exists_in_context"] and citation["lexically_supported"] else "UNVERIFIED"
        print(f"[{status}] {citation['chunk_id']}: {citation['claim']}")


if __name__ == "__main__":
    main()
