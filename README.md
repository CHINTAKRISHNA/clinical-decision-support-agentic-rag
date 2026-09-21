# Clinical Decision Support & Agentic Summarization System

A multi-agent RAG pipeline for querying unstructured clinical notes with
**verifiable citations** back to exact source chunks and **PHI-safe** input/output
handling. Built to show that grounded, auditable answers over clinical text are
achievable without hallucinated claims reaching a clinician.

> All clinical notes in this repo (`data/sample_notes/`) are **synthetic and
> fictional**, written for demo purposes only. No real patient data is included.
> Point `src/data/loader.py` at a de-identified MIMIC-III export to run against
> real (IRB-approved, credentialed) data.

## The problem

EHR/EMR notes are unstructured, siloed, and full of PHI. A naive RAG chatbot
over these notes risks two failure modes: (1) hallucinated claims a clinician
can't verify, and (2) leaking identifiers into logs, prompts, or LLM providers.
This project addresses both directly rather than bolting them on after the fact.

## Architecture

```
 query
   |
   v
 [Redaction Agent]   Presidio strips PHI before anything reaches the LLM
   |
   v
 [Retrieval Agent]   hybrid search (dense + BM25, RRF fusion)
   |                 -> cross-encoder rerank (bge-reranker-large)
   |                 -> prompt-injection guardrail filters retrieved chunks
   |                 -> grounded generation, every claim tagged [chunk_id]
   v
 [Verification /     parses citation tags, confirms each chunk_id was
  Citation Agent]    actually retrieved, checks lexical support -> flags
   |                 unverifiable answers instead of returning them silently
   v
 [Redaction Agent]   Presidio strips PHI from the final answer
   |
   v
 result (answer + per-claim citation ledger + flag status)
```

Orchestrated as a `langgraph.StateGraph` — see [src/graph/pipeline.py](src/graph/pipeline.py).

### Why hybrid retrieval + rerank

BM25 catches exact clinical terms/dosages that dense embeddings can blur
("lisinopril 40mg" vs "40mg lisinopril"); dense search catches semantic
paraphrase. Results are fused with reciprocal rank fusion, then a
cross-encoder reranker (`bge-reranker-large`) re-scores the fused candidates
against the full query before anything reaches the LLM context window.

### The citation engine

The generation prompt requires the model to tag every factual claim with the
exact `chunk_id` it drew from. [src/citation/citation_engine.py](src/citation/citation_engine.py)
then:
1. Parses every `[chunk_id]` tag out of the answer.
2. Confirms the chunk_id was actually part of the retrieved context (catches
   fabricated references outright).
3. Runs a lexical-overlap support check between the claim and its cited chunk
   as a fast pre-filter ahead of the full Ragas faithfulness pass.

Any claim that fails either check flags the whole response rather than
returning a plausible-sounding but unverifiable answer.

### Safety & compliance

- **PII/PHI masking**: [src/redaction/presidio_redactor.py](src/redaction/presidio_redactor.py)
  (Microsoft Presidio) redacts names, dates, MRNs, phone numbers, and
  locations on both the way in and the way out.
- **Prompt-injection guardrails**: [src/guardrails/prompt_injection.py](src/guardrails/prompt_injection.py)
  scans both the user query and every retrieved chunk — injected instructions
  can hide inside stored documents, not just user input — and drops anything
  that matches known injection patterns.

### Evaluation

[src/eval/ragas_eval.py](src/eval/ragas_eval.py) runs a fixed set of clinical
questions through the pipeline and scores them with [Ragas](https://github.com/explodinggradients/ragas):
- **Faithfulness** — does the answer's content actually follow from the
  retrieved context?
- **Context Relevance** — how much of the retrieved context is relevant to
  the question (a proxy for retrieval quality independent of generation)?

## Project layout

```
src/
  agents/         LangGraph node functions (redaction, retrieval, verification)
  graph/          StateGraph wiring the agents into the full pipeline
  retrieval/      Qdrant dense store, BM25 hybrid search, cross-encoder reranker
  redaction/      Presidio-based PHI masking
  guardrails/     Prompt-injection detection
  citation/       Citation parsing + verification engine
  data/           Note loading + chunking (stable, hashable chunk IDs)
  eval/           Ragas evaluation harness
  llm.py          Pluggable LLM backend (mock by default, Claude via Anthropic)
  config.py       Settings (env-driven)
data/sample_notes/  Synthetic fictional clinical notes used for local dev/demo
scripts/ingest.py   One-off ingestion of sample_notes/ into Qdrant
tests/              Offline unit tests (citation parsing, chunking, guardrails)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg   # required by Presidio's NLP engine

# Start Qdrant locally (or point QDRANT_URL at a managed instance)
docker run -p 6333:6333 qdrant/qdrant
```

Copy `.env.example` to `.env` and set `LLM_PROVIDER=anthropic` +
`ANTHROPIC_API_KEY` to generate with Claude; leave unset to use the
deterministic offline `MockLLM` (useful for running tests/demo without an API key).

## Usage

```bash
python scripts/ingest.py
python -m src.main "What is the patient's current dose of lisinopril and why was it changed?"
```

## Testing

```bash
pytest tests/            # offline: citation parsing, chunking, guardrails
python -m src.eval.ragas_eval   # requires Qdrant running + a configured LLM judge
```

## Status

This is a portfolio/reference implementation demonstrating the architecture
end-to-end against a small synthetic corpus. Swapping in a real MIMIC-III
(or institutional) note export requires only pointing `src/data/loader.py`
at the new source and re-running `scripts/ingest.py`.
