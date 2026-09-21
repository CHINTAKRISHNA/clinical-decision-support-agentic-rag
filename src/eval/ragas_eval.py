"""Ragas-based evaluation harness tracking Faithfulness and Context Relevance
across a fixed set of clinical questions against the synthetic note corpus.

Note: Ragas' metrics use an LLM judge internally. Configure it via
RAGAS_LLM / RAGAS_EMBEDDINGS env handling in your ragas version, or wrap
langchain_anthropic.ChatAnthropic as the judge model, before running this.
"""
from __future__ import annotations

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_relevancy, faithfulness

from src.graph.pipeline import default_pipeline

EVAL_QUESTIONS = [
    "What is the patient's current dose of lisinopril and why was it changed?",
    "What was the ejection fraction on the most recent echocardiogram and what is the current assessment?",
    "What is the patient's most recent HbA1c and what medication was added as a result?",
]


def build_eval_dataset() -> Dataset:
    pipeline = default_pipeline()
    rows = {"question": [], "answer": [], "contexts": []}

    for question in EVAL_QUESTIONS:
        result = pipeline.invoke({"query": question})
        rows["question"].append(question)
        rows["answer"].append(result.get("redacted_answer") or result.get("answer", ""))
        rows["contexts"].append(list(result.get("context_by_chunk_id", {}).values()))

    return Dataset.from_dict(rows)


def run_eval() -> dict:
    dataset = build_eval_dataset()
    scores = evaluate(dataset, metrics=[faithfulness, context_relevancy])
    return scores.to_pandas().to_dict(orient="records")


if __name__ == "__main__":
    import json

    print(json.dumps(run_eval(), indent=2, default=str))
