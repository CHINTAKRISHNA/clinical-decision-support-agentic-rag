"""Streamlit UI: ask clinical questions, choose the LLM backend, paste an Anthropic key.

Run:
    streamlit run src/app.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from src.config import settings
from src.graph.pipeline import default_pipeline
from src.llm import AnthropicLLM, MockLLM

MODELS = ["claude-sonnet-5", "claude-opus-5", "claude-haiku-4-5-20251001"]

st.set_page_config(page_title="Clinical Decision Support", page_icon="🩺", layout="wide")
st.title("🩺 Clinical Decision Support")
st.caption("Grounded, cited answers over synthetic clinical notes. PHI is redacted in and out.")


@st.cache_resource(show_spinner="Loading pipeline (embedding + reranker models)...")
def get_pipeline(provider: str, api_key: str, model: str):
    llm = AnthropicLLM(api_key=api_key, model=model) if provider == "Anthropic (Claude)" else MockLLM()
    return default_pipeline(llm)


with st.sidebar:
    st.header("LLM backend")
    default_provider = 0 if settings.llm_provider == "anthropic" else 1
    provider = st.radio("Provider", ["Anthropic (Claude)", "Mock (offline)"], index=default_provider)
    api_key, model = "", MODELS[0]
    if provider.startswith("Anthropic"):
        api_key = st.text_input(
            "Anthropic API key", value=settings.anthropic_api_key, type="password",
            help="Kept in memory for this session only; not written to disk.",
        )
        model = st.selectbox("Model", MODELS, index=MODELS.index(settings.anthropic_model)
                             if settings.anthropic_model in MODELS else 0)
        if not api_key:
            st.warning("Enter an API key to use Claude.")
    st.divider()
    st.caption(f"Qdrant: {settings.qdrant_url}")

query = st.text_area(
    "Question", height=90,
    value="What is the patient's current dose of lisinopril and why was it changed?",
)

if st.button("Ask", type="primary", disabled=provider.startswith("Anthropic") and not api_key):
    try:
        pipeline = get_pipeline(provider, api_key, model)
        with st.spinner("Running pipeline..."):
            result = pipeline.invoke({"query": query})
    except Exception as exc:  # surface Qdrant/API-key errors in the UI
        st.error(f"{type(exc).__name__}: {exc}")
        st.stop()

    st.subheader("Answer")
    st.write(result.get("redacted_answer") or result.get("answer"))
    if result.get("flagged"):
        st.error(f"Flagged: {result.get('flag_reason')}")
    else:
        st.success("All citations verified.")

    st.subheader("Citations")
    for c in result.get("citations", []):
        ok = c["exists_in_context"] and c["lexically_supported"]
        st.markdown(f"{'✅' if ok else '⚠️'} `{c['chunk_id']}` — {c['claim']}")

    with st.expander("Retrieved context"):
        for cid, text in result.get("context_by_chunk_id", {}).items():
            st.markdown(f"**{cid}**")
            st.text(text)
