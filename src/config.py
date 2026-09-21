"""Central configuration for the clinical decision support pipeline."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "clinical_notes")

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-large")

    llm_provider: str = os.getenv("LLM_PROVIDER", "mock")  # "anthropic" or "mock"
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")

    top_k_dense: int = int(os.getenv("TOP_K_DENSE", "10"))
    top_k_bm25: int = int(os.getenv("TOP_K_BM25", "10"))
    top_k_rerank: int = int(os.getenv("TOP_K_RERANK", "5"))

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "300"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))


settings = Settings()
