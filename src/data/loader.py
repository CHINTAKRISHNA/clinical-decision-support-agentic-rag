"""Loads and chunks clinical notes, assigning stable chunk IDs for citation tracking.

Notes directory defaults to the bundled synthetic sample set (data/sample_notes/).
Real deployments would point this at a de-identified MIMIC-III export.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from src.config import settings

DEFAULT_NOTES_DIR = Path(__file__).resolve().parents[2] / "data" / "sample_notes"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    position: int


def _split_words(text: str) -> list[str]:
    return re.split(r"\s+", text.strip())


def chunk_text(doc_id: str, text: str, chunk_size: int, overlap: int) -> list[Chunk]:
    words = _split_words(text)
    chunks: list[Chunk] = []
    step = max(chunk_size - overlap, 1)
    position = 0
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if not window:
            continue
        chunk_text_value = " ".join(window)
        chunk_id = f"{doc_id}::chunk-{position}::{hashlib.sha1(chunk_text_value.encode()).hexdigest()[:8]}"
        chunks.append(Chunk(chunk_id=chunk_id, doc_id=doc_id, text=chunk_text_value, position=position))
        position += 1
        if start + chunk_size >= len(words):
            break
    return chunks


def load_notes(notes_dir: Path | None = None) -> list[Chunk]:
    notes_dir = notes_dir or DEFAULT_NOTES_DIR
    all_chunks: list[Chunk] = []
    for path in sorted(notes_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        doc_id = path.stem
        all_chunks.extend(chunk_text(doc_id, text, settings.chunk_size, settings.chunk_overlap))
    return all_chunks
