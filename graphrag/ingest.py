"""Document loading and chunking.

Supports plain text and PDF input. Chunks are overlapping, sentence-aware
windows so that entity/relation extraction (which operates per-sentence)
never sees a sentence cut in half across chunk boundaries.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
_ABBREVIATIONS_RE = re.compile(
    r"\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr|St|vs|etc|Inc|Corp|Ltd|Co)\.\s+(?=[A-Z])"
)
_PROTECTED_SPACE = "@@ABBR_SPACE@@"


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str


def load_text(path: str | Path) -> str:
    path = Path(path)
    return path.read_text(encoding="utf-8", errors="ignore")


def load_pdf(path: str | Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def load_document(path: str | Path) -> str:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return load_pdf(path)
    return load_text(path)


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    # Protect "Dr. Jane" style abbreviations from being mistaken for a
    # sentence boundary before splitting, then restore the space.
    protected = _ABBREVIATIONS_RE.sub(lambda m: f"{m.group(1)}.{_PROTECTED_SPACE}", text)
    parts = _SENTENCE_SPLIT_RE.split(protected)
    return [p.replace(_PROTECTED_SPACE, " ").strip() for p in parts if p.strip()]


def chunk_text(
    text: str,
    source: str = "document",
    max_sentences: int = 6,
    overlap_sentences: int = 1,
) -> list[Chunk]:
    """Group sentences into overlapping chunks.

    Sentence-based (rather than character-based) chunking guarantees a chunk
    boundary never splits a sentence, which matters because extraction runs
    per-sentence.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    step = max(max_sentences - overlap_sentences, 1)
    idx = 0
    chunk_num = 0
    while idx < len(sentences):
        window = sentences[idx : idx + max_sentences]
        chunk_text_value = " ".join(window)
        chunks.append(
            Chunk(chunk_id=f"{source}::chunk-{chunk_num}", text=chunk_text_value, source=source)
        )
        chunk_num += 1
        if idx + max_sentences >= len(sentences):
            break
        idx += step
    return chunks


def ingest(path: str | Path, max_sentences: int = 6, overlap_sentences: int = 1) -> list[Chunk]:
    path = Path(path)
    text = load_document(path)
    return chunk_text(text, source=path.stem, max_sentences=max_sentences, overlap_sentences=overlap_sentences)


def ingest_text(text: str, source: str = "text", max_sentences: int = 6, overlap_sentences: int = 1) -> list[Chunk]:
    return chunk_text(text, source=source, max_sentences=max_sentences, overlap_sentences=overlap_sentences)


def ingest_path(path: str | Path, max_sentences: int = 6, overlap_sentences: int = 1) -> list[Chunk]:
    """Ingest one file, or every .txt/.pdf file in a directory (one source per file)."""
    path = Path(path)
    if not path.is_dir():
        return ingest(path, max_sentences, overlap_sentences)
    chunks: list[Chunk] = []
    for file in sorted(path.iterdir()):
        if file.suffix.lower() in (".txt", ".pdf"):
            chunks.extend(ingest(file, max_sentences, overlap_sentences))
    return chunks
