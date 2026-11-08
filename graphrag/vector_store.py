"""TF-IDF + cosine similarity semantic-ish search over chunks.

Stands in for an embedding model without needing any download or API key.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .ingest import Chunk


@dataclass
class ScoredChunk:
    chunk_id: str
    text: str
    score: float


class VectorStore:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.chunks: list[Chunk] = []
        self._matrix = None

    def build(self, chunks: list[Chunk]) -> "VectorStore":
        self.chunks = list(chunks)
        texts = [c.text for c in self.chunks] or [""]
        self._matrix = self.vectorizer.fit_transform(texts)
        return self

    def search(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        if not self.chunks or self._matrix is None:
            return []
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix)[0]
        ranked = sorted(zip(self.chunks, scores), key=lambda pair: pair[1], reverse=True)
        return [
            ScoredChunk(chunk_id=c.chunk_id, text=c.text, score=float(score))
            for c, score in ranked[:top_k]
            if score > 0
        ]

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(pickle.dumps({"vectorizer": self.vectorizer, "chunks": self.chunks, "matrix": self._matrix}))

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        data = pickle.loads(Path(path).read_bytes())
        store = cls()
        store.vectorizer = data["vectorizer"]
        store.chunks = data["chunks"]
        store._matrix = data["matrix"]
        return store
