"""BM25 sparse keyword search over chunks, complementing the TF-IDF store."""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from .ingest import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class ScoredChunk:
    chunk_id: str
    text: str
    score: float


class KeywordStore:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None

    def build(self, chunks: list[Chunk]) -> "KeywordStore":
        self.chunks = list(chunks)
        tokenized = [_tokenize(c.text) for c in self.chunks]
        if tokenized:
            self._bm25 = BM25Okapi(tokenized)
        return self

    def search(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        if not self.chunks or self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self.chunks, scores), key=lambda pair: pair[1], reverse=True)
        return [
            ScoredChunk(chunk_id=c.chunk_id, text=c.text, score=float(score))
            for c, score in ranked[:top_k]
            if score > 0
        ]

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(pickle.dumps({"chunks": self.chunks, "bm25": self._bm25}))

    @classmethod
    def load(cls, path: str | Path) -> "KeywordStore":
        data = pickle.loads(Path(path).read_bytes())
        store = cls()
        store.chunks = data["chunks"]
        store._bm25 = data["bm25"]
        return store
