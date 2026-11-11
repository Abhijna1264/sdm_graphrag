"""End-to-end GraphRAG pipeline: ingest -> extract -> index -> route -> retrieve -> generate.

Pass `llm="auto"` (the default) to use Gemini when GEMINI_API_KEY is set, a
GeminiClient to use a specific one, or `llm=None` to stay fully offline.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import generator
from .communities import CommunityStore
from .extraction import extract, llm_extract
from .global_search import global_answer
from .graph_store import GraphStore
from .ingest import Chunk, ingest_path, ingest_text
from .keyword_store import KeywordStore
from .llm import get_llm
from .retriever import HybridRetriever, RetrievalResult, TextResult
from .router import RoutingDecision, classify
from .vector_store import EmbeddingStore, VectorStore

# LLM extraction costs one call per chunk, so chunks are much larger with an
# LLM (~1k tokens, close to the paper's 600-token setting) than for the
# per-sentence regex extractor.
_LLM_CHUNKING = {"max_sentences": 40, "overlap_sentences": 3}
_OFFLINE_CHUNKING = {"max_sentences": 6, "overlap_sentences": 1}


class GraphRAGPipeline:
    def __init__(self, llm="auto") -> None:
        self.llm = get_llm() if llm == "auto" else llm
        self.chunks: list[Chunk] = []
        self.graph_store = GraphStore()
        self.vector_store = VectorStore()
        self.keyword_store = KeywordStore()
        self.embedding_store: EmbeddingStore | None = None
        self.community_store = CommunityStore()
        self.retriever: HybridRetriever | None = None

    @property
    def _chunking(self) -> dict:
        return _LLM_CHUNKING if self.llm is not None else _OFFLINE_CHUNKING

    def _build_from_chunks(self, chunks: list[Chunk]) -> "GraphRAGPipeline":
        self.chunks = chunks
        if self.llm is not None:
            print(f"Extracting entities from {len(chunks)} chunks with {self.llm.model}...", flush=True)
            extraction = llm_extract(chunks, self.llm)
        else:
            extraction = extract(chunks)
        self.graph_store.build(extraction)
        self.community_store.build(self.graph_store, llm=self.llm)
        self.vector_store.build(chunks)
        self.keyword_store.build(chunks)
        if self.llm is not None:
            print("Embedding chunks for the vector RAG baseline...", flush=True)
            self.embedding_store = EmbeddingStore().build(chunks, self.llm)
        self.retriever = HybridRetriever(self.vector_store, self.keyword_store, self.graph_store)
        return self

    def build_from_text(self, text: str, source: str = "text") -> "GraphRAGPipeline":
        return self._build_from_chunks(ingest_text(text, source=source, **self._chunking))

    def build_from_file(self, path: str | Path) -> "GraphRAGPipeline":
        """Build from a .txt/.pdf file or a directory of them."""
        return self._build_from_chunks(ingest_path(path, **self._chunking))

    def ask(self, query: str, top_k: int = 5, route: str | None = None, level: int | None = None) -> dict:
        """Answer a query; `route` forces a strategy instead of the router's choice."""
        if self.retriever is None:
            raise RuntimeError("Pipeline has not been built or loaded yet.")
        decision = classify(query, self.graph_store)
        if route is not None and route != decision.route:
            decision = RoutingDecision(route, decision.matched_entities, f"route forced to {route}")
        if decision.route == "global":
            generation = global_answer(query, self.community_store, level=level, llm=self.llm)
            return {
                "query": query,
                "route": "global",
                "reason": decision.reason,
                "matched_entities": decision.matched_entities,
                "text_hits": [],
                "graph_facts": [],
                "community_level": generation["level"],
                "partial_answers": [(p.community_id, p.title, p.score) for p in generation["partials"]],
                "context": generation.get("context", ""),
                "answer": generation["answer"],
                "backend": generation["backend"],
            }
        retrieval = self.retriever.retrieve(query, decision, top_k=top_k)
        generation = generator.generate_answer(query, retrieval, llm=self.llm)
        return {
            "query": query,
            "route": decision.route,
            "reason": decision.reason,
            "matched_entities": decision.matched_entities,
            "text_hits": [(h.chunk_id, h.text, h.fused_score) for h in retrieval.text_hits],
            "graph_facts": [
                (f.subject, f.predicate, f.obj) for f in retrieval.graph_facts
            ],
            "context": generation["context"],
            "answer": generation["answer"],
            "backend": generation["backend"],
        }

    def ask_baseline(self, query: str, top_k: int = 5) -> dict:
        """Conventional vector RAG baseline: top-k chunks by embedding similarity, no graph.

        Uses Gemini embeddings when available, TF-IDF otherwise.
        """
        if self.retriever is None:
            raise RuntimeError("Pipeline has not been built or loaded yet.")
        store = self.embedding_store if self.embedding_store is not None and self.llm is not None else self.vector_store
        hits = store.search(query, top_k=top_k)
        retrieval = RetrievalResult(
            route="baseline",
            reason="vector RAG baseline",
            text_hits=[TextResult(h.chunk_id, h.text, h.score) for h in hits],
        )
        generation = generator.generate_answer(query, retrieval, llm=self.llm)
        return {
            "query": query,
            "route": "baseline",
            "text_hits": [(h.chunk_id, h.text, h.fused_score) for h in retrieval.text_hits],
            "context": generation["context"],
            "answer": generation["answer"],
            "backend": generation["backend"],
        }

    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.graph_store.save(out_dir / "graph.pkl")
        self.vector_store.save(out_dir / "vector.pkl")
        self.keyword_store.save(out_dir / "keyword.pkl")
        self.community_store.save(out_dir / "communities.pkl")
        if self.embedding_store is not None:
            self.embedding_store.save(out_dir / "embeddings.pkl")
        meta = {"llm_model": getattr(self.llm, "model", None), "chunks": len(self.chunks)}
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, in_dir: str | Path, llm="auto") -> "GraphRAGPipeline":
        in_dir = Path(in_dir)
        pipeline = cls(llm=llm)
        pipeline.graph_store = GraphStore.load(in_dir / "graph.pkl")
        pipeline.vector_store = VectorStore.load(in_dir / "vector.pkl")
        pipeline.keyword_store = KeywordStore.load(in_dir / "keyword.pkl")
        if (in_dir / "communities.pkl").exists():
            pipeline.community_store = CommunityStore.load(in_dir / "communities.pkl")
        if (in_dir / "embeddings.pkl").exists():
            pipeline.embedding_store = EmbeddingStore.load(in_dir / "embeddings.pkl", pipeline.llm)
        pipeline.chunks = pipeline.vector_store.chunks
        pipeline.retriever = HybridRetriever(pipeline.vector_store, pipeline.keyword_store, pipeline.graph_store)
        return pipeline
