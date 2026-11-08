"""End-to-end GraphRAG pipeline: ingest -> extract -> index -> route -> retrieve -> generate."""
from __future__ import annotations

from pathlib import Path

from . import generator
from .communities import CommunityStore
from .extraction import extract
from .global_search import global_answer
from .graph_store import GraphStore
from .ingest import Chunk, ingest, ingest_text
from .keyword_store import KeywordStore
from .retriever import HybridRetriever, RetrievalResult, TextResult
from .router import RoutingDecision, classify
from .vector_store import VectorStore


class GraphRAGPipeline:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.graph_store = GraphStore()
        self.vector_store = VectorStore()
        self.keyword_store = KeywordStore()
        self.community_store = CommunityStore()
        self.retriever: HybridRetriever | None = None

    def _build_from_chunks(self, chunks: list[Chunk], use_ollama: bool = True) -> "GraphRAGPipeline":
        self.chunks = chunks
        extraction = extract(chunks)
        self.graph_store.build(extraction)
        self.community_store.build(self.graph_store, use_ollama=use_ollama)
        self.vector_store.build(chunks)
        self.keyword_store.build(chunks)
        self.retriever = HybridRetriever(self.vector_store, self.keyword_store, self.graph_store)
        return self

    def build_from_text(self, text: str, source: str = "text", use_ollama: bool = True) -> "GraphRAGPipeline":
        return self._build_from_chunks(ingest_text(text, source=source), use_ollama=use_ollama)

    def build_from_file(self, path: str | Path, use_ollama: bool = True) -> "GraphRAGPipeline":
        return self._build_from_chunks(ingest(path), use_ollama=use_ollama)

    def ask(
        self, query: str, top_k: int = 5, use_ollama: bool = True, route: str | None = None,
        level: int | None = None,
    ) -> dict:
        """Answer a query; `route` forces a strategy instead of the router's choice."""
        if self.retriever is None:
            raise RuntimeError("Pipeline has not been built or loaded yet.")
        decision = classify(query, self.graph_store)
        if route is not None and route != decision.route:
            decision = RoutingDecision(route, decision.matched_entities, f"route forced to {route}")
        if decision.route == "global":
            generation = global_answer(query, self.community_store, level=level, use_ollama=use_ollama)
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
        generation = generator.generate_answer(query, retrieval, use_ollama=use_ollama)
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

    def ask_baseline(self, query: str, top_k: int = 5, use_ollama: bool = True) -> dict:
        """Conventional vector RAG baseline: top-k TF-IDF chunks only, no graph."""
        if self.retriever is None:
            raise RuntimeError("Pipeline has not been built or loaded yet.")
        hits = self.vector_store.search(query, top_k=top_k)
        retrieval = RetrievalResult(
            route="baseline",
            reason="vector RAG baseline",
            text_hits=[TextResult(h.chunk_id, h.text, h.score) for h in hits],
        )
        generation = generator.generate_answer(query, retrieval, use_ollama=use_ollama)
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

    @classmethod
    def load(cls, in_dir: str | Path) -> "GraphRAGPipeline":
        in_dir = Path(in_dir)
        pipeline = cls()
        pipeline.graph_store = GraphStore.load(in_dir / "graph.pkl")
        pipeline.vector_store = VectorStore.load(in_dir / "vector.pkl")
        pipeline.keyword_store = KeywordStore.load(in_dir / "keyword.pkl")
        if (in_dir / "communities.pkl").exists():
            pipeline.community_store = CommunityStore.load(in_dir / "communities.pkl")
        pipeline.chunks = pipeline.vector_store.chunks
        pipeline.retriever = HybridRetriever(pipeline.vector_store, pipeline.keyword_store, pipeline.graph_store)
        return pipeline
