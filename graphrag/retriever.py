"""Hybrid retrieval: fuses TF-IDF + BM25 rankings and pulls graph facts."""
from __future__ import annotations

from dataclasses import dataclass, field

from .graph_store import GraphFact, GraphStore
from .keyword_store import KeywordStore
from .router import RoutingDecision
from .vector_store import VectorStore

_RRF_K = 60  # standard reciprocal-rank-fusion damping constant


@dataclass
class TextResult:
    chunk_id: str
    text: str
    fused_score: float


@dataclass
class RetrievalResult:
    route: str
    reason: str
    text_hits: list[TextResult] = field(default_factory=list)
    graph_facts: list[GraphFact] = field(default_factory=list)


def _reciprocal_rank_fusion(*ranked_lists, top_k: int = 5) -> list[TextResult]:
    scores: dict[str, float] = {}
    texts: dict[str, str] = {}
    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
            texts[hit.chunk_id] = hit.text
    ordered = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return [TextResult(chunk_id=cid, text=texts[cid], fused_score=score) for cid, score in ordered[:top_k]]


class HybridRetriever:
    def __init__(self, vector_store: VectorStore, keyword_store: KeywordStore, graph_store: GraphStore) -> None:
        self.vector_store = vector_store
        self.keyword_store = keyword_store
        self.graph_store = graph_store

    def text_search(self, query: str, top_k: int = 5) -> list[TextResult]:
        vector_hits = self.vector_store.search(query, top_k=top_k * 2)
        keyword_hits = self.keyword_store.search(query, top_k=top_k * 2)
        return _reciprocal_rank_fusion(vector_hits, keyword_hits, top_k=top_k)

    def graph_search(self, matched_entities: list[str], hops: int = 2) -> list[GraphFact]:
        if len(matched_entities) >= 2:
            facts: list[GraphFact] = []
            for a, b in zip(matched_entities, matched_entities[1:]):
                facts.extend(self.graph_store.shortest_path_facts(a, b))
            if facts:
                return facts
        facts = []
        for entity in matched_entities:
            facts.extend(self.graph_store.neighbors(entity, hops=hops))
        return facts

    def retrieve(self, query: str, decision: RoutingDecision, top_k: int = 5) -> RetrievalResult:
        result = RetrievalResult(route=decision.route, reason=decision.reason)
        if decision.route in ("vector", "hybrid"):
            result.text_hits = self.text_search(query, top_k=top_k)
        if decision.route in ("graph", "hybrid"):
            result.graph_facts = self.graph_search(decision.matched_entities)
        return result
