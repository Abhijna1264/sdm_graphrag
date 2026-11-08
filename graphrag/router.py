"""Heuristic query routing between vector search, graph traversal, and both.

Mirrors the "simple vs relational vs complex query" idea from typical
GraphRAG designs, implemented here as a small set of independent,
explainable rules rather than an LLM classifier.
"""
from __future__ import annotations

from dataclasses import dataclass

from .graph_store import GraphStore

_RELATIONAL_KEYWORDS = (
    "relationship", "relation", "related", "connect", "connection",
    "between", "link", "linked", "compare", "comparison", "associated",
    "association", "versus", " vs ",
)

# Phrases signalling a corpus-wide "sensemaking" question that no single
# chunk or entity neighborhood can answer.
_GLOBAL_KEYWORDS = (
    "main theme", "main topic", "key theme", "overall", "overview", "summarize",
    "summary of", "across the", "whole corpus", "entire", "big picture",
    "what are the themes", "what is this corpus", "what is the document about",
    "most important", "major themes", "trends",
)

Route = str  # one of "vector", "graph", "hybrid", "global"


@dataclass
class RoutingDecision:
    route: Route
    matched_entities: list[str]
    reason: str


def _has_relational_keyword(query: str) -> bool:
    query_lower = f" {query.lower()} "
    return any(keyword in query_lower for keyword in _RELATIONAL_KEYWORDS)


def _has_global_keyword(query: str) -> bool:
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in _GLOBAL_KEYWORDS)


def classify(query: str, graph: GraphStore) -> RoutingDecision:
    if _has_global_keyword(query):
        return RoutingDecision(
            route="global",
            matched_entities=graph.match_query_entities(query),
            reason="corpus-wide question -> map-reduce over community summaries",
        )
    matched = graph.match_query_entities(query)
    relational = _has_relational_keyword(query)

    if len(matched) >= 2:
        return RoutingDecision(
            route="graph",
            matched_entities=matched,
            reason=f"query mentions {len(matched)} known entities -> graph traversal between them",
        )
    if len(matched) >= 1 and relational:
        return RoutingDecision(
            route="hybrid",
            matched_entities=matched,
            reason="query mentions a known entity and asks about a relationship -> graph + text",
        )
    if len(matched) >= 1:
        return RoutingDecision(
            route="hybrid",
            matched_entities=matched,
            reason="query mentions a known entity -> include its graph context alongside text search",
        )
    return RoutingDecision(
        route="vector",
        matched_entities=[],
        reason="no known entities recognized -> plain semantic/keyword search",
    )
