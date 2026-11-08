"""Knowledge graph storage on top of networkx.MultiDiGraph.

No external graph database is used (see project decisions) — the graph is
small enough for an in-process, in-memory store with pickle persistence.
"""
from __future__ import annotations

import difflib
import pickle
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from .extraction import ExtractionResult


@dataclass
class GraphFact:
    subject: str
    predicate: str
    obj: str
    sentence: str
    chunk_id: str


class GraphStore:
    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()

    def build(self, extraction: ExtractionResult) -> "GraphStore":
        for key, entry in extraction.entities.items():
            self.graph.add_node(key, name=entry["name"], type=entry["type"], mentions=entry["mentions"])

        for triple in extraction.triples:
            sub_key, obj_key = triple.subject.lower(), triple.obj.lower()
            for key, name in ((sub_key, triple.subject), (obj_key, triple.obj)):
                if key not in self.graph:
                    self.graph.add_node(key, name=name, type="ENTITY", mentions=1)
            self.graph.add_edge(
                sub_key,
                obj_key,
                predicate=triple.predicate,
                sentence=triple.sentence,
                chunk_id=triple.chunk_id,
            )
        return self

    @property
    def node_names(self) -> list[str]:
        return [data["name"] for _, data in self.graph.nodes(data=True)]

    def fuzzy_lookup(self, text: str, cutoff: float = 0.8) -> list[str]:
        """Map a free-text mention to known node keys (lowercase)."""
        text_lower = text.lower().strip()
        if not text_lower:
            return []
        if text_lower in self.graph:
            return [text_lower]

        # Below this length, plain substring/fuzzy matching produces false
        # positives (e.g. "is" inside "chief scientist"), so require a
        # minimum length before attempting either.
        if len(text_lower) < 4:
            return []

        substring_hits = [
            key for key in self.graph.nodes
            if len(key) >= 4 and (key in text_lower or text_lower in key)
        ]
        if substring_hits:
            return substring_hits

        close = difflib.get_close_matches(text_lower, list(self.graph.nodes), n=3, cutoff=cutoff)
        return close

    def match_query_entities(self, query: str) -> list[str]:
        """Find known graph node keys mentioned anywhere in a query string."""
        matched: set[str] = set()
        query_lower = query.lower()
        for key, data in self.graph.nodes(data=True):
            name_lower = data["name"].lower()
            if len(name_lower) >= 3 and name_lower in query_lower:
                matched.add(key)
        if not matched:
            for word in query.split():
                for hit in self.fuzzy_lookup(word.strip(".,!?;:'\"")):
                    matched.add(hit)
        return sorted(matched)

    def neighbors(self, node_key: str, hops: int = 1) -> list[GraphFact]:
        if node_key not in self.graph:
            return []
        visited_edges = set()
        facts: list[GraphFact] = []
        frontier = {node_key}
        seen_nodes = {node_key}
        for _ in range(hops):
            next_frontier = set()
            for node in frontier:
                for _, target, data in self.graph.out_edges(node, data=True):
                    edge_id = (node, target, data.get("predicate"), data.get("chunk_id"))
                    if edge_id not in visited_edges:
                        visited_edges.add(edge_id)
                        facts.append(
                            GraphFact(
                                subject=self.graph.nodes[node]["name"],
                                predicate=data["predicate"],
                                obj=self.graph.nodes[target]["name"],
                                sentence=data["sentence"],
                                chunk_id=data["chunk_id"],
                            )
                        )
                    next_frontier.add(target)
                for source, _, data in self.graph.in_edges(node, data=True):
                    edge_id = (source, node, data.get("predicate"), data.get("chunk_id"))
                    if edge_id not in visited_edges:
                        visited_edges.add(edge_id)
                        facts.append(
                            GraphFact(
                                subject=self.graph.nodes[source]["name"],
                                predicate=data["predicate"],
                                obj=self.graph.nodes[node]["name"],
                                sentence=data["sentence"],
                                chunk_id=data["chunk_id"],
                            )
                        )
                    next_frontier.add(source)
            frontier = next_frontier - seen_nodes
            seen_nodes |= next_frontier
            if not frontier:
                break
        return facts

    def shortest_path_facts(self, node_a: str, node_b: str) -> list[GraphFact]:
        if node_a not in self.graph or node_b not in self.graph:
            return []
        undirected = self.graph.to_undirected(as_view=True)
        try:
            path = nx.shortest_path(undirected, node_a, node_b)
        except nx.NetworkXNoPath:
            return []

        facts: list[GraphFact] = []
        for a, b in zip(path, path[1:]):
            data = None
            if self.graph.has_edge(a, b):
                data = next(iter(self.graph.get_edge_data(a, b).values()))
                subj, obj = a, b
            elif self.graph.has_edge(b, a):
                data = next(iter(self.graph.get_edge_data(b, a).values()))
                subj, obj = b, a
            if data is not None:
                facts.append(
                    GraphFact(
                        subject=self.graph.nodes[subj]["name"],
                        predicate=data["predicate"],
                        obj=self.graph.nodes[obj]["name"],
                        sentence=data["sentence"],
                        chunk_id=data["chunk_id"],
                    )
                )
        return facts

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(pickle.dumps(self.graph))

    @classmethod
    def load(cls, path: str | Path) -> "GraphStore":
        store = cls()
        store.graph = pickle.loads(Path(path).read_bytes())
        return store

    def __len__(self) -> int:
        return self.graph.number_of_nodes()
