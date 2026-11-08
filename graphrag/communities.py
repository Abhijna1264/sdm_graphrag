"""Hierarchical community detection + bottom-up community summaries.

This is the "global" half of GraphRAG: the entity graph is partitioned into
nested communities, and each community gets a summary report. Global
questions are then answered from these summaries (see global_search.py)
instead of from individual chunks.

Community detection uses networkx's Louvain partitions, which yield a
hierarchy of successively coarser partitions (level 0 = finest). Louvain is
the predecessor of Leiden and ships with networkx, so no extra dependency is
needed; Leiden mainly adds a guarantee that communities are well-connected.

Summaries are bottom-up: a leaf (level 0) community is summarized from its
own entities, relations and source sentences; a higher-level community is
summarized from the summaries of the sub-communities it contains. If a local
Ollama server is reachable it writes the summaries, otherwise a structured
extractive summary is built.
"""
from __future__ import annotations

import pickle
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

from .graph_store import GraphStore
from .llm import try_ollama


@dataclass
class Community:
    community_id: str
    level: int
    nodes: list[str]  # graph node keys
    children: list[str] = field(default_factory=list)
    title: str = ""
    summary: str = ""
    key_entities: list[str] = field(default_factory=list)
    rank: float = 0.0  # importance: number of relations inside the community


def _louvain_levels(graph: nx.Graph, seed: int) -> list[list[set[str]]]:
    if graph.number_of_nodes() == 0:
        return []
    if graph.number_of_edges() == 0:
        return [[{n} for n in graph.nodes]]
    return [list(level) for level in nx.community.louvain_partitions(graph, seed=seed)]


class CommunityStore:
    def __init__(self) -> None:
        self.communities: dict[str, Community] = {}

    @property
    def levels(self) -> list[int]:
        return sorted({c.level for c in self.communities.values()})

    def at_level(self, level: int) -> list[Community]:
        return [c for c in self.communities.values() if c.level == level]

    def build(self, graph_store: GraphStore, use_ollama: bool = True, seed: int = 42) -> "CommunityStore":
        self.communities = {}
        undirected = nx.Graph(graph_store.graph.to_undirected())
        undirected.remove_edges_from(nx.selfloop_edges(undirected))
        levels = _louvain_levels(undirected, seed=seed)

        previous: list[Community] = []
        for level, partition in enumerate(levels):
            current: list[Community] = []
            for index, nodes in enumerate(sorted(partition, key=len, reverse=True)):
                community = Community(community_id=f"L{level}-C{index}", level=level, nodes=sorted(nodes))
                community.children = [c.community_id for c in previous if set(c.nodes) <= nodes]
                current.append(community)
                self.communities[community.community_id] = community
            previous = current

        for level in self.levels:  # bottom-up: children are summarized before parents
            for community in self.at_level(level):
                self._summarize(community, graph_store, use_ollama)
        return self

    def _summarize(self, community: Community, graph_store: GraphStore, use_ollama: bool) -> None:
        graph = graph_store.graph
        node_set = set(community.nodes)
        edges = [
            (u, v, data) for u, v, data in graph.edges(data=True)
            if u in node_set and v in node_set
        ]
        degree = Counter()
        for u, v, _ in edges:
            degree[u] += 1
            degree[v] += 1
        ranked_nodes = sorted(community.nodes, key=lambda n: (-degree[n], n))
        names = [graph.nodes[n]["name"] for n in ranked_nodes]
        community.key_entities = names[:8]
        community.rank = float(len(edges))
        community.title = ", ".join(names[:3]) + (" and others" if len(names) > 3 else "")

        if community.children:
            child_reports = [self.communities[c] for c in community.children]
            child_reports.sort(key=lambda c: c.rank, reverse=True)
            source = "\n".join(f"- {c.title}: {c.summary}" for c in child_reports)
        else:
            relations = [
                f"- {graph.nodes[u]['name']} --[{data['predicate']}]--> {graph.nodes[v]['name']}"
                for u, v, data in edges[:20]
            ]
            sentences = list(dict.fromkeys(data["sentence"] for _, _, data in edges))[:8]
            source = "Relations:\n" + "\n".join(relations) + "\nEvidence:\n" + "\n".join(
                f"- {s}" for s in sentences
            )

        if use_ollama:
            prompt = (
                "You are writing a report about one community of related entities in a knowledge graph. "
                "Using ONLY the information below, write a concise summary (3-5 sentences) describing "
                "the main entities, how they are connected, and the key facts about them.\n\n"
                f"Entities: {', '.join(names[:15])}\n\n{source}\n\nSummary:"
            )
            llm_summary = try_ollama(prompt, timeout=30.0)
            if llm_summary:
                community.summary = llm_summary
                return

        community.summary = self._extractive_summary(community, edges, graph)

    def _extractive_summary(self, community: Community, edges, graph) -> str:
        if community.children:
            child_reports = sorted(
                (self.communities[c] for c in community.children), key=lambda c: c.rank, reverse=True
            )
            return " ".join(c.summary for c in child_reports[:5])
        if not edges:
            return f"{community.title} appear in the corpus without extracted relations to other entities."
        sentences = list(dict.fromkeys(data["sentence"] for _, _, data in edges))
        return (
            f"This community centers on {', '.join(community.key_entities[:4])}. "
            + " ".join(sentences[:4])
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(pickle.dumps(self.communities))

    @classmethod
    def load(cls, path: str | Path) -> "CommunityStore":
        store = cls()
        store.communities = pickle.loads(Path(path).read_bytes())
        return store

    def __len__(self) -> int:
        return len(self.communities)
