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
summarized from the summaries of the sub-communities it contains.

With a Gemini client the summaries are LLM-written reports (title + summary);
without one they are extractive (key entities plus the source sentences behind
their relations). To save free-tier calls, communities smaller than
`min_llm_size` stay extractive, and a parent identical to its only child
reuses the child's report.
"""
from __future__ import annotations

import pickle
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

_SUMMARY_PROMPT = """You are writing a report about one community of related entities in a
knowledge graph built from a document collection. Using ONLY the information below, return
JSON {{"title": short descriptive title (max 8 words), "summary": 4-6 sentences describing
the main entities, how they are connected, and the most important facts}}.

{source}
"""

from .graph_store import GraphStore


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

    def build(
        self, graph_store: GraphStore, llm=None, seed: int = 42, min_llm_size: int = 3
    ) -> "CommunityStore":
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

        if llm is not None:
            print(f"Writing community summaries for {len(self.levels)} levels...", flush=True)
        for level in self.levels:  # bottom-up: children are summarized before parents
            communities = self.at_level(level)
            for number, community in enumerate(communities, 1):
                self._summarize(community, graph_store, llm, min_llm_size)
                if llm is not None:  # overwrite one line in place: "level 0: 37/167"
                    print(f"\r  level {level}: {number}/{len(communities)} communities", end="", flush=True)
            if llm is not None:
                print(" - done", flush=True)
        return self

    def _summarize(self, community: Community, graph_store: GraphStore, llm, min_llm_size: int) -> None:
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

        if len(community.children) == 1:
            child = self.communities[community.children[0]]
            if set(child.nodes) == node_set:
                community.title, community.summary = child.title, child.summary
                return
        if llm is not None and len(community.nodes) >= min_llm_size:
            report = llm.generate_json(_SUMMARY_PROMPT.format(source=self._llm_source(community, edges, graph)))
            if isinstance(report, dict) and report.get("summary"):
                community.title = str(report.get("title") or community.title)
                community.summary = str(report["summary"])
                return
        community.summary = self._extractive_summary(community, edges, graph)

    def _llm_source(self, community: Community, edges, graph, max_chars: int = 12000) -> str:
        if community.children:
            children = sorted((self.communities[c] for c in community.children), key=lambda c: c.rank, reverse=True)
            text = "Sub-community reports:\n" + "\n".join(f"- {c.title}: {c.summary}" for c in children)
        else:
            entities = [
                f"- {graph.nodes[n]['name']} ({graph.nodes[n].get('type', '')}): {graph.nodes[n].get('description', '')}"
                for n in community.nodes
            ]
            relations = list(dict.fromkeys(
                f"- {graph.nodes[u]['name']} --[{d['predicate']}]--> {graph.nodes[v]['name']}: {d['sentence']}"
                for u, v, d in edges
            ))
            text = "Entities:\n" + "\n".join(entities) + "\n\nRelationships:\n" + "\n".join(relations)
        return text[:max_chars]  # keeps the prompt bounded for very large communities

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
