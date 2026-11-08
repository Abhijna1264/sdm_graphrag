import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graphrag import GraphRAGPipeline
from graphrag.router import classify

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_corpus.txt"


@pytest.fixture(scope="module")
def pipeline() -> GraphRAGPipeline:
    return GraphRAGPipeline().build_from_file(SAMPLE_PATH, use_ollama=False)


def test_build_produces_chunks(pipeline: GraphRAGPipeline) -> None:
    assert len(pipeline.chunks) > 0


def test_graph_has_entities_and_relations(pipeline: GraphRAGPipeline) -> None:
    graph = pipeline.graph_store.graph
    assert graph.number_of_nodes() > 5
    assert graph.number_of_edges() > 5
    assert "nova labs" in graph.nodes
    assert "aiko tanaka" in graph.nodes


def test_vector_search_finds_relevant_chunk(pipeline: GraphRAGPipeline) -> None:
    hits = pipeline.vector_store.search("agricultural drone swarm coordination", top_k=3)
    assert hits
    assert any("kestrel" in hit.text.lower() for hit in hits)


def test_keyword_search_finds_relevant_chunk(pipeline: GraphRAGPipeline) -> None:
    hits = pipeline.keyword_store.search("Singapore industrial sensors", top_k=3)
    assert hits
    assert any("orion dynamics" in hit.text.lower() for hit in hits)


def test_router_picks_graph_for_two_entity_relational_query(pipeline: GraphRAGPipeline) -> None:
    decision = classify("How is Nova Labs related to Aiko Tanaka?", pipeline.graph_store)
    assert decision.route == "graph"
    assert len(decision.matched_entities) >= 2


def test_router_picks_vector_for_generic_query(pipeline: GraphRAGPipeline) -> None:
    decision = classify("What is machine learning used for in general?", pipeline.graph_store)
    assert decision.route == "vector"
    assert decision.matched_entities == []


def test_ask_returns_non_empty_answer_without_ollama(pipeline: GraphRAGPipeline) -> None:
    result = pipeline.ask("How is Nova Labs related to Aiko Tanaka?", use_ollama=False)
    assert result["backend"] == "extractive"
    assert result["answer"].strip()
    assert result["route"] == "graph"


def test_ask_hybrid_route_includes_graph_and_text(pipeline: GraphRAGPipeline) -> None:
    result = pipeline.ask("Tell me about Project Kestrel", use_ollama=False)
    assert result["route"] == "hybrid"
    assert result["graph_facts"] or result["text_hits"]


def test_save_and_load_round_trip(pipeline: GraphRAGPipeline, tmp_path: Path) -> None:
    out_dir = tmp_path / "index"
    pipeline.save(out_dir)
    reloaded = GraphRAGPipeline.load(out_dir)
    result = reloaded.ask("How is Nova Labs related to Aiko Tanaka?", use_ollama=False)
    assert result["answer"].strip()


def test_communities_are_hierarchical(pipeline: GraphRAGPipeline) -> None:
    store = pipeline.community_store
    assert store.levels
    finest = store.at_level(store.levels[0])
    covered = {node for community in finest for node in community.nodes}
    assert covered == set(pipeline.graph_store.graph.nodes)
    assert all(community.summary for community in store.communities.values())
    for level in store.levels[1:]:
        for community in store.at_level(level):
            assert community.children


def test_router_picks_global_for_sensemaking_query(pipeline: GraphRAGPipeline) -> None:
    decision = classify("What are the main themes in this corpus?", pipeline.graph_store)
    assert decision.route == "global"


def test_global_answer_uses_map_reduce(pipeline: GraphRAGPipeline) -> None:
    result = pipeline.ask("What are the main themes in this corpus?", use_ollama=False)
    assert result["route"] == "global"
    assert result["partial_answers"]
    assert result["answer"].strip()


def test_baseline_uses_only_text_chunks(pipeline: GraphRAGPipeline) -> None:
    result = pipeline.ask_baseline("Who invested in Nova Labs?", use_ollama=False)
    assert result["route"] == "baseline"
    assert result["text_hits"]


def test_communities_survive_save_and_load(pipeline: GraphRAGPipeline, tmp_path: Path) -> None:
    pipeline.save(tmp_path / "index")
    reloaded = GraphRAGPipeline.load(tmp_path / "index")
    assert len(reloaded.community_store) == len(pipeline.community_store)
