"""LLM code paths, tested with a fake Gemini client (no network, no API key)."""
import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graphrag import GraphRAGPipeline
from graphrag import llm as llm_module
from graphrag.llm import GeminiClient, QuotaExhausted

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_corpus.txt"

_PEOPLE = ["Aiko Tanaka", "Elena Reyes", "Marcus Webb"]
_ORGS = ["Nova Labs", "Orion Dynamics", "Helio Ventures", "Briar Robotics", "Project Kestrel"]


class FakeLLM:
    model = "fake-gemini"

    def __init__(self) -> None:
        self.calls = 0
        self.cache_hits = 0
        self.prompts: list[str] = []

    def generate(self, prompt: str, json_output: bool = False, temperature: float = 0.0) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        return "Combined answer mentioning Nova Labs and Orion Dynamics."

    def generate_json(self, prompt: str):
        self.calls += 1
        self.prompts.append(prompt)
        if prompt.startswith("Extract a knowledge graph"):
            text = prompt.split("Text:", 1)[1]
            names = [n for n in _PEOPLE + _ORGS if n in text]
            return {
                "entities": [{"name": n, "type": "PERSON" if n in _PEOPLE else "ORGANIZATION",
                              "description": f"{n} appears in the text."} for n in names],
                "relationships": [{"source": a, "target": b, "relation": "works with",
                                   "description": f"{a} works with {b}."} for a, b in zip(names, names[1:])],
            }
        if prompt.startswith("You are writing a report"):
            return {"title": "Drone research network", "summary": "Nova Labs works with Orion Dynamics."}
        if prompt.startswith("Using ONLY the community report"):
            return {"answer": "Nova Labs leads drone research.", "score": 80}
        return None

    def embed(self, texts, task="RETRIEVAL_DOCUMENT"):
        self.calls += 1
        return [[b / 255 for b in hashlib.sha256(t.encode()).digest()[:16]] for t in texts]


@pytest.fixture(scope="module")
def llm_pipeline():
    return GraphRAGPipeline(llm=FakeLLM()).build_from_file(SAMPLE_PATH)


def test_llm_extraction_builds_typed_graph(llm_pipeline) -> None:
    graph = llm_pipeline.graph_store.graph
    assert graph.nodes["aiko tanaka"]["type"] == "PERSON"
    assert graph.nodes["nova labs"]["description"]
    assert graph.number_of_edges() > 0


def test_llm_summaries_are_used(llm_pipeline) -> None:
    summaries = [c.summary for c in llm_pipeline.community_store.communities.values()]
    assert "Nova Labs works with Orion Dynamics." in summaries


def test_llm_global_answer_runs_map_and_reduce(llm_pipeline) -> None:
    result = llm_pipeline.ask("What are the main themes in this corpus?")
    assert result["backend"] == "gemini"
    assert result["partial_answers"]
    assert all(score == 80 for _, _, score in result["partial_answers"])


def test_baseline_uses_embeddings(llm_pipeline) -> None:
    result = llm_pipeline.ask_baseline("Who invested in Nova Labs?")
    assert result["backend"] == "gemini"
    assert result["text_hits"]


def test_embeddings_survive_save_and_load(llm_pipeline, tmp_path: Path) -> None:
    llm_pipeline.save(tmp_path / "index")
    reloaded = GraphRAGPipeline.load(tmp_path / "index", llm=FakeLLM())
    assert reloaded.embedding_store is not None
    assert reloaded.ask_baseline("Who invested in Nova Labs?")["text_hits"]


class _Response:
    def __init__(self, status: int, payload=None, text: str = "") -> None:
        self.status_code, self._payload, self.text = status, payload, text

    def json(self):
        return self._payload


def test_client_caches_responses(monkeypatch, tmp_path: Path) -> None:
    responses = [_Response(200, {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]})]
    monkeypatch.setattr(llm_module.requests, "post", lambda *a, **k: responses.pop(0))
    client = GeminiClient("test-key", rpm=6000, cache_dir=tmp_path)
    assert client.generate("hello") == "hi"
    assert client.generate("hello") == "hi"  # second call served from the cache
    assert client.calls == 1 and client.cache_hits == 1


def test_client_retries_rate_limit_then_stops_on_daily_quota(monkeypatch, tmp_path: Path) -> None:
    responses = [
        _Response(429, text='{"retryDelay": "0s"}'),
        _Response(429, text="Quota exceeded for GenerateRequestsPerDayPerProjectPerModel"),
    ]
    monkeypatch.setattr(llm_module.requests, "post", lambda *a, **k: responses.pop(0))
    monkeypatch.setattr(llm_module.time, "sleep", lambda s: None)
    client = GeminiClient("test-key", rpm=6000, cache_dir=tmp_path)
    with pytest.raises(QuotaExhausted):
        client.generate("hello")
    assert client.calls == 2
