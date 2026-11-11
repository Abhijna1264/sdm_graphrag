#!/usr/bin/env python3
"""CLI: compare GraphRAG global search against a conventional vector RAG baseline.

For each global question both systems answer, and we report:
  - comprehensiveness proxy: share of graph entities mentioned in the answer
  - diversity proxy: number of distinct communities the answer touches
  - context size (characters fed to the generator) and latency

Usage:
    python scripts/evaluate.py --input data/sample_corpus.txt
    python scripts/evaluate.py --index data/index --questions my_questions.txt
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graphrag import GraphRAGPipeline  # noqa: E402

DEFAULT_QUESTIONS = [
    "What are the main themes in this corpus?",
    "Give an overview of the organizations and how they are connected.",
    "What are the most important investments and partnerships described?",
    "Summarize the key people and their roles across the whole corpus.",
]


def entity_coverage(answer: str, pipeline: GraphRAGPipeline) -> float:
    names = pipeline.graph_store.node_names
    if not names:
        return 0.0
    answer_lower = answer.lower()
    return sum(name.lower() in answer_lower for name in names) / len(names)


def community_diversity(answer: str, pipeline: GraphRAGPipeline) -> int:
    store = pipeline.community_store
    if not store.levels:
        return 0
    answer_lower = answer.lower()
    return sum(
        any(name.lower() in answer_lower for name in community.key_entities)
        for community in store.at_level(store.levels[0])
    )


def run_system(fn, question: str) -> tuple[dict, float]:
    start = time.perf_counter()
    result = fn(question)
    return result, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GraphRAG global search vs. vector RAG.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Build a fresh index from this .txt/.pdf file")
    source.add_argument("--index", help="Load a persisted index directory")
    parser.add_argument("--questions", help="File with one global question per line")
    parser.add_argument("--level", type=int, default=None, help="Community level for global search")
    args = parser.parse_args()

    if args.input:
        pipeline = GraphRAGPipeline().build_from_file(args.input)
    else:
        pipeline = GraphRAGPipeline.load(args.index)
    questions = (
        [q.strip() for q in Path(args.questions).read_text(encoding="utf-8").splitlines() if q.strip()]
        if args.questions else DEFAULT_QUESTIONS
    )

    systems = {
        "graphrag": lambda q: pipeline.ask(q, route="global", level=args.level),
        "vector_rag": lambda q: pipeline.ask_baseline(q),
    }
    metrics = {name: {"coverage": [], "diversity": [], "context": [], "latency": []} for name in systems}

    for question in questions:
        print(f"\n=== {question}")
        for name, fn in systems.items():
            result, latency = run_system(fn, question)
            m = metrics[name]
            m["coverage"].append(entity_coverage(result["answer"], pipeline))
            m["diversity"].append(community_diversity(result["answer"], pipeline))
            m["context"].append(len(result.get("context", "")))
            m["latency"].append(latency)
            print(f"\n[{name} | {result['backend']}]\n{result['answer']}")

    print("\n\n=== Summary (mean over questions)")
    print(f"{'system':<12}{'entity cov.':>12}{'communities':>13}{'context chars':>15}{'latency s':>11}")
    for name, m in metrics.items():
        print(
            f"{name:<12}{statistics.mean(m['coverage']):>12.2%}{statistics.mean(m['diversity']):>13.1f}"
            f"{statistics.mean(m['context']):>15.0f}{statistics.mean(m['latency']):>11.3f}"
        )
    store = pipeline.community_store
    print("\nCommunity levels: " + ", ".join(f"L{lv}={len(store.at_level(lv))}" for lv in store.levels))


if __name__ == "__main__":
    main()
