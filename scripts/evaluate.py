#!/usr/bin/env python3
"""CLI: compare GraphRAG global search against a conventional vector RAG baseline.

For each global question both systems answer, and we report:
  - comprehensiveness proxy: share of graph entities mentioned in the answer
  - diversity proxy: number of distinct communities the answer touches
  - context size (characters fed to the generator) and latency
  - optionally, an LLM-as-judge head-to-head (requires a local Ollama server)
    on the four criteria from Edge et al. (2024): comprehensiveness,
    diversity, empowerment, directness

Usage:
    python scripts/evaluate.py --input data/sample_corpus.txt
    python scripts/evaluate.py --index data/index --questions my_questions.txt --judge
"""
import argparse
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graphrag import GraphRAGPipeline  # noqa: E402
from graphrag.llm import try_ollama  # noqa: E402

DEFAULT_QUESTIONS = [
    "What are the main themes in this corpus?",
    "Give an overview of the organizations and how they are connected.",
    "What are the most important investments and partnerships described?",
    "Summarize the key people and their roles across the whole corpus.",
]

CRITERIA = {
    "comprehensiveness": "How much detail does the answer provide to cover all aspects of the question?",
    "diversity": "How varied and rich is the answer in providing different perspectives and insights?",
    "empowerment": "How well does the answer help the reader understand and make informed judgements?",
    "directness": "How specifically and clearly does the answer address the question?",
}


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


def judge(question: str, answer_a: str, answer_b: str, criterion: str) -> str | None:
    prompt = (
        f"Question: {question}\n\nAnswer 1:\n{answer_a}\n\nAnswer 2:\n{answer_b}\n\n"
        f"Criterion - {criterion}: {CRITERIA[criterion]}\n"
        "Which answer is better on this criterion? Reply with exactly one of: 1, 2, TIE."
    )
    response = try_ollama(prompt, timeout=60.0)
    if not response:
        return None
    match = re.search(r"\b(1|2|TIE)\b", response.upper())
    return match.group(1) if match else None


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
    parser.add_argument("--no-ollama", action="store_true", help="Use extractive answers only")
    parser.add_argument("--judge", action="store_true", help="Run LLM-as-judge head-to-head via Ollama")
    args = parser.parse_args()

    use_ollama = not args.no_ollama
    if args.input:
        pipeline = GraphRAGPipeline().build_from_file(args.input, use_ollama=use_ollama)
    else:
        pipeline = GraphRAGPipeline.load(args.index)
    questions = (
        [q.strip() for q in Path(args.questions).read_text(encoding="utf-8").splitlines() if q.strip()]
        if args.questions else DEFAULT_QUESTIONS
    )

    systems = {
        "graphrag": lambda q: pipeline.ask(q, use_ollama=use_ollama, route="global", level=args.level),
        "vector_rag": lambda q: pipeline.ask_baseline(q, use_ollama=use_ollama),
    }
    metrics = {name: {"coverage": [], "diversity": [], "context": [], "latency": []} for name in systems}
    wins = {c: {"graphrag": 0, "vector_rag": 0, "tie": 0} for c in CRITERIA}
    judged = False

    for question in questions:
        print(f"\n=== {question}")
        answers = {}
        for name, fn in systems.items():
            result, latency = run_system(fn, question)
            answers[name] = result["answer"]
            m = metrics[name]
            m["coverage"].append(entity_coverage(result["answer"], pipeline))
            m["diversity"].append(community_diversity(result["answer"], pipeline))
            m["context"].append(len(result.get("context", "")))
            m["latency"].append(latency)
            print(f"\n[{name} | {result['backend']}]\n{result['answer']}")

        if args.judge:
            for criterion in CRITERIA:
                # Judge both orderings to cancel position bias, as in the paper.
                first = judge(question, answers["graphrag"], answers["vector_rag"], criterion)
                second = judge(question, answers["vector_rag"], answers["graphrag"], criterion)
                if first is None or second is None:
                    continue
                judged = True
                score = {"1": 1, "2": -1, "TIE": 0}[first] - {"1": 1, "2": -1, "TIE": 0}[second]
                key = "graphrag" if score > 0 else "vector_rag" if score < 0 else "tie"
                wins[criterion][key] += 1

    print("\n\n=== Summary (mean over questions)")
    print(f"{'system':<12}{'entity cov.':>12}{'communities':>13}{'context chars':>15}{'latency s':>11}")
    for name, m in metrics.items():
        print(
            f"{name:<12}{statistics.mean(m['coverage']):>12.2%}{statistics.mean(m['diversity']):>13.1f}"
            f"{statistics.mean(m['context']):>15.0f}{statistics.mean(m['latency']):>11.3f}"
        )
    store = pipeline.community_store
    print(f"\nCommunity levels: " + ", ".join(f"L{lv}={len(store.at_level(lv))}" for lv in store.levels))

    if args.judge:
        if not judged:
            print("\nLLM judge skipped: no Ollama server reachable at localhost:11434.")
        else:
            print("\n=== LLM-as-judge head-to-head (GraphRAG wins / vector RAG wins / ties)")
            for criterion, w in wins.items():
                print(f"{criterion:<18}{w['graphrag']:>4} /{w['vector_rag']:>4} /{w['tie']:>4}")


if __name__ == "__main__":
    main()
