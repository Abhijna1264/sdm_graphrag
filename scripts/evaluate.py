#!/usr/bin/env python3
"""CLI: compare GraphRAG global search against a conventional vector RAG baseline.

For each global question every system answers, and we report:
  - comprehensiveness proxy: share of graph entities mentioned in the answer
  - diversity proxy: number of distinct level-0 communities the answer touches
  - context size (characters fed to the generator) and latency
  - with Gemini: an LLM-as-judge head-to-head against the baseline on the four
    criteria from Edge et al. (2024): comprehensiveness, diversity,
    empowerment, directness. Each pair is judged in both answer orders to
    cancel position bias.

Results are also written to a JSON file for the report.

Usage:
    python scripts/evaluate.py --index data/index --questions data/questions.txt
    python scripts/evaluate.py --index data/index --levels 0,1,2 --out results/eval.json
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252

from graphrag import GraphRAGPipeline  # noqa: E402
from graphrag.global_search import default_level  # noqa: E402
from graphrag.llm import QuotaExhausted  # noqa: E402

DEFAULT_QUESTIONS = [
    "What are the main themes in this corpus?",
    "Give an overview of the organizations and how they are connected.",
    "What are the most important investments and partnerships described?",
    "Summarize the key people and their roles across the whole corpus.",
]

CRITERIA = {
    "comprehensiveness": "How much detail does the answer provide to cover all aspects and details of the question?",
    "diversity": "How varied and rich is the answer in providing different perspectives and insights on the question?",
    "empowerment": "How well does the answer help the reader understand and make informed judgements about the topic?",
    "directness": "How specifically and clearly does the answer address the question?",
}

_JUDGE_PROMPT = """You are judging two answers to the same question about a document collection.

Question: {question}

Answer 1:
{a}

Answer 2:
{b}

For each criterion below, decide which answer is better: "1", "2", or "tie".
{criteria}

Return JSON mapping each criterion name to {{"winner": "1" | "2" | "tie", "reason": one short sentence}}.
"""


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


def judge(llm, question: str, answer_a: str, answer_b: str) -> dict[str, str]:
    criteria = "\n".join(f"- {name}: {text}" for name, text in CRITERIA.items())
    data = llm.generate_json(_JUDGE_PROMPT.format(question=question, a=answer_a, b=answer_b, criteria=criteria))
    verdicts = {}
    for name in CRITERIA:
        item = data.get(name) if isinstance(data, dict) else None
        winner = str(item.get("winner", "tie") if isinstance(item, dict) else item or "tie").lower()
        verdicts[name] = winner if winner in ("1", "2") else "tie"
    return verdicts


def head_to_head(llm, question: str, graph_answer: str, base_answer: str) -> dict[str, str]:
    """Judge in both orders; a system wins a criterion only if it wins on balance."""
    first = judge(llm, question, graph_answer, base_answer)
    second = judge(llm, question, base_answer, graph_answer)
    value = {"1": 1, "2": -1, "tie": 0}
    outcome = {}
    for name in CRITERIA:
        score = value[first[name]] - value[second[name]]
        outcome[name] = "graphrag" if score > 0 else "baseline" if score < 0 else "tie"
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GraphRAG global search vs. vector RAG.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Build a fresh index from this .txt/.pdf file or directory")
    source.add_argument("--index", help="Load a persisted index directory")
    parser.add_argument("--questions", help="File with one global question per line")
    parser.add_argument("--levels", default=None, help="Comma-separated community levels to test (default: auto)")
    parser.add_argument("--no-judge", action="store_true", help="Skip the LLM judge to save API calls")
    parser.add_argument("--out", default="results/eval.json", help="Where to write the JSON results")
    args = parser.parse_args()

    pipeline = GraphRAGPipeline().build_from_file(args.input) if args.input else GraphRAGPipeline.load(args.index)
    llm = pipeline.llm
    questions = (
        [q.strip() for q in Path(args.questions).read_text(encoding="utf-8").splitlines() if q.strip()]
        if args.questions else DEFAULT_QUESTIONS
    )
    store = pipeline.community_store
    levels = [int(x) for x in args.levels.split(",")] if args.levels else [default_level(store)]

    systems = {f"graphrag_L{lv}": (lambda q, lv=lv: pipeline.ask(q, route="global", level=lv)) for lv in levels}
    systems["vector_rag"] = lambda q: pipeline.ask_baseline(q)
    metrics = {name: {"coverage": [], "diversity": [], "context": [], "latency": []} for name in systems}
    wins = {name: {c: {"graphrag": 0, "baseline": 0, "tie": 0} for c in CRITERIA} for name in systems if name != "vector_rag"}
    records = []
    use_judge = llm is not None and not args.no_judge

    try:
        for number, question in enumerate(questions, 1):
            print(f"\n=== [{number}/{len(questions)}] {question}", flush=True)
            answers = {}
            for name, fn in systems.items():
                start = time.perf_counter()
                result = fn(question)
                latency = time.perf_counter() - start
                answers[name] = result["answer"]
                m = metrics[name]
                m["coverage"].append(entity_coverage(result["answer"], pipeline))
                m["diversity"].append(community_diversity(result["answer"], pipeline))
                m["context"].append(len(result.get("context", "")))
                m["latency"].append(latency)
                print(f"\n[{name} | {result['backend']}]\n{result['answer']}", flush=True)
            record = {"question": question, "answers": answers, "judgements": {}}
            if use_judge:
                for name in wins:
                    outcome = head_to_head(llm, question, answers[name], answers["vector_rag"])
                    record["judgements"][name] = outcome
                    for criterion, winner in outcome.items():
                        wins[name][criterion][winner] += 1
            records.append(record)
    except QuotaExhausted as exc:
        print(f"\n!! {exc}\nReporting the {len(records)} question(s) finished so far.")

    if not records:
        return
    print("\n\n=== Summary (mean over questions)")
    print(f"{'system':<14}{'entity cov.':>12}{'communities':>13}{'context chars':>15}{'latency s':>11}")
    summary = {}
    for name, m in metrics.items():
        n = len(records)
        row = {k: statistics.mean(v[:n]) for k, v in m.items()}
        summary[name] = row
        print(
            f"{name:<14}{row['coverage']:>12.2%}{row['diversity']:>13.1f}"
            f"{row['context']:>15.0f}{row['latency']:>11.2f}"
        )
    print("\nCommunity levels: " + ", ".join(f"L{lv}={len(store.at_level(lv))}" for lv in store.levels))

    if use_judge:
        print("\n=== LLM judge vs. vector RAG: GraphRAG win rate (wins / losses / ties)")
        for name, by_criterion in wins.items():
            print(f"{name}")
            for criterion, w in by_criterion.items():
                total = sum(w.values()) or 1
                print(f"  {criterion:<18}{w['graphrag'] / total:>6.0%}   ({w['graphrag']} / {w['baseline']} / {w['tie']})")
    elif llm is None:
        print("\nLLM judge skipped: set GEMINI_API_KEY to enable it.")
    if llm is not None:
        print(f"\nGemini requests this run: {llm.calls} (cache hits: {llm.cache_hits})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"levels": levels, "summary": summary, "wins": wins, "questions": records}, indent=2),
        encoding="utf-8",
    )
    print(f"Results written to {out}")


if __name__ == "__main__":
    main()
