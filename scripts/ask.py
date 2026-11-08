#!/usr/bin/env python3
"""CLI: ask a question against a persisted GraphRAG index.

Usage:
    python scripts/ask.py --index data/index "How is Nova Labs related to Aiko Tanaka?"
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graphrag import GraphRAGPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question against a persisted GraphRAG index.")
    parser.add_argument("--index", required=True, help="Directory containing a persisted index")
    parser.add_argument("--no-ollama", action="store_true", help="Skip trying a local Ollama server")
    parser.add_argument("--route", choices=["vector", "graph", "hybrid", "global", "baseline"],
                        help="Force a retrieval strategy instead of automatic routing")
    parser.add_argument("--level", type=int, default=None, help="Community level for global search")
    parser.add_argument("question", help="The question to ask")
    args = parser.parse_args()

    pipeline = GraphRAGPipeline.load(args.index)
    if args.route == "baseline":
        result = pipeline.ask_baseline(args.question, use_ollama=not args.no_ollama)
        result.update(reason="vector RAG baseline", matched_entities=[], graph_facts=[])
    else:
        result = pipeline.ask(args.question, use_ollama=not args.no_ollama, route=args.route, level=args.level)

    print(f"Route: {result['route']}  ({result['reason']})")
    if result.get("partial_answers"):
        print(f"
Community level {result['community_level']} partial answers (id, title, score):")
        for community_id, title, score in result["partial_answers"]:
            print(f"  - {community_id} {title}: {score}")
    if result["matched_entities"]:
        print(f"Matched entities: {', '.join(result['matched_entities'])}")
    if result["graph_facts"]:
        print("\nGraph facts used:")
        for subj, pred, obj in result["graph_facts"][:8]:
            print(f"  - {subj} --[{pred}]--> {obj}")
    if result["text_hits"]:
        print("\nText passages used:")
        for chunk_id, text, score in result["text_hits"]:
            print(f"  - ({score:.3f}) {text}")
    print(f"\nAnswer [{result['backend']}]:\n{result['answer']}")


if __name__ == "__main__":
    main()
