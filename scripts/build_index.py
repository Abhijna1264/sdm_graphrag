#!/usr/bin/env python3
"""CLI: ingest a document and persist a GraphRAG index.

Usage:
    python scripts/build_index.py --input data/sample_corpus.txt --out data/index
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graphrag import GraphRAGPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a GraphRAG index from a text or PDF file.")
    parser.add_argument("--input", required=True, help="Path to a .txt or .pdf document")
    parser.add_argument("--out", required=True, help="Output directory for the persisted index")
    parser.add_argument("--no-ollama", action="store_true", help="Write extractive community summaries")
    args = parser.parse_args()

    pipeline = GraphRAGPipeline().build_from_file(args.input, use_ollama=not args.no_ollama)
    pipeline.save(args.out)

    print(f"Indexed {len(pipeline.chunks)} chunks from {args.input}")
    print(f"Knowledge graph: {pipeline.graph_store.graph.number_of_nodes()} entities, "
          f"{pipeline.graph_store.graph.number_of_edges()} relations")
    store = pipeline.community_store
    print("Communities: " + ", ".join(f"level {lv}: {len(store.at_level(lv))}" for lv in store.levels))
    print(f"Index saved to {args.out}")


if __name__ == "__main__":
    main()
