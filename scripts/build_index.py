#!/usr/bin/env python3
"""CLI: ingest a document (or a directory of them) and persist a GraphRAG index.

Uses Gemini when GEMINI_API_KEY is set, otherwise the offline extractors.

Usage:
    python scripts/build_index.py --input data/corpus --out data/index
    python scripts/build_index.py --input data/corpus --estimate   # no API calls
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252

from graphrag import GraphRAGPipeline  # noqa: E402
from graphrag.ingest import ingest_path  # noqa: E402
from graphrag.llm import QuotaExhausted  # noqa: E402
from graphrag.pipeline import _LLM_CHUNKING  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a GraphRAG index from a text/PDF file or directory.")
    parser.add_argument("--input", required=True, help="Path to a .txt/.pdf document or a directory of them")
    parser.add_argument("--out", default="data/index", help="Output directory for the persisted index")
    parser.add_argument("--offline", action="store_true", help="Ignore GEMINI_API_KEY and build offline")
    parser.add_argument("--estimate", action="store_true", help="Only estimate Gemini calls; build nothing")
    args = parser.parse_args()

    if args.estimate:
        chunks = ingest_path(args.input, **_LLM_CHUNKING)
        chars = sum(len(c.text) for c in chunks)
        print(f"{len(chunks)} chunks, ~{chars // 4} tokens")
        print(f"Estimated Gemini requests: {len(chunks)} extraction + roughly {len(chunks)}-{2 * len(chunks)} "
              f"community summaries + {len(chunks) // 100 + 1} embedding batch(es)")
        return

    pipeline = GraphRAGPipeline(llm=None if args.offline else "auto")
    if pipeline.llm is None and not args.offline:
        print("GEMINI_API_KEY not set -> building with the offline extractors.")
    try:
        pipeline.build_from_file(args.input)
    except QuotaExhausted as exc:
        sys.exit(f"\n!! {exc}")
    pipeline.save(args.out)

    print(f"Indexed {len(pipeline.chunks)} chunks from {args.input}")
    print(f"Knowledge graph: {pipeline.graph_store.graph.number_of_nodes()} entities, "
          f"{pipeline.graph_store.graph.number_of_edges()} relations")
    store = pipeline.community_store
    print("Communities: " + ", ".join(f"level {lv}: {len(store.at_level(lv))}" for lv in store.levels))
    if pipeline.llm is not None:
        print(f"Gemini requests: {pipeline.llm.calls} (cache hits: {pipeline.llm.cache_hits})")
    print(f"Index saved to {args.out}")


if __name__ == "__main__":
    main()
