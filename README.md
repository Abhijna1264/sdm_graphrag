# graph-rag (original, fully-offline implementation)

A small, from-scratch GraphRAG pipeline: it ingests a document, builds both a
knowledge graph and a keyword/semantic index from it, routes each incoming
question to the retrieval strategy that fits it, and answers using only the
retrieved context.

This is an independent implementation of the general GraphRAG *pattern*
(chunk → dual index (graph + vector) → route → hybrid retrieve → answer),
written from scratch — no code, notebooks, or text were copied from any
existing GraphRAG project. Every design choice below is a deliberate,
documented tradeoff, not an attempt to reproduce a specific reference
implementation.

## Why it's fully offline

Everything runs with `pip install -r requirements.txt` and no API keys, no
model downloads, and no external database:

- **Entities & relations** are extracted with regex + heuristics (capitalized
  multi-word phrase detection, a small stopword/title/org-suffix list), not
  an NER model or an LLM. See "Design tradeoffs" below.
- **The knowledge graph** lives in memory as a `networkx.MultiDiGraph`
  (no Neo4j / other graph DB).
- **"Semantic" search** is TF-IDF + cosine similarity (`scikit-learn`), not
  a downloaded embedding model.
- **Keyword search** is classic BM25 (`rank_bm25`).
- **Answer generation** is extractive (it quotes the retrieved graph facts
  and passages, clearly labeled `[extractive]`).

## Architecture

```
   file (.txt/.pdf)
         |
         v
   ingest.py  -- sentence-aware chunking (overlapping windows)
         |
         v
   extraction.py -- regex-based entity + (subject, predicate, object) triples
         |                                   |
         v                                   v
   graph_store.py (NetworkX)      vector_store.py (TF-IDF) + keyword_store.py (BM25)
         \                                   /
          \                                 /
                    router.py
      classifies the question: vector | graph | hybrid
                       |
                       v
                 retriever.py
     fuses TF-IDF+BM25 rankings (reciprocal rank fusion)
        and pulls 1-2 hop graph neighbors / shortest paths
                       |
                       v
                 generator.py
      extractive answers from retrieved facts and passages
```

## Global search (communities + map-reduce)

Following Edge et al., *From Local to Global* (2024), the index also holds a
hierarchy of graph communities used to answer corpus-wide questions:

1. **Community detection** (`communities.py`): Louvain partitions from
   networkx give nested levels (level 0 = finest). Leiden is the paper's
   choice; Louvain is its predecessor and needs no extra dependency.
2. **Bottom-up summaries**: leaf communities are summarized from their
   relations and source sentences; higher levels from their children's
   summaries. Summaries are extractive.
3. **Map-reduce answering** (`global_search.py`): each community summary at
   the chosen level yields a partial answer with a 0-100 helpfulness score
   (map); the best ones are packed into a budget and merged (reduce).

`pipeline.ask_baseline()` is a conventional vector RAG baseline (top-k
TF-IDF chunks, no graph), and `scripts/evaluate.py` compares the two on
global questions: entity coverage, community diversity, context size,
and latency.

## Routing logic

`router.py` decides how to answer a question using simple, explainable rules
(no LLM classifier):

| Condition | Route |
|---|---|
| Corpus-wide wording ("main themes", "overview", "summarize", ...) | `global` — map-reduce over community summaries |
| Query mentions 2+ known graph entities | `graph` — walk the shortest path between them |
| Query mentions 1 known entity + a relational word ("related", "between", "connect", ...) | `hybrid` |
| Query mentions 1 known entity | `hybrid` — graph neighbors + text search |
| No known entities recognized | `vector` — plain TF-IDF + BM25 search |

## Design tradeoffs

- **Rule-based extraction over ML/LLM extraction**: keeps the whole project
  dependency-light and instantly runnable, at the cost of extraction quality
  (it will occasionally pick up a noisy "entity" like a capitalized title
  phrase, and relation predicates are just the cleaned text between two
  entity mentions rather than a true semantic relation label). Swapping in
  spaCy or an LLM-based extractor only requires replacing `extraction.py`;
  everything downstream (graph store, router, retriever) is unaffected.
- **TF-IDF instead of embeddings**: no model download, fully deterministic,
  good enough for keyword-adjacent semantic matching on a small corpus.
  A real embedding model would generalize better to paraphrases.
- **Extractive generation**: guarantees the demo always produces an answer
  with zero setup, at the cost of fluency (answers quote source sentences).

## Project layout

```
graphrag/
  ingest.py         # load .txt/.pdf, chunk into overlapping sentence windows
  extraction.py     # entity + relation triple extraction
  graph_store.py     # NetworkX graph: build, neighbors, shortest path, fuzzy lookup, persist
  vector_store.py    # TF-IDF + cosine similarity chunk search
  keyword_store.py   # BM25 chunk search
  router.py          # vector | graph | hybrid classification
  retriever.py       # reciprocal-rank-fusion text search + graph lookups
  generator.py       # context formatting + extractive answer synthesis
  communities.py     # hierarchical community detection + bottom-up summaries
  global_search.py   # map-reduce global answering over community summaries
  pipeline.py        # GraphRAGPipeline: build/save/load/ask/ask_baseline
scripts/
  build_index.py     # CLI: ingest a file, persist the index
  ask.py              # CLI: query a persisted index
  evaluate.py         # CLI: GraphRAG global search vs. vector RAG baseline
data/sample_corpus.txt  # short original demo document (fictional org/people)
tests/test_pipeline.py  # smoke tests, no external services required
```

## Setup

```bash
cd graph-rag
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Build an index from the bundled sample document (or point --input at your own .txt/.pdf)
python scripts/build_index.py --input data/sample_corpus.txt --out data/index

# Ask questions
python scripts/ask.py --index data/index "How is Nova Labs related to Aiko Tanaka?"
python scripts/ask.py --index data/index "Tell me about Project Kestrel"
python scripts/ask.py --index data/index "What does the drone software do for farms?"
python scripts/ask.py --index data/index "What are the main themes in this corpus?"
python scripts/ask.py --index data/index --route baseline "What are the main themes in this corpus?"

# Compare GraphRAG global search with the vector RAG baseline
python scripts/evaluate.py --input data/sample_corpus.txt
```

Or from Python:

```python
from graphrag import GraphRAGPipeline

pipeline = GraphRAGPipeline().build_from_file("data/sample_corpus.txt")
result = pipeline.ask("How is Nova Labs related to Aiko Tanaka?")
print(result["route"], result["answer"])
```

## Tests

```bash
python -m pytest tests -v
```
