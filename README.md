# GraphRAG: from local to global query-focused summarization

SDM term project. A from-scratch implementation of GraphRAG (Edge et al.,
*From Local to Global: A Graph RAG Approach to Query-Focused Summarization*,
2024), compared against a conventional vector RAG baseline on corpus-wide
"sensemaking" questions.

Pipeline: documents → chunks → LLM entity/relation extraction → knowledge
graph → hierarchical communities → bottom-up community summaries →
map-reduce global answers.

The LLM is Google Gemini (free tier is enough). Without an API key the whole
pipeline still runs with offline fallbacks (regex extraction, TF-IDF,
extractive summaries); the tests use that mode.

## Dataset

`data/corpus/` holds 32 English Wikipedia articles on **space exploration**
(Space Race, Apollo, ISS, ESA, ISRO, CNSA, SpaceX, telescopes, rovers, ...),
each truncated to ~8,000 characters: about 65k tokens in total, sized so an
index builds within one day of Gemini free-tier quota. Re-download with
`python scripts/fetch_dataset.py`. Wikipedia text is licensed CC BY-SA 4.0.

`data/questions.txt` holds 12 global questions (themes, trends, comparisons
across the whole corpus) of the kind the paper evaluates.

## Architecture

```
   data/corpus/*.txt
         |
   ingest.py        -- sentence-aware chunks (~1k tokens with Gemini)
         |
   extraction.py    -- Gemini: typed entities + described relationships (1 call/chunk)
         |
   graph_store.py   -- NetworkX knowledge graph
         |
   communities.py   -- Louvain hierarchy (level 0 = finest) + bottom-up Gemini reports
         |
   global_search.py -- map: partial answer + 0-100 score per community
                       reduce: merge the best partial answers into one answer

   baseline: vector_store.py (Gemini embeddings) -> top-k chunks -> Gemini answer
```

Local questions (about specific entities) are also supported: `router.py`
sends them to graph traversal + hybrid TF-IDF/BM25 text search.

| Condition | Route |
|---|---|
| Corpus-wide wording ("main themes", "overview", "summarize", ...) | `global` — map-reduce over community summaries |
| Query mentions 2+ known graph entities | `graph` — shortest path between them |
| Query mentions 1 known entity | `hybrid` — graph neighbors + text search |
| No known entities recognized | `vector` — TF-IDF + BM25 search |

## Free-tier design

`graphrag/llm.py` talks to the Gemini REST API and is built around free-tier
limits:

- **Rate limiting**: requests are spaced to `GEMINI_RPM` (default 10/min).
- **Retries**: 429 and 5xx responses are retried, honouring Gemini's retry delay.
- **Disk cache** (`.cache/gemini/`): every response is cached, so if the
  daily quota runs out mid-build, rerunning the same command resumes from
  where it stopped at no extra cost.
- **Budget**: large chunks (one extraction call each), extractive summaries
  for communities under 3 entities, reused reports when a parent community
  equals its only child, at most 15 communities mapped per question, and one
  judge call per answer order covering all four criteria.
  `build_index.py --estimate` prints the expected call count without calling.

## Setup

```bash
pip install -r requirements.txt
```

Get a free API key at https://aistudio.google.com/apikey and set it as an
environment variable (never put it in the code). On Windows:

```bash
setx GEMINI_API_KEY "your-key"
```

Open a new terminal afterwards. Optional settings: `GEMINI_MODEL` (default
`gemini-2.5-flash-lite`), `GEMINI_EMBED_MODEL` (default
`gemini-embedding-001`), `GEMINI_RPM` (default 10).

## Usage

```bash
# 1. (optional) check the API budget, no calls made
python scripts/build_index.py --input data/corpus --estimate

# 2. build the index (~60 extraction + ~60-120 summary calls; takes a while at 10 RPM)
python scripts/build_index.py --input data/corpus --out data/index

# 3. ask questions
python scripts/ask.py --index data/index "What are the main themes in this collection?"
python scripts/ask.py --index data/index --route baseline "What are the main themes in this collection?"
python scripts/ask.py --index data/index "How is Wernher von Braun related to Saturn V?"

# 4. evaluate GraphRAG vs vector RAG with the LLM judge; add --levels 1,2,3 to compare levels
python scripts/evaluate.py --index data/index --questions data/questions.txt --out results/eval.json
```

`evaluate.py` reports, per system: entity coverage and community diversity
of the answers (cheap proxies), context size, latency, and the LLM-judge
win rate against vector RAG on comprehensiveness, diversity, empowerment and
directness, the paper's four criteria. Each pair is judged in both orders to
cancel position bias. Full answers and verdicts go to the JSON file.

## Differences from the paper

- **Louvain instead of Leiden** for community detection: Louvain ships with
  networkx; Leiden mainly adds a guarantee that communities are connected.
- **Smaller scale**: ~65k tokens instead of ~1M, with no "gleaning" passes
  during extraction and no claim extraction, to fit the free tier.
- **Map step capped** at the 15 largest communities of the chosen level.
- **Offline fallbacks** (regex extraction, TF-IDF, extractive text) exist so
  the code runs and is testable without a key; results for the report should
  come from a Gemini-built index.

## Project layout

```
graphrag/
  llm.py             # Gemini client: rate limit, retries, disk cache, embeddings
  ingest.py          # load .txt/.pdf (file or directory), sentence-aware chunking
  extraction.py      # Gemini entity/relation extraction (+ offline regex fallback)
  graph_store.py     # NetworkX graph: build, neighbors, shortest path, persist
  communities.py     # hierarchical communities + bottom-up summaries
  global_search.py   # map-reduce global answering over community summaries
  vector_store.py    # Gemini embedding store (baseline) + TF-IDF store
  keyword_store.py   # BM25 chunk search
  router.py          # global | graph | hybrid | vector routing
  retriever.py       # reciprocal-rank-fusion text search + graph lookups
  generator.py       # answers from retrieved context (Gemini or extractive)
  pipeline.py        # GraphRAGPipeline: build/save/load/ask/ask_baseline
scripts/
  fetch_dataset.py   # download the Wikipedia corpus
  build_index.py     # build and persist an index
  ask.py             # query an index
  evaluate.py        # GraphRAG vs. vector RAG evaluation
data/
  corpus/            # evaluation corpus (32 Wikipedia articles)
  questions.txt      # global evaluation questions
  sample_corpus.txt  # tiny fictional document used by the tests
tests/               # offline tests + fake-Gemini tests (no network needed)
```

## Tests

```bash
python -m pytest tests -v
```
