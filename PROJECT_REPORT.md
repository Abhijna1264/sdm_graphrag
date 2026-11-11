# GraphRAG — Implementation and Results

**SDM term project.** A from-scratch implementation of GraphRAG
(Edge et al., *From Local to Global: A Graph RAG Approach to Query-Focused
Summarization*, 2024), evaluated against a conventional vector-RAG baseline on
corpus-wide "sensemaking" questions.

This document covers the objective, architecture, every module, the data, the
Gemini free-tier design, the evaluation methodology, and the final results.

---

## 1. Objective

Answer **global** questions that span an entire document collection — "what are
the main themes?", "summarize the key people" — where no single passage contains
the answer. GraphRAG attacks this by building a knowledge graph, partitioning it
into a hierarchy of communities, summarizing each community, and answering
global questions with a map-reduce over those summaries. The project compares
that pipeline head-to-head with a standard vector-RAG baseline (top-k chunks by
embedding similarity) using both cheap automatic proxies and an LLM-as-judge.

Pipeline in one line:

```
documents -> chunks -> LLM entity/relation extraction -> knowledge graph
          -> hierarchical communities -> bottom-up summaries -> map-reduce answers
```

The LLM is Google Gemini (the free tier is sufficient). With no API key the
whole pipeline still runs using offline fallbacks (regex extraction, TF-IDF,
extractive summaries), which is also what the (now-removed) test suite used.

---

## 2. Architecture

### 2.1 Data flow

```
   data/*.txt  (or a directory of .txt/.pdf)
         |
   ingest.py        -- sentence-aware, overlapping chunks
         |              (LLM: 40 sentences/chunk, overlap 3; offline: 6 / 1)
   extraction.py    -- Gemini: typed entities + described relationships (1 call/chunk)
         |
   graph_store.py   -- NetworkX MultiDiGraph knowledge graph
         |
   communities.py   -- Louvain hierarchy (level 0 = finest) + bottom-up summaries
         |
   global_search.py -- map: partial answer + 0-100 score per community
                       reduce: merge the best partial answers into one answer

   baseline: vector_store.py (Gemini embeddings) -> top-k chunks -> Gemini answer
```

Local questions (about specific entities) are also supported: `router.py` sends
them to graph traversal and/or hybrid TF-IDF + BM25 text search.

### 2.2 Module-by-module

**`graphrag/ingest.py` — loading and chunking.**
Reads `.txt` (and `.pdf` via `pypdf`). Splits text into sentences with a regex,
first protecting abbreviations ("Dr.", "Inc.") so they are not mistaken for
sentence boundaries. Sentences are grouped into overlapping windows so a chunk
boundary never cuts a sentence in half (extraction runs per sentence). Chunking
parameters depend on whether an LLM is present:

| mode | `max_sentences` | `overlap_sentences` | rationale |
|---|---|---|---|
| Gemini | 40 (~1k tokens) | 3 | one extraction call per chunk, close to the paper's 600-token setting |
| offline | 6 | 1 | regex extraction is cheap; smaller chunks give finer resolution |

**`graphrag/extraction.py` — entity and relation extraction.**
`llm_extract()` issues one Gemini call per chunk with a JSON-schema prompt that
asks for entities (`name`, `type` in PERSON / ORGANIZATION / MISSION /
SPACECRAFT / LOCATION / EVENT / CONCEPT / OTHER, one-sentence `description`) and
relationships (`source`, `target`, short verb-phrase `relation`, one-sentence
`description`). It caps output at 25 entities and 30 relationships per chunk and
requires relationship endpoints to be declared entities. Names are canonicalised
(whitespace collapsed, surrounding punctuation stripped) and entities are keyed
by lowercase name; duplicate descriptions are concatenated.

The offline path (`extract()`) uses no model: entities are capitalised-phrase
regex matches with heuristic typing (org suffixes, person titles), and a relation
is the cleaned-up text span between two entities co-occurring in a sentence.
This is intentionally simple and inspectable, not state-of-the-art NER/RE.

**`graphrag/graph_store.py` — knowledge graph.**
A `networkx.MultiDiGraph`. Nodes are keyed by lowercase name with attributes
`name`, `type`, `mentions`, `description`; edges carry `predicate`, `sentence`,
`chunk_id`. Provides:
- `match_query_entities()` — known node names appearing verbatim in a query,
  falling back to word-level fuzzy/substring lookup (guarded by a minimum length
  to avoid false positives like "is" inside "chief scientist").
- `neighbors(entity, hops)` — breadth-limited fact expansion.
- `shortest_path_facts(a, b)` — path between two entities, as graph facts.
Persisted with pickle (the graph is small enough for an in-process store; no
external graph database).

**`graphrag/communities.py` — hierarchical communities + summaries.**
Community detection uses `networkx.community.louvain_partitions` (seed 42),
which yields a hierarchy of successively coarser partitions; **level 0 is the
finest**, higher levels coarser. Each community gets:
- `rank` = number of internal relations (importance),
- `key_entities` = up to 8 nodes by degree,
- `title`/`summary` from the LLM.

Summaries are **bottom-up**: a leaf community is summarized from its own
entities, relations, and source sentences; a parent community is summarized from
its children's summaries. Two free-tier savings: communities smaller than
`min_llm_size = 3` stay extractive, and a parent identical to its only child
reuses the child's report. The summary prompt returns JSON
`{"title": ≤8 words, "summary": 4-6 sentences}`; the source text is capped at
12,000 characters.

**`graphrag/global_search.py` — map-reduce global answering.**
- *Map*: for each of the **15 largest** communities at the chosen level, one
  Gemini call returns JSON `{"answer": ≤5 sentences, "score": 0-100}`. Partial
  answers scoring 0 are dropped.
- *Reduce*: the partials are sorted by score, packed into an 8,000-character
  context, and merged into one answer by a single Gemini call.
- `default_level()` picks the coarsest level that still has more than one
  community (fewest summaries to map over while still separating themes).

**`graphrag/router.py` — query routing (heuristic, explainable).**
No LLM classifier; a small set of rules:

| Condition | Route |
|---|---|
| Corpus-wide wording ("main themes", "overview", "summarize", "trends", ...) | `global` — map-reduce over community summaries |
| Query mentions 2+ known graph entities | `graph` — shortest path between them |
| Query mentions 1 known entity + a relational word ("related", "between", ...) | `hybrid` — graph neighbours + text search |
| Query mentions 1 known entity | `hybrid` — include its graph context alongside text search |
| No known entities recognised | `vector` — TF-IDF + BM25 search |

**`graphrag/retriever.py` — hybrid retrieval.**
Fuses TF-IDF and BM25 rankings with **reciprocal-rank fusion** (`k = 60`), then
attaches graph facts: shortest-path facts when 2+ entities matched, otherwise
2-hop neighbourhood facts.

**`graphrag/generator.py` — answer synthesis.**
Builds a context from up to 5 text passages and 8 graph facts, then asks Gemini
to answer using only that context (or produces a clearly-labelled extractive
answer when no key is set).

**`graphrag/vector_store.py` — baseline retrieval.**
- `VectorStore`: TF-IDF + cosine similarity (offline).
- `EmbeddingStore`: Gemini embeddings, L2-normalised; query embedded with task
  type `RETRIEVAL_QUERY`, scored by dot product. This backs the **vector-RAG
  baseline** when a key is available.

**`graphrag/keyword_store.py` — BM25 sparse search** (`rank_bm25.BM25Okapi`)
over chunks, complementing TF-IDF.

**`graphrag/pipeline.py` — orchestration.**
`GraphRAGPipeline(llm="auto"|client|None)` builds (ingest → extract → graph →
communities → vector/keyword/embedding stores), answers (`ask`, `ask_baseline`),
and persists. An index directory contains `graph.pkl`, `vector.pkl`,
`keyword.pkl`, `communities.pkl`, `embeddings.pkl`, and `meta.json`.

**`graphrag/llm.py` — Gemini client** (see §4).

---

## 3. Data

### 3.1 Upstream corpus (shipped with the project)

`data/corpus/` holds **32 English Wikipedia articles on space exploration**
(Space Race, Apollo, ISS, ESA, ISRO, CNSA, SpaceX, telescopes, rovers, ...), each
truncated to ~8,000 characters (~65k tokens total), sized so an index builds
within one day of Gemini free-tier quota. Re-download with
`python scripts/fetch_dataset.py`. Wikipedia text is CC BY-SA 4.0.
`data/questions.txt` holds 12 global evaluation questions.

### 3.2 Podcast corpus (the evaluation dataset used for the reported results)

The final evaluation runs on a **podcast transcript corpus** assembled from
Microsoft's *Behind the Tech with Kevin Scott* episodes:

| Property | Value |
|---|---|
| Episodes used | **Ep-60 (2023 Year in Review), Ep-61 (Lisa Su), Ep-62 (Xyla Foxlin)** |
| Words | **33,821** |
| Full corpus available | 9 episodes, 101,127 words (`data/podcast_corpus_full.txt`) |
| Questions | 5 global questions (`data/podcast_questions.txt`) |

Notes on construction:
- Source `.docx` files were combined by parsing `word/document.xml` (script
  `/tmp/opencode/combine_docx.py`); the episode headers were preserved as
  `# Ep-NN_...` lines.
- `Ep-66` is `CDFV2 Encrypted` and unreadable, so it is excluded from the full
  corpus.
- The current `data/podcast_corpus.txt` is the first three episodes; the full
  nine-episode text is kept at `data/podcast_corpus_full.txt` for scaling up.

The five evaluation questions:

```
What are the main themes across these episodes?
Give an overview of the guests and the topics they discussed.
What are the most important ideas about artificial intelligence in this corpus?
Summarize the key people and their roles across the whole corpus.
What technologies and companies are discussed across the episodes?
```

---

## 4. Gemini client and free-tier design

`graphrag/llm.py` talks to the Gemini REST API (`v1beta`) and is built around
free-tier limits:

- **Key handling** — reads `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) from the
  environment; never from code. A dependency-free `_load_dotenv()` loads a
  project-root `.env` file (existing environment variables win). On native
  Windows, `setx`-saved keys are also read from the registry.
- **Rate limiting** — requests are spaced to `GEMINI_RPM` (default 10/min).
- **Retries** — up to 8 attempts; 429s honour the server's `retryDelay`; 5xx use
  exponential backoff. A 429 flagged as a per-day limit raises `QuotaExhausted`,
  which the CLIs catch and report as a partial result.
- **Disk cache** — every response is cached under `.cache/gemini/` (key =
  SHA-256 of `[kind, payload]`). Reruns resume instead of paying again, and an
  identical question costs **zero** API calls.
- **Embedding batching** — embeddings are sent in token-bounded batches
  (25,000 tokens/min, ≤100 per batch) with a 60 s pause between batches.
- **JSON repair** — model slips such as unquoted object keys are repaired before
  parsing, with one retry for unreadable replies.

Defaults: `GEMINI_MODEL=gemini-3.5-flash-lite`,
`GEMINI_EMBED_MODEL=gemini-embedding-001`, `GEMINI_RPM=10`.

**Budget** (why the design is so frugal): large chunks (one extraction call
each), extractive summaries for tiny communities, report reuse when a parent
equals its only child, at most 15 communities mapped per global question, and one
judge call per answer order covering all four criteria.
`build_index.py --estimate` prints the expected call count without calling.

---

## 5. Setup and configuration

Requirements (`requirements.txt`): `pypdf`, `numpy`, `scikit-learn`,
`rank-bm25`, `networkx`, `requests`.

```bash
pip install -r requirements.txt
```

Get a free key at https://aistudio.google.com/apikey and provide it via a
`.env` file at the project root (added in this project, git-ignored):

```
GEMINI_API_KEY=your-key-here
GEMINI_MODEL=gemini-3.5-flash-lite
```

Environment variables override `.env`. Optional: `GEMINI_EMBED_MODEL`,
`GEMINI_RPM`. The key is never committed (`.env` is in `.gitignore`).

---

## 6. Usage

```bash
# 1. (optional) check the API budget, no calls made
python scripts/build_index.py --input data/corpus --estimate

# 2. build the index (Gemini; use --offline to skip the API)
python scripts/build_index.py --input data/podcast_corpus.txt --out data/index_podcast

# 3. ask questions
python scripts/ask.py --index data/index_podcast "What are the main themes across these episodes?"
python scripts/ask.py --index data/index_podcast "Tell me about Lisa Su"
python scripts/ask.py --index data/index_podcast --route baseline "What are the main themes across these episodes?"

# 4. evaluate GraphRAG vs vector RAG with the LLM judge
python scripts/evaluate.py --index data/index_podcast --questions data/podcast_questions.txt \
    --out results/eval_podcast.json
```

CLI flags: `build_index.py` (`--input`, `--out`, `--offline`, `--estimate`);
`ask.py` (`--index`, `--route {vector,graph,hybrid,global,baseline}`, `--level`);
`evaluate.py` (`--index`/`--input`, `--questions`, `--levels`, `--no-judge`,
`--out`).

---

## 7. Evaluation methodology

`scripts/evaluate.py` compares GraphRAG global search (one system per community
level) against the vector-RAG baseline. For each question every system answers,
and it reports:

**Cheap automatic proxies**
- **Entity coverage** — share of graph entity names mentioned in the answer
  (`entity_coverage`, `evaluate.py:64`).
- **Community diversity** — number of distinct level-0 communities whose key
  entities appear in the answer (`community_diversity`, `evaluate.py:72`).
- **Context size** — characters fed to the generator.
- **Latency** — wall-clock seconds.

**LLM-as-judge head-to-head** (Edge et al. 2024 criteria: comprehensiveness,
diversity, empowerment, directness). Each answer pair is judged in **both
orders** to cancel position bias; a system wins a criterion only if it wins on
balance (`head_to_head`, `evaluate.py:94`). Full answers and verdicts are written
to the JSON output.

---

## 8. Final results

All results below are from the **3-episode podcast corpus** with a
Gemini-built index (`data/index_podcast`), model `gemini-3.5-flash-lite`,
embeddings `gemini-embedding-001`.

### 8.1 Index statistics

| Metric | Value |
|---|---|
| Chunks | **43** |
| Entities (graph nodes) | **128** |
| Relations (graph edges) | **158** |
| Communities (total) | **141** |
| — level 0 (finest) | 57 |
| — level 1 | 43 |
| — level 2 (coarsest, used) | 41 |
| Build API calls | ~83 (15 cache hits) |

The evaluation uses **level 2** (the coarsest level, chosen by
`default_level`), where the map step summarizes the 15 largest of 41
communities.

### 8.2 Quantitative comparison

Mean over the 5 questions (`results/eval_podcast.json`):

| system | entity cov. | communities | context chars | latency (s) |
|---|---|---|---|---|
| **graphrag_L2** | **18.12%** | **12.2** | **1,623** | 114.20 |
| vector_rag | 7.66% | 7.8 | 23,756 | 10.40 |

GraphRAG mentions **~2.4× more graph entities** and touches **~1.6× more
communities**, while feeding the generator **~15× less context** (compact
community summaries vs. raw top-k passages). It is, however, **~11× slower**
because of the map-reduce fan-out (≈16 sequential Gemini calls per global
question).

### 8.3 LLM-judge head-to-head

GraphRAG win rate against vector RAG (wins / losses / ties):

| criterion | win rate | (G / B / tie) |
|---|---|---|
| comprehensiveness | **60%** | 3 / 1 / 1 |
| diversity | **60%** | 3 / 1 / 1 |
| empowerment | **60%** | 3 / 2 / 0 |
| directness | **60%** | 3 / 2 / 0 |

### 8.4 Per-question outcome

| # | Question | GraphRAG vs baseline |
|---|---|---|
| 1 | Main themes across episodes | GraphRAG wins all 4 criteria |
| 2 | Overview of guests and topics | **Baseline wins all 4** |
| 3 | Most important AI ideas | tie, tie, baseline, baseline |
| 4 | Key people and their roles | GraphRAG wins all 4 |
| 5 | Technologies and companies | GraphRAG wins all 4 |

GraphRAG wins 3 of 5 questions decisively and loses one (guest enumeration)
clearly — the 60% is an honest, non-uniform result.

### 8.5 API usage

- **Index build:** ~83 calls (15 cache hits).
- **Original (cold) evaluation:** **101 calls** (18 cache hits) —
  5 global × ~16 + 5 baseline × 2 + 10 judge.
- **Replay of the same questions:** **0 calls** (100 cache hits), ~10 s total.

Per-question API cost by route: `global` ≈ 16 calls (15 map + 1 reduce),
`hybrid`/`vector`/`baseline` = 2 (1 query embedding + 1 answer), `graph` = 1.

---

## 9. Findings and discussion

1. **GraphRAG answers global questions better on the whole.** It covers more
   entities and more communities with far less context, and the judge prefers it
   on 3 of 5 questions across all four criteria.
2. **But not for enumeration.** Question 2 ("overview of the guests") is the
   clearest failure: the level-2 community summaries aggregate away specific
   guest names, whereas vector RAG retrieves the raw passages that name them
   (Tobi Lütke, Bill Gates, Rana el Kaliouby, ...). This is the known tradeoff of
   summarizing before answering.
3. **Efficiency is the cost.** The map-reduce fan-out makes global GraphRAG
   ~11× slower than a single vector-RAG call, though it consumes ~15× less
   context.
4. **Compactness.** GraphRAG's ~1.6k-char context vs. vector RAG's ~24k chars is
   a practical advantage: cheaper downstream generation and less prompt bloat,
   while still scoring higher on coverage and diversity.

---

## 10. Differences from the paper

- **Louvain instead of Leiden** for community detection: Louvain ships with
  networkx; Leiden mainly adds a guarantee that communities are well-connected.
- **Smaller scale**: ~34k-word podcast corpus (or ~65k-token Wikipedia corpus),
  with no "gleaning" passes during extraction and no claim extraction, to fit the
  free tier.
- **Map step capped** at the 15 largest communities of the chosen level.
- **Offline fallbacks** (regex extraction, TF-IDF, extractive summaries) exist so
  the code runs without a key; reported results use a Gemini-built index.
- **Heuristic router** instead of an LLM classifier.

---

## 11. Reproducing the results

```bash
# env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
printf 'GEMINI_API_KEY=your-key\nGEMINI_MODEL=gemini-3.5-flash-lite\n' > .env

# build the index (resumable via .cache/gemini/ if the daily quota runs out)
python scripts/build_index.py --input data/podcast_corpus.txt --out data/index_podcast

# evaluate
python scripts/evaluate.py --index data/index_podcast \
    --questions data/podcast_questions.txt --out results/eval_podcast.json
```

Rerunning the same questions is free (cache hits). To scale to all nine episodes,
point `--input` at `data/podcast_corpus_full.txt` (≈142 chunks, ~350-450 calls;
expect it to span more than one day of free-tier quota — the cache makes it
resumable).

### Saved artefacts

| File | Contents |
|---|---|
| `results/ask_outputs_podcast.txt` | four `ask` runs (global, Lisa Su ×2, baseline) |
| `results/evaluate_podcast.txt` | evaluate stdout: answers, summary table, judge rates |
| `results/eval_podcast.json` | machine-readable answers + verdicts (with a latency note) |
| `results/build_podcast_gemini.log` | index build progress log |
| `data/index_podcast/` | persisted index (graph, stores, communities, meta) |

---

## 12. Project layout

```
graphrag/
  llm.py             # Gemini client: rate limit, retries, disk cache, embeddings, .env loader
  ingest.py          # load .txt/.pdf, sentence-aware chunking
  extraction.py      # Gemini entity/relation extraction (+ offline regex fallback)
  graph_store.py     # NetworkX graph: build, neighbours, shortest path, persist
  communities.py     # hierarchical (Louvain) communities + bottom-up summaries
  global_search.py   # map-reduce global answering over community summaries
  vector_store.py    # TF-IDF store (offline) + Gemini EmbeddingStore (baseline)
  keyword_store.py   # BM25 chunk search
  router.py          # global | graph | hybrid | vector routing
  retriever.py       # reciprocal-rank-fusion text search + graph lookups
  generator.py       # answers from retrieved context (Gemini or extractive)
  pipeline.py        # GraphRAGPipeline: build/save/load/ask/ask_baseline
scripts/
  fetch_dataset.py   # download the Wikipedia corpus
  build_index.py     # build and persist an index (--offline, --estimate)
  ask.py             # query an index
  evaluate.py        # GraphRAG vs. vector RAG evaluation
data/
  corpus/                 # 32 Wikipedia articles (upstream)
  questions.txt           # 12 global questions (upstream)
  podcast_corpus.txt      # 3-episode podcast corpus (used for results)
  podcast_corpus_full.txt # 9-episode backup
  podcast_questions.txt   # 5 global questions for the podcast corpus
  index_podcast/          # persisted Gemini-built index
results/                  # saved run outputs
PROJECT_REPORT.md         # this document
```

---

## 13. Environment notes and troubleshooting

- **Platform used:** WSL (Ubuntu) with the repo on the Windows filesystem.
- **Gemini model availability (free tier):** `gemini-2.5-flash` and
  `gemini-2.5-flash-lite` return 404 ("no longer available to new users");
  `gemini-3.5-flash-lite` is the default and works, but its **daily** quota can be
  exhausted — rerun the next day (the cache resumes the build) or set
  `GEMINI_MODEL` to another available model.
- **429 / per-day quota:** surfaced as `QuotaExhausted`; partial results are
  still reported and cached, so nothing is wasted.
- **Slow global queries:** a global question prints only after its ~16
  sequential Gemini calls finish (roughly 5-10 minutes cold). This is expected,
  not a hang.
- **`.env` loader:** added so keys are read on WSL, where the Windows-registry
  `setx` fallback is unavailable.
```
