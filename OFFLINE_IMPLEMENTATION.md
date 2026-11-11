# GraphRAG — Offline Implementation (No Gemini, No Ollama)

This file documents the **fully offline** version of the pipeline: no Gemini, no
Ollama, no external API or model of any kind. Every stage is replaced by a
deterministic, dependency-light algorithm (regex, TF-IDF, BM25, extractive
text). It is the original "initial implementation" of the project, and it still
runs today as the automatic fallback whenever no `GEMINI_API_KEY` is set.

It is useful as a **control**: it shows the graph/community/map-reduce
*architecture* working end-to-end without any LLM, and it needs **zero API
calls** (so it runs instantly and costs nothing).

---

## 1. What changes when there is no LLM

The pipeline shape is identical; only the three LLM-powered stages swap in
offline algorithms. `GraphRAGPipeline.__init__` calls `get_llm()`, which returns
`None` when no key is present, and every stage checks for that.

| Stage | With Gemini | **Offline (this doc)** |
|---|---|---|
| Chunking | 40 sentences/chunk, overlap 3 | **6 sentences/chunk, overlap 1** (`pipeline.py:27`) |
| Entity/relation extraction | LLM call per chunk | **regex** capitalised-phrase NER + span-based relations (`extraction.py:146`) |
| Community summaries | LLM-written reports | **extractive** sentences (`communities.py:150`) |
| Map (global search) | LLM partial answer + score | **TF-IDF relevance + top-2 sentences** (`global_search.py:63`) |
| Reduce (global search) | LLM merge | **bulleted theme list** (`global_search.py:115`) |
| Answer synthesis | LLM answer | **extractive quote assembly** (`generator.py:33`) |
| Baseline retrieval | Gemini embeddings | **TF-IDF cosine** (`vector_store.py:27`) |
| LLM judge | yes | **skipped** |

The router, retriever (reciprocal-rank fusion of TF-IDF + BM25), graph store,
and community detection are **unchanged** — they never used an LLM.

---

## 2. How each offline stage works

### 2.1 Chunking (`ingest.py`)
Same sentence-aware splitter, but tuned small: `max_sentences=6`,
`overlap_sentences=1`. Because extraction is cheap (regex, not an API call),
smaller chunks give finer resolution — this is why the offline build produces
**far more chunks** than the Gemini build on the same text.

### 2.2 Extraction (`extraction.py::extract`)
No model at all:
- **Entities** = matches of a capitalised-phrase regex `_ENTITY_RE`
  (`extraction.py:15`), with leading discourse words ("The", "However") stripped
  and a minimum length of 2. Typing is heuristic: `PERSON` if preceded by a title
  ("Dr.", "Prof."), `ORG` if it contains an org suffix ("Inc", "University",
  "Labs", ...), otherwise `ENTITY`.
- **Relations** = for two entities that co-occur in the same sentence, the
  cleaned text span *between* them becomes the predicate
  (`extract_from_sentence`, `extraction.py:117`). The span is lowercased, has
  leading/trailing stopwords removed, is capped at 6 words, and defaults to
  `"co-occurs with"` if empty.

This is intentionally simple and inspectable, not state-of-the-art NER/RE.

### 2.3 Community summaries (`communities.py::_extractive_summary`)
Communities are still found with **Louvain** (identical to the Gemini path). The
summary is just:
- **title** = the first three key entities + "and others",
- **summary** = "This community centers on <key entities>. " + up to 4 of the
  source sentences behind its relations.
No calls, and `min_llm_size` is irrelevant (everything is extractive).

### 2.4 Global search (`global_search.py`)
- **Map** (`_extractive_partial`, `:63`): each community's score is
  `round(TF-IDF relevance × 100 + min(rank, 10))`, and its partial answer is the
  2 summary sentences that share the most words with the question. Communities
  scoring 0 are dropped.
- **Reduce** (`reduce_step` with `llm=None`, `:115`): returns
  `"The corpus covers N relevant theme(s):"` followed by each partial answer,
  most relevant first.

So an offline "global answer" is a ranked, bulleted list of quoted community
themes — clearly labeled `[extractive]`, not fluent prose.

### 2.5 Local answering and baseline
Routing and hybrid retrieval are unchanged. The baseline uses the **TF-IDF**
`VectorStore` (no embeddings). The final answer is `_extractive_answer`
(`generator.py:33`): `"From the knowledge graph: ..."` followed by
`"From the retrieved passages: ..."`.

### 2.6 No judge
`evaluate.py` skips the LLM judge when `llm is None` and prints
`LLM judge skipped: set GEMINI_API_KEY to enable it.`

---

## 3. How to run the offline pipeline

```bash
# build an offline index (no API calls)
python scripts/build_index.py --input data/podcast_corpus.txt --out data/index_podcast_offline --offline

# ask (runs offline automatically when no key is set)
python scripts/ask.py --index data/index_podcast_offline "What are the main themes across these episodes?"

# evaluate (judge auto-skipped offline)
python scripts/evaluate.py --index data/index_podcast_offline \
    --questions data/podcast_questions.txt --no-judge --out results/eval_podcast_offline.json
```

You can also force it in code: `GraphRAGPipeline(llm=None)`.

---

## 4. Results

### 4.1 Current 3-episode podcast corpus (fresh offline run)

**Index build** (`results/build_podcast_offline.txt`):

| Metric | Offline |
|---|---|
| Chunks | **315** |
| Entities | **366** |
| Relations | **782** |
| Communities L0 / L1 / L2 | **171 / 116 / 110** |

**Evaluation, level 2** (`results/eval_podcast_offline.json`):

| system | entity cov. | communities | context chars | latency |
|---|---|---|---|---|
| graphrag_L2 | 36.50% | 68.2 | 8,120 | 0.01 s |
| vector_rag | 10.27% | 29.8 | 3,219 | 0.00 s |

LLM judge: **skipped** (offline).

### 4.2 Full 9-episode corpus (earlier offline run, kept in `results/`)

**Index build** (`results/build_index.txt`):

| Metric | Offline |
|---|---|
| Chunks | **1,047** |
| Entities | **963** |
| Relations | **2,319** |
| Communities L0 / L1 / L2 | **450 / 296 / 266** |

**Evaluation** (`results/evaluate.txt`):

| system | entity cov. | communities | context chars | latency |
|---|---|---|---|---|
| graphrag | 14.21% | 97.8 | 4,085 | 0.020 s |
| vector_rag | 5.42% | 42.4 | 2,455 | 0.002 s |

### 4.3 Example offline answer

For *"What are the main themes across these episodes?"* the offline global
answer is an extractive list, e.g.:

```
Answer [extractive]:
The corpus covers 15 relevant theme(s):
- CHRISTINA WARREN, Behind the Tech, MUSIC and others: "If you have anything
  that you would like to share with us, please email us any time ..."
- Kevin Scott, Xyla Foxlin, Laughter and others: "You have to think so out of
  the box ... you have to be always at the cutting edge of technology."
- LISA SU, You, Kevin and others: "when I was a little kid ... the computers
  were incredibly powerful ..."
...
```

It covers the corpus, but it is raw quoted text with noisy entity titles — no
synthesis. That is the whole point of adding the LLM.

---

## 5. Offline vs. Gemini (same 3-episode corpus)

| | Offline | Gemini |
|---|---|---|
| Chunks | 315 | 43 |
| Entities | 366 | 128 |
| Relations | 782 | 158 |
| Communities L0 / L1 / L2 | 171 / 116 / 110 | 57 / 43 / 41 |
| graphrag coverage | 36.50% | 18.12% |
| graphrag community diversity | 68.2 | 12.2 |
| graphrag context chars | 8,120 | 1,623 |
| LLM-judge win rate | n/a | 60% |
| Latency | ~0.01 s | 114.20 s |
| API calls | 0 | ~184 (build + eval) |

**Important caveat — do not compare coverage across modes directly.** The
offline numbers are *not* better answers:
- The offline graph has ~3× more (and noisier) entities, because the regex
  extractor over-generates ("You", "Laughter", "MUSIC", "Like" become nodes), so
  there are more names that *can* appear in an answer.
- The extractive answer **quotes whole passages**, so entity names appear
  verbatim by construction, inflating "coverage" and "diversity" trivially.
- Only the Gemini mode produces fluent, synthesized answers, and only the Gemini
  mode can be scored by the LLM judge.

The offline mode is valuable as a **sanity check and fallback**: the full
graph → communities → map-reduce architecture demonstrably runs and produces
corpus-level output with zero API calls and near-zero latency.

---

## 6. Notes

- The (now-removed) test suite ran entirely on this offline path — 14 tests
  passed in 1.67 s (`results/tests.txt`).
- Offline mode is the default whenever `GEMINI_API_KEY` is absent, so nothing is
  ever "broken" without a key; the pipeline just degrades to regex + TF-IDF +
  extractive text.

### Files produced by this run

| File | Contents |
|---|---|
| `data/index_podcast_offline/` | offline index (3-episode corpus) |
| `results/build_podcast_offline.txt` | offline build stats |
| `results/evaluate_podcast_offline.txt` | offline evaluate stdout |
| `results/eval_podcast_offline.json` | offline evaluate JSON |
| `results/ask_outputs_podcast_offline.txt` | four offline `ask` runs |
| `results/build_index.txt`, `results/evaluate.txt`, `results/ask_outputs.txt` | earlier full-corpus offline run |
| `results/tests.txt` | offline test-suite output (14 passed) |
