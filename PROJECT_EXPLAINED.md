# GraphRAG — Explained From Start to Finish

This file is a plain-language walkthrough of the whole project. If you read only
one thing, read this: it explains **what happens, in order**, using the actual
podcast example, and gives the real numbers (chunks, entities, community levels)
and the final results. For a more formal write-up see `PROJECT_REPORT.md`.

---

## The one-paragraph version

GraphRAG answers "big picture" questions about a collection of documents. It
cannot do that by looking at one paragraph, so it first reads everything and
builds a **knowledge graph** of the people, companies and ideas in the corpus.
It then groups that graph into **communities** (clusters of related things) at
several **levels of detail**, writes a **summary for each community**, and
finally answers a global question by asking the LLM to summarize the most
relevant community summaries. We compare this against the standard approach
(**vector RAG**: embed the text, fetch the most similar passages, answer) using
an LLM as a judge.

---

## Step 0 — The data

We use transcripts of Microsoft's *Behind the Tech with Kevin Scott* podcast.

- Current corpus: **3 episodes** — Ep-60 (2023 Year in Review), Ep-61 (Lisa Su),
  Ep-62 (Xyla Foxlin).
- Size: **33,821 words**.
- A larger 9-episode version (101,127 words) is kept aside
  (`data/podcast_corpus_full.txt`) for scaling up later.
- We ask **5 global questions** (`data/podcast_questions.txt`), e.g. *"What are
  the main themes across these episodes?"*.

---

## Step 1 — Loading and chunking (`ingest.py`)

The raw text is split into sentences (carefully — abbreviations like "Dr." are
not treated as sentence endings), then sentences are packed into **chunks**.

- Because Gemini does one extraction call per chunk, chunks are made large:
  **40 sentences per chunk with a 3-sentence overlap**.
- Result for our corpus: **43 chunks**.

Think of a chunk as "a page the LLM will read in one go." The overlap means a
fact split across a boundary still appears whole in one of the chunks.

---

## Step 2 — Entity and relation extraction (`extraction.py`)

For each of the 43 chunks, we send **one Gemini call** that returns JSON:

- a list of **entities** (typed: PERSON, ORGANIZATION, MISSION, SPACECRAFT,
  LOCATION, EVENT, CONCEPT, OTHER) with a one-sentence description;
- a list of **relationships** between them (a short verb phrase + a sentence).

Result for our corpus:

| | Count |
|---|---|
| Entities | **128** |
| Relations | **158** |

Examples of the relations it found (from the Lisa Su episode):

```
Lisa Su --[is ceo of]--> AMD
Lisa Su --[was elected to]--> National Academy of Engineering
Lisa Su --[received medal from]--> IEEE
Lisa Su --[serves on board for]--> Semiconductor Industry Association
```

If no API key is present, a regex-based extractor runs instead (no model). The
results reported here use Gemini.

---

## Step 3 — The knowledge graph (`graph_store.py`)

All those entities and relations become a graph (a `networkx` MultiDiGraph):
**128 nodes, 158 edges**. It can look up an entity mentioned in a question
(including fuzzy matches), walk neighbours, and find the shortest path between
two entities. It is saved to disk as `graph.pkl`.

---

## Step 4 — Communities and levels (`communities.py`)

The graph is partitioned into **communities** — clusters of tightly related
entities — using the **Louvain** algorithm. Louvain produces a *hierarchy*: the
same graph is partitioned at several resolutions. **Level 0 is the finest**
(smallest groups); higher levels are coarser (bigger, more merged groups).

For our corpus the hierarchy came out as:

| Level | Meaning | Number of communities |
|---|---|---|
| **0** | finest (small clusters) | **57** |
| **1** | medium | **43** |
| **2** | coarsest (big themes) | **41** |

So there are **141 communities in total**. Each community records its size
(how many relations it contains, used as its "rank"/importance) and its up-to-8
key entities.

**Level 2 (41 communities) is the one used for global questions**, because the
coarsest level has the fewest, most "thematic" summaries to reason over. Each
community got a Gemini-written title and 4-6 sentence summary. Summaries are
written **bottom-up**: a small community is summarized from its own entities and
relations; a bigger community is summarized from the summaries of the smaller
communities inside it.

---

## Step 5 — Answering a global question (`global_search.py`)

When you ask a corpus-wide question, the pipeline does **map-reduce**:

1. **Map** — take the **15 largest** level-2 communities and ask Gemini, for
   each one, "using only this community's report, give a short partial answer and
   a 0-100 helpfulness score." → up to **15 calls**.
2. **Reduce** — sort the partial answers by score, pack them into an 8,000-char
   context, and ask Gemini to merge them into one answer. → **1 call**.

Total: **~16 Gemini calls per global question** (this is why global questions are
slow — about 5-10 minutes each when cold).

For *"What are the main themes across these episodes?"* the map step scored
communities like *"Lisa Su and AMD's Leadership in Semiconductors and AI"*
(score 90) and *"Microsoft Leadership and Technology Podcast Network"* (100),
and the reduce step produced a thematic answer covering corporate leadership,
AI, semiconductors, science/energy, and cultural impact.

---

## Step 6 — Answering a local question (`router.py`, `retriever.py`)

Not every question is global. A small set of rules picks the strategy:

| If the question… | Route | What happens |
|---|---|---|
| uses words like "main themes", "overview", "summarize", "trends" | **global** | map-reduce over community summaries |
| mentions **2+ known entities** | **graph** | shortest path between them |
| mentions **1 known entity** + a relational word ("related", "between") | **hybrid** | graph neighbours **and** text search |
| mentions **1 known entity** | **hybrid** | include its graph context alongside text search |
| mentions **no** known entities | **vector** | plain TF-IDF + BM25 search |

For example, *"Tell me about Lisa Su"* routes to **hybrid**: it pulls the graph
facts around "Lisa Su" and the most relevant passages, then Gemini writes the
answer. Text search fuses TF-IDF and BM25 rankings (reciprocal-rank fusion).

---

## Step 7 — The baseline: vector RAG (`vector_store.py`)

The comparison system is deliberately conventional:

1. Embed every chunk with Gemini embeddings (`gemini-embedding-001`).
2. Embed the question.
3. Fetch the **top-5 most similar chunks**.
4. Ask Gemini to answer from those chunks.

No graph, no communities — just "find similar text and answer."

---

## Step 8 — How we compare the two (`evaluate.py`)

For each of the 5 questions, both systems answer, and we measure:

**Cheap automatic numbers**
- **Entity coverage** — what fraction of the graph's entity names appear in the
  answer (a proxy for "did it cover the corpus").
- **Community diversity** — how many distinct level-0 communities the answer
  touches (a proxy for "how varied is it").
- **Context size** — how many characters were fed to the answer-writer.
- **Latency** — how long it took.

**LLM-as-judge (the real test)**
- The judge compares the two answers on the paper's four criteria —
  **comprehensiveness, diversity, empowerment, directness**.
- Each pair is judged **twice, in both orders**, to cancel position bias. A
  system wins a criterion only if it wins on balance.

---

## The final results

### Index built for the podcast corpus

```
43 chunks -> 128 entities, 158 relations -> 141 communities
   level 0: 57   level 1: 43   level 2: 41   (global search uses level 2)
```

### Head-to-head (mean over 5 questions)

| system | entity cov. | communities | context chars | latency |
|---|---|---|---|---|
| **GraphRAG (level 2)** | **18.12%** | **12.2** | **1,623** | 114.20 s |
| vector RAG baseline | 7.66% | 7.8 | 23,756 | 10.40 s |

Read this as: GraphRAG mentions about **2.4× more entities** and touches **1.6×
more communities**, while using about **15× less context** — but it is about
**11× slower** because of the map-reduce fan-out.

### LLM-judge verdicts (GraphRAG win rate vs baseline)

| criterion | win rate | wins / losses / ties |
|---|---|---|
| comprehensiveness | **60%** | 3 / 1 / 1 |
| diversity | **60%** | 3 / 1 / 1 |
| empowerment | **60%** | 3 / 2 / 0 |
| directness | **60%** | 3 / 2 / 0 |

### Per-question, in words

| # | Question | Who won |
|---|---|---|
| 1 | Main themes across episodes | **GraphRAG** (all 4 criteria) |
| 2 | Overview of guests and topics | **Baseline** (all 4) |
| 3 | Most important AI ideas | tie / tie / baseline / baseline |
| 4 | Key people and their roles | **GraphRAG** (all 4) |
| 5 | Technologies and companies | **GraphRAG** (all 4) |

GraphRAG wins 3 of 5 questions clearly and loses one clearly — so the 60% is an
honest mix, not a clean sweep.

---

## Why GraphRAG won some and lost some

- **It won the "themes", "key people", and "technologies/companies" questions.**
  Those need a bird's-eye view of the whole corpus, which is exactly what
  community summaries provide. GraphRAG also produces a compact, well-organised
  answer (≈1,600 characters of context vs the baseline's ≈24,000).
- **It lost the "overview of the guests" question.** That question is really a
  *list of names*, and the coarsest community summaries merge individual guests
  away. The baseline simply retrieves the raw passages that name each guest, so
  it wins. This is the classic tradeoff of summarizing before answering.

---

## What it costs (API calls)

Gemini's free tier is the constraint, so everything is budgeted and cached:

| Action | Gemini calls |
|---|---|
| Building the index | ~83 (43 extraction + summaries) |
| One **global** question | ~16 (15 map + 1 reduce) |
| One **hybrid / vector / baseline** question | 2 (1 query embed + 1 answer) |
| One **graph** question | 1 |
| The full 5-question evaluation (cold) | **101** |
| Re-asking the **same** questions | **0** (served from `.cache/gemini/`) |

Every response is cached on disk, so if the daily quota runs out mid-run you just
rerun the same command later and it resumes. Repeating an identical question
costs nothing.

---

## Mini glossary

- **Chunk** — a ~40-sentence slice of a document; the unit the LLM reads.
- **Entity** — a named thing (person, company, technology, …).
- **Relation / triple** — "A --[relationship]--> B".
- **Knowledge graph** — all entities and relations together.
- **Community** — a cluster of related entities; **level** = how coarse the
  clustering is (0 = finest, 2 = coarsest here).
- **Community summary** — an LLM-written paragraph describing a community.
- **Global search** — map-reduce over community summaries to answer big questions.
- **Vector RAG** — the baseline: fetch similar passages and answer.
- **LLM-as-judge** — using an LLM to decide which of two answers is better.

---

## Where the files live

- Full formal report: `PROJECT_REPORT.md`
- This explanation: `PROJECT_EXPLAINED.md`
- Saved run outputs: `results/` (answers, evaluate log, `eval_podcast.json`)
- Built index: `data/index_podcast/`
- Code: `graphrag/` (library) and `scripts/` (CLI tools)
