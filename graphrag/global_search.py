"""Map-reduce global answering over community summaries.

Map: every community summary at the chosen level is turned into a partial
answer with a 0-100 helpfulness score (by the LLM if available, otherwise by
TF-IDF relevance + the most relevant summary sentences). Partial answers
scoring 0 are dropped.

Reduce: the remaining partial answers are sorted by score, packed into a
context budget, and combined into one global answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .communities import Community, CommunityStore
from .llm import try_ollama

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class PartialAnswer:
    community_id: str
    title: str
    text: str
    score: int


def _relevance(query: str, communities: list[Community]) -> list[float]:
    docs = [f"{c.title}. {c.summary}" for c in communities]
    vectorizer = TfidfVectorizer(stop_words="english").fit(docs + [query])
    return list(cosine_similarity(vectorizer.transform([query]), vectorizer.transform(docs))[0])


def _extractive_partial(query: str, community: Community, relevance: float) -> PartialAnswer:
    sentences = [s for s in _SENTENCE_SPLIT.split(community.summary) if s.strip()]
    query_terms = set(re.findall(r"\w+", query.lower()))
    sentences.sort(key=lambda s: -len(query_terms & set(re.findall(r"\w+", s.lower()))))
    # Global questions ("main themes") often share few words with any summary,
    # so every non-empty community still contributes a small base score
    # weighted by its size (rank), as in "what is this whole corpus about".
    score = int(min(100, round(relevance * 100 + min(community.rank, 10))))
    return PartialAnswer(community.community_id, community.title, " ".join(sentences[:2]), score)


def _llm_partial(query: str, community: Community) -> PartialAnswer | None:
    prompt = (
        "Using ONLY the community report below, write a short partial answer to the question, "
        "then on the last line write 'SCORE: N' where N (0-100) is how helpful the report is "
        "for answering the question. If it is irrelevant, answer 'SCORE: 0'.\n\n"
        f"Report ({community.title}):\n{community.summary}\n\nQuestion: {query}\n"
    )
    response = try_ollama(prompt, timeout=30.0)
    if not response:
        return None
    match = re.search(r"SCORE:\s*(\d+)", response)
    score = int(match.group(1)) if match else 50
    text = response[: match.start()].strip() if match else response
    return PartialAnswer(community.community_id, community.title, text, min(score, 100))


def map_step(query: str, communities: list[Community], use_ollama: bool) -> list[PartialAnswer]:
    if not communities:
        return []
    relevances = _relevance(query, communities)
    partials = []
    for community, relevance in zip(communities, relevances):
        partial = _llm_partial(query, community) if use_ollama else None
        if partial is None:
            partial = _extractive_partial(query, community, relevance)
        if partial.score > 0 and partial.text:
            partials.append(partial)
    return sorted(partials, key=lambda p: p.score, reverse=True)


def reduce_step(
    query: str, partials: list[PartialAnswer], use_ollama: bool, max_chars: int = 4000
) -> dict:
    packed, used = [], 0
    for partial in partials:
        if used + len(partial.text) > max_chars:
            break
        packed.append(partial)
        used += len(partial.text)
    if not packed:
        return {"answer": "No community summaries were relevant to this question.", "backend": "extractive"}

    context = "\n".join(f"[{p.title}] (score {p.score}): {p.text}" for p in packed)
    if use_ollama:
        prompt = (
            "Combine the partial answers below (most helpful first) into one comprehensive answer to "
            "the question. Cover the distinct themes they raise and do not add outside information.\n\n"
            f"{context}\n\nQuestion: {query}\nAnswer:"
        )
        llm_answer = try_ollama(prompt, timeout=60.0)
        if llm_answer:
            return {"answer": llm_answer, "backend": "ollama", "context": context}

    lines = [f"The corpus covers {len(packed)} relevant theme(s):"]
    lines += [f"- {p.title}: {p.text}" for p in packed]
    return {"answer": "\n".join(lines), "backend": "extractive", "context": context}


def global_answer(
    query: str, store: CommunityStore, level: int | None = None, use_ollama: bool = True
) -> dict:
    if not store.levels:
        return {"answer": "No communities have been built.", "backend": "extractive", "level": None, "partials": []}
    # Default to the coarsest level that still has more than one community:
    # fewest summaries to map over while still separating distinct themes.
    if level is None:
        multi = [lv for lv in store.levels if len(store.at_level(lv)) > 1]
        level = multi[-1] if multi else store.levels[0]
    partials = map_step(query, store.at_level(level), use_ollama)
    result = reduce_step(query, partials, use_ollama)
    result.update(level=level, partials=partials)
    return result
