"""Answer synthesis.

Primary path is a clearly-labeled extractive answer built purely from
retrieved text + graph facts, so the pipeline works with zero external
services or API keys.
"""
from __future__ import annotations

from .retriever import RetrievalResult


def build_context(result: RetrievalResult, max_chunks: int = 5, max_facts: int = 8) -> str:
    lines: list[str] = []
    if result.text_hits:
        lines.append("Relevant passages:")
        for hit in result.text_hits[:max_chunks]:
            lines.append(f"- {hit.text}")
    if result.graph_facts:
        lines.append("Knowledge graph facts:")
        seen = set()
        for fact in result.graph_facts:
            key = (fact.subject, fact.predicate, fact.obj)
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- {fact.subject} --[{fact.predicate}]--> {fact.obj}")
            if len(seen) >= max_facts:
                break
    return "\n".join(lines)


def _extractive_answer(result: RetrievalResult) -> str:
    context = build_context(result)
    if not context:
        return "No relevant information was found in the indexed document(s) for this question."

    parts = []
    if result.graph_facts:
        seen = set()
        fact_sentences = []
        for fact in result.graph_facts:
            key = (fact.subject, fact.predicate, fact.obj)
            if key in seen:
                continue
            seen.add(key)
            fact_sentences.append(f"{fact.subject} {fact.predicate} {fact.obj}.")
            if len(fact_sentences) >= 5:
                break
        if fact_sentences:
            parts.append("From the knowledge graph: " + " ".join(fact_sentences))
    if result.text_hits:
        top_text = " ".join(hit.text for hit in result.text_hits[:2])
        parts.append("From the retrieved passages: " + top_text)
    return "\n\n".join(parts)


def generate_answer(query: str, result: RetrievalResult) -> dict:
    return {"answer": _extractive_answer(result), "backend": "extractive", "context": build_context(result)}
