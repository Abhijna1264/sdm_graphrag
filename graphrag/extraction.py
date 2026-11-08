"""Lightweight, fully offline entity + relation extraction.

No ML model or LLM call is used here on purpose (see README "Design
tradeoffs"): entities are detected via capitalized-phrase regex matching,
and relations are the cleaned-up text span between two entities that
co-occur in the same sentence. This is intentionally simple and
inspectable rather than state-of-the-art NER/RE.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .ingest import Chunk, split_sentences

_ENTITY_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9&'.-]*"
    r"(?:\s+(?:of|and|for|the|de|la|van|von)\s+[A-Z][A-Za-z0-9&'.-]*"
    r"|\s+[A-Z][A-Za-z0-9&'.-]*)*"
)

_SENTENCE_STARTER_STOPWORDS = {
    "The", "This", "That", "These", "Those", "It", "They", "He", "She", "We",
    "I", "A", "An", "In", "On", "At", "For", "With", "After", "Before",
    "However", "Also", "Then", "But", "And", "So", "If", "When", "While",
    "Because", "Although", "As", "Its", "Their", "His", "Her", "There",
    "Here", "Some", "Many", "Most", "Both",
}

_ORG_HINTS = {
    "Inc", "Corp", "Corporation", "Ltd", "LLC", "Labs", "Lab", "Company",
    "University", "Institute", "Group", "Systems", "Technologies",
    "Foundation", "Consortium", "Agency",
}

_PERSON_TITLES = {"Dr.", "Mr.", "Mrs.", "Ms.", "Prof."}

_EDGE_STRIP_WORDS = {"the", "a", "an", "of", "in", "on", "at", "for", "with", "and", "to", "as", "by"}


@dataclass
class EntityMention:
    text: str
    entity_type: str
    chunk_id: str
    sentence: str


@dataclass
class Triple:
    subject: str
    predicate: str
    obj: str
    sentence: str
    chunk_id: str


@dataclass
class ExtractionResult:
    entities: dict[str, dict] = field(default_factory=dict)  # key: lowercase name
    triples: list[Triple] = field(default_factory=list)


def _clean_span(text: str) -> str:
    return text.strip(" ,.;:!?\"'()[]{}")


def _type_entity(text: str, preceding_text: str) -> str:
    words = text.split()
    if words and words[0].rstrip(".") in {t.rstrip(".") for t in _PERSON_TITLES}:
        return "PERSON"
    if any(w.rstrip(".") in _ORG_HINTS for w in words):
        return "ORG"
    if preceding_text.rstrip().endswith(tuple(_PERSON_TITLES)):
        return "PERSON"
    return "ENTITY"


def _find_entities(sentence: str) -> list[tuple[str, int, int, str]]:
    """Return (clean_text, start, end, entity_type) for each entity span."""
    found = []
    for match in _ENTITY_RE.finditer(sentence):
        raw = match.group()
        words = raw.split()

        # Drop leading discourse words picked up because they were
        # capitalized (sentence start, "The Osaka office...") so the span
        # starts at the real proper noun.
        start = match.start()
        while len(words) > 1 and words[0] in _SENTENCE_STARTER_STOPWORDS:
            start += len(words[0]) + 1
            words = words[1:]

        if len(words) == 1 and words[0] in _SENTENCE_STARTER_STOPWORDS:
            continue

        clean = _clean_span(" ".join(words))
        if not clean or len(clean) < 2:
            continue
        entity_type = _type_entity(clean, sentence[:start])
        found.append((clean, start, match.end(), entity_type))
    return found


def _clean_predicate(raw: str) -> str:
    raw = raw.strip(" ,;:")
    words = [w for w in re.split(r"\s+", raw) if w]
    while words and words[0].lower() in _EDGE_STRIP_WORDS:
        words.pop(0)
    while words and words[-1].lower() in _EDGE_STRIP_WORDS:
        words.pop()
    words = words[:6]
    cleaned = " ".join(words).lower()
    cleaned = re.sub(r"[^a-z0-9\s'-]", "", cleaned).strip()
    return cleaned


def extract_from_sentence(sentence: str, chunk_id: str) -> tuple[list[EntityMention], list[Triple]]:
    spans = _find_entities(sentence)
    mentions = [
        EntityMention(text=text, entity_type=etype, chunk_id=chunk_id, sentence=sentence)
        for text, _, _, etype in spans
    ]

    triples: list[Triple] = []
    for (text_a, _, end_a, _), (text_b, start_b, _, _) in zip(spans, spans[1:]):
        if text_a.lower() == text_b.lower():
            continue
        between = sentence[end_a:start_b]
        predicate = _clean_predicate(between)
        if not predicate:
            predicate = "co-occurs with"
        triples.append(Triple(subject=text_a, predicate=predicate, obj=text_b, sentence=sentence, chunk_id=chunk_id))
    return mentions, triples


def extract_from_chunk(chunk: Chunk) -> tuple[list[EntityMention], list[Triple]]:
    all_mentions: list[EntityMention] = []
    all_triples: list[Triple] = []
    for sentence in split_sentences(chunk.text):
        mentions, triples = extract_from_sentence(sentence, chunk.chunk_id)
        all_mentions.extend(mentions)
        all_triples.extend(triples)
    return all_mentions, all_triples


def extract(chunks: list[Chunk]) -> ExtractionResult:
    result = ExtractionResult()
    for chunk in chunks:
        mentions, triples = extract_from_chunk(chunk)
        for mention in mentions:
            key = mention.text.lower()
            entry = result.entities.setdefault(
                key, {"name": mention.text, "type": mention.entity_type, "mentions": 0}
            )
            entry["mentions"] += 1
            if entry["type"] == "ENTITY" and mention.entity_type != "ENTITY":
                entry["type"] = mention.entity_type
        result.triples.extend(triples)
    return result
