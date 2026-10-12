"""Deterministic lexical retrieval for short FAQ documents."""

from __future__ import annotations

from collections import Counter
import math
import re
import unicodedata

from .models import Document, RetrievalResult


# Function words only. Negations, time units, numbers, and action words are
# deliberately retained because they can alter the requested fact.
STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "please",
        "the",
        "their",
        "them",
        "to",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "you",
        "your",
    }
)

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")

_CANONICAL = {
    "closure": "close",
    "closing": "close",
    "closed": "close",
    "withdrawal": "withdraw",
    "withdrawals": "withdraw",
    "requirements": "requirement",
    "required": "require",
}


def _light_stem(token: str) -> str:
    if token in _CANONICAL:
        return _CANONICAL[token]
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        stem = token[:-3]
        if len(stem) > 3 and stem[-1:] == stem[-2:-1]:
            stem = stem[:-1]
        return stem
    if len(token) > 4 and token.endswith("ed"):
        stem = token[:-2]
        if len(stem) > 3 and stem[-1:] == stem[-2:-1]:
            stem = stem[:-1]
        return stem
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    # Treat the common spelling variants as one lexical concept.
    normalized = re.sub(r"\btwo[\s-]+factor\b", "2fa", normalized)
    return normalized


def tokenize(text: str, *, remove_stop_words: bool = True) -> tuple[str, ...]:
    tokens = tuple(_light_stem(token) for token in _WORD_RE.findall(normalize_text(text)))
    if remove_stop_words:
        return tuple(token for token in tokens if token not in STOP_WORDS)
    return tokens


def bigrams(tokens: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    return tuple(zip(tokens, tokens[1:]))


def split_sentences(text: str) -> tuple[str, ...]:
    """Split text without rewriting it so snippets remain source substrings."""

    sentences = tuple(part.strip() for part in _SENTENCE_BOUNDARY_RE.split(text) if part.strip())
    return sentences or (text.strip(),)


def _idf(documents: tuple[Document, ...]) -> dict[str, float]:
    frequencies: Counter[str] = Counter()
    for document in documents:
        frequencies.update(set(tokenize(f"{document.title} {document.text}")))
    count = len(documents)
    return {
        term: math.log((count + 1) / (frequency + 1)) + 1.0
        for term, frequency in frequencies.items()
    }


def _query_idf(term: str, values: dict[str, float], count: int) -> float:
    return values.get(term, math.log(count + 1) + 1.0)


def _weighted_vector(
    title_tokens: tuple[str, ...], text_tokens: tuple[str, ...], idf: dict[str, float]
) -> dict[str, float]:
    weighted_counts: Counter[str] = Counter(text_tokens)
    for term, count in Counter(title_tokens).items():
        weighted_counts[term] += 2.0 * count
    return {
        term: (1.0 + math.log(count)) * idf[term]
        for term, count in weighted_counts.items()
    }


def _query_vector(
    query_tokens: tuple[str, ...], idf: dict[str, float], document_count: int
) -> dict[str, float]:
    return {
        term: (1.0 + math.log(count)) * _query_idf(term, idf, document_count)
        for term, count in Counter(query_tokens).items()
    }


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(weight * right.get(term, 0.0) for term, weight in left.items())
    left_norm = math.sqrt(sum(weight * weight for weight in left.values()))
    right_norm = math.sqrt(sum(weight * weight for weight in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def _weighted_coverage(
    query_terms: set[str], evidence_terms: set[str], idf: dict[str, float], count: int
) -> float:
    if not query_terms:
        return 0.0
    total = sum(_query_idf(term, idf, count) for term in query_terms)
    covered = sum(
        _query_idf(term, idf, count) for term in query_terms if term in evidence_terms
    )
    return covered / total if total else 0.0


def _sentence_rank(
    sentence: str, query_tokens: tuple[str, ...], idf: dict[str, float], count: int
) -> tuple[float, int]:
    sentence_tokens = tokenize(sentence)
    sentence_set = set(sentence_tokens)
    coverage = _weighted_coverage(set(query_tokens), sentence_set, idf, count)
    query_bigrams = set(bigrams(query_tokens))
    phrase = (
        len(query_bigrams.intersection(bigrams(sentence_tokens))) / len(query_bigrams)
        if query_bigrams
        else 0.0
    )
    return (0.8 * coverage + 0.2 * phrase, len(sentence))


def retrieve(
    question: str, documents: tuple[Document, ...], *, top_k: int = 3
) -> tuple[RetrievalResult, ...]:
    """Rank documents for *question* and return a small positive-score set."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    query_tokens = tokenize(question)
    if not query_tokens:
        return ()

    idf = _idf(documents)
    query_vector = _query_vector(query_tokens, idf, len(documents))
    query_terms = set(query_tokens)
    query_bigrams = set(bigrams(query_tokens))
    ranked: list[RetrievalResult] = []

    for document in documents:
        title_tokens = tokenize(document.title)
        text_tokens = tokenize(document.text)
        document_tokens = title_tokens + text_tokens
        document_terms = set(document_tokens)
        vector = _weighted_vector(title_tokens, text_tokens, idf)
        cosine = _cosine(query_vector, vector)
        coverage = _weighted_coverage(query_terms, document_terms, idf, len(documents))
        document_bigrams = set(bigrams(document_tokens))
        phrase = (
            len(query_bigrams.intersection(document_bigrams)) / len(query_bigrams)
            if query_bigrams
            else 0.0
        )
        score = min(1.0, 0.70 * cosine + 0.20 * coverage + 0.10 * phrase)
        if score <= 0.0:
            continue

        sentences = split_sentences(document.text)
        sentence_order = {sentence: index for index, sentence in enumerate(sentences)}
        best_sentences = tuple(
            sorted(
                sentences,
                key=lambda sentence: (
                    -_sentence_rank(sentence, query_tokens, idf, len(documents))[0],
                    sentence_order[sentence],
                ),
            )[:3]
        )
        ranked.append(
            RetrievalResult(
                document=document,
                score=score,
                term_coverage=coverage,
                bigram_coverage=phrase,
                matched_terms=tuple(sorted(query_terms.intersection(document_terms))),
                best_sentences=best_sentences,
            )
        )

    ranked.sort(key=lambda item: (-item.score, item.document.id))
    return tuple(ranked[:top_k])

