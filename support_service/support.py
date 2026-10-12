"""Conservative, deterministic evidence support checks."""

from __future__ import annotations

import re

from .models import RetrievalResult, SupportDecision
from .retrieval import bigrams, normalize_text, tokenize


SUPPORT_THRESHOLD = 0.62
MIN_CONCEPT_COVERAGE = 0.60
MIN_CRITICAL_COVERAGE = 0.75

_RESTRICTED_TOPICS = frozenset({"legal", "tax", "taxation"})
_ADVICE_TERMS = frozenset({"advice", "advise", "recommendation", "recommend"})
_GUARANTEE_TERMS = frozenset(
    {"guarantee", "guaranteed", "promise", "promised", "certain", "definitely"}
)
_TIME_UNITS = frozenset(
    {
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "weekend",
        "holiday",
    }
)
_NUMBER_WORDS = frozenset(
    {
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "hundred",
    }
)

# These words frame a question but are not themselves a requested product fact.
_QUERY_FRAMING_TERMS = frozenset(
    {
        "ask",
        "can",
        "could",
        "explain",
        "happen",
        "help",
        "information",
        "know",
        "long",
        "need",
        "provide",
        "question",
        "requirement",
        "say",
        "should",
        "take",
        "tell",
        "want",
        "would",
    }
)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _contains_restricted_advice(question_terms: set[str]) -> bool:
    return bool(question_terms & _RESTRICTED_TOPICS) and bool(
        question_terms & _ADVICE_TERMS
    )


def _is_guarantee_request(question_terms: set[str]) -> bool:
    return bool(question_terms & _GUARANTEE_TERMS)


def _meaningful_terms(question: str) -> set[str]:
    return set(tokenize(question)) - _QUERY_FRAMING_TERMS


def _numeric_or_time_terms(terms: set[str]) -> set[str]:
    return {
        term
        for term in terms
        if term.isdigit() or term in _NUMBER_WORDS or term in _TIME_UNITS
    }


def _has_coherent_sentence(question: str, result: RetrievalResult) -> bool:
    query_terms = _meaningful_terms(question)
    query_bigrams = set(bigrams(tokenize(question)))
    for sentence in result.best_sentences:
        sentence_tokens = tokenize(sentence)
        sentence_terms = set(sentence_tokens)
        if len(query_terms.intersection(sentence_terms)) >= 2:
            return True
        if query_bigrams.intersection(bigrams(sentence_tokens)):
            return True
    return False


def _has_conflicting_number(question_terms: set[str], evidence_terms: set[str]) -> bool:
    requested_numbers = {
        term for term in question_terms if term.isdigit() or term in _NUMBER_WORDS
    }
    if not requested_numbers:
        return False
    return not requested_numbers.issubset(evidence_terms)


def check_support(
    question: str, results: tuple[RetrievalResult, ...]
) -> SupportDecision:
    """Decide whether the highest-ranked evidence supports the full question.

    Retrieval scores help rank likely evidence, but never bypass critical-facet
    and policy checks. Only the top coherent document is used for an answer;
    unrelated fragments are not combined to manufacture support.
    """

    if not results:
        return SupportDecision(False, 0.0, "NO_EVIDENCE", ())

    top = results[0]
    question_terms = set(tokenize(question))
    evidence_text = f"{top.document.title} {top.document.text}"
    evidence_terms = set(tokenize(evidence_text))
    meaningful = _meaningful_terms(question)
    matched_meaningful = meaningful.intersection(evidence_terms)
    concept_coverage = (
        len(matched_meaningful) / len(meaningful) if meaningful else 0.0
    )

    critical_terms = {
        term
        for term in meaningful
        if term not in _QUERY_FRAMING_TERMS
    }
    critical_coverage = (
        len(critical_terms.intersection(evidence_terms)) / len(critical_terms)
        if critical_terms
        else 0.0
    )
    phrase_consistency = top.bigram_coverage
    support_score = _clamp(
        0.40 * top.score
        + 0.30 * concept_coverage
        + 0.20 * critical_coverage
        + 0.10 * phrase_consistency
    )
    evidence = top.best_sentences

    if _contains_restricted_advice(question_terms):
        return SupportDecision(False, support_score, "RESTRICTED_ADVICE", evidence)

    if _is_guarantee_request(question_terms):
        normalized_evidence = set(tokenize(evidence_text, remove_stop_words=False))
        if not normalized_evidence.intersection(_GUARANTEE_TERMS):
            return SupportDecision(
                False, support_score, "UNSUPPORTED_GUARANTEE", evidence
            )

    if _has_conflicting_number(question_terms, evidence_terms):
        return SupportDecision(False, support_score, "MISSING_CRITICAL_FACET", evidence)

    # A time unit in the request must be present in the evidence. A request for
    # an unspecified duration ("how long") is allowed when the evidence supplies
    # a concrete number and time unit.
    requested_time = _numeric_or_time_terms(question_terms) & _TIME_UNITS
    if requested_time and not requested_time.issubset(evidence_terms):
        return SupportDecision(False, support_score, "MISSING_CRITICAL_FACET", evidence)

    unmatched = critical_terms - evidence_terms
    if unmatched:
        return SupportDecision(False, support_score, "MISSING_CRITICAL_FACET", evidence)

    if concept_coverage < MIN_CONCEPT_COVERAGE:
        return SupportDecision(False, support_score, "LOW_CONCEPT_COVERAGE", evidence)
    if critical_coverage < MIN_CRITICAL_COVERAGE:
        return SupportDecision(False, support_score, "MISSING_CRITICAL_FACET", evidence)
    if not _has_coherent_sentence(question, top):
        return SupportDecision(False, support_score, "INCOHERENT_EVIDENCE", evidence)
    if support_score < SUPPORT_THRESHOLD:
        return SupportDecision(False, support_score, "LOW_SCORE", evidence)

    # Detect explicit polarity conflicts for the small set of critical
    # negations without exposing internal reasoning in the public result.
    normalized_question = normalize_text(question)
    normalized_evidence = normalize_text(evidence_text)
    if re.search(r"\b(cannot|never|not)\b", normalized_question) and not re.search(
        r"\b(cannot|never|not)\b", normalized_evidence
    ):
        return SupportDecision(False, support_score, "CONFLICTING_EVIDENCE", evidence)

    return SupportDecision(True, support_score, "SUPPORTED", evidence)

