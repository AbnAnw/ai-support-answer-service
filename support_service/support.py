"""Passage-scoped, deterministic evidence support checks."""

from __future__ import annotations

from dataclasses import dataclass
import re

from .models import RetrievalResult, SupportDecision
from .retrieval import bigrams, normalize_text, split_sentences, tokenize


SUPPORT_THRESHOLD = 0.62

# Words that frame a request without identifying the fact that must be found.
# Certainty, polarity, quantities, units, actors, actions, and objects are not
# discarded here.
_FRAMING_TERMS = frozenset(
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
        "say",
        "soon",
        "take",
        "tell",
        "want",
        "would",
    }
)
_POLARITY_TERMS = frozenset({"not", "never", "cannot", "without"})
_STRONG_CERTAINTY_TERMS = frozenset(
    {"always", "certain", "certainly", "definite", "definitely", "guarantee", "will"}
)
_WEAK_QUALIFIERS = frozenset({"may", "might", "usually", "can", "longer"})
_ADVICE_TERMS = frozenset({"advice", "advise", "counsel", "guidance"})
_ACTION_TERMS = frozenset(
    {
        "change",
        "close",
        "download",
        "enable",
        "expire",
        "provide",
        "remove",
        "require",
        "reset",
        "send",
        "update",
        "use",
    }
)
_NUMBER_OR_TIME_RE = re.compile(
    r"^(?:\d+|second|minute|hour|day|week|month|year|weekend|holiday)$"
)
_CLAUSE_BOUNDARY_RE = re.compile(r"\s*[,;]\s*(?:and\s+)?", re.IGNORECASE)


@dataclass(frozen=True)
class _ClauseMatch:
    sentence: str
    sentence_index: int
    coverage: float
    phrase_score: float


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _split_clauses(question: str) -> tuple[str, ...]:
    """Split explicit multi-part questions without breaking noun lists."""

    parts = [part.strip() for part in _CLAUSE_BOUNDARY_RE.split(question) if part.strip()]
    clauses: list[str] = []
    for part in parts:
        # Split conjunctions that clearly introduce another requested action or
        # interrogative. "legal and tax advice" remains one clause.
        subparts = re.split(
            r"\s+and\s+(?=(?:where|when|how|what|who|why|can|could|does|do|will|"
            r"reset|change|send|remove|close|enable|use|download|update)\b)",
            part,
            flags=re.IGNORECASE,
        )
        clauses.extend(subpart.strip() for subpart in subparts if subpart.strip())
    return tuple(clauses) or (question.strip(),)


def _critical_terms(clause: str) -> set[str]:
    terms = set(tokenize(clause)) - _FRAMING_TERMS
    normalized = normalize_text(clause)
    # In "reset link expires/is sent", reset names the kind of link; expire or
    # send is the requested relationship. Password remains as the topic context.
    if "reset" in terms and "link" in terms and terms.intersection({"expire", "send"}):
        terms.remove("reset")
    if re.search(r"\bhow\s+(?:long|soon)\b", normalized):
        terms.add("__duration__")
    return terms


def _title_context(title: str) -> set[str]:
    """Use only a single-topic title as local sentence context.

    A compound title such as "Refunds and delivery" cannot attach the refund
    subject to a delivery sentence and manufacture a relationship.
    """

    normalized = normalize_text(title)
    if re.search(r"\b(?:and|or)\b|[/&]", normalized):
        return set()
    # A title may establish a local topic (for example 2FA), but an action in a
    # title may not be attached to a sentence about a different actor/action.
    return set(tokenize(title)) - _ACTION_TERMS


def _term_is_satisfied(term: str, evidence_terms: set[str]) -> bool:
    if term in evidence_terms:
        return True
    if term == "__duration__":
        has_number = any(item.isdigit() for item in evidence_terms)
        has_unit = any(
            item in {"second", "minute", "hour", "day", "week", "month", "year"}
            for item in evidence_terms
        )
        return has_number and has_unit
    if term == "require" and "must" in evidence_terms:
        return True
    # "Supported methods are ... SMS" directly supports "Can I use SMS?"
    if term == "use" and {"support", "method"}.issubset(evidence_terms):
        return True
    return False


def _coverage(required: set[str], evidence_terms: set[str]) -> float:
    if not required:
        return 0.0
    matched = sum(_term_is_satisfied(term, evidence_terms) for term in required)
    return matched / len(required)


def _phrase_score(clause: str, sentence: str) -> float:
    query_bigrams = set(bigrams(tokenize(clause)))
    if not query_bigrams:
        return 0.0
    sentence_bigrams = set(bigrams(tokenize(sentence)))
    return len(query_bigrams.intersection(sentence_bigrams)) / len(query_bigrams)


def _has_quantity_mismatch(required: set[str], evidence_terms: set[str]) -> bool:
    quantities = {term for term in required if _NUMBER_OR_TIME_RE.match(term)}
    return any(not _term_is_satisfied(term, evidence_terms) for term in quantities)


def _has_polarity_conflict(clause: str, evidence_terms: set[str]) -> bool:
    query_terms = set(tokenize(clause, remove_stop_words=False))
    requested_polarity = query_terms.intersection(_POLARITY_TERMS)
    if not requested_polarity:
        # A neutral capability question may be answered by explicit negative
        # evidence, e.g. "Can support send my existing password?" → "cannot".
        return False
    return not requested_polarity.issubset(evidence_terms)


def _has_certainty_conflict(clause: str, evidence_terms: set[str]) -> bool:
    query_terms = set(tokenize(clause, remove_stop_words=False))
    requested_certainty = query_terms.intersection(_STRONG_CERTAINTY_TERMS)
    if not requested_certainty:
        return False
    explicit_certainty = evidence_terms.intersection(_STRONG_CERTAINTY_TERMS)
    return not explicit_certainty or bool(evidence_terms.intersection(_WEAK_QUALIFIERS))


def _is_evidence_restriction(clause: str, evidence_terms: set[str]) -> bool:
    """Recognize a retrieved prohibition without hardcoding a topic."""

    clause_terms = set(tokenize(clause))
    asks_for_advice = bool(clause_terms.intersection(_ADVICE_TERMS))
    prohibits_provision = "not" in evidence_terms and "provide" in evidence_terms
    return asks_for_advice and prohibits_provision


def _match_clause(
    clause: str,
    sentences: tuple[str, ...],
    title_terms: set[str],
) -> tuple[_ClauseMatch | None, float, str | None]:
    required = _critical_terms(clause)
    if not required:
        return None, 0.0, "NO_MEANINGFUL_TERMS"

    best_score = 0.0
    best_reason: str | None = None
    for index, sentence in enumerate(sentences):
        sentence_terms = set(tokenize(sentence, remove_stop_words=False))
        evidence_terms = sentence_terms.union(title_terms)
        coverage = _coverage(required, evidence_terms)
        phrase = _phrase_score(clause, sentence)
        candidate_score = 0.85 * coverage + 0.15 * phrase
        if candidate_score > best_score:
            best_score = candidate_score

        # All critical actor/action/object/quantity facets must coexist in this
        # sentence (with safe, single-topic title context).
        if coverage < 1.0:
            continue
        if _has_quantity_mismatch(required, evidence_terms):
            best_reason = "QUANTITY_MISMATCH"
            continue
        if _has_polarity_conflict(clause, evidence_terms):
            best_reason = "POLARITY_CONFLICT"
            continue
        if _has_certainty_conflict(clause, evidence_terms):
            best_reason = "QUALIFICATION_CONFLICT"
            continue
        if _is_evidence_restriction(clause, evidence_terms):
            return (
                _ClauseMatch(sentence, index, coverage, phrase),
                candidate_score,
                "EVIDENCE_RESTRICTION",
            )
        return _ClauseMatch(sentence, index, coverage, phrase), candidate_score, None

    return None, best_score, best_reason or "MISSING_PASSAGE_FACETS"


def _passage_is_tight(matches: tuple[_ClauseMatch, ...]) -> bool:
    indexes = sorted({match.sentence_index for match in matches})
    if len(indexes) <= 1:
        return True
    return indexes[-1] - indexes[0] + 1 == len(indexes)


def _passage_score(
    matches: tuple[_ClauseMatch, ...], retrieval_score: float
) -> float:
    if not matches:
        return 0.0
    facet_coverage = sum(match.coverage for match in matches) / len(matches)
    phrase_consistency = sum(match.phrase_score for match in matches) / len(matches)
    # Facet coverage is one signal, not duplicated as both concept and critical
    # coverage. Retrieval is deliberately a small tie-breaking contribution.
    return _clamp(
        0.75 * facet_coverage
        + 0.15 * phrase_consistency
        + 0.10 * retrieval_score
    )


def check_support(
    question: str, results: tuple[RetrievalResult, ...]
) -> SupportDecision:
    """Approve only a passage that supports every requested clause locally."""

    if not results:
        return SupportDecision(False, 0.0, "NO_EVIDENCE", ())

    clauses = _split_clauses(question)
    best_rejected_score = 0.0
    best_rejected_reason = "MISSING_PASSAGE_FACETS"
    best_rejected_evidence: tuple[str, ...] = ()
    best_rejected_document: str | None = None

    for result in results:
        sentences = split_sentences(result.document.text)
        title_terms = _title_context(result.document.title)
        matches: list[_ClauseMatch] = []
        rejected = False

        for clause in clauses:
            match, clause_score, reason = _match_clause(clause, sentences, title_terms)
            if reason == "EVIDENCE_RESTRICTION" and match is not None:
                score = _clamp(0.90 * clause_score + 0.10 * result.score)
                if score > best_rejected_score:
                    best_rejected_score = score
                    best_rejected_reason = reason
                    best_rejected_evidence = (match.sentence,)
                    best_rejected_document = result.document.id
                rejected = True
                break
            if match is None:
                score = _clamp(0.90 * clause_score + 0.10 * result.score)
                if score > best_rejected_score:
                    best_rejected_score = score
                    best_rejected_reason = reason or "MISSING_PASSAGE_FACETS"
                    best_rejected_evidence = ()
                    best_rejected_document = result.document.id
                rejected = True
                break
            matches.append(match)

        if rejected:
            continue

        match_tuple = tuple(matches)
        score = _passage_score(match_tuple, result.score)
        if not _passage_is_tight(match_tuple):
            if score > best_rejected_score:
                best_rejected_score = score
                best_rejected_reason = "DISJOINT_PASSAGE"
                best_rejected_document = result.document.id
            continue
        if score < SUPPORT_THRESHOLD:
            if score > best_rejected_score:
                best_rejected_score = score
                best_rejected_reason = "LOW_PASSAGE_SCORE"
                best_rejected_document = result.document.id
            continue

        selected_indexes = sorted({match.sentence_index for match in match_tuple})
        evidence = tuple(sentences[index] for index in selected_indexes)
        return SupportDecision(True, score, "SUPPORTED", evidence, result.document.id)

    return SupportDecision(
        False,
        best_rejected_score,
        best_rejected_reason,
        best_rejected_evidence,
        best_rejected_document,
    )

