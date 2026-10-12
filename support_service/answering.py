"""Extractive answer and citation construction."""

from __future__ import annotations

from .models import Citation, RetrievalResult, SupportDecision
from .retrieval import tokenize


ABSTENTION_MESSAGE = (
    "I don't have enough supported information in the knowledge base to answer "
    "that. Please contact human support for assistance."
)


def _select_sentences(
    question: str, result: RetrievalResult, decision: SupportDecision
) -> tuple[str, ...]:
    question_terms = set(tokenize(question))
    candidates = decision.evidence or result.best_sentences
    selected: list[str] = []
    covered: set[str] = set()

    for sentence in candidates:
        contribution = question_terms.intersection(tokenize(sentence)) - covered
        if contribution or not selected:
            selected.append(sentence)
            covered.update(question_terms.intersection(tokenize(sentence)))
        if question_terms.issubset(covered) or len(selected) == 2:
            break
    return tuple(selected)


def generate_answer(
    question: str, result: RetrievalResult, decision: SupportDecision
) -> tuple[str, list[Citation]]:
    """Generate an extractive answer after, and only after, support approval."""

    if not decision.supported:
        raise ValueError("answer generation requires an affirmative support decision")

    sentences = _select_sentences(question, result, decision)
    if not sentences:
        raise ValueError("supported answer has no evidence sentences")
    answer = " ".join(sentences)
    citations = [
        Citation(
            id=result.document.id,
            title=result.document.title,
            snippet=sentence,
        )
        for sentence in sentences
    ]
    return answer, citations


def restriction_citation(
    question: str, result: RetrievalResult | None, decision: SupportDecision
) -> list[Citation]:
    """Return a citation only when a source explicitly states the restriction."""

    if result is None or decision.reason_code != "RESTRICTED_ADVICE":
        return []
    question_terms = set(tokenize(question))
    for sentence in result.best_sentences:
        terms = set(tokenize(sentence))
        if terms.intersection({"legal", "tax", "taxation"}) and terms.intersection(
            {"advice", "advise"}
        ):
            return [
                Citation(result.document.id, result.document.title, sentence)
            ]
    return []

