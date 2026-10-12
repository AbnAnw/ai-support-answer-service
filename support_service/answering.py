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
    # The support stage has already approved the exact sentence-level passage.
    # Re-selecting here could reintroduce unrelated document evidence.
    return decision.evidence


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
    _question: str, result: RetrievalResult | None, decision: SupportDecision
) -> list[Citation]:
    """Return a citation only when a source explicitly states the restriction."""

    if result is None or decision.reason_code != "EVIDENCE_RESTRICTION":
        return []
    for sentence in decision.evidence:
        terms = set(tokenize(sentence))
        if (
            terms.intersection({"advice", "advise", "counsel", "guidance"})
            and "not" in terms
            and "provide" in terms
        ):
            return [
                Citation(result.document.id, result.document.title, sentence)
            ]
    return []

