"""Pipeline orchestration and public result contract."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .answering import ABSTENTION_MESSAGE, generate_answer, restriction_citation
from .kb import load_knowledge_base
from .models import Citation, Document
from .retrieval import retrieve
from .support import check_support


def _validate_result(result: dict[str, Any], documents: tuple[Document, ...]) -> None:
    expected_keys = {"question", "decision", "answer", "citations", "debug"}
    if set(result) != expected_keys:
        raise RuntimeError("result has an invalid top-level schema")
    if result["decision"] not in {"answer", "abstain"}:
        raise RuntimeError("result has an invalid decision")
    if not isinstance(result["question"], str) or not isinstance(result["answer"], str):
        raise RuntimeError("question and answer must be strings")
    if not isinstance(result["citations"], list):
        raise RuntimeError("citations must be a list")
    if set(result["debug"]) != {"retrieved_ids", "support_score"}:
        raise RuntimeError("result has an invalid debug schema")
    score = result["debug"]["support_score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise RuntimeError("support_score must be numeric")
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise RuntimeError("support_score must be finite and within [0, 1]")

    by_id = {document.id: document for document in documents}
    retrieved_ids = result["debug"]["retrieved_ids"]
    if not isinstance(retrieved_ids, list) or not all(
        isinstance(item, str) and item in by_id for item in retrieved_ids
    ):
        raise RuntimeError("retrieved_ids must reference loaded documents")

    for citation in result["citations"]:
        if not isinstance(citation, dict) or set(citation) != {"id", "title", "snippet"}:
            raise RuntimeError("citation has an invalid schema")
        document = by_id.get(citation["id"])
        if document is None:
            raise RuntimeError("citation references an unknown document")
        if citation["title"] != document.title:
            raise RuntimeError("citation title does not match its document")
        if not isinstance(citation["snippet"], str) or citation["snippet"] not in document.text:
            raise RuntimeError("citation snippet is not present in its document")

    if result["decision"] == "answer" and not result["citations"]:
        raise RuntimeError("answered results require at least one citation")


def answer_question(
    question: str, kb_path: str | Path, *, top_k: int = 3
) -> dict[str, Any]:
    """Execute LOAD_KB → RETRIEVE → CHECK_SUPPORT → ANSWER/ABSTAIN."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string")
    clean_question = question.strip()

    # Stage 1: LOAD_KB
    documents = load_knowledge_base(kb_path)
    # Stage 2: RETRIEVE
    results = retrieve(clean_question, documents, top_k=top_k)
    # Stage 3: CHECK_SUPPORT
    support = check_support(clean_question, results)

    citations: list[Citation]
    if support.supported:
        # Stage 4a: ANSWER
        answer, citations = generate_answer(clean_question, results[0], support)
        decision = "answer"
    else:
        # Stage 4b: ABSTAIN
        answer = ABSTENTION_MESSAGE
        citations = restriction_citation(
            clean_question, results[0] if results else None, support
        )
        decision = "abstain"

    result: dict[str, Any] = {
        "question": clean_question,
        "decision": decision,
        "answer": answer,
        "citations": [citation.as_dict() for citation in citations],
        "debug": {
            "retrieved_ids": [item.document.id for item in results],
            "support_score": round(support.score, 6),
        },
    }
    _validate_result(result, documents)
    return result

