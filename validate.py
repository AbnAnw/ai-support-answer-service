"""Run representative end-to-end checks against the sample knowledge base."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from support_service import KnowledgeBaseError, answer_question, load_knowledge_base


KB_PATH = Path(__file__).resolve().with_name("kb.json")
CASES = (
    ("How do I reset my password?", "answer"),
    ("What happens if I lose access to my second factor?", "answer"),
    ("How long do withdrawals take in review?", "answer"),
    ("What are the requirements before I close my account?", "answer"),
    ("Can support change my email address for me?", "abstain"),
    ("What tax advice do you provide for closed accounts?", "abstain"),
    ("Can you guarantee my withdrawal will finish in 2 hours?", "abstain"),
)


def schema_errors(result: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if set(result) != {"question", "decision", "answer", "citations", "debug"}:
        errors.append("invalid top-level keys")
    if result.get("decision") not in {"answer", "abstain"}:
        errors.append("invalid decision")
    if not isinstance(result.get("question"), str):
        errors.append("question is not a string")
    if not isinstance(result.get("answer"), str):
        errors.append("answer is not a string")
    if not isinstance(result.get("citations"), list):
        errors.append("citations is not a list")
    debug = result.get("debug")
    if not isinstance(debug, dict) or set(debug) != {"retrieved_ids", "support_score"}:
        errors.append("invalid debug object")
    elif (
        not isinstance(debug["retrieved_ids"], list)
        or isinstance(debug["support_score"], bool)
        or not isinstance(debug["support_score"], (int, float))
        or not 0.0 <= debug["support_score"] <= 1.0
    ):
        errors.append("invalid debug field types or range")
    return errors


def main() -> int:
    try:
        documents = load_knowledge_base(KB_PATH)
    except KnowledgeBaseError as exc:
        print(f"KB LOAD FAILED: {exc}")
        return 1

    by_id = {document.id: document for document in documents}
    failures: list[str] = []
    correct_decisions = 0
    answered = 0
    answered_with_citations = 0

    print(f"Loaded {len(documents)} knowledge-base documents from {KB_PATH.name}.")
    for index, (question, expected) in enumerate(CASES, start=1):
        result = answer_question(question, KB_PATH)
        actual = result["decision"]
        case_failures = schema_errors(result)
        if actual == expected:
            correct_decisions += 1
        else:
            case_failures.append(f"expected {expected}, got {actual}")

        if actual == "answer":
            answered += 1
            citations = result["citations"]
            if citations:
                answered_with_citations += 1
            else:
                case_failures.append("answered result has no citation")
            for citation in citations:
                source = by_id.get(citation.get("id"))
                if source is None:
                    case_failures.append("citation has unknown document ID")
                elif (
                    citation.get("title") != source.title
                    or citation.get("snippet") not in source.text
                ):
                    case_failures.append("citation is not grounded in its source")

        status = "PASS" if not case_failures else "FAIL"
        print(f"{index}. {status}: {question} -> {actual}")
        failures.extend(f"Case {index}: {failure}" for failure in case_failures)

    decision_accuracy = correct_decisions / len(CASES)
    citation_rate = answered_with_citations / answered if answered else 1.0
    print(f"Decision accuracy: {correct_decisions}/{len(CASES)} ({decision_accuracy:.1%})")
    print(
        "Citation presence rate for answered cases: "
        f"{answered_with_citations}/{answered} ({citation_rate:.1%})"
    )
    if failures:
        print("Failures:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("All validation checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

