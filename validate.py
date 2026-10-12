"""Run representative and adversarial end-to-end pipeline checks."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any

from support_service import KnowledgeBaseError, answer_question, load_knowledge_base


PROJECT_ROOT = Path(__file__).resolve().parent
KB_PATH = PROJECT_ROOT / "kb.json"
SAMPLE_CASES = (
    ("Password reset", "How do I reset my password?", "answer"),
    (
        "2FA recovery",
        "What happens if I lose access to my second factor?",
        "answer",
    ),
    ("Withdrawal timing", "How long do withdrawals take in review?", "answer"),
    (
        "Account closure requirements",
        "What are the requirements before I close my account?",
        "answer",
    ),
    ("Reset expiry paraphrase", "How soon does the password reset link expire?", "answer"),
    (
        "Number-word equivalence",
        "Does the password reset link expire in thirty minutes?",
        "answer",
    ),
    ("Enable morphology", "How do I enable 2FA?", "answer"),
    ("Supported method paraphrase", "Can I use SMS for two-factor authentication?", "answer"),
    ("Zero equivalence", "Can I close my account with a balance of 0?", "answer"),
    (
        "Supported negative capability",
        "Can support send me my existing password?",
        "answer",
    ),
    (
        "Complete multi-part answer",
        "Where do I reset my password, where is the link sent, and when does it expire?",
        "answer",
    ),
    ("Unsupported email action", "Can support change my email address for me?", "abstain"),
    ("Explicit tax restriction", "What tax advice does support provide?", "abstain"),
    (
        "Unsupported two-hour guarantee",
        "Can you guarantee my withdrawal will finish in 2 hours?",
        "abstain",
    ),
    (
        "Unsupported support reset actor",
        "Can support agents reset my password?",
        "abstain",
    ),
    (
        "Unsupported support link actor",
        "Can support agents send a reset link?",
        "abstain",
    ),
    (
        "Qualified timing is not definite",
        "Will my withdrawal definitely finish within 24 hours?",
        "abstain",
    ),
    (
        "Closure requirement conflict",
        "Can I close my account without closing open positions?",
        "abstain",
    ),
    (
        "2FA requirement conflict",
        "Can I remove 2FA without identity verification?",
        "abstain",
    ),
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


def _write_kb(path: Path, payload: list[dict[str, str]]) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _run_cases(cases: list[tuple[str, str, str, Path]]) -> int:
    failures: list[str] = []
    correct_decisions = 0
    answered = 0
    answered_with_citations = 0
    false_answers = 0
    false_abstentions = 0
    documents_by_path: dict[Path, dict[str, Any]] = {}

    for _, _, _, path in cases:
        if path not in documents_by_path:
            try:
                documents_by_path[path] = {
                    document.id: document for document in load_knowledge_base(path)
                }
            except KnowledgeBaseError as exc:
                print(f"KB LOAD FAILED ({path}): {exc}")
                return 1

    print(f"Loaded {len(documents_by_path)} knowledge-base fixture(s).")
    for index, (name, question, expected, path) in enumerate(cases, start=1):
        result = answer_question(question, path)
        actual = result["decision"]
        case_failures = schema_errors(result)
        if actual == expected:
            correct_decisions += 1
        else:
            case_failures.append(f"expected {expected}, got {actual}")
            if expected == "abstain" and actual == "answer":
                false_answers += 1
            elif expected == "answer" and actual == "abstain":
                false_abstentions += 1

        if actual == "answer":
            answered += 1
            citations = result["citations"]
            if citations:
                answered_with_citations += 1
            else:
                case_failures.append("answered result has no citation")
            by_id = documents_by_path[path]
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
        print(f"{index}. {status}: {name} -> {actual}")
        failures.extend(f"Case {index} ({name}): {failure}" for failure in case_failures)

    decision_accuracy = correct_decisions / len(cases)
    citation_rate = answered_with_citations / answered if answered else 1.0
    print(f"Decision accuracy: {correct_decisions}/{len(cases)} ({decision_accuracy:.1%})")
    print(
        "Citation presence rate for answered cases: "
        f"{answered_with_citations}/{answered} ({citation_rate:.1%})"
    )
    print(f"False-answer count: {false_answers}")
    print(f"False-abstention count: {false_abstentions}")
    if failures:
        print("Failures:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("All validation checks passed.")
    return 0


def main() -> int:
    with tempfile.TemporaryDirectory(prefix=".validate-", dir=PROJECT_ROOT) as directory:
        fixture_root = Path(directory)
        shipping_kb = _write_kb(
            fixture_root / "shipping.json",
            [
                {
                    "id": "shipping",
                    "title": "Refunds and delivery",
                    "text": (
                        "Refunds are available within 30 days. "
                        "Express delivery is available within 2 hours."
                    ),
                }
            ],
        )
        actions_kb = _write_kb(
            fixture_root / "actions.json",
            [
                {
                    "id": "actions",
                    "title": "Account actions",
                    "text": (
                        "Alice can reset her password. "
                        "Bob can change the email address."
                    ),
                }
            ],
        )
        policy_kb = _write_kb(
            fixture_root / "policy.json",
            [
                {
                    "id": "policy",
                    "title": "Tax guidance",
                    "text": (
                        "Support provides general tax guidance about downloading "
                        "annual tax statements."
                    ),
                }
            ],
        )

        cases = [(name, question, expected, KB_PATH) for name, question, expected in SAMPLE_CASES]
        cases.extend(
            [
                (
                    "Cross-sentence quantity mismatch",
                    "Are refunds available within 2 hours?",
                    "abstain",
                    shipping_kb,
                ),
                (
                    "Cross-sentence actor/action mismatch",
                    "Can Alice change the email address?",
                    "abstain",
                    actions_kb,
                ),
                (
                    "Replacement KB permits tax guidance",
                    "What tax guidance does support provide?",
                    "answer",
                    policy_kb,
                ),
            ]
        )
        return _run_cases(cases)


if __name__ == "__main__":
    raise SystemExit(main())

