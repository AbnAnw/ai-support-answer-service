from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from support_service import answer_question, load_knowledge_base
from support_service.answering import ABSTENTION_MESSAGE


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_KB = PROJECT_ROOT / "kb.json"


class ServiceTests(unittest.TestCase):
    def assert_valid_schema(self, result: dict[str, object]) -> None:
        self.assertEqual(
            set(result), {"question", "decision", "answer", "citations", "debug"}
        )
        self.assertIsInstance(result["question"], str)
        self.assertIn(result["decision"], {"answer", "abstain"})
        self.assertIsInstance(result["answer"], str)
        self.assertIsInstance(result["citations"], list)
        debug = result["debug"]
        self.assertIsInstance(debug, dict)
        self.assertEqual(set(debug), {"retrieved_ids", "support_score"})
        self.assertIsInstance(debug["retrieved_ids"], list)
        self.assertIsInstance(debug["support_score"], (int, float))
        self.assertNotIsInstance(debug["support_score"], bool)
        self.assertGreaterEqual(debug["support_score"], 0.0)
        self.assertLessEqual(debug["support_score"], 1.0)
        for citation in result["citations"]:
            self.assertEqual(set(citation), {"id", "title", "snippet"})

    def assert_citations_are_grounded(self, result: dict[str, object], kb_path: Path) -> None:
        documents = {document.id: document for document in load_knowledge_base(kb_path)}
        citations = result["citations"]
        self.assertTrue(citations)
        for citation in citations:
            self.assertIn(citation["id"], documents)
            source = documents[citation["id"]]
            self.assertEqual(citation["title"], source.title)
            self.assertIn(citation["snippet"], source.text)

    def test_answerable_password_question_has_grounded_citations(self) -> None:
        result = answer_question("How do I reset my password?", SAMPLE_KB)
        self.assertEqual(result["decision"], "answer")
        self.assert_valid_schema(result)
        self.assert_citations_are_grounded(result, SAMPLE_KB)
        self.assertEqual(result["debug"]["retrieved_ids"][0], "doc_1")

    def test_polite_framing_does_not_cause_false_abstention(self) -> None:
        result = answer_question("Please tell me how to reset my password.", SAMPLE_KB)
        self.assertEqual(result["decision"], "answer")
        self.assert_citations_are_grounded(result, SAMPLE_KB)

    def test_answerable_two_factor_recovery(self) -> None:
        result = answer_question(
            "What happens if I lose access to my second factor?", SAMPLE_KB
        )
        self.assertEqual(result["decision"], "answer")
        self.assert_citations_are_grounded(result, SAMPLE_KB)

    def test_answerable_withdrawal_review_timing(self) -> None:
        result = answer_question("How long do withdrawals take in review?", SAMPLE_KB)
        self.assertEqual(result["decision"], "answer")
        self.assert_citations_are_grounded(result, SAMPLE_KB)
        self.assertIn("24 hours", result["answer"])

    def test_answerable_account_closure_requirements(self) -> None:
        result = answer_question(
            "What are the requirements before I close my account?", SAMPLE_KB
        )
        self.assertEqual(result["decision"], "answer")
        self.assert_citations_are_grounded(result, SAMPLE_KB)
        self.assertIn("balance must be zero", result["answer"])

    def test_unrelated_question_abstains(self) -> None:
        result = answer_question("What is today's weather in Singapore?", SAMPLE_KB)
        self.assertEqual(result["decision"], "abstain")
        self.assertEqual(result["answer"], ABSTENTION_MESSAGE)
        self.assert_valid_schema(result)

    def test_unsupported_guarantee_abstains(self) -> None:
        result = answer_question(
            "Can you guarantee my withdrawal will finish in 2 hours?", SAMPLE_KB
        )
        self.assertEqual(result["decision"], "abstain")

    def test_tax_advice_abstains_with_optional_grounded_restriction(self) -> None:
        result = answer_question(
            "What tax advice do you provide for closed accounts?", SAMPLE_KB
        )
        self.assertEqual(result["decision"], "abstain")
        if result["citations"]:
            self.assert_citations_are_grounded(result, SAMPLE_KB)

    def test_unsupported_email_change_action_abstains(self) -> None:
        result = answer_question(
            "Can support change my email address for me?", SAMPLE_KB
        )
        self.assertEqual(result["decision"], "abstain")

    def test_partially_supported_multi_part_question_abstains(self) -> None:
        result = answer_question(
            "How do I reset my password and change my username?", SAMPLE_KB
        )
        self.assertEqual(result["decision"], "abstain")

    def test_output_contains_exact_required_fields(self) -> None:
        self.assert_valid_schema(answer_question("How do I reset my password?", SAMPLE_KB))
        self.assert_valid_schema(answer_question("How do I change my username?", SAMPLE_KB))

    def test_temporary_replacement_kb_changes_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replacement.json"
            payload = [
                {
                    "id": "novel_1",
                    "title": "Moonstone invoices",
                    "text": "Moonstone invoices can be downloaded from the lunar billing page.",
                }
            ]
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = answer_question(
                "Where can I download Moonstone invoices?", path
            )
            self.assertEqual(result["decision"], "answer")
            self.assertEqual(result["citations"][0]["id"], "novel_1")
            self.assertIn("Moonstone", result["answer"])

    def test_unicode_question_and_answer_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unicode.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "id": "cafe",
                            "title": "Café résumé",
                            "text": "Le résumé du café est disponible dans le profil.",
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            result = answer_question("Where is the café résumé?", path)
            self.assertEqual(result["decision"], "answer")
            self.assertIn("café", result["answer"])

    def test_blank_question_fails_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            answer_question("  ", SAMPLE_KB)


if __name__ == "__main__":
    unittest.main()

