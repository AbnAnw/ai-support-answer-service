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
    def write_temporary_kb(self, directory: str, payload: object) -> Path:
        path = Path(directory) / "replacement.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

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

    def test_support_agent_actor_action_mismatches_abstain(self) -> None:
        questions = (
            "Can support agents reset my password?",
            "Can support agents send a reset link?",
        )
        for question in questions:
            with self.subTest(question=question):
                self.assertEqual(
                    answer_question(question, SAMPLE_KB)["decision"], "abstain"
                )

    def test_quantity_cannot_move_between_unrelated_sentences(self) -> None:
        payload = [
            {
                "id": "shipping",
                "title": "Refunds and delivery",
                "text": (
                    "Refunds are available within 30 days. "
                    "Express delivery is available within 2 hours."
                ),
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_temporary_kb(directory, payload)
            result = answer_question("Are refunds available within 2 hours?", path)
            self.assertEqual(result["decision"], "abstain")

    def test_actor_action_object_cannot_move_between_sentences(self) -> None:
        payload = [
            {
                "id": "actions",
                "title": "Account actions",
                "text": (
                    "Alice can reset her password. "
                    "Bob can change the email address."
                ),
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_temporary_kb(directory, payload)
            result = answer_question("Can Alice change the email address?", path)
            self.assertEqual(result["decision"], "abstain")

    def test_supported_negative_capability_answers_from_exact_sentence(self) -> None:
        result = answer_question(
            "Can support send me my existing password?", SAMPLE_KB
        )
        self.assertEqual(result["decision"], "answer")
        self.assertIn("cannot", result["answer"])
        self.assertIn("existing passwords", result["answer"])
        self.assert_citations_are_grounded(result, SAMPLE_KB)

    def test_paraphrases_and_number_words_are_supported(self) -> None:
        questions = (
            "How soon does the password reset link expire?",
            "Does the password reset link expire in thirty minutes?",
            "How do I enable 2FA?",
            "Can I use SMS for two-factor authentication?",
            "Can I close my account with a balance of 0?",
            "How long do withdrawal reviews usually take?",
        )
        for question in questions:
            with self.subTest(question=question):
                result = answer_question(question, SAMPLE_KB)
                self.assertEqual(result["decision"], "answer")
                self.assert_citations_are_grounded(result, SAMPLE_KB)

    def test_qualification_and_requirement_conflicts_abstain(self) -> None:
        questions = (
            "Can you guarantee my withdrawal will finish in 2 hours?",
            "Will my withdrawal definitely finish within 24 hours?",
            "Can I close my account without closing open positions?",
            "Can I remove 2FA without identity verification?",
        )
        for question in questions:
            with self.subTest(question=question):
                self.assertEqual(
                    answer_question(question, SAMPLE_KB)["decision"], "abstain"
                )

    def test_complete_multi_part_answer_uses_all_approved_sentences(self) -> None:
        result = answer_question(
            "Where do I reset my password, where is the link sent, and when does it expire?",
            SAMPLE_KB,
        )
        self.assertEqual(result["decision"], "answer")
        self.assertIn("login page", result["answer"])
        self.assertIn("registered email address", result["answer"])
        self.assertIn("30 minutes", result["answer"])
        self.assertEqual(len(result["citations"]), 3)
        self.assert_citations_are_grounded(result, SAMPLE_KB)

    def test_replacement_kb_policy_is_authoritative(self) -> None:
        payload = [
            {
                "id": "policy",
                "title": "Tax guidance",
                "text": (
                    "Support provides general tax guidance about downloading "
                    "annual tax statements."
                ),
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_temporary_kb(directory, payload)
            result = answer_question("What tax guidance does support provide?", path)
            self.assertEqual(result["decision"], "answer")
            self.assertEqual(result["citations"][0]["id"], "policy")
            self.assert_citations_are_grounded(result, path)

    def test_sample_kb_advice_prohibition_still_abstains(self) -> None:
        result = answer_question("What tax advice does support provide?", SAMPLE_KB)
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
            payload = [
                {
                    "id": "novel_1",
                    "title": "Moonstone invoices",
                    "text": "Moonstone invoices can be downloaded from the lunar billing page.",
                }
            ]
            path = self.write_temporary_kb(directory, payload)
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

