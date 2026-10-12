from __future__ import annotations

import unittest

from support_service.models import Document
from support_service.retrieval import retrieve, tokenize


class RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = (
            Document("a", "Orchid invoices", "Download orchid invoices from Billing."),
            Document("b", "General help", "Ask support for general account help."),
            Document("c", "Orchid colors", "Orchid themes change display colors."),
        )

    def test_phrase_and_specific_terms_rank_best_document_first(self) -> None:
        results = retrieve("Where can I download orchid invoices?", self.documents)
        self.assertEqual(results[0].document.id, "a")
        self.assertGreater(results[0].score, results[1].score)

    def test_results_are_deterministic_and_limited(self) -> None:
        first = retrieve("orchid", self.documents, top_k=2)
        second = retrieve("orchid", self.documents, top_k=2)
        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 2)

    def test_stop_words_do_not_create_false_results(self) -> None:
        self.assertEqual(retrieve("the and of", self.documents), ())

    def test_normalization_handles_inflection_and_two_factor(self) -> None:
        self.assertEqual(tokenize("Withdrawals"), ("withdraw",))
        self.assertEqual(tokenize("Two-factor authentication"), ("2fa", "authentication"))


if __name__ == "__main__":
    unittest.main()

