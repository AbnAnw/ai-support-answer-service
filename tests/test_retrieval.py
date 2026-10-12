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

    def test_required_morphology_and_number_words_are_canonical(self) -> None:
        groups = (
            (("enable", "enabled", "enabling"), "enable"),
            (("remove", "removed", "removing"), "remove"),
            (("use", "used", "using"), "use"),
            (("close", "closed", "closure"), "close"),
            (("expire", "expires", "expired", "expiry"), "expire"),
            (("send", "sent", "sending"), "send"),
            (("withdraw", "withdrawal"), "withdraw"),
            (("require", "required", "requirement", "requirements"), "require"),
            (("0", "zero"), "0"),
            (("2", "two"), "2"),
            (("30", "thirty"), "30"),
        )
        for variants, expected in groups:
            with self.subTest(variants=variants):
                self.assertEqual(
                    {tokenize(variant)[0] for variant in variants}, {expected}
                )


if __name__ == "__main__":
    unittest.main()

