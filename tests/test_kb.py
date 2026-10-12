from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from support_service.kb import KnowledgeBaseError, load_knowledge_base


class KnowledgeBaseTests(unittest.TestCase):
    def write_json(self, directory: str, value: object) -> Path:
        path = Path(directory) / "kb.json"
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return path

    def test_loads_valid_unicode_knowledge_base(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_json(
                directory,
                [{"id": "unicode", "title": "Café", "text": "你好, café."}],
            )
            documents = load_knowledge_base(path)
            self.assertEqual(documents[0].title, "Café")
            self.assertEqual(documents[0].text, "你好, café.")

    def test_missing_file_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(KnowledgeBaseError, "not found"):
                load_knowledge_base(Path(directory) / "missing.json")

    def test_invalid_json_reports_location(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kb.json"
            path.write_text("[{", encoding="utf-8")
            with self.assertRaisesRegex(KnowledgeBaseError, r"line 1, column"):
                load_knowledge_base(path)

    def test_non_array_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_json(directory, {"id": "x"})
            with self.assertRaisesRegex(KnowledgeBaseError, "JSON array"):
                load_knowledge_base(path)

    def test_empty_array_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_json(directory, [])
            with self.assertRaisesRegex(KnowledgeBaseError, "must not be empty"):
                load_knowledge_base(path)

    def test_invalid_records_are_rejected(self) -> None:
        invalid_cases = (
            (["not an object"], "expected a JSON object"),
            ([{"id": "x", "title": "Title"}], "missing required field 'text'"),
            ([{"id": 1, "title": "Title", "text": "Text"}], "must be a string"),
            ([{"id": "x", "title": " ", "text": "Text"}], "must not be blank"),
        )
        for payload, message in invalid_cases:
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as directory:
                path = self.write_json(directory, payload)
                with self.assertRaisesRegex(KnowledgeBaseError, message):
                    load_knowledge_base(path)

    def test_duplicate_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_json(
                directory,
                [
                    {"id": "same", "title": "First", "text": "One"},
                    {"id": "same", "title": "Second", "text": "Two"},
                ],
            )
            with self.assertRaisesRegex(KnowledgeBaseError, "duplicate document id 'same'"):
                load_knowledge_base(path)


if __name__ == "__main__":
    unittest.main()

