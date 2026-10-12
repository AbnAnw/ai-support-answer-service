from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP = PROJECT_ROOT / "app.py"


class CliTests(unittest.TestCase):
    def run_cli(self, *arguments: str, cwd: str | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(APP), *arguments],
            cwd=cwd,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )

    def test_cli_emits_valid_json(self) -> None:
        completed = self.run_cli("How do I reset my password?")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["decision"], "answer")

    def test_default_kb_resolves_outside_project_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            completed = self.run_cli("How do I reset my password?", cwd=directory)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["decision"], "answer")

    def test_invalid_kb_uses_stderr_and_nonzero_exit(self) -> None:
        completed = self.run_cli("--kb", "does-not-exist.json", "Question")
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(completed.stdout, "")
        self.assertIn("not found", completed.stderr)

    def test_blank_question_uses_stderr_and_nonzero_exit(self) -> None:
        completed = self.run_cli("   ")
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(completed.stdout, "")
        self.assertIn("non-empty", completed.stderr)


if __name__ == "__main__":
    unittest.main()

