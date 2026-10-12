"""Command-line interface for the AI support answer service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from support_service import KnowledgeBaseError, answer_question


DEFAULT_KB_PATH = Path(__file__).resolve().with_name("kb.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Answer a question using evidence from a local FAQ knowledge base."
    )
    parser.add_argument("question", help="Non-empty support question")
    parser.add_argument(
        "--kb",
        type=Path,
        default=DEFAULT_KB_PATH,
        help="Path to kb.json (default: repository kb.json)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = answer_question(args.question, args.kb)
    except (KnowledgeBaseError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        print(f"internal error: {exc}", file=sys.stderr)
        return 1

    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

