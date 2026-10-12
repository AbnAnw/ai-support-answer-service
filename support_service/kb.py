"""Knowledge-base loading and schema validation."""

from __future__ import annotations

import json
from pathlib import Path

from .models import Document


class KnowledgeBaseError(ValueError):
    """Raised when a knowledge-base file cannot be loaded or validated."""


def _location(path: Path, index: int | None = None) -> str:
    if index is None:
        return str(path)
    return f"{path} (record {index})"


def load_knowledge_base(path: str | Path) -> tuple[Document, ...]:
    """Load and validate a JSON FAQ knowledge base from *path*.

    The loader intentionally performs disk I/O on every call. This keeps the
    service responsive to replacement KB files and prevents sample content
    from becoming an implicit source of truth.
    """

    kb_path = Path(path)
    try:
        raw_text = kb_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise KnowledgeBaseError(f"Knowledge base file not found: {kb_path}") from exc
    except OSError as exc:
        raise KnowledgeBaseError(
            f"Could not read knowledge base '{kb_path}': {exc}"
        ) from exc

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise KnowledgeBaseError(
            f"Invalid JSON in knowledge base '{kb_path}' at "
            f"line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(payload, list):
        raise KnowledgeBaseError(
            f"Knowledge base root must be a JSON array: {kb_path}"
        )
    if not payload:
        raise KnowledgeBaseError(f"Knowledge base must not be empty: {kb_path}")

    documents: list[Document] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(payload):
        location = _location(kb_path, index)
        if not isinstance(record, dict):
            raise KnowledgeBaseError(f"{location}: expected a JSON object")

        values: dict[str, str] = {}
        for field in ("id", "title", "text"):
            if field not in record:
                raise KnowledgeBaseError(f"{location}: missing required field '{field}'")
            value = record[field]
            if not isinstance(value, str):
                raise KnowledgeBaseError(f"{location}: field '{field}' must be a string")
            value = value.strip()
            if not value:
                raise KnowledgeBaseError(
                    f"{location}: field '{field}' must not be blank"
                )
            values[field] = value

        if values["id"] in seen_ids:
            raise KnowledgeBaseError(
                f"{location}: duplicate document id '{values['id']}'"
            )
        seen_ids.add(values["id"])
        documents.append(Document(**values))

    return tuple(documents)

