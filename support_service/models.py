"""Typed data passed between the pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class RetrievalResult:
    document: Document
    score: float
    term_coverage: float
    bigram_coverage: float
    matched_terms: tuple[str, ...]
    best_sentences: tuple[str, ...]


@dataclass(frozen=True)
class SupportDecision:
    supported: bool
    score: float
    reason_code: str
    evidence: tuple[str, ...]
    document_id: str | None = None


@dataclass(frozen=True)
class Citation:
    id: str
    title: str
    snippet: str

    def as_dict(self) -> dict[str, str]:
        return {"id": self.id, "title": self.title, "snippet": self.snippet}


Decision = Literal["answer", "abstain"]

