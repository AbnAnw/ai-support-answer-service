"""Offline, evidence-grounded support answer service."""

from .kb import KnowledgeBaseError, load_knowledge_base
from .service import answer_question

__all__ = ["KnowledgeBaseError", "answer_question", "load_knowledge_base"]

