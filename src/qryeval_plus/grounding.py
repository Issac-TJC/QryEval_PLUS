"""Deterministic, auditable answer-support checks."""

from __future__ import annotations

import re
import string
from typing import Iterable


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "that", "the", "to", "was", "were", "with",
}


def normalize_text(value: str) -> str:
    punctuation = set(string.punctuation + "‘’´`")
    lowered = str(value or "").replace("_", " ").lower()
    cleaned = "".join(char if char not in punctuation else " " for char in lowered)
    return " ".join(cleaned.split())


def grounding_status(answer: str, passages: Iterable[str], threshold: float = 0.8) -> str:
    """Return answered, needs_review, or abstained without another model call."""
    normalized_answer = normalize_text(answer)
    if not normalized_answer:
        return "abstained"
    evidence = normalize_text(" ".join(str(item or "") for item in passages))
    if not evidence:
        return "abstained"
    if normalized_answer in evidence:
        return "answered"
    answer_tokens = [
        token for token in re.findall(r"[a-z0-9]+", normalized_answer)
        if token not in _STOPWORDS
    ]
    if not answer_tokens:
        return "needs_review"
    evidence_tokens = set(re.findall(r"[a-z0-9]+", evidence))
    supported = sum(token in evidence_tokens for token in answer_tokens) / len(answer_tokens)
    return "answered" if supported >= float(threshold) else "needs_review"
