"""Shared heuristic for labeling and triggering on refusal text."""

from __future__ import annotations

import re

REFUSAL = re.compile(
    r"\b(?:i(?:'m| am) sorry|i (?:cannot|can't|am unable|won't)|"
    r"i (?:can't|cannot) help|as an ai|unable to assist)\b",
    flags=re.IGNORECASE,
)


def is_refusal(text: str) -> bool:
    return bool(REFUSAL.search(text[:300]))
