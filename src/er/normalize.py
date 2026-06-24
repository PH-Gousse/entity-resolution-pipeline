"""Text normalization for entity resolution."""

from __future__ import annotations

import re

# Legal and stop tokens to remove (common in software product names).
_STOP_TOKENS = frozenset({
    "inc", "ltd", "corp", "llc", "the", "a", "an", "and", "or", "of", "for",
    "by", "to", "in", "on", "at", "is", "it", "co", "corporation",
})

# Pattern: one or more non-alphanumeric, non-space characters.
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)

# Pattern: digit-letter or letter-digit boundary (e.g., "v5" -> "v 5").
_DIGIT_SPLIT_RE = re.compile(r"(?<=\d)(?=[a-zA-Z])|(?<=[a-zA-Z])(?=\d)")

# Pattern: two or more whitespace characters.
_MULTI_SPACE_RE = re.compile(r"\s{2,}")


def normalize(text: str | None) -> str:
    """Normalize a product name for entity resolution.

    Steps:
      1. Lowercase
      2. Strip punctuation
      3. Split digit/letter boundaries (e.g., "v5" -> "v 5")
      4. Remove stop/legal tokens
      5. Collapse whitespace and strip
    """
    if not text:
        return ""

    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _DIGIT_SPLIT_RE.sub(" ", text)

    tokens = text.split()
    tokens = [t for t in tokens if t not in _STOP_TOKENS]

    text = " ".join(tokens)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()
