"""Deterministic query construction (no LLM, no translation of its own).

The existing pipeline already detects the language and translates non-English
claims to English before retrieval. This module only cleans text and builds up
to ``max_queries`` search queries: the English normalised claim and, for
non-English claims, the original-language claim as a second query.
"""
from __future__ import annotations

import re
import unicodedata
from typing import List, Optional

from backend.rag.text_utils import STOPWORDS, dedupe_preserve_order

_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_BOILERPLATE = re.compile(
    r"(forwarded\s+as\s+received|forwarded\s+many\s+times|fwd:|please\s+share(\s+(this|with\s+everyone))?|"
    r"share\s+(this\s+)?(message|post)?\s*(with\s+(everyone|all|your\s+friends))?|"
    r"copy\s*(&|and)?\s*paste|received\s+from\s+a\s+friend|urgent\s*[:!]|breaking\s*[:!])",
    re.IGNORECASE,
)
_ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\u2060\ufeff\ufe0f]")


def clean_claim(text: str) -> str:
    """Strip URLs, forwarded-message boilerplate, emoji/symbols and extra whitespace."""
    if not text:
        return ""
    text = _ZERO_WIDTH.sub("", text)
    text = _URL.sub(" ", text)
    text = _BOILERPLATE.sub(" ", text)
    # Drop emoji / pictographs / control characters but keep letters, marks, digits, punctuation.
    text = "".join(ch for ch in text if unicodedata.category(ch) not in ("So", "Sk", "Cc", "Cs", "Co"))
    return re.sub(r"\s+", " ", text).strip()


def _fit_to_length(text: str, max_chars: int) -> str:
    """Shorten over-long input (e.g. whole articles or OCR text) to a usable query."""
    if len(text) <= max_chars:
        return text
    words = text.split()
    content = [w for w in words if w.lower().strip(".,;:!?\"'()") not in STOPWORDS]
    out: List[str] = []
    length = 0
    for word in content:
        if length + len(word) + 1 > max_chars:
            break
        out.append(word)
        length += len(word) + 1
    return " ".join(out) if out else text[:max_chars].rsplit(" ", 1)[0]


def build_queries(
    normalized_claim: str,
    original_claim: Optional[str] = None,
    original_language: str = "en",
    max_queries: int = 2,
    max_chars: int = 200,
) -> List[str]:
    queries: List[str] = []
    primary = _fit_to_length(clean_claim(normalized_claim), max_chars)
    if primary:
        queries.append(primary)
    if original_language and original_language.lower() != "en" and original_claim:
        secondary = _fit_to_length(clean_claim(original_claim), max_chars)
        if secondary:
            queries.append(secondary)
    return dedupe_preserve_order(queries)[: max(1, max_queries)]
