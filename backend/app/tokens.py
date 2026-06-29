"""Token counting used by the chunker (M3).

Uses tiktoken when available for an accurate count; otherwise falls back to a
deterministic word-based estimate. The chunker only needs a stable, monotonic
measure to enforce the 100–600 token bounds, so either is acceptable.
"""

from __future__ import annotations

import re

try:  # pragma: no cover - import guard
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover
    _ENC = None

_WORD_RE = re.compile(r"\w+|[^\w\s]")


def count_tokens(text: str) -> int:
    if not text:
        return 0
    if _ENC is not None:
        return len(_ENC.encode(text))
    # Fallback: count word-ish tokens. Close enough for boundary control.
    return len(_WORD_RE.findall(text))
