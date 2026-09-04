from __future__ import annotations

import json
from typing import Any


def compact_json(value: Any) -> str:
    """Return token-compact JSON for LLM prompts.

    Keep run artifacts pretty elsewhere; this helper is intentionally for prompt
    payloads where whitespace costs tokens but content must be unchanged.
    """
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def stable_json(value: Any) -> str:
    """Return deterministic compact JSON for cache keys/hashes."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
