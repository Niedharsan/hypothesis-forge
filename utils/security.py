from __future__ import annotations

import os
import re
from typing import Any

_SECRET_ENV_NAMES = (
    "GEMINI_API_KEY",
    "NCBI_API_KEY",
    "OPENALEX_API_KEY",
    "SEMANTIC_SCHOLAR_API_KEY",
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(api[_-]?key|access[_-]?token|auth(?:orization)?|token|key)=([^&\s]+)"
)
_BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")


def redact_sensitive_text(value: Any) -> str:
    """Return log/UI-safe text with common credential forms removed."""
    text = str(value or "")
    for env_name in _SECRET_ENV_NAMES:
        secret = os.getenv(env_name)
        if secret and len(secret) >= 4:
            text = text.replace(secret, "[REDACTED]")
    text = _SECRET_ASSIGNMENT.sub(lambda match: f"{match.group(1)}=[REDACTED]", text)
    text = _BEARER.sub("Bearer [REDACTED]", text)
    return text
