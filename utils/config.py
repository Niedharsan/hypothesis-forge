from __future__ import annotations
import os
from pathlib import Path
from typing import Any
import yaml

class ConfigValidationError(RuntimeError): pass

def load_config(path: str = 'configs/config.yaml') -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding='utf-8')) or {}

def validate_config(config: dict[str, Any], *, runtime_mode: str | None = None, strict: bool = False) -> list[str]:
    warnings: list[str] = []
    mode = str(runtime_mode or ((config.get('runtime') or {}).get('mode')) or 'normal').lower()
    if mode not in {'normal','dry_run'}:
        warnings.append(f'invalid runtime mode: {mode}')
    limit = ((config.get('runtime') or {}).get('limits') or {}).get('max_llm_calls_per_run', 8)
    try:
        if int(limit) < 1: warnings.append('runtime.limits.max_llm_calls_per_run must be >= 1')
    except Exception:
        warnings.append('runtime.limits.max_llm_calls_per_run must be an integer')
    if mode != 'dry_run' and not os.getenv('GEMINI_API_KEY'):
        warnings.append('GEMINI_API_KEY is not set')
    if warnings and strict:
        raise ConfigValidationError('\n'.join(warnings))
    return warnings
