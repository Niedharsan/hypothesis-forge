from __future__ import annotations

import inspect
import json
try:
    from json_repair import repair_json
except Exception:  # dependency should be installed from requirements; keep imports robust for dry-run/static checks
    def repair_json(text, return_objects=False):
        if return_objects:
            return json.loads(text)
        return text
import os
import re
import time
from pathlib import Path
from typing import Any

from utils.run_logger import log_gemini_call
from utils.security import redact_sensitive_text

import yaml
from dotenv import load_dotenv

load_dotenv()


def gemini_available() -> bool:
    if not os.getenv("GEMINI_API_KEY"):
        return False
    try:
        from google import genai  # noqa: F401
    except Exception:
        return False
    return True


def ask_gemini_json(prompt: str, model: str | None = None, *, agent: str | None = None, purpose: str | None = None, temperature: float | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("Gemini API key is not configured.")
    try:
        from google import genai
    except Exception as exc:
        raise RuntimeError("google-genai is not installed.") from exc

    selected_model = model or _default_model()
    caller = f"{agent}.{purpose}" if agent and purpose else (agent or purpose or _caller_name())
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    started = time.perf_counter()
    max_attempts = _retry_attempts()
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            gen_config = {"response_mime_type": "application/json"}
            if temperature is not None:
                gen_config["temperature"] = float(temperature)
            max_output_tokens = os.getenv("GEMINI_MAX_OUTPUT_TOKENS")
            if max_output_tokens:
                try:
                    gen_config["max_output_tokens"] = int(max_output_tokens)
                except ValueError:
                    pass
            response = client.models.generate_content(
                model=selected_model,
                contents=prompt,
                config=gen_config,
            )
            break
        except Exception as exc:
            last_exc = exc
            retryable = _is_retryable_gemini_error(exc)
            is_final = attempt >= max_attempts or not retryable
            log_gemini_call(
                caller=caller,
                model=selected_model,
                prompt=prompt,
                duration_s=time.perf_counter() - started,
                error=redact_sensitive_text(exc),
                metadata={**(metadata or {}), "attempt": attempt, "max_attempts": max_attempts, "retryable": retryable, "will_retry": not is_final},
            )
            if is_final:
                raise
            time.sleep(_retry_delay(attempt))
    else:
        raise last_exc or RuntimeError("Gemini request failed before response was created.")
    text = getattr(response, "text", None) or ""
    log_gemini_call(
        caller=caller,
        model=selected_model,
        prompt=prompt,
        response_text=text,
        usage=_usage_dict(getattr(response, "usage_metadata", None)),
        duration_s=time.perf_counter() - started,
        metadata=metadata or {},
    )
    return _extract_json(text)



def _retry_attempts() -> int:
    try:
        return max(1, int(os.getenv("GEMINI_RETRY_ATTEMPTS", "1")))
    except Exception:
        return 1


def _retry_delay(attempt: int) -> float:
    # Exponential backoff with a small deterministic jitter-free delay by default.
    raw = os.getenv("GEMINI_RETRY_DELAYS", "4")
    try:
        parts = [float(x.strip()) for x in raw.split(",") if x.strip()]
        if attempt - 1 < len(parts):
            return parts[attempt - 1]
    except Exception:
        pass
    return min(60.0, 8.0 * (2 ** max(0, attempt - 1)))


def _is_retryable_gemini_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(term in msg for term in [
        "503", "unavailable", "overloaded", "high demand",
        "429", "rate limit", "quota",
        "500", "502", "504", "deadline", "timeout",
    ])

def _caller_name() -> str:
    for frame in inspect.stack()[2:8]:
        module = inspect.getmodule(frame.frame)
        mod_name = module.__name__ if module else Path(frame.filename).stem
        if mod_name != __name__:
            return f"{mod_name}.{frame.function}"
    return "unknown"


def _usage_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    out: dict[str, Any] = {}
    for key in [
        "prompt_token_count",
        "candidates_token_count",
        "total_token_count",
        "cached_content_token_count",
        "thoughts_token_count",
    ]:
        value = getattr(usage, key, None)
        if value is not None:
            out[key] = value
    return out


def _default_model(config_path: str | Path = "configs/config.yaml") -> str:
    path = Path(config_path)
    if not path.exists():
        return "gemini-2.5-flash-lite"
    try:
        with path.open(encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except Exception:
        return "gemini-2.5-flash-lite"
    return config.get("llm", {}).get("cheap_model", "gemini-2.5-flash-lite")



def _repair_invalid_json_escapes(text: str) -> str:
    """Escape invalid backslashes inside JSON strings, e.g. \\alpha -> \\\\alpha."""
    out: list[str] = []
    in_string = False
    escaped = False
    valid = set('"\\/bfnrtu')
    for ch in text:
        if escaped:
            if ch not in valid:
                out.append('\\')
            out.append(ch)
            escaped = False
            continue
        if ch == '\\' and in_string:
            escaped = True
            continue
        if ch == '"':
            # This is a best-effort parser; generated JSON is simple enough for this repair.
            in_string = not in_string
        out.append(ch)
    if escaped:
        out.append('\\')
    return ''.join(out)



def _extract_json(text: str):
    stripped = text.strip()

    # Remove markdown fences if present
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()

    candidates = [stripped]

    # Add object substring candidate
    first_obj = stripped.find("{")
    last_obj = stripped.rfind("}")
    if first_obj != -1 and last_obj != -1 and last_obj > first_obj:
        candidates.append(stripped[first_obj:last_obj + 1])

    # Add array substring candidate
    first_arr = stripped.find("[")
    last_arr = stripped.rfind("]")
    if first_arr != -1 and last_arr != -1 and last_arr > first_arr:
        candidates.append(stripped[first_arr:last_arr + 1])

    last_error = None
    for c in candidates:
        try:
            return json.loads(c)
        except Exception as e:
            last_error = e

        try:
            return repair_json(c, return_objects=True)
        except Exception as e:
            last_error = e

    raise last_error
