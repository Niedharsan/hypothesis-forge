from __future__ import annotations

import hashlib
from typing import Any

from llm.gemini_client import ask_gemini_json as _ask_real_gemini_json
from llm.gemini_client import gemini_available as _real_gemini_available
from llm.mock_client import ask_mock_json, mock_llm_available
from runtime.context import current_runtime, increment_llm_call_count
from utils.run_logger import log_event


_LOW_TEMPERATURE_STAGES = {
    ("evidence_selector", "subtopic_evidence_selection"),
    ("evidence_selector", "axis_batch_subtopic_evidence_selection"),
    ("query_reviewer", "axis_query_family_review"),
}


def gemini_available() -> bool:
    """Compatibility name: true in dry-run so existing agents can continue."""
    runtime = current_runtime()
    if runtime.is_dry_run and runtime.dry_run.mock_llm_outputs:
        return mock_llm_available()
    return _real_gemini_available()


def llm_available() -> bool:
    return gemini_available()


def ask_gemini_json(
    prompt: str,
    model: str | None = None,
    *,
    agent: str | None = None,
    purpose: str | None = None,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper for all JSON LLM calls.

    The provider enforces the global call limit and preserves agent/purpose
    labels for cost logging. Temperature is intentionally opt-in/stage-specific:
    deterministic selector/reviewer calls can use low temperature, but novelty
    generation/reflection are not globally forced to temperature=0.
    """
    runtime = current_runtime()
    call_count = increment_llm_call_count()
    if call_count > runtime.limits.max_llm_calls_per_run:
        log_event(
            "runtime",
            "llm_call_limit_exceeded",
            {
                "limit": runtime.limits.max_llm_calls_per_run,
                "mode": runtime.mode,
                "attempted_call_count": call_count,
                "agent": agent,
                "purpose": purpose,
            },
            status="error",
        )
        raise RuntimeError(f"LLM call limit exceeded: {runtime.limits.max_llm_calls_per_run}")

    resolved_temperature = _stage_temperature(agent=agent, purpose=purpose, explicit=temperature)
    metadata = {
        "agent": agent,
        "purpose": purpose,
        "stage": _stage_label(agent, purpose),
        "llm_call_index": call_count,
        "prompt_sha256": hashlib.sha256((prompt or "").encode("utf-8")).hexdigest(),
        "temperature": resolved_temperature,
        "runtime_mode": runtime.mode,
    }

    if runtime.is_dry_run and runtime.dry_run.mock_llm_outputs:
        return ask_mock_json(
            prompt,
            model=model,
            agent=agent,
            purpose=purpose,
            temperature=resolved_temperature,
            metadata=metadata,
        )

    return _ask_real_gemini_json(
        prompt,
        model=model,
        agent=agent,
        purpose=purpose,
        temperature=resolved_temperature,
        metadata=metadata,
    )


def ask_llm_json(
    prompt: str,
    model: str | None = None,
    *,
    agent: str | None = None,
    purpose: str | None = None,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Provider-neutral alias for future multi-provider routing."""
    return ask_gemini_json(prompt, model=model, agent=agent, purpose=purpose, temperature=temperature)


def _stage_temperature(*, agent: str | None, purpose: str | None, explicit: float | None) -> float | None:
    if explicit is not None:
        return explicit
    key = (agent or "", purpose or "")
    if key in _LOW_TEMPERATURE_STAGES:
        return 0.0
    return None


def _stage_label(agent: str | None, purpose: str | None) -> str:
    if agent and purpose:
        return f"{agent}.{purpose}"
    return agent or purpose or "unknown"
