from __future__ import annotations

from typing import Any


def build_generation_supervisor_view(supervisor_config: dict[str, Any], *, temporary_constraints: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return only the Supervisor fields needed by GenerationAgent.

    Literature/reflection-specific guidance is intentionally omitted. Temporary
    benchmark constraints are separated from Supervisor guidance so they can be
    deleted later without changing the Supervisor contract.
    """
    view = _compact({
        "goal_summary": supervisor_config.get("goal_summary"),
        "objective_type": supervisor_config.get("objective_type"),
        "target_context": supervisor_config.get("target_context"),
        "constraints": supervisor_config.get("constraints", []),
        "success_criteria": supervisor_config.get("success_criteria", []),
        "transferability_criteria": supervisor_config.get("transferability_criteria", []),
        "generation_guidance": supervisor_config.get("generation_guidance", {}),
    })
    if temporary_constraints:
        view["temporary_constraints"] = _compact(temporary_constraints)
    return view


def _compact(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _compact(v) for k, v in obj.items() if _keep(v)}
    if isinstance(obj, list):
        return [_compact(v) for v in obj if _keep(v)]
    return obj


def _keep(value: Any) -> bool:
    if value is None:
        return False
    if value == "":
        return False
    if value == []:
        return False
    if value == {}:
        return False
    return True
