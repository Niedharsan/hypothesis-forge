from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import json

from llm.provider import ask_llm_json
from utils.prompt_loader import load_prompt


@dataclass
class SupervisorConfig:
    goal_summary: str
    objective_type: str
    target_context: str
    constraints: list[str]
    success_criteria: list[str]
    transferability_criteria: list[str]
    generation_guidance: dict[str, Any]
    literature_guidance: dict[str, Any]
    reflection_guidance: dict[str, Any]
    models: dict[str, str]
    generation_config: dict[str, Any]
    literature_config: dict[str, Any]
    reflection_config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SupervisorConfigAgent:
    """LLM-based compact Supervisor configuration.

    The Supervisor interprets the research goal and writes global scientific
    guidance for downstream agents. It does not generate axes, literature
    syntheses, hypotheses, reviews, rankings, or evolved hypotheses.

    There is intentionally no heuristic fallback: if the Supervisor cannot
    produce valid JSON, the run should stop rather than silently continuing with
    weak deterministic assumptions.
    """

    def configure(
        self,
        objective: str,
        *,
        axes: int = 10,
        use_literature: bool = True,
        model: str = "gemini-2.5-flash-lite",
    ) -> SupervisorConfig:
        objective_clean = objective.strip()
        prompt = load_prompt(
            "v31/supervisor_config.md",
            objective=objective_clean,
        )
        payload = ask_llm_json(prompt, model=model, agent="supervisor", purpose="configure_research_goal")

        return SupervisorConfig(
            goal_summary=_string(payload.get("goal_summary"), objective_clean),
            objective_type=_string(payload.get("objective_type"), "general_hypothesis_generation"),
            target_context=_string(payload.get("target_context"), "general"),
            constraints=_string_list(payload.get("constraints")),
            success_criteria=_string_list(payload.get("success_criteria")),
            transferability_criteria=_string_list(payload.get("transferability_criteria")),
            generation_guidance=_dict(payload.get("generation_guidance")),
            literature_guidance=_dict(payload.get("literature_guidance")),
            reflection_guidance=_dict(payload.get("reflection_guidance")),
            models={
                "supervisor": model,
                "generation": model,
                "literature": model,
                "reflection": model,
                "evolution": model,
                "ranking": model,
                "proximity": model,
            },
            generation_config={
                "axis_count": int(axes),
                "hypothesis_count_policy": "minimum 10 requested; more allowed if genuinely distinct and evidence-grounded",
                "use_literature": bool(use_literature),
                "web_search": False,
            },
            literature_config={
                "decomposition_style": "axis_to_subtopics_before_retrieval",
                "uses_new_retrieval": bool(use_literature),
            },
            reflection_config={
                "reflection_mode": "batch_basic",
                "uses_new_retrieval": False,
                "review_dimensions": [
                    "correctness",
                    "novelty_risk",
                    "evidence_support",
                    "mechanistic_specificity",
                    "testability",
                    "explicit_and_implicit_assumptions",
                    "alternative_explanations",
                    "refinement_needs",
                ],
                "allowed_recommendations": ["keep", "revise", "merge", "reject", "needs_more_literature"],
            },
        )


def _string(value: Any, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return [str(value).strip()] if str(value).strip() else []


def _dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    return {"value": value}
