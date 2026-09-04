from __future__ import annotations

from typing import Any

from llm.provider import ask_llm_json
from utils.prompt_loader import render_prompt
from utils.json_compact import compact_json


class EvolutionAgent:
    """Repo-style hypothesis evolution agent.

    This agent follows the simple evolution modes used by public co-scientist
    reimplementations: simplify, feasibility improve, combine, and out-of-box
    reimagine. It does not run retrieval itself. Stage-specific human guidance
    and any optional retrieval context are supplied by the orchestrating script.
    """

    def __init__(self, model: str = "gemini-2.5-flash-lite") -> None:
        self.model = model

    def simplify(
        self,
        *,
        goal: str,
        preferences: dict[str, Any] | str,
        hypothesis: dict[str, Any],
        review: dict[str, Any] | None = None,
        proximity_context: dict[str, Any] | None = None,
        paper_memory_compact: dict[str, Any] | None = None,
        human_stage_guidance: str = "",
    ) -> dict[str, Any]:
        prompt = render_prompt(
            "v31/evolution_simplify.md",
            goal=goal,
            preferences=compact_json(preferences),
            hypothesis_json=compact_json(hypothesis),
            review_json=compact_json(review or {}),
            proximity_json=compact_json(proximity_context or {}),
            paper_memory_compact_json=compact_json(paper_memory_compact or {}),
            human_stage_guidance=human_stage_guidance or "",
        )
        return ask_llm_json(prompt, model=self.model, agent="evolution", purpose="simplify")

    def feasibility(
        self,
        *,
        goal: str,
        preferences: dict[str, Any] | str,
        hypothesis: dict[str, Any],
        review: dict[str, Any] | None = None,
        proximity_context: dict[str, Any] | None = None,
        paper_memory_compact: dict[str, Any] | None = None,
        human_stage_guidance: str = "",
    ) -> dict[str, Any]:
        prompt = render_prompt(
            "v31/evolution_feasibility.md",
            goal=goal,
            preferences=compact_json(preferences),
            hypothesis_json=compact_json(hypothesis),
            review_json=compact_json(review or {}),
            proximity_json=compact_json(proximity_context or {}),
            paper_memory_compact_json=compact_json(paper_memory_compact or {}),
            human_stage_guidance=human_stage_guidance or "",
        )
        return ask_llm_json(prompt, model=self.model, agent="evolution", purpose="feasibility")

    def combine(
        self,
        *,
        goal: str,
        preferences: dict[str, Any] | str,
        hypothesis_a: dict[str, Any],
        hypothesis_b: dict[str, Any],
        review_a: dict[str, Any] | None = None,
        review_b: dict[str, Any] | None = None,
        proximity_context: dict[str, Any] | None = None,
        paper_memory_compact: dict[str, Any] | None = None,
        human_stage_guidance: str = "",
    ) -> dict[str, Any]:
        prompt = render_prompt(
            "v31/evolution_combine.md",
            goal=goal,
            preferences=compact_json(preferences),
            hypothesis_a_json=compact_json(hypothesis_a),
            hypothesis_b_json=compact_json(hypothesis_b),
            review_a_json=compact_json(review_a or {}),
            review_b_json=compact_json(review_b or {}),
            proximity_json=compact_json(proximity_context or {}),
            paper_memory_compact_json=compact_json(paper_memory_compact or {}),
            human_stage_guidance=human_stage_guidance or "",
        )
        return ask_llm_json(prompt, model=self.model, agent="evolution", purpose="combine")

    def out_of_box(
        self,
        *,
        goal: str,
        preferences: dict[str, Any] | str,
        hypotheses: list[dict[str, Any]],
        reviews: list[dict[str, Any]] | None = None,
        proximity_context: dict[str, Any] | None = None,
        paper_memory_compact: dict[str, Any] | None = None,
        human_stage_guidance: str = "",
    ) -> dict[str, Any]:
        prompt = render_prompt(
            "v31/evolution_out_of_box.md",
            goal=goal,
            preferences=compact_json(preferences),
            hypotheses_json=compact_json(hypotheses),
            reviews_json=compact_json(reviews or []),
            proximity_json=compact_json(proximity_context or {}),
            paper_memory_compact_json=compact_json(paper_memory_compact or {}),
            human_stage_guidance=human_stage_guidance or "",
        )
        return ask_llm_json(prompt, model=self.model, agent="evolution", purpose="out_of_box")
