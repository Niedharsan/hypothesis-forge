from __future__ import annotations

import json
from typing import Any

from llm.provider import ask_llm_json
from utils.prompt_loader import render_prompt
from utils.json_compact import compact_json


class ReflectionAgent:
    """Supervisor-guided batch reflection over generated hypotheses.

    Reflection does not create initial hypotheses and does not run new retrieval.
    It critiques generated hypotheses against the Supervisor objective, supplied
    synthesis, and compact run evidence memory. Domain-specific novelty/relevance
    definitions must come from the Supervisor view/config, not from this agent.
    """

    def __init__(self, model: str = "gemini-2.5-flash-lite") -> None:
        self.model = model

    def review_axis_hypotheses(
        self,
        *,
        supervisor_config: dict[str, Any],
        generation_supervisor_view: dict[str, Any],
        axis: dict[str, Any],
        axis_synthesis: dict[str, Any],
        hypotheses_payload: dict[str, Any],
        paper_memory_compact: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = render_prompt(
            "v31/reflection_supervisor_guided_batch.md",
            supervisor_config_json=compact_json(supervisor_config),
            generation_supervisor_view_json=compact_json(generation_supervisor_view),
            axis_json=compact_json(axis),
            axis_synthesis_json=compact_json(axis_synthesis),
            hypotheses_json=compact_json(hypotheses_payload),
            paper_memory_compact_json=compact_json(paper_memory_compact),
        )
        return ask_llm_json(prompt, model=self.model, agent="reflection", purpose="supervisor_guided_axis_hypothesis_reflection")

    def review_global_hypotheses(
        self,
        *,
        supervisor_config: dict[str, Any],
        generation_supervisor_view: dict[str, Any],
        axes_payload: dict[str, Any],
        global_synthesis: dict[str, Any],
        axis_syntheses: list[dict[str, Any]],
        hypotheses_payload: dict[str, Any],
        paper_memory_compact: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = render_prompt(
            "v31/reflection_supervisor_guided_global_batch.md",
            supervisor_config_json=compact_json(supervisor_config),
            generation_supervisor_view_json=compact_json(generation_supervisor_view),
            axes_json=compact_json(axes_payload),
            global_synthesis_json=compact_json(global_synthesis),
            axis_syntheses_json=compact_json(axis_syntheses),
            hypotheses_json=compact_json(hypotheses_payload),
            paper_memory_compact_json=compact_json(paper_memory_compact),
        )
        return ask_llm_json(prompt, model=self.model, agent="reflection", purpose="supervisor_guided_global_hypothesis_reflection")


    def review_global_hypotheses_with_proximity(
        self,
        *,
        supervisor_config: dict[str, Any],
        generation_supervisor_view: dict[str, Any],
        axes_payload: dict[str, Any],
        global_synthesis: dict[str, Any],
        axis_syntheses: list[dict[str, Any]],
        hypotheses_payload: dict[str, Any],
        proximity_payload: dict[str, Any],
        paper_memory_compact: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = render_prompt(
            "v31/reflection_supervisor_guided_global_with_proximity.md",
            supervisor_config_json=compact_json(supervisor_config),
            generation_supervisor_view_json=compact_json(generation_supervisor_view),
            axes_json=compact_json(axes_payload),
            global_synthesis_json=compact_json(global_synthesis),
            axis_syntheses_json=compact_json(axis_syntheses),
            hypotheses_json=compact_json(hypotheses_payload),
            proximity_json=compact_json(proximity_payload),
            paper_memory_compact_json=compact_json(paper_memory_compact),
        )
        return ask_llm_json(prompt, model=self.model, agent="reflection", purpose="supervisor_guided_global_with_proximity")

    # Backwards-compatible method used by older scripts.
    def review_batch(
        self,
        *,
        supervisor_config: dict[str, Any],
        global_synthesis: dict[str, Any],
        axis_syntheses: list[dict[str, Any]],
        hypotheses_payload: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = render_prompt(
            "v31/reflection_batch_review.md",
            supervisor_config=compact_json(supervisor_config),
            global_synthesis_json=compact_json(global_synthesis),
            axis_syntheses_json=compact_json(axis_syntheses),
            hypotheses_json=compact_json(hypotheses_payload),
        )
        return ask_llm_json(prompt, model=self.model, agent="reflection", purpose="legacy_batch_reflection")
