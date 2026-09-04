from __future__ import annotations

from typing import Any

import app.orchestrator as base
from agents.evolution_agent import EvolutionAgent


def run_evolution(
    run: dict[str, Any],
    parents: list[dict[str, Any]],
    output_count: int,
    guidance: str,
):
    """Evolution stage with parent/review lineage kept as one paired record."""
    base._prepare_stage(run, "evolution")
    supervisor = base._artifact_data(run, "01_supervisor_config.json") or {}
    proximity = base._artifact_data(run, "08_proximity_clusters.json") or {}
    compact = base._artifact_data(run, "07b_paper_memory_compact.json") or {}

    pairs: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for parent in parents:
        payload = parent.get("payload", {}) if isinstance(parent.get("payload"), dict) else {}
        hypothesis = payload.get("hypothesis")
        if not isinstance(hypothesis, dict) or not hypothesis:
            continue
        review = payload.get("review") if isinstance(payload.get("review"), dict) else {}
        pairs.append((parent, hypothesis, review))

    hypotheses = [hypothesis for _, hypothesis, _ in pairs]
    reviews = [review for _, _, review in pairs]
    focused = {}
    if run.get("enable_evolution_retrieval") and hypotheses:
        focused, focused_compact = base._focused_evolution_retrieval(run, hypotheses, guidance)
        compact = {"base_memory": compact, "evolution_focused_retrieval_memory": focused_compact}

    evo = EvolutionAgent(model=run["model"])
    cards: list[dict[str, Any]] = []
    for i, (parent, hypothesis, review) in enumerate(pairs[:output_count], 1):
        result = evo.feasibility(
            goal=run["objective"],
            preferences=supervisor,
            hypothesis=hypothesis,
            review=review,
            proximity_context=proximity,
            paper_memory_compact=compact,
            human_stage_guidance=guidance,
        )
        hypothesis_id = str(hypothesis.get("hypothesis_id") or f"H{i:03d}")
        cards.append(
            base._card(
                run,
                "evolution",
                f"EVO-{hypothesis_id}",
                f"Evolved: {hypothesis.get('title') or hypothesis_id}",
                base._summary_text(result),
                {"source_hypothesis": hypothesis, "source_review": review, "evolution": result},
                [parent["id"]],
            )
        )

    if len(hypotheses) > 1 and len(cards) < output_count:
        result = evo.out_of_box(
            goal=run["objective"],
            preferences=supervisor,
            hypotheses=hypotheses,
            reviews=reviews,
            proximity_context=proximity,
            paper_memory_compact=compact,
            human_stage_guidance=guidance,
        )
        cards.append(
            base._card(
                run,
                "evolution",
                "EVO-OUT-OF-BOX",
                "Out-of-box evolution",
                base._summary_text(result),
                {"evolution": result, "source_hypotheses": hypotheses},
                [parent["id"] for parent, _, _ in pairs],
            )
        )

    artifacts = {"10_evolution_outputs.json": [card["payload"] for card in cards]}
    if focused:
        artifacts["09b_evolution_focused_retrieval.json"] = focused
    return cards, artifacts, base._finish_usage("evolution")
