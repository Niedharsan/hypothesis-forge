from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from llm.provider import ask_gemini_json
from utils.prompt_loader import render_prompt
from utils.json_compact import compact_json
from schemas.hypothesis_strategy import HypothesisStrategy
from schemas.evidence_packet import EvidencePacket
from agents.multi_source_literature_agent import MultiSourceLiteratureAgent, RouteLiteratureResult


@dataclass
class GenerationRunOutput:
    axes: dict[str, Any] = field(default_factory=dict)
    routes: dict[str, Any] = field(default_factory=dict)
    literature_results: list[RouteLiteratureResult] = field(default_factory=list)
    hypotheses_payload: dict[str, Any] = field(default_factory=dict)
    strategies: list[HypothesisStrategy] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class RewiredGenerationAgent:
    """Generation does discovery: axes -> route decomposition -> hypotheses.

    Supervisor only configures. Literature is optional and route-specific.
    """

    def __init__(self, model: str = "gemini-2.5-flash-lite") -> None:
        self.model = model

    def run(
        self,
        *,
        objective: str,
        supervisor_config: dict[str, Any],
        sources: list[str],
        config_path: str = "configs/config.yaml",
        enable_literature: bool = False,
        max_queries_per_axis: int = 2,
        papers_per_axis: int = 5,
    ) -> GenerationRunOutput:
        axes = self.generate_axes(objective, supervisor_config)
        routes = self.decompose_axes(objective, axes)
        literature_results: list[RouteLiteratureResult] = []
        evidence_packets: list[EvidencePacket] = []
        if enable_literature:
            lit = MultiSourceLiteratureAgent(config_path=config_path)
            for route in routes.get("routes", []):
                if not isinstance(route, dict):
                    continue
                axis_id = str(route.get("axis_id") or "").strip()
                queries = route.get("search_queries") or []
                result = lit.investigate_route(
                    axis_id=axis_id,
                    objective=objective,
                    queries=[str(q) for q in queries if str(q).strip()],
                    sources=sources,
                    max_queries=max_queries_per_axis,
                    papers_per_axis=papers_per_axis,
                )
                literature_results.append(result)
                evidence_packets.extend(result.evidence_packets)
        literature_context = self._format_literature_context(evidence_packets) if evidence_packets else "No literature retrieval was enabled for this run."
        hypotheses_payload = self.generate_hypotheses(objective, supervisor_config, routes, literature_context)
        strategies = self._coerce_hypotheses(hypotheses_payload, objective, evidence_packets)
        return GenerationRunOutput(
            axes=axes,
            routes=routes,
            literature_results=literature_results,
            hypotheses_payload=hypotheses_payload,
            strategies=strategies,
        )

    def generate_axes(self, objective: str, supervisor_config: dict[str, Any]) -> dict[str, Any]:
        prompt = render_prompt(
            "v31/generation_axes.md",
            objective=objective,
            supervisor_config=compact_json(supervisor_config),
        )
        return ask_gemini_json(prompt, model=self.model, agent="generation", purpose="generate_axes")

    def decompose_axes(self, objective: str, axes: dict[str, Any]) -> dict[str, Any]:
        prompt = render_prompt(
            "v31/generation_decompose.md",
            objective=objective,
            axes_json=compact_json(axes),
        )
        return ask_gemini_json(prompt, model=self.model, agent="generation", purpose="decompose_axes")

    def generate_hypotheses(
        self,
        objective: str,
        supervisor_config: dict[str, Any],
        routes: dict[str, Any],
        literature_context: str,
    ) -> dict[str, Any]:
        prompt = render_prompt(
            "v31/generation_hypotheses.md",
            objective=objective,
            supervisor_config=compact_json(supervisor_config),
            routes_json=compact_json(routes),
            literature_context=literature_context,
        )
        return ask_gemini_json(prompt, model=self.model, agent="generation", purpose="generate_hypotheses")


    def generate_hypotheses_from_literature(
        self,
        objective: str,
        supervisor_config: dict[str, Any],
        literature_syntheses: list[dict[str, Any]],
        global_synthesis: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        prompt = render_prompt(
            "v31/generation_hypotheses_from_literature.md",
            objective=objective,
            supervisor_config=compact_json(supervisor_config),
            literature_syntheses_json=compact_json(literature_syntheses),
            global_synthesis_json=compact_json(global_synthesis or {}),
        )
        return ask_gemini_json(prompt, model=self.model, agent="generation", purpose="hypotheses_from_literature")


    def generate_hypotheses_from_axis_literature(
        self,
        objective: str,
        supervisor_config: dict[str, Any],
        axes_payload: dict[str, Any],
        literature_syntheses: list[dict[str, Any]],
        global_synthesis: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        prompt = render_prompt(
            "v31/generation_hypotheses_from_axis_literature.md",
            objective=objective,
            supervisor_config=compact_json(supervisor_config),
            axes_json=compact_json(axes_payload),
            literature_syntheses_json=compact_json(literature_syntheses),
            global_synthesis_json=compact_json(global_synthesis or {}),
        )
        return ask_gemini_json(prompt, model=self.model, agent="generation", purpose="hypotheses_from_axis_literature")

    @staticmethod
    def _format_literature_context(packets: list[EvidencePacket], limit: int = 40) -> str:
        lines = []
        for packet in packets[:limit]:
            lines.append(
                f"[{packet.evidence_id}] axis={packet.metadata.get('axis_id','')} title={packet.title}\n"
                f"paper_id={packet.paper_id}\n{packet.text[:1200]}"
            )
        return "\n\n".join(lines)

    def _coerce_hypotheses(
        self,
        payload: dict[str, Any],
        objective: str,
        evidence_packets: list[EvidencePacket],
    ) -> list[HypothesisStrategy]:
        items = payload.get("hypotheses", [])
        if not isinstance(items, list):
            return []
        fallback_ids = [p.evidence_id for p in evidence_packets[:4]]
        strategies: list[HypothesisStrategy] = []
        for idx, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or f"Generated hypothesis {idx}").strip()
            hyp = str(item.get("hypothesis") or item.get("core_hypothesis") or "").strip()
            candidate = str(
                item.get("candidate_drug_or_class")
                or item.get("candidate_intervention_or_focus")
                or item.get("hypothesis")
                or ""
            ).strip()
            rationale = str(item.get("rationale") or item.get("mechanistic_rationale") or hyp).strip()
            if not hyp and not candidate:
                continue

            evidence_grounding = item.get("evidence_grounding") if isinstance(item.get("evidence_grounding"), dict) else {}
            supporting_ids = evidence_grounding.get("supporting_evidence_ids") if isinstance(evidence_grounding, dict) else None
            if not isinstance(supporting_ids, list) or not supporting_ids:
                supporting_ids = fallback_ids

            novelty_or_gap = item.get("novelty_or_gap") if isinstance(item.get("novelty_or_gap"), dict) else {}
            novelty_rationale = (
                novelty_or_gap.get("proposed_new_connection")
                or novelty_or_gap.get("novelty_risk")
                or item.get("novelty_risk")
                or "unknown"
            )

            validation = item.get("validation_approach")
            if isinstance(validation, dict):
                first_experiment = str(validation.get("approach") or validation.get("experiment") or validation)
            else:
                first_experiment = str(validation or item.get("first_validation_step") or "Run a decisive validation assay.")

            predictions = item.get("testable_predictions") or item.get("falsifiable_predictions") or []
            pred_strings: list[str] = []
            if isinstance(predictions, list):
                for pred in predictions:
                    if isinstance(pred, dict):
                        p_text = str(pred.get("prediction") or pred.get("expected_result") or "").strip()
                        d_text = str(pred.get("disconfirming_result") or "").strip()
                        if p_text and d_text:
                            pred_strings.append(f"{p_text} Disconfirmed by: {d_text}")
                        elif p_text:
                            pred_strings.append(p_text)
                    else:
                        pred_strings.append(str(pred))

            risks = []
            for key in ("assumptions", "uncertainties", "must_verify_later"):
                values = item.get(key, [])
                if isinstance(values, list):
                    risks.extend(str(x) for x in values if str(x).strip())

            strategies.append(HypothesisStrategy(
                strategy_id=f"S{idx:03d}",
                title=title,
                strategy_type="generation_rewired",
                proposed_intervention=candidate or hyp,
                target_or_pathway=str(item.get("axis_id") or ",".join(item.get("source_subtopic_ids", [])) or "literature-linked hypothesis"),
                mechanism=rationale,
                why_selective_for_pathogen_or_disease_state=f"Generated for objective: {objective}",
                why_host_sparing="Requires Reflection/literature verification.",
                supporting_evidence_ids=[str(x) for x in supporting_ids],
                prior_art_overlap="unknown; ReflectionAgent must verify.",
                novelty_rationale=str(novelty_rationale),
                risks=risks or ["Requires evidence and novelty verification."],
                falsification_test="; ".join(pred_strings) or "Fails if predicted route effect is absent.",
                first_experiment=first_experiment,
                family="axis_linked_generation",
                generation_round=0,
                status="candidate",
                metadata={
                    "axis_id": str(item.get("axis_id") or ""),
                    "source_subtopic_ids": item.get("source_subtopic_ids", []),
                    "assumptions": item.get("assumptions", []),
                    "uncertainties": item.get("uncertainties", []),
                    "raw_generation_item": item,
                },
            ))
        return strategies
