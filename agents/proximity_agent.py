from __future__ import annotations

from typing import Any

from llm.provider import ask_llm_json
from utils.prompt_loader import render_prompt
from utils.json_compact import compact_json


class ProximityAgent:
    """Analyze hypothesis relatedness and redundancy before reflection.

    Proximity is a no-retrieval agent: it should not call LiteratureAgent or
    EvidenceSelector. It detects duplicate/near-duplicate hypotheses, groups
    related-but-distinct hypotheses, and emits focused seeds only for useful
    absorbed/lost material.
    """

    def __init__(self, model: str = "gemini-2.5-flash-lite") -> None:
        self.model = model

    def cluster_merge_salvage(
        self,
        *,
        supervisor_config: dict[str, Any],
        generation_supervisor_view: dict[str, Any],
        hypotheses_payload: dict[str, Any],
        paper_memory_compact: dict[str, Any] | None = None,
        max_focus_seeds: int = 8,
    ) -> dict[str, Any]:
        prompt = render_prompt(
            "v31/proximity_cluster_merge_salvage.md",
            supervisor_config_json=compact_json(supervisor_config),
            generation_supervisor_view_json=compact_json(generation_supervisor_view),
            hypotheses_json=compact_json(hypotheses_payload),
            paper_memory_compact_json=compact_json(paper_memory_compact or {}),
            max_focus_seeds=max_focus_seeds,
        )
        return ask_llm_json(
            prompt,
            model=self.model,
            agent="proximity",
            purpose="relatedness_redundancy_focus_seed",
        )


def _raw_hypothesis_index(raw_hypotheses_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    hyps = raw_hypotheses_payload.get("hypotheses", []) if isinstance(raw_hypotheses_payload, dict) else []
    out: dict[str, dict[str, Any]] = {}
    for h in hyps:
        if isinstance(h, dict) and h.get("hypothesis_id"):
            out[str(h["hypothesis_id"])] = h
    return out


def _coerce_survivor_from_raw(raw: dict[str, Any], proximity_payload: dict[str, Any]) -> dict[str, Any]:
    hyp_id = str(raw.get("hypothesis_id", "")).strip()
    group_id = None
    retained_components: list[Any] = []
    for group in proximity_payload.get("proximity_groups", []) if isinstance(proximity_payload, dict) else []:
        if not isinstance(group, dict):
            continue
        ids = group.get("hypothesis_ids", []) or []
        reps = group.get("canonical_representative_ids", []) or []
        if hyp_id in ids or hyp_id in reps:
            group_id = group.get("group_id")
            break
    return {
        "hypothesis_id": hyp_id,
        "title": raw.get("title", ""),
        "hypothesis": raw.get("hypothesis", ""),
        "candidate_intervention_or_focus": raw.get("candidate_intervention_or_focus", ""),
        "mechanistic_rationale": raw.get("mechanistic_rationale", raw.get("hypothesis", "")),
        "source_hypothesis_ids": [hyp_id],
        "retained_components": retained_components,
        "proximity_group_id": group_id,
        "survivor_type": "original",
        "proximity_metadata": {"source": "survivor_hypothesis_ids"},
    }


def build_hypotheses_payload_from_proximity(
    raw_hypotheses_payload: dict[str, Any],
    proximity_payload: dict[str, Any],
) -> dict[str, Any]:
    """Return a hypotheses payload suitable for Reflection after Proximity.

    Prefer survivor_hypothesis_ids so Reflection receives original specific
    hypotheses rather than broad rewritten summaries. If ids are unavailable,
    fall back to survivor_hypotheses. If that also fails, return the raw input.
    """
    raw_index = _raw_hypothesis_index(raw_hypotheses_payload)

    survivor_ids = proximity_payload.get("survivor_hypothesis_ids") if isinstance(proximity_payload, dict) else None
    if isinstance(survivor_ids, list) and survivor_ids:
        coerced = []
        seen: set[str] = set()
        for sid in survivor_ids:
            sid_s = str(sid).strip()
            if not sid_s or sid_s in seen:
                continue
            seen.add(sid_s)
            raw = raw_index.get(sid_s)
            if raw:
                coerced.append(_coerce_survivor_from_raw(raw, proximity_payload))
        if coerced:
            return {
                "hypotheses": coerced,
                "source_payload": "proximity_survivor_ids",
                "raw_input_hypothesis_count": len(raw_index),
                "survivor_count": len(coerced),
            }

    survivors = proximity_payload.get("survivor_hypotheses") if isinstance(proximity_payload, dict) else None
    if not isinstance(survivors, list) or not survivors:
        return raw_hypotheses_payload

    coerced: list[dict[str, Any]] = []
    for idx, item in enumerate(survivors, start=1):
        if not isinstance(item, dict):
            continue
        hyp_id = str(item.get("hypothesis_id") or f"P{idx:02d}").strip()
        # If the survivor points to an original hypothesis, preserve the raw text.
        if hyp_id in raw_index:
            coerced.append(_coerce_survivor_from_raw(raw_index[hyp_id], proximity_payload))
            continue
        title = str(item.get("title") or f"Proximity survivor {idx}").strip()
        hypothesis = str(item.get("hypothesis") or item.get("merged_hypothesis") or "").strip()
        if not hypothesis:
            continue
        coerced.append({
            "hypothesis_id": hyp_id,
            "title": title,
            "hypothesis": hypothesis,
            "candidate_intervention_or_focus": item.get("candidate_intervention_or_focus") or item.get("candidate_drug_or_class") or "",
            "mechanistic_rationale": item.get("mechanistic_rationale") or item.get("rationale") or hypothesis,
            "source_hypothesis_ids": item.get("source_hypothesis_ids", []),
            "retained_components": item.get("retained_components", []),
            "proximity_group_id": item.get("proximity_group_id") or item.get("proximity_cluster_id") or hyp_id,
            "survivor_type": item.get("survivor_type") or "minimal_merge",
            "proximity_metadata": item,
        })

    if not coerced:
        return raw_hypotheses_payload

    return {
        "hypotheses": coerced,
        "source_payload": "proximity_survivors",
        "raw_input_hypothesis_count": len(raw_index),
        "survivor_count": len(coerced),
    }
