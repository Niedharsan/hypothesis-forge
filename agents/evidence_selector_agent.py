from __future__ import annotations

import json
import hashlib
from typing import Any

from llm.provider import ask_llm_json
from utils.prompt_loader import render_prompt
from utils.json_compact import compact_json, stable_json
from utils.run_logger import log_event


class EvidenceSelectorAgent:
    """Branch-aware evidence curator.

    This worker selects a compact set of papers from a balanced candidate slate
    before synthesis. It is intentionally separate from synthesis: it chooses
    what evidence gets read/synthesized, but it does not write the synthesis.
    """

    def __init__(self, model: str = "gemini-2.5-flash-lite") -> None:
        self.model = model
        self._axis_batch_cache: dict[str, dict[str, Any]] = {}

    def select_subtopic_evidence(
        self,
        *,
        subtopic: dict[str, Any],
        candidate_cards: list[dict[str, Any]],
        target_papers: int = 3,
    ) -> dict[str, Any]:
        subtopic_id = str(subtopic.get("subtopic_id") or "UNKNOWN")
        prompt = render_prompt(
            "v31/evidence_selector_subtopic.md",
            subtopic_json=compact_json(subtopic),
            candidate_cards_json=compact_json(candidate_cards),
            target_papers=str(max(1, int(target_papers))),
        )
        payload = ask_llm_json(prompt, model=self.model, agent="evidence_selector", purpose="subtopic_evidence_selection")
        if not isinstance(payload, dict) or not isinstance(payload.get("selected_candidate_ids"), list):
            log_event(
                "evidence_selector",
                "subtopic_selection_invalid",
                {"subtopic_id": subtopic_id, "reason": "missing selected_candidate_ids; using deterministic fallback"},
                status="warning",
            )
            return {
                "subtopic_id": subtopic_id,
                "selection_decision": "fallback_deterministic",
                "selected_candidate_ids": [str(c.get("candidate_id")) for c in candidate_cards[: max(1, int(target_papers))]],
                "covered_branches": [],
                "uncovered_branches": [],
                "selection_notes": [
                    {"issue": "invalid_selector_output", "details": "EvidenceSelector output did not contain selected_candidate_ids."}
                ],
            }
        payload.setdefault("subtopic_id", subtopic_id)
        payload.setdefault("selection_decision", "selected")
        return payload

    def select_axis_subtopic_evidence(
        self,
        *,
        axis: dict[str, Any],
        subtopics: list[dict[str, Any]],
        candidates_by_subtopic: dict[str, list[dict[str, Any]]],
        target_papers_per_subtopic: int = 3,
    ) -> dict[str, Any]:
        """Select evidence for all subtopics of one axis in a single LLM call.

        Retrieval and slate construction still happen independently per subtopic.
        This method only batches the EvidenceSelector reasoning call so the model
        returns one compact selection list for each subtopic.
        """
        axis_id = str(axis.get("axis_id") or "UNKNOWN")
        prompt = render_prompt(
            "v31/evidence_selector_axis_batch.md",
            axis_json=compact_json(axis),
            subtopics_json=compact_json(subtopics),
            candidates_by_subtopic_json=compact_json(candidates_by_subtopic),
            target_papers_per_subtopic=str(max(1, int(target_papers_per_subtopic))),
        )
        cache_key = hashlib.sha256(stable_json({
            "prompt_template": "v31/evidence_selector_axis_batch.md",
            "model": self.model,
            "axis": axis,
            "subtopics": subtopics,
            "candidates_by_subtopic": candidates_by_subtopic,
            "target_papers_per_subtopic": max(1, int(target_papers_per_subtopic)),
        }).encode("utf-8")).hexdigest()
        cached_payload = self._axis_batch_cache.get(cache_key)
        if cached_payload is not None:
            log_event(
                "evidence_selector",
                "axis_batch_cache_hit",
                {"axis_id": axis_id, "cache_key": cache_key, "subtopics": len(subtopics)},
            )
            payload = dict(cached_payload)
        else:
            log_event(
                "evidence_selector",
                "axis_batch_cache_miss",
                {"axis_id": axis_id, "cache_key": cache_key, "subtopics": len(subtopics)},
            )
            payload = ask_llm_json(prompt, model=self.model, agent="evidence_selector", purpose="axis_batch_subtopic_evidence_selection")
            if isinstance(payload, dict):
                self._axis_batch_cache[cache_key] = dict(payload)
        if not isinstance(payload, dict) or not isinstance(payload.get("subtopic_selections"), list):
            log_event(
                "evidence_selector",
                "axis_batch_selection_invalid",
                {"axis_id": axis_id, "reason": "missing subtopic_selections; using deterministic fallback per subtopic"},
                status="warning",
            )
            return {
                "axis_id": axis_id,
                "selection_scope": "axis_batch",
                "selection_decision": "fallback_deterministic",
                "subtopic_selections": [
                    {
                        "subtopic_id": sid,
                        "selection_decision": "fallback_deterministic",
                        "selected_candidate_ids": [str(c.get("candidate_id")) for c in cards[: max(1, int(target_papers_per_subtopic))]],
                        "covered_branches": [],
                        "uncovered_branches": [],
                        "selection_notes": [
                            {"issue": "invalid_axis_batch_selector_output", "details": "Axis-batch EvidenceSelector output did not contain subtopic_selections."}
                        ],
                    }
                    for sid, cards in candidates_by_subtopic.items()
                ],
                "axis_level_selection_notes": [
                    {"issue": "invalid_selector_output", "details": "Axis-batch output did not contain subtopic_selections."}
                ],
            }
        payload.setdefault("axis_id", axis_id)
        payload.setdefault("selection_scope", "axis_batch")
        return payload
