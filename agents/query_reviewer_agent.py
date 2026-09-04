from __future__ import annotations

import json
from typing import Any

from llm.provider import ask_llm_json
from utils.prompt_loader import render_prompt
from utils.json_compact import compact_json
from utils.run_logger import log_event


class QueryReviewerAgent:
    """Neutral query hygiene worker.

    The reviewer is intentionally separate from Supervisor. It reviews a full
    generated query set at a given stage and returns a revised set for retrieval.
    """

    def __init__(self, model: str = "gemini-2.5-flash-lite") -> None:
        self.model = model

    def review_axis_query_families(
        self,
        *,
        axis: dict[str, Any],
        query_families_payload: dict[str, Any],
        max_query_families: int = 6,
    ) -> dict[str, Any]:
        axis_id = str(axis.get("axis_id") or query_families_payload.get("axis_id") or "A00")
        prompt = render_prompt(
            "v31/query_reviewer_axis_query_families.md",
            scope_label="axis",
            scope_id=axis_id,
            scope_json=compact_json(axis),
            query_families_json=compact_json(query_families_payload),
            max_query_families=str(max_query_families),
        )
        reviewed = ask_llm_json(prompt, model=self.model, agent="query_reviewer", purpose="axis_query_family_review")
        if not isinstance(reviewed, dict) or not isinstance(reviewed.get("query_families"), list):
            log_event(
                "query_reviewer",
                "axis_query_family_review_invalid",
                {"axis_id": axis_id, "reason": "missing query_families list; using unreviewed payload"},
                status="warning",
            )
            fallback = dict(query_families_payload)
            fallback["query_review"] = {
                "review_decision": "fallback_unreviewed",
                "review_notes": [
                    {"issue": "invalid_reviewer_output", "details": "Reviewer output did not contain a query_families list.", "change_made": "Used original query-family payload."}
                ],
            }
            return fallback
        reviewed.setdefault("axis_id", axis_id)
        reviewed.setdefault("review_decision", "revised")
        reviewed["query_reviewed"] = True
        return reviewed
