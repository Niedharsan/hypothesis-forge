from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from app.storage import read_json, run_dir, write_run


def create_focus_seed(run: dict[str, Any], request: Any) -> dict[str, Any]:
    """Persist an unambiguous focus seed with explicit stage provenance."""
    matches: list[dict[str, Any]] = []
    for stage, cards in run.get("stages", {}).items():
        if request.source_stage is not None and stage != request.source_stage:
            continue
        for card in cards:
            if str(card.get("id")) == request.source_card_id:
                matches.append(card)

    if not matches:
        raise KeyError(request.source_card_id)
    if len(matches) > 1:
        raise ValueError(
            f"Card id {request.source_card_id!r} exists in multiple stages; provide source_stage explicitly"
        )

    source = matches[0]
    seed = {
        "seed_id": f"seed-{uuid.uuid4().hex[:10]}",
        "source_card_id": request.source_card_id,
        "source_stage": source.get("stage"),
        "title": request.title or source.get("title", ""),
        "summary": request.summary or source.get("summary", ""),
        "guidance": request.guidance,
        "payload": source.get("payload", {}),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    seeds = read_json(run_dir(run["run_id"]) / "focus_seeds.json", []) or []
    seeds.append(seed)

    run.setdefault("artifacts", [])
    run["artifacts"] = [item for item in run["artifacts"] if item.get("filename") != "focus_seeds.json"]
    run["artifacts"].append(
        {
            "id": "focus-seeds",
            "label": "Focus seeds",
            "filename": "focus_seeds.json",
            "stage": "generation",
            "data": seeds,
        }
    )
    write_run(run, {"focus_seeds.json": seeds})
    return seed
