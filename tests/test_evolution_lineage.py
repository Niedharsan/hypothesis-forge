from __future__ import annotations

import app.stage_overrides as stage_overrides


class StubEvolutionAgent:
    def __init__(self, model: str):
        self.model = model

    def feasibility(self, **kwargs):
        return {"summary": kwargs["hypothesis"]["hypothesis_id"]}

    def out_of_box(self, **kwargs):
        return {"summary": "combined"}


def test_evolution_keeps_parent_review_hypothesis_pairing(monkeypatch):
    monkeypatch.setattr(stage_overrides, "EvolutionAgent", StubEvolutionAgent)
    monkeypatch.setattr(stage_overrides.base, "_prepare_stage", lambda *args, **kwargs: None)
    monkeypatch.setattr(stage_overrides.base, "_artifact_data", lambda *args, **kwargs: {})
    monkeypatch.setattr(stage_overrides.base, "_finish_usage", lambda stage: {"stage": stage, "calls": 0})

    run = {
        "run_id": "hf-lineage-test",
        "model": "mock",
        "objective": "test objective",
        "enable_evolution_retrieval": False,
    }
    parents = [
        {"id": "REF-MISSING", "payload": {"hypothesis": {}, "review": {"note": "skip"}}},
        {"id": "REF-H002", "payload": {"hypothesis": {"hypothesis_id": "H002", "title": "Valid"}, "review": {"note": "paired"}}},
    ]

    cards, _, _ = stage_overrides.run_evolution(run, parents, output_count=10, guidance="")

    assert len(cards) == 1
    assert cards[0]["id"] == "EVO-H002"
    assert cards[0]["parent_ids"] == ["REF-H002"]
    assert cards[0]["payload"]["source_review"] == {"note": "paired"}
