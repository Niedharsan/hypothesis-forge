from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.storage as storage
from app import orchestrator
from app.main import app


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(storage, "RUNS_DIR", tmp_path / "runs")
    storage.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    # orchestrator imports the storage functions, whose globals resolve the patched RUNS_DIR.
    return TestClient(app)


def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["service"] == "hypothesis-forge"


def test_dry_run_start_persists_checkpoint(client: TestClient):
    response = client.post("/runs", json={
        "research_objective": "Find mechanistically distinct explanations for a synthetic biology phenotype.",
        "runtime_mode": "dry_run",
        "model": "gemini-2.5-flash-lite",
        "output_count": 10,
    })
    assert response.status_code == 200, response.text
    run = response.json()
    assert run["current_stage"] == "axis_generation"
    assert run["status"] == "checkpoint_ready"
    assert run["stages"]["axis_generation"]
    loaded = client.get(f"/runs/{run['run_id']}")
    assert loaded.status_code == 200
    assert loaded.json()["run_id"] == run["run_id"]


def test_checkpoint_selection_is_retained(client: TestClient):
    run = client.post("/runs", json={
        "research_objective": "Generate testable explanations for a signaling phenotype.",
        "runtime_mode": "dry_run",
        "output_count": 10,
    }).json()
    card_id = run["stages"]["axis_generation"][0]["id"]
    response = client.post(f"/runs/{run['run_id']}/selection", json={
        "stage": "axis_generation",
        "saved_ids": [card_id],
    })
    assert response.status_code == 200
    card = next(c for c in response.json()["stages"]["axis_generation"] if c["id"] == card_id)
    assert card["status"] == "saved"


def test_complete_dry_run_traverses_all_stages(client: TestClient):
    response = client.post("/runs", json={
        "research_objective": "Identify mechanistically distinct, testable explanations for a synthetic biology phenotype.",
        "runtime_mode": "dry_run",
        "model": "gemini-2.5-flash-lite",
        "output_count": 10,
    })
    assert response.status_code == 200, response.text
    run = response.json()
    source = "axis_generation"

    for stage in orchestrator.STAGES[1:]:
        response = client.post(f"/runs/{run['run_id']}/stage", json={
            "stage": stage,
            "source_stage": source,
            "include_all": True,
            "output_count": 10,
            "stage_guidance": "",
        })
        assert response.status_code == 200, f"{stage}: {response.text}"
        run = response.json()
        assert run["stages"][stage], f"No outputs for {stage}"
        source = stage

    assert run["current_stage"] == "candidate_ranking"
    assert run["usage"]["calls"] > 0
    assert len(run["artifacts"]) >= 10


def test_focus_seed_requires_stage_when_card_id_ambiguous(client: TestClient):
    run = client.post("/runs", json={
        "research_objective": "Identify mechanistically distinct, testable explanations for a synthetic biology phenotype.",
        "runtime_mode": "dry_run",
        "model": "gemini-2.5-flash-lite",
        "output_count": 10,
    }).json()
    source = "axis_generation"
    for stage in ["subtopic_generation", "literature_retrieval", "synthesis", "hypothesis_generation", "proximity"]:
        run = client.post(f"/runs/{run['run_id']}/stage", json={
            "stage": stage,
            "source_stage": source,
            "include_all": True,
            "output_count": 10,
            "stage_guidance": "",
        }).json()
        source = stage

    dup_id = sorted({c["id"] for c in run["stages"]["hypothesis_generation"]} & {c["id"] for c in run["stages"]["proximity"]})[0]
    ambiguous = client.post(f"/runs/{run['run_id']}/focus-seed", json={"source_card_id": dup_id})
    assert ambiguous.status_code == 400
    assert "provide source_stage" in ambiguous.json()["detail"]

    hyp_seed = client.post(f"/runs/{run['run_id']}/focus-seed", json={"source_card_id": dup_id, "source_stage": "hypothesis_generation"})
    prox_seed = client.post(f"/runs/{run['run_id']}/focus-seed", json={"source_card_id": dup_id, "source_stage": "proximity"})
    assert hyp_seed.status_code == 200
    assert prox_seed.status_code == 200
    assert hyp_seed.json()["source_stage"] == "hypothesis_generation"
    assert prox_seed.json()["source_stage"] == "proximity"


def test_llm_call_budget_accumulates_across_stage_checkpoints(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        orchestrator,
        "load_config",
        lambda *_args, **_kwargs: {
            "runtime": {
                "mode": "normal",
                "limits": {"max_llm_calls_per_run": 30},
                "dry_run": {"mock_llm_outputs": True},
                "pricing": {},
            }
        },
    )
    response = client.post("/runs", json={
        "research_objective": "Identify mechanistically distinct, testable explanations for a synthetic biology phenotype.",
        "runtime_mode": "dry_run",
        "model": "gemini-2.5-flash-lite",
        "output_count": 10,
    })
    assert response.status_code == 200
    run = response.json()

    source = "axis_generation"
    blocked = None
    for stage in ["subtopic_generation", "literature_retrieval", "synthesis"]:
        response = client.post(f"/runs/{run['run_id']}/stage", json={
            "stage": stage,
            "source_stage": source,
            "include_all": True,
            "output_count": 10,
            "stage_guidance": "",
        })
        if response.status_code != 200:
            blocked = response
            break
        run = response.json()
        source = stage

    assert blocked is not None
    assert blocked.status_code == 502
    assert "LLM call limit exceeded: 30" in blocked.json()["detail"]
