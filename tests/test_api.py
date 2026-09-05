from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.storage as storage
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


def test_run_archive_lists_persisted_questions(client: TestClient):
    first = client.post("/runs", json={
        "research_objective": "Archive question one",
        "runtime_mode": "dry_run",
        "output_count": 10,
    }).json()
    second = client.post("/runs", json={
        "research_objective": "Archive question two",
        "runtime_mode": "dry_run",
        "output_count": 10,
    }).json()

    response = client.get("/runs?limit=100")
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 2
    by_id = {item["run_id"]: item for item in payload["runs"]}
    assert set(by_id) == {first["run_id"], second["run_id"]}
    assert by_id[first["run_id"]]["objective"] == "Archive question one"
    assert by_id[first["run_id"]]["current_stage"] == "axis_generation"
    assert by_id[first["run_id"]]["stage_counts"]["axis_generation"] == 10
    assert "stages" not in by_id[first["run_id"]]


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
    from app.orchestrator import STAGES

    response = client.post("/runs", json={
        "research_objective": "Identify mechanistically distinct, testable explanations for a synthetic biology phenotype.",
        "runtime_mode": "dry_run",
        "model": "gemini-2.5-flash-lite",
        "output_count": 10,
    })
    assert response.status_code == 200, response.text
    run = response.json()
    source = "axis_generation"

    for stage in STAGES[1:]:
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
