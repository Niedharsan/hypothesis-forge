from __future__ import annotations

from types import SimpleNamespace

import pytest
import requests
from pydantic import ValidationError

from app.focus_seeds import create_focus_seed
from app.models import FocusSeedRequest, SelectionRequest, StageRequest
from retrieval.api_client import CachedAPIClient
from utils.security import redact_sensitive_text


def test_stage_request_rejects_backward_or_same_stage_transitions():
    with pytest.raises(ValidationError):
        StageRequest(stage="synthesis", source_stage="reflection")
    with pytest.raises(ValidationError):
        StageRequest(stage="synthesis", source_stage="synthesis")
    assert StageRequest(stage="synthesis", source_stage="literature_retrieval").source_stage == "literature_retrieval"


def test_selection_request_rejects_overlapping_card_states():
    with pytest.raises(ValidationError):
        SelectionRequest(stage="axis_generation", selected_ids=["A01"], saved_ids=["A01"])


def test_focus_seed_requires_stage_when_card_id_is_ambiguous(tmp_path, monkeypatch):
    import app.focus_seeds as focus_seeds
    import app.storage as storage

    monkeypatch.setattr(storage, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(focus_seeds, "run_dir", storage.run_dir)
    monkeypatch.setattr(focus_seeds, "write_run", storage.write_run)
    monkeypatch.setattr(focus_seeds, "read_json", storage.read_json)
    storage.RUNS_DIR.mkdir(parents=True, exist_ok=True)

    run = {
        "run_id": "hf-test-focus",
        "stages": {
            "hypothesis_generation": [{"id": "H001", "stage": "hypothesis_generation", "title": "Original", "summary": "A", "payload": {"v": 1}}],
            "proximity": [{"id": "H001", "stage": "proximity", "title": "Survivor", "summary": "B", "payload": {"v": 2}}],
        },
        "artifacts": [],
    }

    with pytest.raises(ValueError, match="multiple stages"):
        create_focus_seed(run, FocusSeedRequest(source_card_id="H001"))

    seed = create_focus_seed(run, FocusSeedRequest(source_card_id="H001", source_stage="proximity"))
    assert seed["source_stage"] == "proximity"
    assert seed["payload"] == {"v": 2}


def test_sensitive_text_redaction_covers_query_keys_bearer_and_env(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "super-secret-openalex")
    text = "https://example.test?q=x&api_key=abc123 Authorization=def456 Bearer tok_789 super-secret-openalex"
    redacted = redact_sensitive_text(text)
    assert "abc123" not in redacted
    assert "def456" not in redacted
    assert "tok_789" not in redacted
    assert "super-secret-openalex" not in redacted


def test_http_errors_do_not_expose_query_parameter_api_keys(tmp_path, monkeypatch):
    response = requests.Response()
    response.status_code = 401
    response.url = "https://example.test/search?api_key=super-secret"
    response.request = requests.Request("GET", response.url).prepare()

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: response)
    client = CachedAPIClient(cache_dir=tmp_path / "cache", min_interval_seconds=0)

    with pytest.raises(RuntimeError) as exc_info:
        client.get_json(
            "https://example.test/search",
            params={"api_key": "super-secret"},
            max_retries=0,
        )

    assert "super-secret" not in str(exc_info.value)
