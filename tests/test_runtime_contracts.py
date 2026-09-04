import pytest
from pydantic import ValidationError

from app.models import StartRunRequest
from llm.provider import ask_llm_json
from runtime.context import configure_runtime, current_llm_call_count, seed_llm_call_count


def test_initial_axis_count_is_explicitly_fixed_at_ten():
    request = StartRunRequest(research_objective="Test a scientific hypothesis space.")
    assert request.output_count == 10

    with pytest.raises(ValidationError):
        StartRunRequest(
            research_objective="Test a scientific hypothesis space.",
            output_count=5,
        )


def test_seeded_llm_budget_is_enforced_across_checkpoint_runtime():
    config = {
        "runtime": {
            "mode": "dry_run",
            "limits": {"max_llm_calls_per_run": 3},
            "dry_run": {"mock_llm_outputs": True},
        }
    }
    seed_llm_call_count(2)
    configure_runtime(config, mode_override="dry_run")
    assert current_llm_call_count() == 2

    # The first call consumes the final available slot.
    ask_llm_json(
        'Return strict JSON: {"ok": true}',
        model="gemini-2.5-flash-lite",
        agent="test",
        purpose="cumulative_budget",
    )
    assert current_llm_call_count() == 3

    # A further call is rejected using the cumulative run count, rather than a
    # fresh per-stage counter.
    with pytest.raises(RuntimeError, match="LLM call limit exceeded: 3"):
        ask_llm_json(
            'Return strict JSON: {"ok": true}',
            model="gemini-2.5-flash-lite",
            agent="test",
            purpose="cumulative_budget",
        )


def test_llm_budget_seed_is_one_shot_and_does_not_leak():
    config = {
        "runtime": {
            "mode": "dry_run",
            "limits": {"max_llm_calls_per_run": 10},
            "dry_run": {"mock_llm_outputs": True},
        }
    }
    seed_llm_call_count(4)
    configure_runtime(config, mode_override="dry_run")
    assert current_llm_call_count() == 4

    configure_runtime(config, mode_override="dry_run")
    assert current_llm_call_count() == 0
