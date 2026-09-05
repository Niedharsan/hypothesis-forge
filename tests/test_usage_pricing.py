from pathlib import Path

from runtime.context import configure_runtime
from utils.config import load_config
from utils.run_logger import _estimate_cost, collect_llm_usage_summary, log_gemini_call, start_run_log


ROOT = Path(__file__).resolve().parents[1]


def _configure_pricing() -> None:
    configure_runtime(load_config(ROOT / "configs" / "config.yaml"), mode_override="normal")


def test_every_selectable_gemini_model_has_cost_pricing() -> None:
    _configure_pricing()
    assert _estimate_cost("gemini-2.5-flash-lite", 1_000_000, 1_000_000) == 0.5
    assert _estimate_cost("gemini-2.5-flash", 1_000_000, 1_000_000) == 2.8
    assert _estimate_cost("gemini-2.5-pro", 200_000, 100_000) == 1.25
    assert _estimate_cost("gemini-2.5-pro", 1_000_000, 1_000_000) == 17.5


def test_thinking_tokens_are_included_in_billable_output(tmp_path: Path) -> None:
    _configure_pricing()
    start_run_log("usage test", run_root=tmp_path)
    log_gemini_call(
        caller="test.pricing",
        model="gemini-2.5-flash-lite",
        prompt="test prompt",
        response_text="{}",
        usage={
            "prompt_token_count": 100,
            "candidates_token_count": 50,
            "thoughts_token_count": 25,
            "total_token_count": 175,
        },
    )
    summary = collect_llm_usage_summary()
    call = summary["calls"][0]
    assert call["input_tokens"] == 100
    assert call["output_tokens"] == 75
    assert call["thoughts_token_count"] == 25
    assert call["estimated_cost_usd"] == 0.00004
