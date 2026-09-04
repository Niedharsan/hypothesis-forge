#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.proximity_agent import ProximityAgent, build_hypotheses_payload_from_proximity
from agents.reflection_agent import ReflectionAgent
from runtime.context import configure_runtime, current_llm_call_count
from utils.config import load_config, validate_config
from utils.run_logger import start_run_log, log_event, finalize_run, collect_llm_usage_summary


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _ids(payload: dict, key: str) -> list[str]:
    val = payload.get(key, []) if isinstance(payload, dict) else []
    return [str(x) for x in val] if isinstance(val, list) else []


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run Proximity then Reflection on the merged 30-hypothesis AML set."
    )
    ap.add_argument(
        "--goal",
        default="Identify novel drug candidates or therapeutic routes for acute myeloid leukemia that have not previously been used for AML.",
    )
    ap.add_argument("--input-dir", default="inputs/merged_v70_30_with_v73_ire1")
    ap.add_argument("--out-dir", default="runs/v76_merged_30_proximity_reflection_cutoff2023")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--model", default="gemini-2.5-flash-lite")
    ap.add_argument("--proximity-model", default="")
    ap.add_argument("--reflection-model", default="")
    ap.add_argument("--max-llm-calls", type=int, default=20)
    ap.add_argument("--max-proximity-focus-seeds", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--clean-out-dir", action="store_true")
    args = ap.parse_args()

    if args.max_proximity_focus_seeds < 0 or args.max_proximity_focus_seeds > 30:
        raise SystemExit("--max-proximity-focus-seeds must be 0-30")

    inp = Path(args.input_dir)
    out = Path(args.out_dir)
    if args.clean_out_dir and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    config.setdefault("runtime", {}).setdefault("limits", {})["max_llm_calls_per_run"] = args.max_llm_calls
    warnings = validate_config(config, runtime_mode="dry_run" if args.dry_run else None, strict=False)
    configure_runtime(config, mode_override="dry_run" if args.dry_run else None)
    start_run_log(args.goal, "v76_merged_30_proximity_reflection")
    for warning in warnings:
        log_event("config", "warning", {"warning": warning}, status="warning")

    supervisor_config = load_json(inp / "01_supervisor_config.json")
    generation_supervisor_view = load_json(inp / "01d_hypothesis_generation_supervisor_view.json")
    axes_payload = load_json(inp / "02_generation_axes_merged.json")
    global_synthesis = load_json(inp / "06_global_synthesis_merged.json")
    axis_syntheses = load_json(inp / "05_axis_literature_syntheses_merged.json")
    hypotheses_payload = load_json(inp / "07_generation_hypotheses_payload_merged_30.json")
    paper_memory_compact = load_json(inp / "07b_paper_memory_compact_for_reflection_merged.json")

    proximity_payload = ProximityAgent(model=args.proximity_model or args.model).cluster_merge_salvage(
        supervisor_config=supervisor_config,
        generation_supervisor_view=generation_supervisor_view,
        hypotheses_payload=hypotheses_payload,
        paper_memory_compact=paper_memory_compact,
        max_focus_seeds=args.max_proximity_focus_seeds,
    )

    reflection_input_payload = build_hypotheses_payload_from_proximity(
        hypotheses_payload,
        proximity_payload if isinstance(proximity_payload, dict) else {},
    )

    reflection_payload = ReflectionAgent(model=args.reflection_model or args.model).review_global_hypotheses_with_proximity(
        supervisor_config=supervisor_config,
        generation_supervisor_view=generation_supervisor_view,
        axes_payload=axes_payload,
        global_synthesis=global_synthesis,
        axis_syntheses=axis_syntheses if isinstance(axis_syntheses, list) else axis_syntheses.get("axis_syntheses", []),
        hypotheses_payload=reflection_input_payload,
        proximity_payload=proximity_payload if isinstance(proximity_payload, dict) else {},
        paper_memory_compact=paper_memory_compact,
    )

    for name in [
        "merge_manifest.json",
        "07_generation_hypotheses_payload_merged_30.json",
        "replaced_original_v70_H001.json",
        "inserted_original_v73_A04_H01.json",
    ]:
        src = inp / name
        if src.exists():
            shutil.copy2(src, out / name)

    (out / "08_proximity_analysis.json").write_text(
        json.dumps(proximity_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "08a_proximity_reflection_input_hypotheses.json").write_text(
        json.dumps(reflection_input_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "09_reflection_reviews_with_proximity.json").write_text(
        json.dumps(reflection_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Convenience files for next stages.
    keep_ids = _ids(reflection_payload, "keep_for_evolution") + _ids(reflection_payload, "revise_for_evolution")
    keep_ids = list(dict.fromkeys([x for x in keep_ids if x and x != "..."]))
    focus_seeds = reflection_payload.get("focus_seeds", []) if isinstance(reflection_payload, dict) else []
    prox_focus_seeds = proximity_payload.get("focus_seeds", []) if isinstance(proximity_payload, dict) else []
    (out / "09a_reflection_keep_for_evolution_ids.json").write_text(
        json.dumps({"keep_or_revise_for_evolution": keep_ids}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "09b_reflection_focus_seeds.json").write_text(
        json.dumps({"reflection_focus_seeds": focus_seeds, "proximity_focus_seeds": prox_focus_seeds}, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    llm_usage = collect_llm_usage_summary()
    summary = {
        "stage_stopped_after": "reflection_with_proximity",
        "mode": "merged_30_proximity_reflection",
        "base_run": "v70_full_reflection_cutoff2023",
        "inserted_run": "v73_fixed_weak_axis_hypotheses_cutoff2023",
        "hypotheses_input_count": len(hypotheses_payload.get("hypotheses", [])) if isinstance(hypotheses_payload, dict) else None,
        "proximity_survivor_count": len(proximity_payload.get("survivor_hypotheses", [])) if isinstance(proximity_payload, dict) else 0,
        "proximity_group_count": len(proximity_payload.get("proximity_groups", [])) if isinstance(proximity_payload, dict) else 0,
        "proximity_focus_seed_count": len(proximity_payload.get("focus_seeds", [])) if isinstance(proximity_payload, dict) else 0,
        "reflection_input_hypotheses": len(reflection_input_payload.get("hypotheses", [])) if isinstance(reflection_input_payload, dict) else 0,
        "reflection_review_count": len(reflection_payload.get("hypothesis_reviews", [])) if isinstance(reflection_payload, dict) else 0,
        "reflection_keep_for_evolution_count": len(_ids(reflection_payload, "keep_for_evolution")),
        "reflection_revise_for_evolution_count": len(_ids(reflection_payload, "revise_for_evolution")),
        "reflection_rejected_count": len(_ids(reflection_payload, "rejected_hypotheses")),
        "reflection_focus_seed_count": len(focus_seeds) if isinstance(focus_seeds, list) else 0,
        "llm_calls_used": current_llm_call_count(),
        "llm_usage_totals": llm_usage.get("totals", {}),
        "llm_usage_by_stage": llm_usage.get("by_stage", {}),
        "top_prompt_calls": llm_usage.get("top_prompt_calls", []),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "llm_usage.json").write_text(json.dumps(llm_usage, indent=2, ensure_ascii=False), encoding="utf-8")
    log_event("workflow", "merged_30_proximity_reflection_complete", summary)
    finalize_run(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
