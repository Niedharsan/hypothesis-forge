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
from runtime.context import configure_runtime, current_llm_call_count
from utils.config import load_config, validate_config
from utils.run_logger import start_run_log, log_event, finalize_run, collect_llm_usage_summary


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run Proximity only on the merged 30-hypothesis AML set."
    )
    ap.add_argument(
        "--goal",
        default="Identify novel drug candidates or therapeutic routes for acute myeloid leukemia that have not previously been used for AML.",
    )
    ap.add_argument("--input-dir", default="inputs/merged_v70_30_with_v73_ire1")
    ap.add_argument("--out-dir", default="runs/v75_merged_30_proximity_cutoff2023")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--model", default="gemini-2.5-flash-lite")
    ap.add_argument("--max-llm-calls", type=int, default=20)
    # Deprecated compatibility flag; Proximity no longer uses a survivor target.
    ap.add_argument("--max-proximity-survivors", type=int, default=None)
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
    start_run_log(args.goal, "v75_merged_30_proximity_only")
    for warning in warnings:
        log_event("config", "warning", {"warning": warning}, status="warning")

    supervisor_config = load_json(inp / "01_supervisor_config.json")
    generation_supervisor_view = load_json(inp / "01d_hypothesis_generation_supervisor_view.json")
    hypotheses_payload = load_json(inp / "07_generation_hypotheses_payload_merged_30.json")
    paper_memory_compact = load_json(inp / "07b_paper_memory_compact_for_reflection_merged.json")

    proximity_payload = ProximityAgent(model=args.model).cluster_merge_salvage(
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

    for name in [
        "merge_manifest.json",
        "07_generation_hypotheses_payload_merged_30.json",
        "replaced_original_v70_H001.json",
        "inserted_original_v73_A04_H01.json",
    ]:
        src = inp / name
        if src.exists():
            shutil.copy2(src, out / name)

    (out / "08_proximity_clusters.json").write_text(
        json.dumps(proximity_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "08a_proximity_reflection_input_hypotheses.json").write_text(
        json.dumps(reflection_input_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    llm_usage = collect_llm_usage_summary()
    summary = {
        "stage_stopped_after": "proximity",
        "mode": "merged_30_proximity_only",
        "base_run": "v70_full_reflection_cutoff2023",
        "inserted_run": "v73_fixed_weak_axis_hypotheses_cutoff2023",
        "hypotheses_input_count": len(hypotheses_payload.get("hypotheses", [])) if isinstance(hypotheses_payload, dict) else None,
        "proximity_survivor_count": len(proximity_payload.get("survivor_hypotheses", [])) if isinstance(proximity_payload, dict) else 0,
        "proximity_focus_seed_count": len(proximity_payload.get("focus_seeds", [])) if isinstance(proximity_payload, dict) else 0,
        "reflection_input_hypotheses": len(reflection_input_payload.get("hypotheses", [])) if isinstance(reflection_input_payload, dict) else 0,
        "llm_calls_used": current_llm_call_count(),
        "llm_usage_totals": llm_usage.get("totals", {}),
        "llm_usage_by_stage": llm_usage.get("by_stage", {}),
        "top_prompt_calls": llm_usage.get("top_prompt_calls", []),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "llm_usage.json").write_text(json.dumps(llm_usage, indent=2, ensure_ascii=False), encoding="utf-8")
    log_event("workflow", "merged_30_proximity_complete", summary)
    finalize_run(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
