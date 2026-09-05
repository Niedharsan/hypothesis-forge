from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.supervisor_config_agent import SupervisorConfigAgent
from agents.literature_agent import LiteratureAgent
from agents.generation_rewired import RewiredGenerationAgent
from runtime.context import configure_runtime
from utils.config import load_config, validate_config
from utils.run_logger import start_run_log, finalize_run, log_event, collect_llm_usage_summary
from utils.supervisor_views import build_generation_supervisor_view

DEFAULT_SOURCES = "PubMed,EuropePMC,OpenAlex,Crossref"
SUPPORTED_SOURCES = {"PubMed", "EuropePMC", "OpenAlex", "Crossref", "SemanticScholar"}


def parse_sources(raw: str) -> list[str]:
    values = [s.strip() for s in raw.split(",") if s.strip()]
    for s in values:
        if s not in SUPPORTED_SOURCES:
            raise SystemExit(f"Unsupported source: {s}. Supported: {sorted(SUPPORTED_SOURCES)}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the v33 literature-first HypothesisForge generation workflow")
    parser.add_argument("--goal", required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--out-dir", default="runs/v33_literature_first")
    parser.add_argument("--sources", default=DEFAULT_SOURCES)
    parser.add_argument("--max-subtopics", type=int, default=10)
    parser.add_argument("--max-queries-per-subtopic", type=int, default=3)
    parser.add_argument("--raw-papers-per-source-query", type=int, default=5)
    parser.add_argument("--ai-papers-per-subtopic", type=int, default=5)
    parser.add_argument("--disable-retrieval", action="store_true")
    parser.add_argument("--model", default="gemini-2.5-flash-lite")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.max_subtopics < 1 or args.max_subtopics > 12:
        raise SystemExit("--max-subtopics must be 1–12")
    if args.max_queries_per_subtopic < 1 or args.max_queries_per_subtopic > 5:
        raise SystemExit("--max-queries-per-subtopic must be 1–5")
    if args.raw_papers_per_source_query < 1 or args.raw_papers_per_source_query > 10:
        raise SystemExit("--raw-papers-per-source-query must be 1–10")
    if args.ai_papers_per_subtopic < 1 or args.ai_papers_per_subtopic > 15:
        raise SystemExit("--ai-papers-per-subtopic must be 1–15")

    config = load_config(args.config)
    warnings = validate_config(config, runtime_mode="dry_run" if args.dry_run else None, strict=False)
    configure_runtime(config, mode_override="dry_run" if args.dry_run else None)
    run_id = start_run_log(args.goal, "v33_literature_first_generation")
    for warning in warnings:
        log_event("config", "warning", {"warning": warning}, status="warning")

    sources = parse_sources(args.sources)
    supervisor = SupervisorConfigAgent().configure(
        args.goal,
        axes=args.max_subtopics,
        use_literature=not args.disable_retrieval,
        model=args.model,
    )

    supervisor_dict = supervisor.to_dict()
    hypothesis_generation_supervisor_view = build_generation_supervisor_view(supervisor_dict)

    literature = LiteratureAgent(model=args.model).run(
        objective=args.goal,
        sources=sources,
        max_subtopics=args.max_subtopics,
        max_queries_per_subtopic=args.max_queries_per_subtopic,
        raw_papers_per_source_query=args.raw_papers_per_source_query,
        ai_papers_per_subtopic=args.ai_papers_per_subtopic,
        enable_retrieval=not args.disable_retrieval,
    )

    generator = RewiredGenerationAgent(model=args.model)
    hypotheses_payload = generator.generate_hypotheses_from_literature(
        args.goal,
        hypothesis_generation_supervisor_view,
        literature.syntheses,
        literature.global_synthesis,
    )
    evidence_packets = [p for r in literature.retrieval_results for p in r.evidence_packets]
    strategies = generator._coerce_hypotheses(hypotheses_payload, args.goal, evidence_packets)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "01_supervisor_config.json").write_text(json.dumps(supervisor_dict, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "01a_literature_input_policy.json").write_text(json.dumps({"supervisor_fields_passed_to_literature": [], "policy": "LiteratureAgent receives only objective/axis/subtopic/retrieved-evidence inputs; Supervisor constraints are not passed to retrieval or synthesis."}, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "01b_hypothesis_generation_supervisor_view.json").write_text(json.dumps(hypothesis_generation_supervisor_view, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "02_literature_subtopics.json").write_text(json.dumps(literature.subtopics_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "03_retrieval_results.json").write_text(json.dumps([
        {
            "subtopic_id": r.subtopic_id,
            "queries": r.queries,
            "records_returned_to_ai": len(r.evidence_packets),
            "evidence_packets": [p.to_dict() for p in r.evidence_packets],
            "warnings": r.warnings,
        }
        for r in literature.retrieval_results
    ], indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "04_literature_syntheses.json").write_text(json.dumps(literature.syntheses, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "05_global_synthesis.json").write_text(json.dumps(literature.global_synthesis, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "06_generation_hypotheses_payload.json").write_text(json.dumps(hypotheses_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "initial_candidates.json").write_text(json.dumps([s.to_dict() for s in strategies], indent=2, ensure_ascii=False), encoding="utf-8")

    llm_usage = collect_llm_usage_summary()
    (out / "llm_usage.json").write_text(json.dumps(llm_usage, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "run_id": run_id,
        "goal": args.goal,
        "model": args.model,
        "subtopics_returned": len(literature.subtopics_payload.get("subtopics", [])) if isinstance(literature.subtopics_payload, dict) else None,
        "literature_syntheses_returned": len(literature.syntheses),
        "global_synthesis_present": bool(literature.global_synthesis),
        "hypotheses_returned": len(strategies),
        "sources": sources,
        "max_subtopics": args.max_subtopics,
        "max_queries_per_subtopic": args.max_queries_per_subtopic,
        "raw_papers_per_source_query": args.raw_papers_per_source_query,
        "ai_papers_per_subtopic": args.ai_papers_per_subtopic,
        "retrieval_enabled": not args.disable_retrieval,
        "llm_usage_totals": llm_usage.get("totals", {}),
        "llm_usage_by_stage": llm_usage.get("by_stage", {}),
        "llm_usage_by_model": llm_usage.get("by_model", {}),
        "top_prompt_calls": llm_usage.get("top_prompt_calls", []),
        "budget_usage": llm_usage.get("budget_usage", {}),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log_event("workflow", "v33_literature_first_complete", summary)
    finalize_run(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
