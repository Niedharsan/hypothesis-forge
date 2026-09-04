from __future__ import annotations

import argparse
import json
import os
import sys
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.literature_agent import LiteratureAgent, _subtopics, _select_balanced_packets_by_subtopic
from agents.query_reviewer_agent import QueryReviewerAgent
from agents.supervisor_config_agent import SupervisorConfigAgent
from agents.reflection_agent import ReflectionAgent
from llm.provider import ask_llm_json
from utils.prompt_loader import render_prompt
from utils.json_compact import compact_json
from runtime.context import configure_runtime
from utils.config import load_config, validate_config
from utils.run_logger import start_run_log, finalize_run, log_event, collect_llm_usage_summary
from utils.supervisor_views import build_generation_supervisor_view
from utils.evidence_memory import build_paper_memory, compact_memory_for_reflection

DEFAULT_SOURCES = "PubMed,EuropePMC,OpenAlex,Crossref"
SUPPORTED_SOURCES = {"PubMed", "EuropePMC", "OpenAlex", "Crossref", "SemanticScholar"}

DEFAULT_GOAL = "Identify novel drug repurposing candidates for acute myeloid leukemia that have not previously been used for AML."
DEFAULT_AXIS_JSON = "inputs/weakest_upr_axis_v52_v39_mech_granularity_rep2.json"


def parse_sources(raw: str) -> list[str]:
    values = [s.strip() for s in raw.split(",") if s.strip()]
    for s in values:
        if s not in SUPPORTED_SOURCES:
            raise SystemExit(f"Unsupported source: {s}. Supported: {sorted(SUPPORTED_SOURCES)}")
    return values


def generate_single_axis_hypotheses(
    *,
    objective: str,
    axis: dict[str, Any],
    subtopics_payload: dict[str, Any],
    axis_synthesis: dict[str, Any],
    cutoff_year: int,
    model: str,
    generation_supervisor_view: dict[str, Any],
) -> dict[str, Any]:
    prompt = render_prompt(
        "v31/generation_hypotheses_from_single_axis_literature.md",
        objective=objective,
        generation_supervisor_view_json=compact_json(generation_supervisor_view),
        cutoff_rule=f"When judging whether something is known or new, use only supplied evidence published through December 31, {cutoff_year}. Do not use later evidence for novelty decisions.",
        axis_json=compact_json(axis),
        subtopics_json=compact_json(subtopics_payload),
        axis_synthesis_json=compact_json(axis_synthesis),
    )
    return ask_llm_json(prompt, model=model, agent="generation", purpose="single_axis_axis_local_hypotheses")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one fixed axis through v2 entity-map subtopic generation, retrieval, axis-batched evidence selection, axis synthesis, and axis-local hypothesis generation.")
    parser.add_argument("--goal", default=DEFAULT_GOAL)
    parser.add_argument("--axis-json", default=DEFAULT_AXIS_JSON)
    parser.add_argument("--resume-from-subtopics", default="", help="Load an existing selected-subtopics JSON and resume from targeted subtopic retrieval/synthesis. Skips axis query-family generation, broad entity-map retrieval, and subtopic generation.")
    parser.add_argument("--resume-from-context", default="", help="Optional existing 00c_subtopic_generation_context.json to copy into output for provenance when resuming from subtopics.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--out-dir", default="runs/v68_3_curator_hypothesis_filtered")
    parser.add_argument("--clean-out-dir", action="store_true", help="Delete --out-dir before writing new outputs. Use for clean reruns only.")
    parser.add_argument("--sources", default=DEFAULT_SOURCES)
    parser.add_argument("--max-subtopics-per-axis", type=int, default=5)
    parser.add_argument("--max-queries-per-subtopic", type=int, default=5)
    parser.add_argument("--subtopic-mode", choices=["v2", "anchor"], default="v2", help="v2=axis-first plus entity-map evidence. anchor keeps the v56 anchor-first diagnostic. Legacy v1/compare/old axis-only modes have been removed in v69.1.")
    parser.add_argument("--selected-subtopic-version", choices=["v2"], default="v2", help="Deprecated compatibility flag; v69.1 always uses v2 when --subtopic-mode v2.")
    parser.add_argument("--stop-after-query-families", action="store_true", help="Only generate and write axis query families, then stop before entity-map retrieval/subtopic generation.")
    parser.add_argument("--disable-query-reviewer", action="store_true", help="Disable QueryReviewerAgent for axis query-family review.")
    parser.add_argument("--query-reviewer-model", default="", help="Optional model override for QueryReviewerAgent. Defaults to --model.")
    parser.add_argument("--stop-after-subtopics", action="store_true", help="Only write subtopic-generation outputs and stop before targeted retrieval/synthesis.")
    parser.add_argument("--stop-after-synthesis", action="store_true", help="Run targeted retrieval and axis synthesis, then stop before hypothesis generation.")
    parser.add_argument("--stop-after-evidence-slate", action="store_true", help="Run targeted retrieval, dedupe, PubTator/lexical filtering, and balanced slate construction, then stop before EvidenceSelector and synthesis.")
    parser.add_argument("--stop-after-reflection", action="store_true", help="Run through hypothesis generation and ReflectionAgent, then stop. This is mainly a guardrail for staged tests.")
    parser.add_argument("--max-axis-anchor-queries", type=int, default=3)
    parser.add_argument("--axis-anchor-papers", type=int, default=2)
    parser.add_argument("--max-axis-query-families", type=int, default=6)
    parser.add_argument("--entity-map-candidates-per-family", type=int, default=10)
    parser.add_argument("--raw-papers-per-source-query", type=int, default=5)
    parser.add_argument("--use-pubtator", action="store_true")
    parser.add_argument("--disable-evidence-selector", action="store_true", help="Disable EvidenceSelectorAgent. By default v68 uses branch-aware evidence selection before synthesis.")
    parser.add_argument("--disable-pmid-branch-tag-filter", action="store_true", help="Disable PMID-only PubTator/branch-signal guardrail for the pre-curator slate.")
    parser.add_argument("--evidence-selector-model", default="", help="Optional model override for EvidenceSelectorAgent. Defaults to --model.")
    parser.add_argument("--evidence-selector-initial-depth-per-query", type=int, default=3, help="Round-robin slate depth: top K deduped candidates per subtopic query are shown to EvidenceSelectorAgent.")
    parser.add_argument("--evidence-selector-max-depth-per-query", type=int, default=10, help="Maximum deduped candidates saved per query branch for debugging/refill experiments.")
    parser.add_argument("--pubtator-max-candidates-per-subtopic", type=int, default=100)
    parser.add_argument("--ai-papers-per-subtopic", type=int, default=3)
    parser.add_argument("--ai-papers-per-axis", type=int, default=15, help="Safety cap only; default equals 5 subtopics × 3 papers.")
    parser.add_argument("--cutoff-year", type=int, default=2025)
    parser.add_argument("--axis-synthesis-abstract-chars", type=int, default=1200, help="Characters per selected evidence packet passed into the axis synthesis LLM call. Lower values reduce cost; default 1200.")
    parser.add_argument("--exclusion-rule", default="", help=argparse.SUPPRESS)  # deprecated; objective-specific exclusions now belong in Supervisor/objective text.
    parser.add_argument("--disable-retrieval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", default="gemini-2.5-flash-lite")
    parser.add_argument("--run-reflection", action="store_true", help="After hypothesis generation, run one supervisor-guided ReflectionAgent batch review for this axis.")
    parser.add_argument("--reflection-model", default="", help="Optional model override for ReflectionAgent. Defaults to --model.")
    args = parser.parse_args()

    if args.max_subtopics_per_axis < 1 or args.max_subtopics_per_axis > 8:
        raise SystemExit("--max-subtopics-per-axis must be 1–8")
    if args.max_queries_per_subtopic < 1 or args.max_queries_per_subtopic > 5:
        raise SystemExit("--max-queries-per-subtopic must be 1–5")
    if args.max_axis_anchor_queries < 1 or args.max_axis_anchor_queries > 5:
        raise SystemExit("--max-axis-anchor-queries must be 1–5")
    if args.axis_anchor_papers < 1 or args.axis_anchor_papers > 5:
        raise SystemExit("--axis-anchor-papers must be 1–5")
    if args.raw_papers_per_source_query < 1 or args.raw_papers_per_source_query > 10:
        raise SystemExit("--raw-papers-per-source-query must be 1–10")
    if args.max_axis_query_families < 1 or args.max_axis_query_families > 8:
        raise SystemExit("--max-axis-query-families must be 1–8")
    if args.entity_map_candidates_per_family < 1 or args.entity_map_candidates_per_family > 25:
        raise SystemExit("--entity-map-candidates-per-family must be 1–25")
    if args.pubtator_max_candidates_per_subtopic < 1 or args.pubtator_max_candidates_per_subtopic > 500:
        raise SystemExit("--pubtator-max-candidates-per-subtopic must be 1–500")
    if args.evidence_selector_initial_depth_per_query < 1 or args.evidence_selector_initial_depth_per_query > 10:
        raise SystemExit("--evidence-selector-initial-depth-per-query must be 1–10")
    if args.evidence_selector_max_depth_per_query < 1 or args.evidence_selector_max_depth_per_query > 25:
        raise SystemExit("--evidence-selector-max-depth-per-query must be 1–25")
    if args.ai_papers_per_subtopic < 1 or args.ai_papers_per_subtopic > 5:
        raise SystemExit("--ai-papers-per-subtopic must be 1–5")
    if args.ai_papers_per_axis < 1 or args.ai_papers_per_axis > 30:
        raise SystemExit("--ai-papers-per-axis must be 1–30")
    if args.axis_synthesis_abstract_chars < 500 or args.axis_synthesis_abstract_chars > 2200:
        raise SystemExit("--axis-synthesis-abstract-chars must be 500–2200")

    config = load_config(args.config)
    warnings = validate_config(config, runtime_mode="dry_run" if args.dry_run else None, strict=False)
    configure_runtime(config, mode_override="dry_run" if args.dry_run else None)
    run_id = start_run_log(args.goal, "v68_4_reflection_memory_single_axis")
    for warning in warnings:
        log_event("config", "warning", {"warning": warning}, status="warning")

    # Supervisor is intentionally not created before literature/query planning.
    # This guarantees Supervisor guidance cannot be passed into LiteratureAgent or QueryReviewerAgent.
    supervisor_dict: dict[str, Any] = {}
    generation_supervisor_view: dict[str, Any] = {}

    axis_path = Path(args.axis_json)
    axis = json.loads(axis_path.read_text(encoding="utf-8"))
    resume_from_subtopics = bool(str(args.resume_from_subtopics).strip())
    resume_subtopics_path = Path(args.resume_from_subtopics) if resume_from_subtopics else None
    resume_context_payload: dict[str, Any] = {}
    if str(args.resume_from_context).strip():
        resume_context_path = Path(args.resume_from_context)
        if not resume_context_path.exists():
            raise SystemExit(f"--resume-from-context not found: {resume_context_path}")
        resume_context_payload = json.loads(resume_context_path.read_text(encoding="utf-8"))
    sources = parse_sources(args.sources)
    os.environ["AXIS_SYNTHESIS_ABSTRACT_CHARS"] = str(args.axis_synthesis_abstract_chars)
    lit = LiteratureAgent(model=args.model)  # LiteratureAgent receives no Supervisor fields; Supervisor guidance is used by Generation, not retrieval planning
    query_reviewer = QueryReviewerAgent(model=args.query_reviewer_model or args.model)

    axis_id = str(axis.get("axis_id") or "A00")
    anchor_queries_payload: dict[str, Any] = {}
    anchor_retrieval = None
    anchor_packets = []
    query_families_payload: dict[str, Any] = {}
    raw_query_families_payload: dict[str, Any] = {}
    query_reviewer_payload: dict[str, Any] = {}
    entity_concept_inventory: dict[str, Any] = {}
    subtopics_v2: dict[str, Any] = {}

    if resume_from_subtopics:
        if resume_subtopics_path is None or not resume_subtopics_path.exists():
            raise SystemExit(f"--resume-from-subtopics not found: {args.resume_from_subtopics}")
        subtopics_payload = json.loads(resume_subtopics_path.read_text(encoding="utf-8"))
        if not _subtopics(subtopics_payload):
            raise SystemExit(f"--resume-from-subtopics contains zero recognizable subtopics: {resume_subtopics_path}")
        subtopics_v2 = subtopics_payload if args.subtopic_mode == "v2" else {}
        entity_concept_inventory = {
            "resume_from_subtopics": True,
            "resume_from_subtopics_path": str(resume_subtopics_path),
            "notes": [
                "Loaded selected subtopics from a previous run.",
                "Skipped axis query-family generation, QueryReviewer, broad entity-map retrieval, and subtopic generation.",
                "Continuing directly to targeted subtopic retrieval/synthesis unless stopped earlier.",
            ],
        }
    elif args.subtopic_mode == "anchor":
        anchor_queries_payload = lit.generate_axis_anchor_queries(axis, max_anchor_queries=args.max_axis_anchor_queries)
        anchor_queries = [str(q).strip() for q in anchor_queries_payload.get("anchor_search_queries", []) if str(q).strip()]
        if args.disable_retrieval:
            from agents.literature_agent import SubtopicRetrievalResult
            anchor_retrieval = SubtopicRetrievalResult(subtopic_id=f"{axis_id}_ANCHOR", queries=anchor_queries, warnings=["retrieval disabled"])
        else:
            anchor_retrieval = lit.retrieve_axis_anchors(
                axis_id=axis_id,
                queries=anchor_queries,
                sources=sources,
                max_queries=args.max_axis_anchor_queries,
                raw_papers_per_source_query=args.raw_papers_per_source_query,
                max_anchor_papers=args.axis_anchor_papers,
                cutoff_year=args.cutoff_year,
            )
        anchor_packets = anchor_retrieval.evidence_packets if anchor_retrieval else []
        subtopics_payload = lit.decompose_axis_from_anchors(
            axis,
            anchor_packets=anchor_packets,
            max_subtopics=args.max_subtopics_per_axis,
            max_queries_per_subtopic=args.max_queries_per_subtopic,
        )
    else:
        raw_query_families_payload = lit.generate_axis_query_families(axis, max_query_families=args.max_axis_query_families)
        query_families_payload = raw_query_families_payload
        if not args.disable_query_reviewer:
            query_families_payload = query_reviewer.review_axis_query_families(
                axis=axis,
                query_families_payload=raw_query_families_payload,
                max_query_families=args.max_axis_query_families,
            )
            query_reviewer_payload = {
                "enabled": True,
                "stage": "axis_query_families",
                "model": args.query_reviewer_model or args.model,
                "reviewed_payload": query_families_payload,
                "raw_query_families_payload": raw_query_families_payload,
            }
        else:
            query_reviewer_payload = {"enabled": False, "stage": "axis_query_families", "raw_query_families_payload": raw_query_families_payload}
        if args.stop_after_query_families:
            entity_concept_inventory = {
                "query_family_generation_only": True,
                "query_families_payload": query_families_payload,
                "notes": [
                    "Stopped immediately after axis query-family generation and QueryReviewer review.",
                    "No Supervisor config, literature retrieval, or subtopic generation was run in this query-only mode.",
                ],
            }
            subtopics_v2 = {"axis_id": axis_id, "subtopics": []}
            subtopics_payload = subtopics_v2
        else:
            if args.disable_retrieval:
                entity_concept_inventory = {
                    "retrieval_disabled": True,
                    "query_families_payload": query_families_payload,
                    "notes": ["Entity-map retrieval disabled; v2 decomposition will use query-family plan only."],
                }
            else:
                entity_concept_inventory = lit.build_axis_entity_concept_inventory(
                    axis,
                    query_families_payload=query_families_payload,
                    sources=sources,
                    candidates_per_family=args.entity_map_candidates_per_family,
                    cutoff_year=args.cutoff_year,
                    use_pubtator=args.use_pubtator,
                )
            subtopics_v2 = lit.decompose_axis_v2_entity_map(
                axis,
                concept_inventory=entity_concept_inventory,
                max_subtopics=args.max_subtopics_per_axis,
                max_queries_per_subtopic=args.max_queries_per_subtopic,
            )
            subtopics_payload = subtopics_v2

    supervisor_needed = not (args.stop_after_query_families or args.stop_after_subtopics or args.stop_after_synthesis or args.stop_after_evidence_slate)
    if supervisor_needed:
        supervisor = SupervisorConfigAgent().configure(args.goal, axes=1, use_literature=True, model=args.model)
        supervisor_dict = supervisor.to_dict()
        generation_supervisor_view = build_generation_supervisor_view(
            supervisor_dict,
            temporary_constraints={
                "novelty_reference_rule": f"When judging whether something is known or new, use only supplied evidence published through December 31, {args.cutoff_year}. Do not use later evidence for novelty decisions.",
            },
        )
    else:
        skipped_reasons = []
        if args.stop_after_query_families:
            skipped_reasons.append("stop_after_query_families mode")
        if args.stop_after_subtopics:
            skipped_reasons.append("stop_after_subtopics mode")
        if args.stop_after_synthesis:
            skipped_reasons.append("stop_after_synthesis mode")
        supervisor_dict = {
            "skipped": True,
            "reason": ", ".join(skipped_reasons) + ": Supervisor is not needed because no hypothesis-generation step will run. Supervisor is also never passed into LiteratureAgent or QueryReviewerAgent.",
        }
        generation_supervisor_view = {}

    out = Path(args.out_dir)
    if args.clean_out_dir and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "00_supervisor_config.json").write_text(json.dumps(supervisor_dict, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "00a_literature_input_policy.json").write_text(json.dumps({"supervisor_fields_passed_to_literature": [], "policy": "LiteratureAgent and QueryReviewerAgent receive only axis/query/subtopic/retrieved-evidence inputs. Supervisor constraints are not passed into query-family generation, query review, retrieval, or literature synthesis."}, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "00b_generation_supervisor_view.json").write_text(json.dumps(generation_supervisor_view, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "00_fixed_axis_input.json").write_text(json.dumps(axis, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "00c_subtopic_generation_context.json").write_text(json.dumps({
        "subtopic_mode": args.subtopic_mode,
        "selected_subtopic_version": args.subtopic_mode,
        "resume_from_subtopics": resume_from_subtopics,
        "resume_from_subtopics_path": str(resume_subtopics_path) if resume_subtopics_path else "",
        "resume_context_payload": resume_context_payload,
        "anchor_queries_payload": anchor_queries_payload,
        "axis_anchor_retrieval": {
            "subtopic_id": anchor_retrieval.subtopic_id,
            "queries": anchor_retrieval.queries,
            "raw_records_count": getattr(anchor_retrieval, "raw_records_count", None),
            "deduped_candidate_count": getattr(anchor_retrieval, "deduped_candidate_count", None),
            "selected_anchor_papers": [p.to_dict() for p in anchor_packets],
            "warnings": anchor_retrieval.warnings,
        } if anchor_retrieval else None,
        "raw_query_families_payload": raw_query_families_payload,
        "query_reviewer_payload": query_reviewer_payload,
        "query_families_payload": query_families_payload,
        "entity_concept_inventory": entity_concept_inventory,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    if subtopics_v2:
        (out / "01b_literature_subtopics_v2_entity_map.json").write_text(json.dumps(subtopics_v2, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "01_literature_subtopics_selected.json").write_text(json.dumps(subtopics_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.stop_after_query_families:
        llm_usage = collect_llm_usage_summary()
        query_families = query_families_payload.get("query_families", []) if isinstance(query_families_payload, dict) else []
        summary = {
            "run_id": run_id,
            "goal": args.goal,
            "fixed_axis_name": axis.get("axis_name"),
            "stage_stopped_after": "axis_query_family_generation",
            "subtopic_mode": args.subtopic_mode,
            "max_axis_query_families": args.max_axis_query_families,
            "query_reviewer_enabled": not args.disable_query_reviewer,
            "supervisor_was_run": False,
            "query_families_returned": len(query_families),
            "retrieval_was_run": False,
            "subtopic_generation_was_run": False,
            "model": args.model,
            "evidence_selector_enabled": (not args.disable_evidence_selector and not args.stop_after_evidence_slate),
        "evidence_selector_batching": "per_axis",
        "pmid_branch_tag_filter_enabled": not args.disable_pmid_branch_tag_filter,
        "stop_after_evidence_slate": args.stop_after_evidence_slate,
            "evidence_selector_initial_depth_per_query": args.evidence_selector_initial_depth_per_query,
            "llm_usage_totals": llm_usage.get("totals", {}),
        "llm_usage_by_stage": llm_usage.get("by_stage", {}),
        "llm_usage_by_model": llm_usage.get("by_model", {}),
        "top_prompt_calls": llm_usage.get("top_prompt_calls", []),
        "budget_usage": llm_usage.get("budget_usage", {}),
            "budget_usage": llm_usage.get("budget_usage", {}),
            "llm_usage_by_stage": llm_usage.get("by_stage", {}),
            "llm_usage_by_model": llm_usage.get("by_model", {}),
            "top_prompt_calls": llm_usage.get("top_prompt_calls", []),
        "budget_usage": llm_usage.get("budget_usage", {}),
            "budget_usage": llm_usage.get("budget_usage", {}),
        }
        (out / "llm_usage.json").write_text(json.dumps(llm_usage, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        log_event("workflow", "axis_query_family_generation_complete", summary)
        finalize_run(summary)
        print(json.dumps(summary, indent=2))
        return

    selected_subtopics = _subtopics(subtopics_payload)
    if not selected_subtopics:
        raise RuntimeError(
            "Subtopic generation returned zero subtopics. "
            "This usually means the LLM/mock response did not match the expected v2 subtopic schema."
        )

    if args.stop_after_subtopics:
        llm_usage = collect_llm_usage_summary()
        summary = {
            "run_id": run_id,
            "goal": args.goal,
            "fixed_axis_name": axis.get("axis_name"),
            "stage_stopped_after": "subtopic_generation_v2",
            "subtopic_mode": args.subtopic_mode,
            "selected_subtopic_version": args.subtopic_mode,
            "subtopics_returned_selected": len(selected_subtopics),
            "resume_from_subtopics": resume_from_subtopics,
            "subtopics_v2_returned": len(_subtopics(subtopics_v2)) if subtopics_v2 else None,
            "entity_map_candidates_per_family": args.entity_map_candidates_per_family,
            "candidate_pool_total_after_dedupe_cutoff": entity_concept_inventory.get("candidate_pool_total_after_dedupe_cutoff") if entity_concept_inventory else None,
            "pubtator_annotated_count": entity_concept_inventory.get("pubtator_annotated_count") if entity_concept_inventory else None,
            "llm_usage_totals": llm_usage.get("totals", {}),
        "llm_usage_by_stage": llm_usage.get("by_stage", {}),
        "llm_usage_by_model": llm_usage.get("by_model", {}),
        "top_prompt_calls": llm_usage.get("top_prompt_calls", []),
        "budget_usage": llm_usage.get("budget_usage", {}),
            "budget_usage": llm_usage.get("budget_usage", {}),
            "llm_usage_by_stage": llm_usage.get("by_stage", {}),
            "llm_usage_by_model": llm_usage.get("by_model", {}),
            "top_prompt_calls": llm_usage.get("top_prompt_calls", []),
        "budget_usage": llm_usage.get("budget_usage", {}),
            "budget_usage": llm_usage.get("budget_usage", {}),
        }
        (out / "llm_usage.json").write_text(json.dumps(llm_usage, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        log_event("workflow", "v69_1_subtopic_generation_v2_complete", summary)
        finalize_run(summary)
        print(json.dumps(summary, indent=2))
        return

    retrieval_results = []
    packets_by_subtopic = {}
    for subtopic in selected_subtopics:
        sid = str(subtopic.get("subtopic_id") or "").strip()
        queries = [str(q).strip() for q in subtopic.get("search_queries", []) if str(q).strip()]
        if args.disable_retrieval:
            from agents.literature_agent import SubtopicRetrievalResult
            result = SubtopicRetrievalResult(subtopic_id=sid, queries=queries, warnings=["retrieval disabled"])
        else:
            result = lit.retrieve_subtopic(
                subtopic_id=sid,
                queries=queries,
                sources=sources,
                max_queries=args.max_queries_per_subtopic,
                raw_papers_per_source_query=args.raw_papers_per_source_query,
                ai_papers_per_subtopic=args.ai_papers_per_subtopic,
                cutoff_year=args.cutoff_year,
                use_pubtator=args.use_pubtator,
                pubtator_max_candidates_per_subtopic=args.pubtator_max_candidates_per_subtopic,
                use_evidence_selector=(not args.disable_evidence_selector and not args.stop_after_evidence_slate),
                defer_evidence_selector=(not args.disable_evidence_selector and not args.stop_after_evidence_slate),
                evidence_selector_model=args.evidence_selector_model or args.model,
                evidence_selector_initial_depth_per_query=args.evidence_selector_initial_depth_per_query,
                evidence_selector_max_depth_per_query=args.evidence_selector_max_depth_per_query,
                enable_pmid_branch_tag_filter=not args.disable_pmid_branch_tag_filter,
                subtopic_payload=subtopic,
            )
        retrieval_results.append(result)
        packets_by_subtopic[sid] = result.evidence_packets

    if (not args.disable_retrieval) and (not args.disable_evidence_selector) and (not args.stop_after_evidence_slate):
        lit.select_axis_evidence_batch(
            axis=axis,
            subtopics_payload=subtopics_payload,
            retrieval_results=retrieval_results,
            ai_papers_per_subtopic=args.ai_papers_per_subtopic,
            evidence_selector_model=args.evidence_selector_model or args.model,
        )
        packets_by_subtopic = {r.subtopic_id: r.evidence_packets for r in retrieval_results}

    if args.stop_after_evidence_slate:
        llm_usage = collect_llm_usage_summary()
        retrieval_payload = {
            "cutoff_year": args.cutoff_year,
            "sources": sources,
            "use_pubtator": args.use_pubtator,
            "pmid_branch_tag_filter_enabled": not args.disable_pmid_branch_tag_filter,
            "evidence_selector_enabled": False,
            "stop_after_evidence_slate": True,
            "evidence_selector_initial_depth_per_query": args.evidence_selector_initial_depth_per_query,
            "evidence_selector_max_depth_per_query": args.evidence_selector_max_depth_per_query,
            "subtopic_retrieval": [
                {
                    "subtopic_id": r.subtopic_id,
                    "queries": r.queries,
                    "raw_records_count": getattr(r, "raw_records_count", None),
                    "deduped_candidate_count": getattr(r, "deduped_candidate_count", None),
                    "pubtator_annotated_count": getattr(r, "pubtator_annotated_count", None),
                    "raw_candidate_pool_by_query": getattr(r, "raw_candidate_pool_by_query", {}),
                    "candidate_pool_by_query": getattr(r, "candidate_pool_by_query", {}),
                    "filtered_candidate_pool_by_query": getattr(r, "filtered_candidate_pool_by_query", {}),
                    "pmid_branch_filter_excluded": getattr(r, "pmid_branch_filter_excluded", {}),
                    "non_pmid_sanity_excluded": getattr(r, "non_pmid_sanity_excluded", {}),
                    "deduped_candidate_pool": getattr(r, "deduped_candidate_pool", []),
                    "balanced_candidate_slate": getattr(r, "balanced_candidate_slate", []),
                    "evidence_selector_payload": getattr(r, "evidence_selector_payload", {}),
                    "resolved_evidence_selection": getattr(r, "resolved_evidence_selection", {}),
                    "warnings": r.warnings,
                }
                for r in retrieval_results
            ],
        }
        (out / "02_retrieval_results.json").write_text(json.dumps(retrieval_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "llm_usage.json").write_text(json.dumps(llm_usage, indent=2, ensure_ascii=False), encoding="utf-8")
        summary = {
            "run_id": run_id,
            "goal": args.goal,
            "fixed_axis_name": axis.get("axis_name"),
            "stage_stopped_after": "filtered_evidence_slate",
            "subtopic_mode": args.subtopic_mode,
            "sources": sources,
            "use_pubtator": args.use_pubtator,
            "pmid_branch_tag_filter_enabled": not args.disable_pmid_branch_tag_filter,
            "evidence_selector_enabled": False,
            "resume_from_subtopics": resume_from_subtopics,
            "model": args.model,
            "llm_usage_totals": llm_usage.get("totals", {}),
        "llm_usage_by_stage": llm_usage.get("by_stage", {}),
        "llm_usage_by_model": llm_usage.get("by_model", {}),
        "top_prompt_calls": llm_usage.get("top_prompt_calls", []),
        "budget_usage": llm_usage.get("budget_usage", {}),
            "budget_usage": llm_usage.get("budget_usage", {}),
            "llm_usage_by_stage": llm_usage.get("by_stage", {}),
            "llm_usage_by_model": llm_usage.get("by_model", {}),
            "top_prompt_calls": llm_usage.get("top_prompt_calls", []),
        "budget_usage": llm_usage.get("budget_usage", {}),
            "budget_usage": llm_usage.get("budget_usage", {}),
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        log_event("workflow", "filtered_evidence_slate_complete", summary)
        finalize_run(summary)
        print(json.dumps(summary, indent=2))
        return

    selected_packets = _select_balanced_packets_by_subtopic(
        packets_by_subtopic,
        max_total=args.ai_papers_per_axis,
    )
    synthesis_input_payload = {
        "axis_id": axis.get("axis_id"),
        "subtopics_payload": subtopics_payload,
        "selected_packet_ids": [p.evidence_id for p in selected_packets],
        "selected_packets": [p.to_dict() for p in selected_packets],
        "selector_branch_metadata_by_subtopic": {
            r.subtopic_id: {
                "evidence_selector_payload": getattr(r, "evidence_selector_payload", {}),
                "resolved_evidence_selection": getattr(r, "resolved_evidence_selection", {}),
            }
            for r in retrieval_results
        },
    }
    paper_memory = build_paper_memory(
        retrieval_results=retrieval_results,
        selected_packets=selected_packets,
        axis_id=str(axis.get("axis_id") or ""),
        axis_name=str(axis.get("axis_name") or ""),
        cutoff_year=args.cutoff_year,
    )
    paper_memory_compact = compact_memory_for_reflection(paper_memory, max_entries=45)
    axis_synthesis = lit.synthesize_axis(axis, subtopics_payload, selected_packets)
    hypotheses_payload = None
    reflection_payload = None

    # Subtopic-generation files were already written above.
    (out / "02_retrieval_results.json").write_text(json.dumps({
        "cutoff_year": args.cutoff_year,
        "ai_papers_per_subtopic": args.ai_papers_per_subtopic,
        "ai_papers_per_axis_safety_cap": args.ai_papers_per_axis,
        "use_pubtator": args.use_pubtator,
        "pubtator_max_candidates_per_subtopic": args.pubtator_max_candidates_per_subtopic if args.use_pubtator else None,
        "evidence_selector_enabled": (not args.disable_evidence_selector and not args.stop_after_evidence_slate),
        "evidence_selector_batching": "per_axis",
        "pmid_branch_tag_filter_enabled": not args.disable_pmid_branch_tag_filter,
        "stop_after_evidence_slate": args.stop_after_evidence_slate,
        "evidence_selector_model": args.evidence_selector_model or args.model,
        "evidence_selector_initial_depth_per_query": args.evidence_selector_initial_depth_per_query,
        "evidence_selector_max_depth_per_query": args.evidence_selector_max_depth_per_query,
        "selected_total_for_axis_synthesis": len(selected_packets),
        "evidence_selection_by_subtopic": {
            r.subtopic_id: sum(1 for p in selected_packets if p.metadata.get("selected_from_subtopic") == r.subtopic_id)
            for r in retrieval_results
        },
        "subtopic_retrieval": [
            {
                "subtopic_id": r.subtopic_id,
                "queries": r.queries,
                "raw_records_count": getattr(r, "raw_records_count", None),
                "deduped_candidate_count": getattr(r, "deduped_candidate_count", None),
                "pubtator_annotated_count": getattr(r, "pubtator_annotated_count", None),
                "raw_candidate_pool_by_query": getattr(r, "raw_candidate_pool_by_query", {}),
                "candidate_pool_by_query": getattr(r, "candidate_pool_by_query", {}),
                "filtered_candidate_pool_by_query": getattr(r, "filtered_candidate_pool_by_query", {}),
                "pmid_branch_filter_excluded": getattr(r, "pmid_branch_filter_excluded", {}),
                "non_pmid_sanity_excluded": getattr(r, "non_pmid_sanity_excluded", {}),
                "deduped_candidate_pool": getattr(r, "deduped_candidate_pool", []),
                "balanced_candidate_slate": getattr(r, "balanced_candidate_slate", []),
                "evidence_selector_payload": getattr(r, "evidence_selector_payload", {}),
                "resolved_evidence_selection": getattr(r, "resolved_evidence_selection", {}),
                "records_returned_to_ai_before_axis_cap": len(r.evidence_packets),
                "selected_for_axis_synthesis": sum(1 for p in selected_packets if p.metadata.get("selected_from_subtopic") == r.subtopic_id),
                "evidence_packets": [p.to_dict() for p in r.evidence_packets],
                "warnings": r.warnings,
            }
            for r in retrieval_results
        ],
        "selected_evidence_packets": [p.to_dict() for p in selected_packets],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "02a_paper_memory.json").write_text(json.dumps(paper_memory, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "02b_paper_memory_compact_for_reflection.json").write_text(json.dumps(paper_memory_compact, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "03a_axis_synthesis_inputs.json").write_text(json.dumps(synthesis_input_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "03_axis_literature_synthesis.json").write_text(json.dumps(axis_synthesis, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.stop_after_synthesis:
        llm_usage = collect_llm_usage_summary()
        (out / "llm_usage.json").write_text(json.dumps(llm_usage, indent=2, ensure_ascii=False), encoding="utf-8")
        summary = {
            "run_id": run_id,
            "goal": args.goal,
            "fixed_axis_name": axis.get("axis_name"),
            "stage_stopped_after": "axis_literature_synthesis",
            "subtopic_mode": args.subtopic_mode,
            "selected_subtopic_version": args.subtopic_mode,
            "max_axis_query_families": args.max_axis_query_families if args.subtopic_mode in {"v2", "compare"} else 0,
            "entity_map_candidates_per_family": args.entity_map_candidates_per_family if args.subtopic_mode in {"v2", "compare"} else 0,
            "candidate_pool_total_after_dedupe_cutoff": entity_concept_inventory.get("candidate_pool_total_after_dedupe_cutoff") if entity_concept_inventory else None,
            "entity_map_pubtator_annotated_count": entity_concept_inventory.get("pubtator_annotated_count") if entity_concept_inventory else None,
            "selected_total_for_axis_synthesis": len(selected_packets),
            "resume_from_subtopics": resume_from_subtopics,
            "model": args.model,
            "llm_usage_totals": llm_usage.get("totals", {}),
        "llm_usage_by_stage": llm_usage.get("by_stage", {}),
        "llm_usage_by_model": llm_usage.get("by_model", {}),
        "top_prompt_calls": llm_usage.get("top_prompt_calls", []),
        "budget_usage": llm_usage.get("budget_usage", {}),
            "budget_usage": llm_usage.get("budget_usage", {}),
            "llm_usage_by_stage": llm_usage.get("by_stage", {}),
            "llm_usage_by_model": llm_usage.get("by_model", {}),
            "top_prompt_calls": llm_usage.get("top_prompt_calls", []),
        "budget_usage": llm_usage.get("budget_usage", {}),
            "budget_usage": llm_usage.get("budget_usage", {}),
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        log_event("workflow", "single_axis_literature_synthesis_complete", summary)
        finalize_run(summary)
        print(json.dumps(summary, indent=2))
        return

    hypotheses_payload = generate_single_axis_hypotheses(
        objective=args.goal,
        axis=axis,
        subtopics_payload=subtopics_payload,
        axis_synthesis=axis_synthesis,
        cutoff_year=args.cutoff_year,
        model=args.model,
        generation_supervisor_view=generation_supervisor_view,
    )
    (out / "04_axis_local_generation_hypotheses.json").write_text(json.dumps(hypotheses_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.run_reflection:
        reflection = ReflectionAgent(model=args.reflection_model or args.model)
        reflection_payload = reflection.review_axis_hypotheses(
            supervisor_config=supervisor_dict,
            generation_supervisor_view=generation_supervisor_view,
            axis=axis,
            axis_synthesis=axis_synthesis,
            hypotheses_payload=hypotheses_payload if isinstance(hypotheses_payload, dict) else {},
            paper_memory_compact=paper_memory_compact,
        )
        (out / "05_reflection_supervisor_guided_reviews.json").write_text(json.dumps(reflection_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    llm_usage = collect_llm_usage_summary()
    (out / "llm_usage.json").write_text(json.dumps(llm_usage, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "run_id": run_id,
        "goal": args.goal,
        "fixed_axis_name": axis.get("axis_name"),
        "stage_stopped_after": "reflection" if args.stop_after_reflection and args.run_reflection else "axis_local_generation_hypotheses",
        "subtopic_mode": args.subtopic_mode,
        "selected_subtopic_version": args.subtopic_mode,
        "resume_from_subtopics": resume_from_subtopics,
        "resume_from_subtopics_path": str(resume_subtopics_path) if resume_subtopics_path else "",
        "resume_context_payload": resume_context_payload,
        "axis_anchor_papers": args.axis_anchor_papers if args.subtopic_mode == "anchor" else 0,
        "max_axis_anchor_queries": args.max_axis_anchor_queries if args.subtopic_mode == "anchor" else 0,
        "max_axis_query_families": args.max_axis_query_families if args.subtopic_mode in {"v2", "compare"} else 0,
        "entity_map_candidates_per_family": args.entity_map_candidates_per_family if args.subtopic_mode in {"v2", "compare"} else 0,
        "candidate_pool_total_after_dedupe_cutoff": entity_concept_inventory.get("candidate_pool_total_after_dedupe_cutoff") if entity_concept_inventory else None,
        "entity_map_pubtator_annotated_count": entity_concept_inventory.get("pubtator_annotated_count") if entity_concept_inventory else None,
        "model": args.model,
        "sources": sources,
        "cutoff_year": args.cutoff_year,
        "max_subtopics_per_axis": args.max_subtopics_per_axis,
        "subtopics_returned": len(_subtopics(subtopics_payload)),
        "max_queries_per_subtopic": args.max_queries_per_subtopic,
        "raw_papers_per_source_query": args.raw_papers_per_source_query,
        "ai_papers_per_subtopic": args.ai_papers_per_subtopic,
        "ai_papers_per_axis_safety_cap": args.ai_papers_per_axis,
        "use_pubtator": args.use_pubtator,
        "pubtator_max_candidates_per_subtopic": args.pubtator_max_candidates_per_subtopic if args.use_pubtator else None,
        "evidence_selector_enabled": (not args.disable_evidence_selector and not args.stop_after_evidence_slate),
        "evidence_selector_batching": "per_axis",
        "pmid_branch_tag_filter_enabled": not args.disable_pmid_branch_tag_filter,
        "stop_after_evidence_slate": args.stop_after_evidence_slate,
        "evidence_selector_model": args.evidence_selector_model or args.model,
        "evidence_selector_initial_depth_per_query": args.evidence_selector_initial_depth_per_query,
        "evidence_selector_max_depth_per_query": args.evidence_selector_max_depth_per_query,
        "selected_total_for_axis_synthesis": len(selected_packets),
        "resume_from_subtopics": resume_from_subtopics,
        "hypotheses_returned": len(hypotheses_payload.get("hypotheses", [])) if isinstance(hypotheses_payload, dict) else None,
        "paper_memory_entries": paper_memory.get("counts", {}).get("memory_entries") if isinstance(paper_memory, dict) else None,
        "paper_memory_used_in_synthesis": paper_memory.get("counts", {}).get("used_in_synthesis") if isinstance(paper_memory, dict) else None,
        "reflection_enabled": bool(args.run_reflection),
        "reflection_reviews_returned": len(reflection_payload.get("reflection_reviews", [])) if isinstance(reflection_payload, dict) else None,
        "llm_usage_totals": llm_usage.get("totals", {}),
        "llm_usage_by_stage": llm_usage.get("by_stage", {}),
        "llm_usage_by_model": llm_usage.get("by_model", {}),
        "top_prompt_calls": llm_usage.get("top_prompt_calls", []),
        "budget_usage": llm_usage.get("budget_usage", {}),
            "budget_usage": llm_usage.get("budget_usage", {}),
            "llm_usage_by_stage": llm_usage.get("by_stage", {}),
            "llm_usage_by_model": llm_usage.get("by_model", {}),
            "top_prompt_calls": llm_usage.get("top_prompt_calls", []),
        "budget_usage": llm_usage.get("budget_usage", {}),
            "budget_usage": llm_usage.get("budget_usage", {}),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log_event("workflow", "v58_axisfirst_entity_map_single_axis_complete", summary)
    finalize_run(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
