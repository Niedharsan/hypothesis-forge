from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.supervisor_config_agent import SupervisorConfigAgent
from llm.provider import ask_llm_json
from agents.generation_rewired import RewiredGenerationAgent
from agents.literature_agent import LiteratureAgent
from agents.reflection_agent import ReflectionAgent
from agents.proximity_agent import ProximityAgent, build_hypotheses_payload_from_proximity
from runtime.context import configure_runtime
from utils.config import load_config, validate_config
from utils.run_logger import start_run_log, finalize_run, log_event, collect_llm_usage_summary
from utils.supervisor_views import build_generation_supervisor_view
from utils.evidence_memory import build_paper_memory, compact_memory_for_reflection

DEFAULT_SOURCES = "PubMed,EuropePMC,OpenAlex,Crossref"
SUPPORTED_SOURCES = {"PubMed", "EuropePMC", "OpenAlex", "Crossref", "SemanticScholar"}


def parse_sources(raw: str) -> list[str]:
    values = [s.strip() for s in raw.split(",") if s.strip()]
    for s in values:
        if s not in SUPPORTED_SOURCES:
            raise SystemExit(f"Unsupported source: {s}. Supported: {sorted(SUPPORTED_SOURCES)}")
    return values


def axis_success_criteria_preset(preset: str) -> list[str]:
    """Return v52 criteria variants.

    v52 tests supervisor-controlled granularity for criterion 2 only versus
    supervisor-controlled granularity for both criterion 1 and criterion 2.
    The functional/mechanistic route variant intentionally removes the older
    "appropriate axis granularity" line to avoid duplicated granularity pressure.
    """
    presets = {
        # Original v39 criteria, with supervisor controlling criterion 2 only.
        "v52_v39_mech_granularity": [
            "biological diversity",
            "mechanistic specificity",
            "non-overlap between parent axes",
            "novelty potential",
            "testability",
            "clear falsifiable predictions",
        ],
        # High-ceiling functional/mechanistic wording, without the old duplicate
        # "appropriate axis granularity" criterion, with supervisor controlling criterion 2 only.
        "v52_functional_mech_granularity": [
            "biological diversity across distinct functional and mechanistic routes",
            "mechanistic specificity",
            "non-overlap between parent axes",
            "novelty potential",
            "testability",
            "clear falsifiable predictions",
        ],
        # Original v39 criteria, with supervisor controlling both criterion 1 and criterion 2.
        "v52_v39_bio_and_mech_granularity": [
            "biological diversity",
            "mechanistic specificity",
            "non-overlap between parent axes",
            "novelty potential",
            "testability",
            "clear falsifiable predictions",
        ],
        # Functional/mechanistic wording, with supervisor controlling both criterion 1 and criterion 2.
        "v52_functional_bio_and_mech_granularity": [
            "biological diversity across distinct functional and mechanistic routes",
            "mechanistic specificity",
            "non-overlap between parent axes",
            "novelty potential",
            "testability",
            "clear falsifiable predictions",
        ],
    }
    if preset not in presets:
        raise SystemExit(f"Unsupported --axis-criteria-preset: {preset}. Choose one of: {', '.join(sorted(presets))}")
    return presets[preset]


def preset_uses_criterion_1_granularity(preset: str) -> bool:
    return preset in {
        "v52_v39_bio_and_mech_granularity",
        "v52_functional_bio_and_mech_granularity",
    }


BIO_GRANULARITY_LEVELS = {
    "broad_level": "Use broad biological domains when the user goal is mainly asking for overview, field mapping, or high-level coverage.",
    "route_level": "Use distinct biological research routes when the user goal is discovery-oriented and each axis should open a different search, experiment, readout, model, or hypothesis path.",
    "subroute_level": "Use narrower biological branches when the user goal is already focused on a specific process, mechanism, system, or hypothesis space.",
    "multi_scale_level": "Use a deliberate mix of broad domains and narrower routes when the user goal requires both field coverage and discovery of hidden or underexplored routes.",
}

BIO_GRANULARITY_INSTRUCTIONS = {
    "broad_level": (
        "Criterion 1 granularity: Interpret biological diversity at broad-level granularity. "
        "Axes may represent major biological domains when high-level field coverage is most useful."
    ),
    "route_level": (
        "Criterion 1 granularity: Interpret biological diversity at route-level granularity. "
        "Axes should represent distinct biological research routes rather than only broad textbook categories."
    ),
    "subroute_level": (
        "Criterion 1 granularity: Interpret biological diversity at subroute-level granularity. "
        "Axes may focus on narrower biological branches when fine-grained hypothesis generation is most useful."
    ),
    "multi_scale_level": (
        "Criterion 1 granularity: Interpret biological diversity at multi-scale granularity. "
        "Use broad axes where field coverage is needed and narrower axes where distinct research routes would otherwise be hidden."
    ),
}

MECH_GRANULARITY_LEVELS = {
    "broad_mechanism_level": "Use broad mechanism families when the user goal needs high-level coverage more than detailed pathway separation.",
    "pathway_level": "Use named pathways or processes when the user goal needs mechanisms that can guide search, testing, or hypothesis generation.",
    "branch_level": "Use mechanistic branches within pathways or processes when the user goal needs fine-grained routes that may lead to different searches, perturbations, readouts, or hypotheses.",
    "multi_scale_level": "Use a deliberate mix of broad mechanisms, named pathways, and narrower branches when the user goal needs both coverage and discovery of hidden mechanistic routes.",
}

MECH_GRANULARITY_INSTRUCTIONS = {
    "broad_mechanism_level": (
        "Criterion 2 granularity: Interpret mechanistic specificity at broad-mechanism granularity. "
        "Axes may describe broad mechanism families when high-level coverage is most useful."
    ),
    "pathway_level": (
        "Criterion 2 granularity: Interpret mechanistic specificity at pathway-level granularity. "
        "Axes should name relevant biological pathways or processes when they help distinguish different search, testing, or hypothesis paths."
    ),
    "branch_level": (
        "Criterion 2 granularity: Interpret mechanistic specificity at branch-level granularity. "
        "When an axis contains a pathway or process, include the main mechanistic branches needed to distinguish different search, perturbation, readout, or hypothesis paths."
    ),
    "multi_scale_level": (
        "Criterion 2 granularity: Interpret mechanistic specificity at multi-scale granularity. "
        "Use broad mechanisms where coverage is needed, and narrower branches where important mechanistic routes would otherwise be hidden."
    ),
}


def determine_criterion_1_granularity(
    *,
    objective: str,
    supervisor_config: dict,
    model: str = "gemini-2.5-flash-lite",
) -> dict:
    """Ask the LLM for only the granularity level of criterion 1.

    The returned enum is mapped to fixed neutral text before axis generation.
    Raw supervisor prose is not passed to the Generation Agent.
    """
    objective_type = str(supervisor_config.get("objective_type") or "").strip()
    target_context = str(supervisor_config.get("target_context") or "").strip()
    levels_text = "\n".join(f"- {k}: {v}" for k, v in BIO_GRANULARITY_LEVELS.items())
    prompt = f"""Determine only the granularity level for criterion 1: "biological diversity".

Use the user goal and the stable supervisor fields only to decide how broad or specific biological diversity should be for axis generation.

User goal:
{objective}

Stable supervisor fields:
objective_type: {objective_type}
target_context: {target_context}

Allowed values and definitions:
{levels_text}

Return exactly one JSON object with one field:
criterion_1_granularity_level

Allowed values:
- broad_level
- route_level
- subroute_level
- multi_scale_level

Do not include explanations, disease names, pathways, targets, drug classes, organisms, candidate mechanisms, candidate interventions, or example axes.

Expected strict JSON shape:
{{"criterion_1_granularity_level": "route_level"}}
"""
    payload = ask_llm_json(prompt, model=model, agent="supervisor", purpose="criterion_1_biological_diversity_granularity")
    level = str(payload.get("criterion_1_granularity_level") or "").strip()
    if level not in BIO_GRANULARITY_INSTRUCTIONS:
        raise RuntimeError(
            "Supervisor returned invalid criterion_1_granularity_level: "
            f"{level!r}. Expected one of: {', '.join(BIO_GRANULARITY_INSTRUCTIONS)}"
        )
    return {
        "criterion_1_granularity_level": level,
        "criterion_1_granularity_instruction": BIO_GRANULARITY_INSTRUCTIONS[level],
    }


def determine_criterion_2_granularity(
    *,
    objective: str,
    supervisor_config: dict,
    model: str = "gemini-2.5-flash-lite",
) -> dict:
    """Ask the LLM for only the granularity level of criterion 2.

    The returned enum is mapped to fixed neutral text before axis generation.
    Raw supervisor prose is not passed to the Generation Agent.
    """
    objective_type = str(supervisor_config.get("objective_type") or "").strip()
    target_context = str(supervisor_config.get("target_context") or "").strip()
    levels_text = "\n".join(f"- {k}: {v}" for k, v in MECH_GRANULARITY_LEVELS.items())
    prompt = f"""Determine only the granularity level for criterion 2: "mechanistic specificity".

Use the user goal and the stable supervisor fields only to decide how broad or detailed mechanistic specificity should be for axis generation.

User goal:
{objective}

Stable supervisor fields:
objective_type: {objective_type}
target_context: {target_context}

Allowed values and definitions:
{levels_text}

Return exactly one JSON object with one field:
criterion_2_granularity_level

Allowed values:
- broad_mechanism_level
- pathway_level
- branch_level
- multi_scale_level

Do not include explanations, disease names, pathways, targets, drug classes, organisms, candidate mechanisms, candidate interventions, or example axes.

Expected strict JSON shape:
{{"criterion_2_granularity_level": "branch_level"}}
"""
    payload = ask_llm_json(prompt, model=model, agent="supervisor", purpose="criterion_2_mechanistic_specificity_granularity")
    level = str(payload.get("criterion_2_granularity_level") or "").strip()
    if level not in MECH_GRANULARITY_INSTRUCTIONS:
        raise RuntimeError(
            "Supervisor returned invalid criterion_2_granularity_level: "
            f"{level!r}. Expected one of: {', '.join(MECH_GRANULARITY_INSTRUCTIONS)}"
        )
    return {
        "criterion_2_granularity_level": level,
        "criterion_2_granularity_instruction": MECH_GRANULARITY_INSTRUCTIONS[level],
    }


def minimal_generation_supervisor_view(
    supervisor_config: dict,
    *,
    axis_criteria_preset: str = "v52_v39_mech_granularity",
    criterion_1_granularity: dict | None = None,
    criterion_2_granularity: dict | None = None,
) -> dict:
    """Return the compact v39-style view used by axis generation.

    The raw user goal is already passed to the axis prompt as `objective`.
    Do not pass LLM-paraphrased goal_summary, objective_type, target_context,
    constraints, generation_guidance, literature_guidance, or reflection_guidance
    into axis generation. Only fixed neutral granularity instructions are passed.
    """
    model = (supervisor_config.get("models") or {}).get("generation", "gemini-2.5-flash-lite")
    return {
        "axis_criteria_preset": axis_criteria_preset,
        "success_criteria": axis_success_criteria_preset(axis_criteria_preset),
        "criterion_1_granularity_level": (criterion_1_granularity or {}).get("criterion_1_granularity_level"),
        "criterion_1_granularity_instruction": (criterion_1_granularity or {}).get("criterion_1_granularity_instruction"),
        "criterion_2_granularity_level": (criterion_2_granularity or {}).get("criterion_2_granularity_level"),
        "criterion_2_granularity_instruction": (criterion_2_granularity or {}).get("criterion_2_granularity_instruction"),
        "generation_config": {
            "axis_count": 10,
            "hypothesis_count_policy": "dynamic; generate as many useful hypotheses as supported by each route",
            "use_literature": True,
            "model": model,
            "web_search": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v69 full axis-first v2 literature-to-hypothesis workflow")
    parser.add_argument("--goal", required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--out-dir", default="runs/v69_full_reflection_cutoff2023")
    parser.add_argument("--clean-out-dir", action="store_true", help="Delete --out-dir before writing new outputs. Use for clean reruns only.")
    parser.add_argument("--sources", default=DEFAULT_SOURCES)
    parser.add_argument("--axes", type=int, default=10)
    parser.add_argument("--max-subtopics-per-axis", type=int, default=5)
    parser.add_argument("--max-queries-per-subtopic", type=int, default=5)
    parser.add_argument("--raw-papers-per-source-query", type=int, default=5)
    parser.add_argument("--cutoff-year", type=int, default=2023)
    parser.add_argument("--use-pubtator", action="store_true")
    parser.add_argument("--max-axis-query-families", type=int, default=6)
    parser.add_argument("--entity-map-candidates-per-family", type=int, default=10)
    parser.add_argument("--disable-query-reviewer", action="store_true")
    parser.add_argument("--query-reviewer-model", default="")
    parser.add_argument("--ai-papers-per-subtopic", type=int, default=3)
    parser.add_argument("--ai-papers-per-axis", type=int, default=15)
    parser.add_argument("--disable-evidence-selector", action="store_true", help="Disable branch-aware EvidenceSelector before synthesis. Enabled by default.")
    parser.add_argument("--evidence-selector-model", default="")
    parser.add_argument("--evidence-selector-initial-depth-per-query", type=int, default=3)
    parser.add_argument("--evidence-selector-max-depth-per-query", type=int, default=10)
    parser.add_argument("--disable-pmid-branch-tag-filter", action="store_true")
    parser.add_argument("--disable-retrieval", action="store_true")
    parser.add_argument("--model", default="gemini-2.5-flash-lite")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-llm-calls", type=int, default=200, help="Safety cap for LLM calls in one run. Full v2 10-axis runs still need >30; v69.1 batches EvidenceSelector once per axis.")
    parser.add_argument("--disable-reflection", action="store_true")
    parser.add_argument("--disable-proximity", action="store_true", help="Disable Proximity clustering/merge/salvage before Reflection. Enabled by default.")
    parser.add_argument("--proximity-model", default="", help="Model for ProximityAgent; defaults to --model.")
    parser.add_argument("--max-proximity-survivors", type=int, default=None, help="Deprecated compatibility flag; Proximity no longer uses a survivor target.")
    parser.add_argument("--max-proximity-focus-seeds", type=int, default=8, help="Target cap for focused seeds emitted by Proximity for later Generation.")
    parser.add_argument("--stop-after", choices=["axes", "subtopics", "proximity", "reflection"], default=None, help="Stop after the requested stage. Supports: axes, subtopics, proximity, reflection")
    parser.add_argument("--axis-criteria-preset", default="v52_v39_mech_granularity", choices=["v52_v39_mech_granularity", "v52_functional_mech_granularity", "v52_v39_bio_and_mech_granularity", "v52_functional_bio_and_mech_granularity"], help="Axis-generation success-criteria preset for v52 supervised granularity testing")
    args = parser.parse_args()

    if args.axes != 10:
        raise SystemExit("v35 keeps --axes fixed at 10 for this diagnostic")
    if args.stop_after == "reflection" and args.disable_reflection:
        raise SystemExit("--stop-after reflection requires reflection to be enabled; remove --disable-reflection")
    if args.stop_after == "proximity" and args.disable_proximity:
        raise SystemExit("--stop-after proximity requires Proximity to be enabled; remove --disable-proximity")
    if args.max_subtopics_per_axis < 1 or args.max_subtopics_per_axis > 5:
        raise SystemExit("--max-subtopics-per-axis must be 1–5")
    if args.max_queries_per_subtopic < 1 or args.max_queries_per_subtopic > 5:
        raise SystemExit("--max-queries-per-subtopic must be 1–5")
    if args.raw_papers_per_source_query < 1 or args.raw_papers_per_source_query > 10:
        raise SystemExit("--raw-papers-per-source-query must be 1–10")
    if args.max_axis_query_families < 1 or args.max_axis_query_families > 8:
        raise SystemExit("--max-axis-query-families must be 1–8")
    if args.entity_map_candidates_per_family < 1 or args.entity_map_candidates_per_family > 25:
        raise SystemExit("--entity-map-candidates-per-family must be 1–25")
    if args.evidence_selector_initial_depth_per_query < 1 or args.evidence_selector_initial_depth_per_query > 10:
        raise SystemExit("--evidence-selector-initial-depth-per-query must be 1–10")
    if args.evidence_selector_max_depth_per_query < 1 or args.evidence_selector_max_depth_per_query > 25:
        raise SystemExit("--evidence-selector-max-depth-per-query must be 1–25")
    if args.ai_papers_per_subtopic < 1 or args.ai_papers_per_subtopic > 10:
        raise SystemExit("--ai-papers-per-subtopic must be 1–10")
    if args.ai_papers_per_axis < 1 or args.ai_papers_per_axis > 30:
        raise SystemExit("--ai-papers-per-axis must be 1–30")
    if args.max_llm_calls < 1 or args.max_llm_calls > 500:
        raise SystemExit("--max-llm-calls must be 1–500")
    if args.max_proximity_focus_seeds < 0 or args.max_proximity_focus_seeds > 30:
        raise SystemExit("--max-proximity-focus-seeds must be 0–30")

    out = Path(args.out_dir)
    if args.clean_out_dir and out.exists():
        import shutil
        shutil.rmtree(out)

    config = load_config(args.config)
    config.setdefault("runtime", {}).setdefault("limits", {})["max_llm_calls_per_run"] = args.max_llm_calls
    warnings = validate_config(config, runtime_mode="dry_run" if args.dry_run else None, strict=False)
    configure_runtime(config, mode_override="dry_run" if args.dry_run else None)
    run_id = start_run_log(args.goal, "v73_full_axis_first_v2_v70_granularity_selector_pool_nonredundancy")
    for warning in warnings:
        log_event("config", "warning", {"warning": warning}, status="warning")

    sources = parse_sources(args.sources)
    supervisor = SupervisorConfigAgent().configure(
        args.goal,
        axes=args.axes,
        use_literature=not args.disable_retrieval,
        model=args.model,
    )

    generator = RewiredGenerationAgent(model=args.model)
    supervisor_dict = supervisor.to_dict()
    hypothesis_generation_supervisor_view = build_generation_supervisor_view(supervisor_dict)
    criterion_1_granularity = None
    if preset_uses_criterion_1_granularity(args.axis_criteria_preset):
        criterion_1_granularity = determine_criterion_1_granularity(
            objective=args.goal,
            supervisor_config=supervisor_dict,
            model=args.model,
        )
    criterion_2_granularity = determine_criterion_2_granularity(
        objective=args.goal,
        supervisor_config=supervisor_dict,
        model=args.model,
    )
    axis_supervisor_view = minimal_generation_supervisor_view(
        supervisor_dict,
        axis_criteria_preset=args.axis_criteria_preset,
        criterion_1_granularity=criterion_1_granularity,
        criterion_2_granularity=criterion_2_granularity,
    )
    axes_payload = generator.generate_axes(args.goal, axis_supervisor_view)

    if args.stop_after == "axes":
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "01_supervisor_config.json").write_text(json.dumps(supervisor_dict, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "01b_axis_generation_supervisor_view.json").write_text(json.dumps(axis_supervisor_view, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "01c_literature_input_policy.json").write_text(json.dumps({"supervisor_fields_passed_to_literature": [], "policy": "Full axis-first LiteratureAgent uses the v2 branch-preserving route only: query families -> QueryReviewer -> entity/concept inventory -> v2 entity-map subtopics -> targeted retrieval -> axis-batched EvidenceSelector -> synthesis. Supervisor constraints are not passed to retrieval or synthesis."}, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "01d_hypothesis_generation_supervisor_view.json").write_text(json.dumps(hypothesis_generation_supervisor_view, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "02_generation_axes.json").write_text(json.dumps(axes_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        llm_usage = collect_llm_usage_summary()
        (out / "llm_usage.json").write_text(json.dumps(llm_usage, indent=2, ensure_ascii=False), encoding="utf-8")
        summary = {
            "run_id": run_id,
            "goal": args.goal,
            "model": args.model,
            "stage_stopped_after": "axes",
            "axis_criteria_preset": args.axis_criteria_preset,
            "criterion_1_granularity_level": (criterion_1_granularity or {}).get("criterion_1_granularity_level"),
            "criterion_1_granularity_instruction": (criterion_1_granularity or {}).get("criterion_1_granularity_instruction"),
            "criterion_2_granularity_level": criterion_2_granularity.get("criterion_2_granularity_level"),
            "criterion_2_granularity_instruction": criterion_2_granularity.get("criterion_2_granularity_instruction"),
            "axes_fixed": args.axes,
            "axes_returned": len(axes_payload.get("axes", [])) if isinstance(axes_payload, dict) else None,
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
        log_event("workflow", "v69_axes_complete", summary)
        finalize_run(summary)
        print(json.dumps(summary, indent=2))
        return

    literature = LiteratureAgent(model=args.model).run_axis_first(
        objective=args.goal,
        axes_payload=axes_payload,
        sources=sources,
        max_subtopics_per_axis=args.max_subtopics_per_axis,
        max_queries_per_subtopic=args.max_queries_per_subtopic,
        raw_papers_per_source_query=args.raw_papers_per_source_query,
        ai_papers_per_subtopic=args.ai_papers_per_subtopic,
        ai_papers_per_axis=args.ai_papers_per_axis,
        cutoff_year=args.cutoff_year,
        enable_retrieval=not args.disable_retrieval,
        use_pubtator=args.use_pubtator,
        max_axis_query_families=args.max_axis_query_families,
        entity_map_candidates_per_family=args.entity_map_candidates_per_family,
        use_query_reviewer=not args.disable_query_reviewer,
        query_reviewer_model=args.query_reviewer_model or args.model,
        use_evidence_selector=not args.disable_evidence_selector,
        evidence_selector_model=args.evidence_selector_model or args.model,
        evidence_selector_initial_depth_per_query=args.evidence_selector_initial_depth_per_query,
        evidence_selector_max_depth_per_query=args.evidence_selector_max_depth_per_query,
        enable_pmid_branch_tag_filter=not args.disable_pmid_branch_tag_filter,
        subtopics_only=args.stop_after == "subtopics",
    )

    if args.stop_after == "subtopics":
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "01_supervisor_config.json").write_text(json.dumps(supervisor_dict, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "01b_axis_generation_supervisor_view.json").write_text(json.dumps(axis_supervisor_view, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "01c_literature_input_policy.json").write_text(json.dumps({"supervisor_fields_passed_to_literature": [], "policy": "v73 subtopic-only test: v70 granularity baseline plus one added axis-generation instruction: Generate axes that maximize biological non-redundancy across the full search space. Stops before retrieval, EvidenceSelector, synthesis, hypothesis generation, and reflection."}, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "01d_hypothesis_generation_supervisor_view.json").write_text(json.dumps(hypothesis_generation_supervisor_view, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "02_generation_axes.json").write_text(json.dumps(axes_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "03_axis_literature_subtopics.json").write_text(json.dumps([
            {
                "axis_id": r.axis_id,
                "axis": r.axis,
                "subtopics_payload": r.subtopics_payload,
            }
            for r in literature.axis_results
        ], indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "03a_axis_subtopic_generation_contexts.json").write_text(json.dumps([
            {
                "axis_id": r.axis_id,
                "axis_name": r.axis.get("axis_name"),
                "subtopic_generation_context": r.subtopic_generation_context,
            }
            for r in literature.axis_results
        ], indent=2, ensure_ascii=False), encoding="utf-8")
        llm_usage = collect_llm_usage_summary()
        (out / "llm_usage.json").write_text(json.dumps(llm_usage, indent=2, ensure_ascii=False), encoding="utf-8")
        summary = {
            "run_id": run_id,
            "goal": args.goal,
            "model": args.model,
            "stage_stopped_after": "subtopics",
            "axis_criteria_preset": args.axis_criteria_preset,
            "criterion_1_granularity_level": (criterion_1_granularity or {}).get("criterion_1_granularity_level"),
            "criterion_1_granularity_instruction": (criterion_1_granularity or {}).get("criterion_1_granularity_instruction"),
            "criterion_2_granularity_level": criterion_2_granularity.get("criterion_2_granularity_level"),
            "criterion_2_granularity_instruction": criterion_2_granularity.get("criterion_2_granularity_instruction"),
            "axes_fixed": args.axes,
            "axes_returned": len(axes_payload.get("axes", [])) if isinstance(axes_payload, dict) else None,
            "axis_subtopic_payloads_returned": len(literature.axis_results),
            "subtopic_route": "v2_query_family_reviewed_entity_map_only",
            "v1_axis_decomposition_used": False,
            "query_reviewer_enabled": not args.disable_query_reviewer,
            "evidence_selector_enabled": False,
            "retrieval_stage_after_subtopics_enabled": False,
            "entity_map_retrieval_enabled": not args.disable_retrieval,
            "llm_usage_totals": llm_usage.get("totals", {}),
            "llm_usage_by_stage": llm_usage.get("by_stage", {}),
            "llm_usage_by_model": llm_usage.get("by_model", {}),
            "top_prompt_calls": llm_usage.get("top_prompt_calls", []),
            "budget_usage": llm_usage.get("budget_usage", {}),
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        log_event("workflow", "v73_axis_first_subtopics_complete", summary)
        finalize_run(summary)
        print(json.dumps(summary, indent=2))
        return

    hypotheses_payload = generator.generate_hypotheses_from_axis_literature(
        args.goal,
        hypothesis_generation_supervisor_view,
        axes_payload,
        literature.syntheses,
        literature.global_synthesis,
    )
    evidence_packets = [p for axis in literature.axis_results for p in axis.evidence_packets]
    strategies = generator._coerce_hypotheses(hypotheses_payload, args.goal, evidence_packets)

    all_retrieval_results = []
    for axis_result in literature.axis_results:
        all_retrieval_results.extend(axis_result.retrieval_results)
    paper_memory = build_paper_memory(
        retrieval_results=all_retrieval_results,
        selected_packets=evidence_packets,
        axis_id="ALL",
        axis_name="all_axes",
    )
    paper_memory_compact = compact_memory_for_reflection(paper_memory, max_entries=80)

    proximity_payload = {}
    reflection_hypotheses_payload = hypotheses_payload if isinstance(hypotheses_payload, dict) else {}
    if not args.disable_proximity:
        proximity_payload = ProximityAgent(model=args.proximity_model or args.model).cluster_merge_salvage(
            supervisor_config=supervisor_dict,
            generation_supervisor_view=hypothesis_generation_supervisor_view,
            hypotheses_payload=hypotheses_payload if isinstance(hypotheses_payload, dict) else {},
            paper_memory_compact=paper_memory_compact,
            max_focus_seeds=args.max_proximity_focus_seeds,
        )
        reflection_hypotheses_payload = build_hypotheses_payload_from_proximity(
            hypotheses_payload if isinstance(hypotheses_payload, dict) else {},
            proximity_payload if isinstance(proximity_payload, dict) else {},
        )

    if args.stop_after == "proximity":
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "01_supervisor_config.json").write_text(json.dumps(supervisor_dict, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "01d_hypothesis_generation_supervisor_view.json").write_text(json.dumps(hypothesis_generation_supervisor_view, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "07_generation_hypotheses_payload.json").write_text(json.dumps(hypotheses_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "07b_paper_memory_compact_for_reflection.json").write_text(json.dumps(paper_memory_compact, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "08_proximity_clusters.json").write_text(json.dumps(proximity_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "08a_proximity_reflection_input_hypotheses.json").write_text(json.dumps(reflection_hypotheses_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        llm_usage = collect_llm_usage_summary()
        (out / "llm_usage.json").write_text(json.dumps(llm_usage, indent=2, ensure_ascii=False), encoding="utf-8")
        summary = {
            "run_id": run_id,
            "goal": args.goal,
            "model": args.model,
            "stage_stopped_after": "proximity",
            "hypotheses_returned": len(strategies),
            "proximity_enabled": not args.disable_proximity,
            "proximity_survivor_count": len(proximity_payload.get("survivor_hypotheses", [])) if isinstance(proximity_payload, dict) else 0,
            "proximity_focus_seed_count": len(proximity_payload.get("focus_seeds", [])) if isinstance(proximity_payload, dict) else 0,
            "reflection_input_hypotheses": len(reflection_hypotheses_payload.get("hypotheses", [])) if isinstance(reflection_hypotheses_payload, dict) else 0,
            "llm_usage_totals": llm_usage.get("totals", {}),
            "llm_usage_by_stage": llm_usage.get("by_stage", {}),
            "llm_usage_by_model": llm_usage.get("by_model", {}),
            "top_prompt_calls": llm_usage.get("top_prompt_calls", []),
            "budget_usage": llm_usage.get("budget_usage", {}),
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        log_event("workflow", "v74_proximity_complete", summary)
        finalize_run(summary)
        print(json.dumps(summary, indent=2))
        return

    reflection_payload = {}
    if not args.disable_reflection:
        reflection_payload = ReflectionAgent(model=args.model).review_global_hypotheses(
            supervisor_config=supervisor_dict,
            generation_supervisor_view=hypothesis_generation_supervisor_view,
            axes_payload=axes_payload if isinstance(axes_payload, dict) else {},
            global_synthesis=literature.global_synthesis if isinstance(literature.global_synthesis, dict) else {},
            axis_syntheses=literature.syntheses if isinstance(literature.syntheses, list) else [],
            hypotheses_payload=reflection_hypotheses_payload if isinstance(reflection_hypotheses_payload, dict) else {},
            paper_memory_compact=paper_memory_compact,
        )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "01_supervisor_config.json").write_text(json.dumps(supervisor_dict, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "01b_axis_generation_supervisor_view.json").write_text(json.dumps(axis_supervisor_view, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "01c_literature_input_policy.json").write_text(json.dumps({"supervisor_fields_passed_to_literature": [], "policy": "Full axis-first LiteratureAgent uses the v2 branch-preserving route only: query families -> QueryReviewer -> entity/concept inventory -> v2 entity-map subtopics -> targeted retrieval -> axis-batched EvidenceSelector -> synthesis. Supervisor constraints are not passed to retrieval or synthesis."}, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "01d_hypothesis_generation_supervisor_view.json").write_text(json.dumps(hypothesis_generation_supervisor_view, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "02_generation_axes.json").write_text(json.dumps(axes_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "03_axis_literature_subtopics.json").write_text(json.dumps([
        {
            "axis_id": r.axis_id,
            "axis": r.axis,
            "subtopics_payload": r.subtopics_payload,
        }
        for r in literature.axis_results
    ], indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "03a_axis_subtopic_generation_contexts.json").write_text(json.dumps([
        {
            "axis_id": r.axis_id,
            "axis_name": r.axis.get("axis_name"),
            "subtopic_generation_context": r.subtopic_generation_context,
        }
        for r in literature.axis_results
    ], indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "04_retrieval_results.json").write_text(json.dumps([
        {
            "axis_id": r.axis_id,
            "records_returned_to_ai": len(r.evidence_packets),
            "evidence_packets": [p.to_dict() for p in r.evidence_packets],
            "evidence_selection_by_subtopic": {
                sr.subtopic_id: sum(
                    1 for p in r.evidence_packets
                    if p.metadata.get("selected_from_subtopic") == sr.subtopic_id
                )
                for sr in r.retrieval_results
            },
            "subtopic_retrieval": [
                {
                    "subtopic_id": sr.subtopic_id,
                    "queries": sr.queries,
                    "raw_records_count": sr.raw_records_count,
                    "deduped_candidate_count": sr.deduped_candidate_count,
                    "pubtator_annotated_count": sr.pubtator_annotated_count,
                    "selected_for_axis_synthesis": sum(
                        1 for p in r.evidence_packets
                        if p.metadata.get("selected_from_subtopic") == sr.subtopic_id
                    ),
                    "balanced_candidate_slate": sr.balanced_candidate_slate,
                    "resolved_evidence_selection": sr.resolved_evidence_selection,
                    "pmid_branch_filter_excluded": sr.pmid_branch_filter_excluded,
                    "non_pmid_sanity_excluded": sr.non_pmid_sanity_excluded,
                    "warnings": sr.warnings,
                }
                for sr in r.retrieval_results
            ],
        }
        for r in literature.axis_results
    ], indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "05_axis_literature_syntheses.json").write_text(json.dumps(literature.syntheses, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "06_global_synthesis.json").write_text(json.dumps(literature.global_synthesis, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "07_generation_hypotheses_payload.json").write_text(json.dumps(hypotheses_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "07a_paper_memory.json").write_text(json.dumps(paper_memory, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "07b_paper_memory_compact_for_reflection.json").write_text(json.dumps(paper_memory_compact, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "08_proximity_clusters.json").write_text(json.dumps(proximity_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "08a_proximity_reflection_input_hypotheses.json").write_text(json.dumps(reflection_hypotheses_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "initial_candidates.json").write_text(json.dumps([s.to_dict() for s in strategies], indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "09_reflection_reviews.json").write_text(json.dumps(reflection_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    llm_usage = collect_llm_usage_summary()
    (out / "llm_usage.json").write_text(json.dumps(llm_usage, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "run_id": run_id,
        "goal": args.goal,
        "model": args.model,
        "axis_criteria_preset": args.axis_criteria_preset,
        "criterion_1_granularity_level": (criterion_1_granularity or {}).get("criterion_1_granularity_level"),
        "criterion_1_granularity_instruction": (criterion_1_granularity or {}).get("criterion_1_granularity_instruction"),
        "criterion_2_granularity_level": criterion_2_granularity.get("criterion_2_granularity_level"),
        "criterion_2_granularity_instruction": criterion_2_granularity.get("criterion_2_granularity_instruction"),
        "stage_stopped_after": "reflection" if args.stop_after == "reflection" and not args.disable_reflection else "generation_reflection_pipeline",
        "axes_fixed": args.axes,
        "axes_returned": len(axes_payload.get("axes", [])) if isinstance(axes_payload, dict) else None,
        "axis_syntheses_returned": len(literature.syntheses),
        "global_synthesis_present": bool(literature.global_synthesis),
        "hypotheses_returned": len(strategies),
        "hypothesis_count_policy": "up to 30 total across all axes; generate fewer if not supported; no forced per-axis quota",
        "paper_memory_entries": paper_memory.get("counts", {}).get("memory_entries") if isinstance(paper_memory, dict) else None,
        "paper_memory_used_in_synthesis": paper_memory.get("counts", {}).get("used_in_synthesis") if isinstance(paper_memory, dict) else None,
        "proximity_enabled": not args.disable_proximity,
        "proximity_survivor_count": len(proximity_payload.get("survivor_hypotheses", [])) if isinstance(proximity_payload, dict) else 0,
        "proximity_focus_seed_count": len(proximity_payload.get("focus_seeds", [])) if isinstance(proximity_payload, dict) else 0,
        "reflection_input_hypotheses": len(reflection_hypotheses_payload.get("hypotheses", [])) if isinstance(reflection_hypotheses_payload, dict) else 0,
        "reflection_enabled": not args.disable_reflection,
        "reflection_reviews_returned": len(reflection_payload.get("reflection_reviews", [])) if isinstance(reflection_payload, dict) else 0,
        "sources": sources,
        "cutoff_year": args.cutoff_year,
        "use_pubtator": args.use_pubtator,
        "subtopic_route": "v2_query_family_reviewed_entity_map_only",
        "v1_axis_decomposition_used": False,
        "query_reviewer_enabled": not args.disable_query_reviewer,
        "evidence_selector_enabled": not args.disable_evidence_selector,
        "evidence_selector_batching": "per_axis",
        "max_subtopics_per_axis": args.max_subtopics_per_axis,
        "max_queries_per_subtopic": args.max_queries_per_subtopic,
        "raw_papers_per_source_query": args.raw_papers_per_source_query,
        "ai_papers_per_subtopic": args.ai_papers_per_subtopic,
        "ai_papers_per_axis": args.ai_papers_per_axis,
        "retrieval_enabled": not args.disable_retrieval,
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
    log_event("workflow", "v45_axis_first_literature_reflection_complete", summary)
    finalize_run(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
