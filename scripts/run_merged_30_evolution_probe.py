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

from agents.evolution_agent import EvolutionAgent
from agents.literature_agent import LiteratureAgent
from runtime.context import configure_runtime, current_llm_call_count
from utils.config import load_config, validate_config
from utils.run_logger import start_run_log, log_event, finalize_run, collect_llm_usage_summary
from utils.evidence_memory import build_paper_memory, compact_memory_for_reflection


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _hypothesis_id(h: dict) -> str:
    return str(h.get("hypothesis_id") or h.get("id") or "")


def _hypothesis_index(payload: dict) -> dict[str, dict]:
    return {_hypothesis_id(h): h for h in payload.get("hypotheses", []) if isinstance(h, dict)}


def _reviews_index(payload: dict) -> dict[str, dict]:
    return {str(r.get("hypothesis_id")): r for r in payload.get("hypothesis_reviews", []) if isinstance(r, dict) and r.get("hypothesis_id")}


def _relevant_proximity(prox: dict, ids: list[str]) -> dict:
    groups = []
    ids_set = set(ids)
    for g in prox.get("proximity_groups", []) if isinstance(prox, dict) else []:
        hids = set(str(x) for x in g.get("hypothesis_ids", []))
        if ids_set & hids:
            groups.append(g)
    return {
        "relevant_groups": groups,
        "focus_seeds": prox.get("focus_seeds", []) if isinstance(prox, dict) else [],
        "absorbed_or_duplicate_hypotheses": prox.get("absorbed_or_duplicate_hypotheses", []) if isinstance(prox, dict) else [],
    }




def _extract_candidate_terms(hypothesis: dict, limit: int = 8) -> list[str]:
    """Conservative text-derived terms for focused Evolution retrieval.

    This intentionally does not use answer-specific hardcoded terms. It extracts
    compact uppercase gene/protein/pathway-like tokens from the supplied
    hypothesis text and title, plus a few informative title words.
    """
    import re

    text = " ".join(str(hypothesis.get(k, "")) for k in ["title", "hypothesis", "mechanistic_rationale", "candidate_intervention_or_focus"])
    tokens: list[str] = []
    stop = {"AML", "DNA", "RNA", "ROS", "CD", "AND", "THE", "FOR", "WITH", "IN", "OF", "TO"}
    for tok in re.findall(r"\b[A-Z][A-Z0-9α-ωΑ-Ω/-]{1,12}\b", text):
        clean = tok.strip("-/")
        if clean and clean not in stop and clean not in tokens:
            tokens.append(clean)
    if len(tokens) < 4:
        for tok in re.findall(r"\b[a-zA-Z][a-zA-Z0-9-]{4,}\b", str(hypothesis.get("title", ""))):
            low = tok.lower()
            if low not in {"targeting", "acute", "myeloid", "leukemia", "therapy", "vulnerability", "hypothesis"} and tok not in tokens:
                tokens.append(tok)
            if len(tokens) >= limit:
                break
    return tokens[:limit]


def _build_evolution_retrieval_queries(hypothesis: dict, *, goal: str, guidance: str, max_queries: int) -> list[str]:
    terms = _extract_candidate_terms(hypothesis)
    title = str(hypothesis.get("title") or hypothesis.get("hypothesis_id") or "").strip()
    base = " ".join(terms[:5]) if terms else title
    queries: list[str] = []
    if base:
        queries.append(f"{base} acute myeloid leukemia")
    guidance_l = guidance.lower()
    goal_l = goal.lower()
    if any(x in guidance_l + " " + goal_l for x in ["drug", "compound", "inhibitor", "therapeutic", "candidate", "probe"]):
        if base:
            queries.append(f"{base} acute myeloid leukemia inhibitor drug candidate")
            queries.append(f"{base} AML therapeutic target compound probe")
    else:
        if base:
            queries.append(f"{base} AML mechanism therapeutic target")
    if title and title.lower() not in " ".join(q.lower() for q in queries):
        queries.append(f"{title} AML")
    # Deduplicate and cap.
    out: list[str] = []
    for q in queries:
        q = " ".join(str(q).split())
        if q and q.lower() not in {x.lower() for x in out}:
            out.append(q)
        if len(out) >= max(1, int(max_queries)):
            break
    return out


def _run_evolution_retrieval(
    *,
    selected_hypotheses: list[dict],
    goal: str,
    guidance: str,
    model: str,
    config_path: str,
    sources: list[str],
    cutoff_year: int,
    max_queries: int,
    raw_papers_per_source_query: int,
    selected_papers_per_hypothesis: int,
    use_pubtator: bool,
    use_evidence_selector: bool,
) -> tuple[dict, dict]:
    lit = LiteratureAgent(model=model, config_path=config_path)
    retrieval_results = []
    retrieval_by_hypothesis: list[dict] = []
    selected_packets = []
    for hyp in selected_hypotheses:
        hid = _hypothesis_id(hyp) or "HYP"
        queries = _build_evolution_retrieval_queries(hyp, goal=goal, guidance=guidance, max_queries=max_queries)
        subtopic_payload = {
            "subtopic_id": f"EVOL_{hid}",
            "name": str(hyp.get("title") or hid),
            "coverage_intent": "Focused Evolution retrieval for candidate-level hypothesis maturation.",
            "search_queries": queries,
            "focus_seed": hyp,
            "human_stage_guidance": guidance,
        }
        result = lit.retrieve_subtopic(
            subtopic_id=f"EVOL_{hid}",
            queries=queries,
            sources=sources,
            max_queries=max_queries,
            raw_papers_per_source_query=raw_papers_per_source_query,
            ai_papers_per_subtopic=selected_papers_per_hypothesis,
            cutoff_year=cutoff_year,
            use_pubtator=use_pubtator,
            use_evidence_selector=use_evidence_selector,
            evidence_selector_model=model,
            evidence_selector_initial_depth_per_query=3,
            evidence_selector_max_depth_per_query=8,
            enable_pmid_branch_tag_filter=True,
            subtopic_payload=subtopic_payload,
        )
        retrieval_results.append(result)
        selected_packets.extend(result.evidence_packets)
        retrieval_by_hypothesis.append({
            "hypothesis_id": hid,
            "queries": queries,
            "warnings": result.warnings,
            "raw_records_count": result.raw_records_count,
            "deduped_candidate_count": result.deduped_candidate_count,
            "selected_evidence_packets": [p.to_dict() for p in result.evidence_packets],
            "resolved_evidence_selection": result.resolved_evidence_selection,
            "evidence_selector_payload": result.evidence_selector_payload,
        })
    memory = build_paper_memory(
        retrieval_results=retrieval_results,
        selected_packets=selected_packets,
        axis_id="evolution_probe",
        axis_name="Evolution focused retrieval",
        cutoff_year=cutoff_year,
    )
    compact = compact_memory_for_reflection(memory, max_entries=max(40, len(selected_hypotheses) * selected_papers_per_hypothesis * 3))
    return {"retrieval_by_hypothesis": retrieval_by_hypothesis, "paper_memory": memory}, compact


def main() -> None:
    ap = argparse.ArgumentParser(description="Run repo-style Evolution probes on selected hypotheses from merged-30 AML run.")
    ap.add_argument("--goal", default="Identify novel drug candidates or therapeutic routes for acute myeloid leukemia that have not previously been used for AML, using literature only up to cutoff year 2023.")
    ap.add_argument("--input-dir", default="inputs/merged_v70_30_with_v73_ire1")
    ap.add_argument("--reflection-run-dir", default="", help="Optional run folder containing 08_proximity_analysis.json and 09_reflection_reviews_with_proximity.json")
    ap.add_argument("--out-dir", default="runs/v77_evolution_probe_cutoff2023")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--model", default="gemini-2.5-flash-lite")
    ap.add_argument("--hypothesis-ids", nargs="+", default=["H001", "H005", "H018", "H030"], help="Hypotheses to evolve")
    ap.add_argument("--strategies", nargs="+", choices=["simplify", "feasibility", "out_of_box", "combine"], default=["simplify", "feasibility", "out_of_box", "combine"])
    ap.add_argument("--combine-pairs", nargs="*", default=["H001,H018", "H005,H030"], help="Pairs for combine strategy, e.g. H001,H018")
    ap.add_argument("--max-llm-calls", type=int, default=40)
    ap.add_argument("--evolution-guidance", default="", help="Optional human stage guidance injected into Evolution. Empty by default; use to steer the next stage without naming answers.")
    ap.add_argument("--enable-evolution-retrieval", action="store_true", help="Run focused literature retrieval for selected hypotheses before Evolution and store compact retrieved-paper memory.")
    ap.add_argument("--evolution-retrieval-sources", nargs="+", default=["PubMed", "EuropePMC", "OpenAlex", "SemanticScholar"], help="Literature sources for optional Evolution retrieval.")
    ap.add_argument("--evolution-retrieval-max-queries", type=int, default=3)
    ap.add_argument("--evolution-retrieval-raw-papers", type=int, default=4)
    ap.add_argument("--evolution-retrieval-selected-papers", type=int, default=2)
    ap.add_argument("--use-pubtator", action="store_true")
    ap.add_argument("--disable-evolution-selector", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--clean-out-dir", action="store_true")
    args = ap.parse_args()

    inp = Path(args.input_dir)
    out = Path(args.out_dir)
    if args.clean_out_dir and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    config.setdefault("runtime", {}).setdefault("limits", {})["max_llm_calls_per_run"] = args.max_llm_calls
    warnings = validate_config(config, runtime_mode="dry_run" if args.dry_run else None, strict=False)
    configure_runtime(config, mode_override="dry_run" if args.dry_run else None)
    start_run_log(args.goal, "v77_evolution_probe")
    for warning in warnings:
        log_event("config", "warning", {"warning": warning}, status="warning")

    supervisor_config = load_json(inp / "01_supervisor_config.json")
    generation_supervisor_view = load_json(inp / "01d_hypothesis_generation_supervisor_view.json")
    if args.evolution_guidance:
        supervisor_config.setdefault("human_stage_guidance", {})["evolution"] = args.evolution_guidance
    hypotheses_payload = load_json(inp / "07_generation_hypotheses_payload_merged_30.json")
    paper_memory_compact = load_json(inp / "07b_paper_memory_compact_for_reflection_merged.json")
    hyp_by_id = _hypothesis_index(hypotheses_payload)

    proximity_payload = {}
    reflection_payload = {}
    if args.reflection_run_dir:
        rr = Path(args.reflection_run_dir)
        if (rr / "08_proximity_analysis.json").exists():
            proximity_payload = load_json(rr / "08_proximity_analysis.json")
        if (rr / "09_reflection_reviews_with_proximity.json").exists():
            reflection_payload = load_json(rr / "09_reflection_reviews_with_proximity.json")
    reviews_by_id = _reviews_index(reflection_payload)

    selected_ids = [x for x in args.hypothesis_ids if x in hyp_by_id]
    if not selected_ids:
        raise SystemExit("No requested hypothesis IDs were found in the input hypothesis payload.")
    selected_hypotheses = [hyp_by_id[x] for x in selected_ids]
    selected_reviews = [reviews_by_id.get(x, {}) for x in selected_ids]
    relevant_prox = _relevant_proximity(proximity_payload, selected_ids)

    base_paper_memory_compact = paper_memory_compact
    evolution_retrieval_payload: dict = {"enabled": False}
    evolution_retrieval_compact: dict = {}
    if args.enable_evolution_retrieval:
        evolution_retrieval_payload, evolution_retrieval_compact = _run_evolution_retrieval(
            selected_hypotheses=selected_hypotheses,
            goal=args.goal,
            guidance=args.evolution_guidance,
            model=args.model,
            config_path=args.config,
            sources=args.evolution_retrieval_sources,
            cutoff_year=2023,
            max_queries=args.evolution_retrieval_max_queries,
            raw_papers_per_source_query=args.evolution_retrieval_raw_papers,
            selected_papers_per_hypothesis=args.evolution_retrieval_selected_papers,
            use_pubtator=args.use_pubtator,
            use_evidence_selector=not args.disable_evolution_selector,
        )
        evolution_retrieval_payload["enabled"] = True
        base_paper_memory_compact = {
            "original_run_memory": paper_memory_compact,
            "evolution_focused_retrieval_memory": evolution_retrieval_compact,
        }

    evo = EvolutionAgent(model=args.model)
    outputs: list[dict] = []

    for hid, hyp, rev in zip(selected_ids, selected_hypotheses, selected_reviews):
        if "simplify" in args.strategies:
            payload = evo.simplify(goal=args.goal, preferences=supervisor_config, hypothesis=hyp, review=rev, proximity_context=relevant_prox, paper_memory_compact=base_paper_memory_compact, human_stage_guidance=args.evolution_guidance)
            payload.setdefault("parent_ids", [hid])
            outputs.append({"input_hypothesis_id": hid, "strategy": "simplify", "output": payload})
        if "feasibility" in args.strategies:
            payload = evo.feasibility(goal=args.goal, preferences=supervisor_config, hypothesis=hyp, review=rev, proximity_context=relevant_prox, paper_memory_compact=base_paper_memory_compact, human_stage_guidance=args.evolution_guidance)
            payload.setdefault("parent_ids", [hid])
            outputs.append({"input_hypothesis_id": hid, "strategy": "feasibility", "output": payload})

    if "out_of_box" in args.strategies:
        payload = evo.out_of_box(goal=args.goal, preferences=supervisor_config, hypotheses=selected_hypotheses, reviews=selected_reviews, proximity_context=relevant_prox, paper_memory_compact=base_paper_memory_compact, human_stage_guidance=args.evolution_guidance)
        payload.setdefault("parent_ids", selected_ids)
        outputs.append({"input_hypothesis_id": "+".join(selected_ids), "strategy": "out_of_box", "output": payload})

    if "combine" in args.strategies:
        for raw_pair in args.combine_pairs:
            try:
                a, b = [x.strip() for x in raw_pair.split(",", 1)]
            except ValueError:
                continue
            if a not in hyp_by_id or b not in hyp_by_id:
                continue
            payload = evo.combine(
                goal=args.goal,
                preferences=supervisor_config,
                hypothesis_a=hyp_by_id[a],
                hypothesis_b=hyp_by_id[b],
                review_a=reviews_by_id.get(a, {}),
                review_b=reviews_by_id.get(b, {}),
                proximity_context=_relevant_proximity(proximity_payload, [a, b]),
                paper_memory_compact=base_paper_memory_compact,
                human_stage_guidance=args.evolution_guidance,
            )
            payload.setdefault("parent_ids", [a, b])
            outputs.append({"input_hypothesis_id": f"{a}+{b}", "strategy": "combine", "output": payload})

    if evolution_retrieval_payload.get("enabled"):
        (out / "09b_evolution_focused_retrieval.json").write_text(json.dumps(evolution_retrieval_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "09c_evolution_focused_retrieval_compact_memory.json").write_text(json.dumps(evolution_retrieval_compact, indent=2, ensure_ascii=False), encoding="utf-8")

    (out / "10_evolution_probe_outputs.json").write_text(json.dumps({"evolution_outputs": outputs}, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "10a_evolution_inputs.json").write_text(json.dumps({
        "selected_hypothesis_ids": selected_ids,
        "selected_hypotheses": selected_hypotheses,
        "selected_reviews": selected_reviews,
        "relevant_proximity_context": relevant_prox,
        "human_stage_guidance": {"evolution": args.evolution_guidance},
        "evolution_retrieval_enabled": bool(evolution_retrieval_payload.get("enabled")),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    llm_usage = collect_llm_usage_summary()
    summary = {
        "stage_stopped_after": "evolution_probe",
        "mode": "repo_style_evolution_probe_with_optional_stage_guidance_and_retrieval",
        "selected_hypothesis_ids": selected_ids,
        "strategies": args.strategies,
        "evolution_output_count": len(outputs),
        "human_evolution_guidance_present": bool(args.evolution_guidance),
        "evolution_retrieval_enabled": bool(evolution_retrieval_payload.get("enabled")),
        "evolution_retrieval_hypothesis_count": len(evolution_retrieval_payload.get("retrieval_by_hypothesis", [])) if isinstance(evolution_retrieval_payload, dict) else 0,
        "used_reflection_context": bool(reflection_payload),
        "used_proximity_context": bool(proximity_payload),
        "llm_calls_used": current_llm_call_count(),
        "llm_usage_totals": llm_usage.get("totals", {}),
        "llm_usage_by_stage": llm_usage.get("by_stage", {}),
        "top_prompt_calls": llm_usage.get("top_prompt_calls", []),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "llm_usage.json").write_text(json.dumps(llm_usage, indent=2, ensure_ascii=False), encoding="utf-8")
    log_event("workflow", "evolution_probe_complete", summary)
    finalize_run(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
