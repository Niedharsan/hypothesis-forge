from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.evolution_agent import EvolutionAgent
from agents.generation_rewired import RewiredGenerationAgent
from agents.literature_agent import LiteratureAgent, SubtopicRetrievalResult
from agents.proximity_agent import ProximityAgent, build_hypotheses_payload_from_proximity
from agents.reflection_agent import ReflectionAgent
from agents.supervisor_config_agent import SupervisorConfigAgent
from llm.provider import ask_llm_json
from runtime.context import configure_runtime
from schemas.evidence_packet import EvidencePacket
from schemas.paper_record import PaperRecord
from utils.config import load_config
from utils.evidence_memory import build_paper_memory, compact_memory_for_reflection
from utils.json_compact import compact_json
from utils.run_logger import collect_llm_usage_summary, finalize_run, start_run_log
from utils.supervisor_views import build_generation_supervisor_view

from app.storage import read_json, run_dir, write_run

STAGES = [
    "axis_generation",
    "subtopic_generation",
    "literature_retrieval",
    "synthesis",
    "hypothesis_generation",
    "proximity",
    "reflection",
    "evolution",
    "candidate_ranking",
]
SUPPORTED_SOURCES = {"PubMed", "EuropePMC", "OpenAlex", "Crossref", "SemanticScholar"}
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def start_run(payload: Any) -> dict[str, Any]:
    run_id = f"hf-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    sources = [s for s in payload.literature_sources if s in SUPPORTED_SOURCES]
    if not sources:
        sources = ["PubMed", "EuropePMC", "OpenAlex", "Crossref"]
    run: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "current_stage": "axis_generation",
        "created_at": now(),
        "objective": payload.research_objective,
        "cutoff_year": payload.cutoff_year,
        "model": payload.model,
        "runtime_mode": payload.runtime_mode,
        "literature_sources": sources,
        "use_pubtator": bool(payload.use_pubtator),
        "enable_evolution_retrieval": bool(payload.enable_evolution_retrieval),
        "stages": {},
        "artifacts": [],
        "logs": [],
        "usage": {"calls": 0, "input_tokens": 0, "output_tokens": 0, "estimated_cost_usd": 0.0},
        "usage_by_stage": [],
        "human_stage_guidance": {},
    }
    write_run(run)
    outputs, artifacts, stage_usage = _run_axis_generation(run, payload.output_count)
    _finish_stage(run, "axis_generation", outputs, artifacts, stage_usage, "Created the initial research axes from the v78 Supervisor + Generation path.")
    return run


def run_stage(run: dict[str, Any], request: Any) -> dict[str, Any]:
    stage = request.stage
    if stage == "axis_generation":
        return run
    target_index = STAGES.index(stage)
    source_stage = request.source_stage or STAGES[target_index - 1]
    if source_stage not in run.get("stages", {}):
        raise ValueError(f"Source stage has not been run: {source_stage}")
    parents = _selected_parents(run, source_stage, request.selected_ids, request.include_all)
    if not parents:
        raise ValueError("No parent outputs selected")
    run.setdefault("human_stage_guidance", {})[stage] = request.stage_guidance
    for card in run["stages"][source_stage]:
        if card["id"] in {p["id"] for p in parents}:
            card.update(status="advanced", selected_for_next_stage=True, selection_source=request.selection_source)
        elif card.get("status") == "active":
            card.update(status="saved", selected_for_next_stage=False, selection_source=request.selection_source)

    dispatch = {
        "subtopic_generation": _run_subtopic_generation,
        "literature_retrieval": _run_literature_retrieval,
        "synthesis": _run_synthesis,
        "hypothesis_generation": _run_hypothesis_generation,
        "proximity": _run_proximity,
        "reflection": _run_reflection,
        "evolution": _run_evolution,
        "candidate_ranking": _run_candidate_ranking,
    }
    outputs, artifacts, usage = dispatch[stage](run, parents, request.output_count, request.stage_guidance)
    _finish_stage(run, stage, outputs, artifacts, usage, f"Completed {stage} from {source_stage} using the standalone v78-derived engine.")
    return run


def update_selection(run: dict[str, Any], request: Any) -> dict[str, Any]:
    cards = run.get("stages", {}).get(request.stage, [])
    ids = {str(c.get("id")) for c in cards}
    requested = set(request.selected_ids) | set(request.rejected_ids) | set(request.saved_ids)
    missing = requested - ids
    if missing:
        raise KeyError(", ".join(sorted(missing)))
    selected, rejected, saved = set(request.selected_ids), set(request.rejected_ids), set(request.saved_ids)
    for card in cards:
        cid = card["id"]
        if cid in selected:
            card.update(status="active", selected_for_next_stage=True, selection_source=request.selection_source, selection_reason="Selected at checkpoint.")
        elif cid in rejected:
            card.update(status="rejected", selected_for_next_stage=False, selection_source=request.selection_source, selection_reason="Hidden from active path; retained in run state.")
        elif cid in saved:
            card.update(status="saved", selected_for_next_stage=False, selection_source=request.selection_source, selection_reason="Saved for a later branch.")
    write_run(run)
    return run


def create_focus_seed(run: dict[str, Any], request: Any) -> dict[str, Any]:
    source = None
    for cards in run.get("stages", {}).values():
        for card in cards:
            if card.get("id") == request.source_card_id:
                source = card
                break
    if source is None:
        raise KeyError(request.source_card_id)
    seed = {
        "seed_id": f"seed-{uuid.uuid4().hex[:10]}",
        "source_card_id": request.source_card_id,
        "source_stage": source.get("stage"),
        "title": request.title or source.get("title", ""),
        "summary": request.summary or source.get("summary", ""),
        "guidance": request.guidance,
        "payload": source.get("payload", {}),
        "created_at": now(),
    }
    seeds = read_json(run_dir(run["run_id"]) / "focus_seeds.json", []) or []
    seeds.append(seed)
    _upsert_artifact(run, "focus-seeds", "Focus seeds", "focus_seeds.json", "generation", seeds)
    write_run(run, {"focus_seeds.json": seeds})
    return seed


def _prepare_stage(run: dict[str, Any], stage: str) -> None:
    config = load_config(CONFIG_PATH)
    configure_runtime(config, mode_override=run.get("runtime_mode", "normal"))
    start_run_log(run["objective"], f"hypothesis_forge_{stage}", run_root=run_dir(run["run_id"]) / "logs")


def _finish_usage(stage: str) -> dict[str, Any]:
    usage = collect_llm_usage_summary()
    totals = usage.get("totals", {})
    finalize_run({"stage": stage})
    return {
        "stage": stage,
        "calls": int(totals.get("calls") or 0),
        "input_tokens": int(totals.get("input_tokens") or 0),
        "output_tokens": int(totals.get("output_tokens") or 0),
        "estimated_cost_usd": float(totals.get("estimated_cost_usd") or 0.0),
        "updated_at": now(),
    }


def _run_axis_generation(run: dict[str, Any], output_count: int):
    _prepare_stage(run, "axis_generation")
    supervisor = SupervisorConfigAgent().configure(run["objective"], axes=output_count, use_literature=True, model=run["model"])
    supervisor_dict = supervisor.to_dict()
    generation_view = build_generation_supervisor_view(supervisor_dict)
    axes = RewiredGenerationAgent(model=run["model"]).generate_axes(run["objective"], generation_view)
    items = axes.get("axes", []) if isinstance(axes, dict) else []
    cards = [_card(run, "axis_generation", str(x.get("axis_id") or f"A{i:02d}"), str(x.get("axis_name") or x.get("name") or x.get("title") or f"Axis {i}"), str(x.get("description") or x.get("rationale") or ""), x) for i, x in enumerate(items, 1) if isinstance(x, dict)]
    return cards, {"01_supervisor_config.json": supervisor_dict, "01d_generation_supervisor_view.json": generation_view, "02_generation_axes.json": axes}, _finish_usage("axis_generation")


def _run_subtopic_generation(run: dict[str, Any], parents: list[dict[str, Any]], output_count: int, guidance: str):
    _prepare_stage(run, "subtopic_generation")
    axes_payload = {"axes": [p.get("payload", {}) for p in parents]}
    lit = LiteratureAgent(model=run["model"], config_path=str(CONFIG_PATH))
    out = lit.run_axis_first(
        objective=run["objective"], axes_payload=axes_payload, sources=run["literature_sources"],
        max_subtopics_per_axis=min(5, max(1, output_count)), max_queries_per_subtopic=5,
        cutoff_year=run["cutoff_year"], enable_retrieval=run.get("runtime_mode") != "dry_run",
        use_pubtator=run.get("use_pubtator", False), subtopics_only=True,
    )
    cards: list[dict[str, Any]] = []
    contexts = []
    subtopics_by_axis = []
    for axis_result in out.axis_results:
        contexts.append({"axis_id": axis_result.axis_id, "context": axis_result.subtopic_generation_context})
        subtopics_by_axis.append({"axis_id": axis_result.axis_id, "subtopics_payload": axis_result.subtopics_payload})
        for idx, sub in enumerate(axis_result.subtopics_payload.get("subtopics", []), 1):
            if not isinstance(sub, dict):
                continue
            sid = str(sub.get("subtopic_id") or f"{axis_result.axis_id}-S{idx:02d}")
            payload = {"axis_id": axis_result.axis_id, "axis": axis_result.axis, "subtopic": sub}
            cards.append(_card(run, "subtopic_generation", sid, str(sub.get("name") or sub.get("title") or sid), str(sub.get("coverage_intent") or sub.get("description") or ""), payload, [str(p["id"]) for p in parents if str(p.get("id")) == axis_result.axis_id]))
    return cards, {"03_axis_literature_subtopics.json": subtopics_by_axis, "03a_axis_subtopic_generation_contexts.json": contexts}, _finish_usage("subtopic_generation")


def _run_literature_retrieval(run: dict[str, Any], parents: list[dict[str, Any]], output_count: int, guidance: str):
    _prepare_stage(run, "literature_retrieval")
    lit = LiteratureAgent(model=run["model"], config_path=str(CONFIG_PATH))
    cards: list[dict[str, Any]] = []
    serialized_results: list[dict[str, Any]] = []
    for parent in parents:
        payload = parent.get("payload", {})
        sub = payload.get("subtopic", {}) if isinstance(payload, dict) else {}
        sid = str(sub.get("subtopic_id") or parent["id"])
        queries = [str(q) for q in sub.get("search_queries", []) if str(q).strip()]
        if run.get("runtime_mode") == "dry_run":
            result = SubtopicRetrievalResult(subtopic_id=sid, queries=queries, warnings=["Network retrieval skipped in dry_run mode."])
        else:
            result = lit.retrieve_subtopic(
                subtopic_id=sid, queries=queries, sources=run["literature_sources"], max_queries=5,
                raw_papers_per_source_query=5, ai_papers_per_subtopic=min(10, max(1, output_count)),
                cutoff_year=run["cutoff_year"], use_pubtator=run.get("use_pubtator", False),
                use_evidence_selector=True, evidence_selector_model=run["model"], subtopic_payload=sub,
            )
        ser = _serialize_retrieval(result, payload.get("axis_id"), payload.get("axis", {}), sub)
        serialized_results.append(ser)
        summary = f"{ser['deduped_candidate_count']} deduplicated candidates; {len(ser['evidence_packets'])} selected evidence packets."
        cards.append(_card(run, "literature_retrieval", f"LIT-{sid}", str(sub.get("name") or sid), summary, ser, [parent["id"]]))
    return cards, {"04_literature_results.json": serialized_results}, _finish_usage("literature_retrieval")


def _run_synthesis(run: dict[str, Any], parents: list[dict[str, Any]], output_count: int, guidance: str):
    _prepare_stage(run, "synthesis")
    lit = LiteratureAgent(model=run["model"], config_path=str(CONFIG_PATH))
    by_axis: dict[str, list[dict[str, Any]]] = {}
    for parent in parents:
        by_axis.setdefault(str(parent.get("payload", {}).get("axis_id") or "unknown"), []).append(parent.get("payload", {}))
    axis_syntheses: list[dict[str, Any]] = []
    cards: list[dict[str, Any]] = []
    for axis_id, payloads in by_axis.items():
        axis = next((p.get("axis") for p in payloads if isinstance(p.get("axis"), dict)), {"axis_id": axis_id})
        subtopics = [p.get("subtopic", {}) for p in payloads]
        packets = [EvidencePacket(**ep) for p in payloads for ep in p.get("evidence_packets", []) if isinstance(ep, dict)]
        synthesis = lit.synthesize_axis(axis, {"axis_id": axis_id, "subtopics": subtopics}, packets)
        axis_syntheses.append({"axis_id": axis_id, "axis": axis, "synthesis": synthesis})
        cards.append(_card(run, "synthesis", f"SYN-{axis_id}", str(axis.get("name") or axis_id), _summary_text(synthesis), {"axis_id": axis_id, "axis": axis, "synthesis": synthesis}, [p["id"] for p in parents if str(p.get("payload", {}).get("axis_id")) == axis_id]))
    global_synthesis = lit.synthesize_global(run["objective"], [x["synthesis"] for x in axis_syntheses])
    cards.append(_card(run, "synthesis", "SYN-GLOBAL", "Global synthesis", _summary_text(global_synthesis), {"axis_id": "GLOBAL", "synthesis": global_synthesis}, [p["id"] for p in parents]))
    return cards, {"05_axis_literature_syntheses.json": axis_syntheses, "06_global_synthesis.json": global_synthesis}, _finish_usage("synthesis")


def _run_hypothesis_generation(run: dict[str, Any], parents: list[dict[str, Any]], output_count: int, guidance: str):
    _prepare_stage(run, "hypothesis_generation")
    supervisor = _artifact_data(run, "01_supervisor_config.json") or {}
    generation_view = _artifact_data(run, "01d_generation_supervisor_view.json") or build_generation_supervisor_view(supervisor)
    axes = _artifact_data(run, "02_generation_axes.json") or {"axes": []}
    selected_syntheses = [p.get("payload", {}).get("synthesis", {}) for p in parents if p.get("id") != "SYN-GLOBAL"]
    global_synthesis = next((p.get("payload", {}).get("synthesis", {}) for p in parents if p.get("id") == "SYN-GLOBAL"), _artifact_data(run, "06_global_synthesis.json") or {})
    if guidance:
        generation_view = {**generation_view, "human_stage_guidance": {"hypothesis_generation": guidance}}
    payload = RewiredGenerationAgent(model=run["model"]).generate_hypotheses_from_axis_literature(run["objective"], generation_view, axes, selected_syntheses, global_synthesis)
    hyps = payload.get("hypotheses", []) if isinstance(payload, dict) else []
    cards = []
    for i, hyp in enumerate(hyps[:output_count], 1):
        if not isinstance(hyp, dict):
            continue
        hid = str(hyp.get("hypothesis_id") or f"H{i:03d}")
        cards.append(_card(run, "hypothesis_generation", hid, str(hyp.get("title") or hid), str(hyp.get("hypothesis") or hyp.get("mechanistic_rationale") or ""), hyp, [p["id"] for p in parents]))
    return cards, {"07_generation_hypotheses_payload.json": payload}, _finish_usage("hypothesis_generation")


def _run_proximity(run: dict[str, Any], parents: list[dict[str, Any]], output_count: int, guidance: str):
    _prepare_stage(run, "proximity")
    supervisor = _artifact_data(run, "01_supervisor_config.json") or {}
    generation_view = _artifact_data(run, "01d_generation_supervisor_view.json") or {}
    hypotheses_payload = {"hypotheses": [p.get("payload", {}) for p in parents]}
    paper_memory = _paper_memory_from_run(run)
    compact = compact_memory_for_reflection(paper_memory, max_entries=80) if paper_memory else {}
    proximity = ProximityAgent(model=run["model"]).cluster_merge_salvage(
        supervisor_config=supervisor, generation_supervisor_view=generation_view,
        hypotheses_payload=hypotheses_payload, paper_memory_compact=compact,
        max_focus_seeds=min(30, max(0, output_count)),
    )
    survivors = build_hypotheses_payload_from_proximity(hypotheses_payload, proximity)
    cards = []
    for i, hyp in enumerate(survivors.get("hypotheses", []), 1):
        if not isinstance(hyp, dict):
            continue
        hid = str(hyp.get("hypothesis_id") or f"P{i:03d}")
        cards.append(_card(run, "proximity", hid, str(hyp.get("title") or hid), str(hyp.get("hypothesis") or ""), hyp, [p["id"] for p in parents]))
    return cards, {"08_proximity_clusters.json": proximity, "08a_proximity_survivors.json": survivors, "07b_paper_memory_compact.json": compact}, _finish_usage("proximity")


def _run_reflection(run: dict[str, Any], parents: list[dict[str, Any]], output_count: int, guidance: str):
    _prepare_stage(run, "reflection")
    supervisor = _artifact_data(run, "01_supervisor_config.json") or {}
    generation_view = _artifact_data(run, "01d_generation_supervisor_view.json") or {}
    axes = _artifact_data(run, "02_generation_axes.json") or {}
    global_synthesis = _artifact_data(run, "06_global_synthesis.json") or {}
    axis_syntheses_raw = _artifact_data(run, "05_axis_literature_syntheses.json") or []
    axis_syntheses = [x.get("synthesis", {}) for x in axis_syntheses_raw if isinstance(x, dict)]
    proximity = _artifact_data(run, "08_proximity_clusters.json") or {}
    hypotheses_payload = {"hypotheses": [p.get("payload", {}) for p in parents]}
    compact = _artifact_data(run, "07b_paper_memory_compact.json") or {}
    if guidance:
        supervisor = {**supervisor, "human_stage_guidance": {**supervisor.get("human_stage_guidance", {}), "reflection": guidance}}
    review = ReflectionAgent(model=run["model"]).review_global_hypotheses_with_proximity(
        supervisor_config=supervisor, generation_supervisor_view=generation_view, axes_payload=axes,
        global_synthesis=global_synthesis, axis_syntheses=axis_syntheses,
        hypotheses_payload=hypotheses_payload, proximity_payload=proximity, paper_memory_compact=compact,
    )
    reviews = (review.get("reflection_reviews") or review.get("hypothesis_reviews") or []) if isinstance(review, dict) else []
    hyp_index = {str(h.get("hypothesis_id")): h for h in hypotheses_payload.get("hypotheses", []) if isinstance(h, dict)}
    cards = []
    for i, item in enumerate(reviews[:output_count], 1):
        if not isinstance(item, dict):
            continue
        hid = str(item.get("hypothesis_id") or f"R{i:03d}")
        payload = {"hypothesis": hyp_index.get(hid, {}), "review": item}
        cards.append(_card(run, "reflection", f"REF-{hid}", str(hyp_index.get(hid, {}).get("title") or hid), _summary_text(item), payload, [p["id"] for p in parents if p["id"] == hid]))
    if not cards:
        cards.append(_card(run, "reflection", "REF-BATCH", "Reflection batch", _summary_text(review), {"hypothesis": {}, "review": review}, [p["id"] for p in parents]))
    return cards, {"09_reflection_reviews.json": review}, _finish_usage("reflection")


def _run_evolution(run: dict[str, Any], parents: list[dict[str, Any]], output_count: int, guidance: str):
    _prepare_stage(run, "evolution")
    supervisor = _artifact_data(run, "01_supervisor_config.json") or {}
    proximity = _artifact_data(run, "08_proximity_clusters.json") or {}
    compact = _artifact_data(run, "07b_paper_memory_compact.json") or {}
    selected_hypotheses = [p.get("payload", {}).get("hypothesis", {}) for p in parents if isinstance(p.get("payload", {}).get("hypothesis"), dict) and p.get("payload", {}).get("hypothesis")]
    selected_reviews = [p.get("payload", {}).get("review", {}) for p in parents]
    focused = {}
    if run.get("enable_evolution_retrieval") and selected_hypotheses:
        focused, focused_compact = _focused_evolution_retrieval(run, selected_hypotheses, guidance)
        compact = {"base_memory": compact, "evolution_focused_retrieval_memory": focused_compact}
    evo = EvolutionAgent(model=run["model"])
    cards = []
    for i, hyp in enumerate(selected_hypotheses[:output_count], 1):
        review = selected_reviews[i - 1] if i - 1 < len(selected_reviews) else {}
        result = evo.feasibility(goal=run["objective"], preferences=supervisor, hypothesis=hyp, review=review, proximity_context=proximity, paper_memory_compact=compact, human_stage_guidance=guidance)
        hid = str(hyp.get("hypothesis_id") or f"H{i:03d}")
        cards.append(_card(run, "evolution", f"EVO-{hid}", f"Evolved: {hyp.get('title') or hid}", _summary_text(result), {"source_hypothesis": hyp, "source_review": review, "evolution": result}, [parents[i - 1]["id"]]))
    if len(selected_hypotheses) > 1 and len(cards) < output_count:
        result = evo.out_of_box(goal=run["objective"], preferences=supervisor, hypotheses=selected_hypotheses, reviews=selected_reviews, proximity_context=proximity, paper_memory_compact=compact, human_stage_guidance=guidance)
        cards.append(_card(run, "evolution", "EVO-OUT-OF-BOX", "Out-of-box evolution", _summary_text(result), {"evolution": result, "source_hypotheses": selected_hypotheses}, [p["id"] for p in parents]))
    artifacts = {"10_evolution_outputs.json": [c["payload"] for c in cards]}
    if focused:
        artifacts["09b_evolution_focused_retrieval.json"] = focused
    return cards, artifacts, _finish_usage("evolution")


def _run_candidate_ranking(run: dict[str, Any], parents: list[dict[str, Any]], output_count: int, guidance: str):
    _prepare_stage(run, "candidate_ranking")
    prompt = f"""Rank the supplied evolved scientific candidates for the user's stated research objective.\n\nObjective:\n{run['objective']}\n\nHuman stage guidance:\n{guidance or '(none)'}\n\nCandidates:\n{compact_json([p.get('payload', {}) for p in parents])}\n\nReturn strict JSON: {{\"ranked_candidates\":[{{\"candidate_id\":\"...\",\"rank\":1,\"title\":\"...\",\"rationale\":\"...\",\"major_uncertainties\":[\"...\"],\"next_decisive_test\":\"...\"}}]}}. Do not invent evidence not present in the candidate payloads."""
    ranked = ask_llm_json(prompt, model=run["model"], agent="ranking", purpose="candidate_ranking")
    items = ranked.get("ranked_candidates", []) if isinstance(ranked, dict) else []
    cards = []
    for i, item in enumerate(items[:output_count], 1):
        if not isinstance(item, dict):
            continue
        cid = str(item.get("candidate_id") or f"C{i:03d}")
        cards.append(_card(run, "candidate_ranking", f"RANK-{i:02d}-{cid}", str(item.get("title") or cid), str(item.get("rationale") or ""), item, [p["id"] for p in parents]))
    return cards, {"11_candidate_ranking.json": ranked}, _finish_usage("candidate_ranking")


def _focused_evolution_retrieval(run: dict[str, Any], hypotheses: list[dict[str, Any]], guidance: str) -> tuple[dict[str, Any], dict[str, Any]]:
    lit = LiteratureAgent(model=run["model"], config_path=str(CONFIG_PATH))
    retrieval_results: list[SubtopicRetrievalResult] = []
    selected_packets: list[EvidencePacket] = []
    by_hypothesis = []
    objective_terms = _informative_terms(run["objective"], 8)
    for hyp in hypotheses:
        hid = str(hyp.get("hypothesis_id") or "HYP")
        hyp_terms = _informative_terms(" ".join(str(hyp.get(k, "")) for k in ("title", "hypothesis", "mechanistic_rationale", "candidate_intervention_or_focus")), 8)
        base = " ".join((hyp_terms + objective_terms)[:8])
        queries = [base]
        if guidance:
            queries.append(f"{base} {guidance[:140]}")
        queries = [" ".join(q.split()) for q in queries if q.strip()][:3]
        if run.get("runtime_mode") == "dry_run":
            result = SubtopicRetrievalResult(subtopic_id=f"EVOL_{hid}", queries=queries, warnings=["Network retrieval skipped in dry_run mode."])
        else:
            result = lit.retrieve_subtopic(
                subtopic_id=f"EVOL_{hid}", queries=queries, sources=run["literature_sources"], max_queries=3,
                raw_papers_per_source_query=4, ai_papers_per_subtopic=2, cutoff_year=run["cutoff_year"],
                use_pubtator=run.get("use_pubtator", False), use_evidence_selector=True,
                evidence_selector_model=run["model"], subtopic_payload={"subtopic_id": f"EVOL_{hid}", "search_queries": queries, "focus_seed": hyp},
            )
        retrieval_results.append(result)
        selected_packets.extend(result.evidence_packets)
        by_hypothesis.append({"hypothesis_id": hid, "queries": queries, "warnings": result.warnings, "evidence_packets": [p.to_dict() for p in result.evidence_packets]})
    memory = build_paper_memory(retrieval_results=retrieval_results, selected_packets=selected_packets, axis_id="evolution", axis_name="Focused Evolution retrieval", cutoff_year=run["cutoff_year"])
    compact = compact_memory_for_reflection(memory, max_entries=max(40, len(hypotheses) * 6))
    return {"retrieval_by_hypothesis": by_hypothesis, "paper_memory": memory}, compact


def _informative_terms(text: str, limit: int) -> list[str]:
    stop = {"with", "from", "that", "this", "using", "into", "have", "been", "were", "their", "which", "identify", "research", "objective", "hypothesis", "candidate", "novel"}
    out = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text):
        if token.lower() in stop or token.lower() in {x.lower() for x in out}:
            continue
        out.append(token)
        if len(out) >= limit:
            break
    return out


def _paper_memory_from_run(run: dict[str, Any]) -> dict[str, Any]:
    raw = _artifact_data(run, "04_literature_results.json") or []
    entries = []
    for item in raw:
        for ep in item.get("evidence_packets", []) if isinstance(item, dict) else []:
            entries.append({"stable_id": ep.get("paper_id"), "title": ep.get("title"), "abstract": ep.get("text"), "source_apis": [ep.get("source")], "used_in_synthesis": True, "selected_evidence_ids": [ep.get("evidence_id")], "statuses": ["used_in_synthesis"]})
    return {"counts": {"memory_entries": len(entries), "used_in_synthesis": len(entries)}, "entries": entries}


def _selected_parents(run: dict[str, Any], source_stage: str, selected_ids: list[str], include_all: bool) -> list[dict[str, Any]]:
    cards = [c for c in run.get("stages", {}).get(source_stage, []) if c.get("status") == "active"]
    if include_all:
        return cards
    ids = set(selected_ids)
    if not ids:
        return [c for c in cards if c.get("selected_for_next_stage")]
    return [c for c in cards if c.get("id") in ids]


def _card(run: dict[str, Any], stage: str, card_id: str, title: str, summary: str, payload: dict[str, Any], parent_ids: list[str] | None = None) -> dict[str, Any]:
    return {
        "id": card_id, "title": title or card_id, "summary": summary or "", "source_stage": stage,
        "status": "active", "selected_for_next_stage": False, "selection_source": "auto", "selection_reason": "Generated by stage.",
        "parent_ids": parent_ids or [], "run_id": run["run_id"], "stage": stage, "created_at": now(), "payload": payload,
    }


def _serialize_retrieval(result: SubtopicRetrievalResult, axis_id: str | None, axis: dict[str, Any], subtopic: dict[str, Any]) -> dict[str, Any]:
    return {
        "axis_id": axis_id, "subtopic": subtopic, "subtopic_id": result.subtopic_id, "queries": result.queries,
        "records": [r.to_dict() for r in result.records], "evidence_packets": [p.to_dict() for p in result.evidence_packets],
        "warnings": result.warnings, "raw_records_count": result.raw_records_count,
        "deduped_candidate_count": result.deduped_candidate_count, "pubtator_annotated_count": result.pubtator_annotated_count,
        "balanced_candidate_slate": result.balanced_candidate_slate, "evidence_selector_payload": result.evidence_selector_payload,
        "resolved_evidence_selection": result.resolved_evidence_selection,
        "axis": axis if isinstance(axis, dict) else {},
    }


def _summary_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload[:1200]
    if not isinstance(payload, dict):
        return str(payload)[:1200]
    for key in ("summary", "synthesis", "rationale", "recommendation", "overall_assessment", "hypothesis"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:1200]
    return json.dumps(payload, ensure_ascii=False, default=str)[:1200]


def _artifact_data(run: dict[str, Any], filename: str) -> Any:
    path = run_dir(run["run_id"]) / filename
    return read_json(path)


def _upsert_artifact(run: dict[str, Any], artifact_id: str, label: str, filename: str, stage: str, data: Any) -> None:
    run["artifacts"] = [a for a in run.get("artifacts", []) if a.get("filename") != filename]
    run["artifacts"].append({"id": artifact_id, "label": label, "filename": filename, "stage": stage, "data": data})


def _finish_stage(run: dict[str, Any], stage: str, outputs: list[dict[str, Any]], artifacts: dict[str, Any], usage: dict[str, Any], message: str) -> None:
    run.setdefault("stages", {})[stage] = outputs
    run["current_stage"] = stage
    run["status"] = "checkpoint_ready"
    for filename, data in artifacts.items():
        _upsert_artifact(run, filename.replace("_", "-").replace(".json", ""), filename.replace("_", " ").replace(".json", "").title(), filename, stage, data)
    run.setdefault("logs", []).append({"id": f"log-{stage}-{uuid.uuid4().hex[:6]}", "timestamp": now(), "level": "info", "agent": stage, "message": message})
    prior = [u for u in run.get("usage_by_stage", []) if u.get("stage") != stage]
    run["usage_by_stage"] = [*prior, usage]
    run["usage"] = {
        "calls": sum(int(u.get("calls") or 0) for u in run["usage_by_stage"]),
        "input_tokens": sum(int(u.get("input_tokens") or 0) for u in run["usage_by_stage"]),
        "output_tokens": sum(int(u.get("output_tokens") or 0) for u in run["usage_by_stage"]),
        "estimated_cost_usd": round(sum(float(u.get("estimated_cost_usd") or 0) for u in run["usage_by_stage"]), 6),
    }
    write_run(run, artifacts)
