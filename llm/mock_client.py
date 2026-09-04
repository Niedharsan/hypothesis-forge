from __future__ import annotations

import inspect
import json
import re
import time
from pathlib import Path
from typing import Any

from runtime.context import current_runtime, current_llm_call_count
from utils.run_logger import log_event, log_gemini_call


def mock_llm_available() -> bool:
    return True


def ask_mock_json(prompt: str, model: str | None = None, *, agent: str | None = None, purpose: str | None = None, temperature: float | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return schema-shaped fake JSON so dry-run can traverse the real workflow.

    The content is intentionally generic: it is for estimating call counts/tokens and
    testing orchestration, not for producing scientific claims.
    """
    runtime = current_runtime()
    call_index = current_llm_call_count()
    selected_model = model or "mock-gemini-dry-run"
    caller = f"{agent}.{purpose}" if agent and purpose else (agent or purpose or _caller_name())
    started = time.perf_counter()
    payload = _payload_for_prompt(prompt, call_index=call_index)
    text = json.dumps(payload, ensure_ascii=False)
    input_tokens = max(1, len(prompt or "") // 4)
    output_tokens = max(runtime.dry_run.mock_response_tokens, len(text) // 4)
    log_gemini_call(
        caller=caller,
        model=f"dry-run:{selected_model}",
        prompt=prompt,
        response_text=text,
        usage={"input_tokens": input_tokens, "output_tokens": output_tokens, "total_token_count": input_tokens + output_tokens},
        duration_s=time.perf_counter() - started,
        metadata={**(metadata or {}), "agent": agent, "purpose": purpose, "stage": f"{agent}.{purpose}" if agent and purpose else (agent or purpose or caller), "temperature": temperature, "mock": True},
    )
    log_event(
        "dry_run",
        "mock_llm_json",
        {
            "caller": caller,
            "agent": agent,
            "purpose": purpose,
            "model": selected_model,
            "call_index": call_index,
            "input_tokens_estimated": input_tokens,
            "output_tokens_estimated": output_tokens,
            "response_kind": _response_kind(prompt),
        },
    )
    return payload


def _payload_for_prompt(prompt: str, call_index: int) -> dict[str, Any]:
    kind = _response_kind(prompt)
    objective = (_extract_after(prompt, "Original objective:") or _extract_after(prompt, "Research objective:") or _extract_after(prompt, "Objective:") or "the research objective")
    objective_short = _short(objective, 100)


    # HypothesisForge final candidate ranking dry-run contract.
    if "rank the supplied evolved scientific candidates" in prompt.lower() and "ranked_candidates" in prompt:
        ids = list(dict.fromkeys(re.findall(r'"id"\s*:\s*"([^"]+)"', prompt))) or ["EVO-S001"]
        return {"ranked_candidates": [
            {"candidate_id": cid, "rank": i, "title": f"Dry-run ranked candidate {i}", "rationale": "Dry-run structural ranking only; normal mode is required for scientific assessment.", "major_uncertainties": ["Scientific evidence not evaluated in dry-run mode."], "next_decisive_test": "Run the next objective-specific validation step."}
            for i, cid in enumerate(ids[:10], 1)
        ]}

    # v52 granularity supervisor helper prompts
    if "criterion_2_granularity_level" in prompt and "Allowed values" in prompt:
        return {"criterion_2_granularity_level": "pathway_level"}
    if "criterion_1_granularity_level" in prompt and "Allowed values" in prompt:
        return {"criterion_1_granularity_level": "biological_system_level"}

    # v58/v60 subtopic coverage prompts
    if "VERSION 2: AXIS-FIRST + ENTITY-MAP-INFORMED COVERAGE MAP" in prompt:
        axis_id = _axis_id(prompt) or "A01"
        return _mock_axis_coverage_subtopics(axis_id, include_entity_enrichment=True)

    if "creating broad query families for axis-level entity/concept mapping" in prompt.lower() or "query-family" in prompt.lower() and "query_families" in prompt.lower():
        return {"query_families": [
            {"family_id": "QF01", "family_name": "axis general", "query": f"{objective_short} mechanism", "purpose": "broad axis coverage"},
            {"family_id": "QF02", "family_name": "branch mechanisms", "query": f"{objective_short} pathway branches", "purpose": "branch coverage"},
            {"family_id": "QF03", "family_name": "actionable handles", "query": f"{objective_short} perturbation assay", "purpose": "testable handles"}
        ], "coverage_rationale": "Dry-run query families."}

    if "structured, evidence-preserving synthesis" in prompt and "subtopic_coverage" in prompt and "unexplored_intersections" in prompt:
        axis_id = _axis_id(prompt) or "A01"
        return _mock_axis_synthesis(axis_id)

    if "AXIS-BATCH selection call" in prompt and "subtopic_selections" in prompt:
        axis_id = _axis_id(prompt) or "A01"
        target = _target_int(prompt, "Target number of papers per subtopic", default=3)
        grouped: dict[str, list[str]] = {}
        for cid in re.findall(r'"candidate_id"\s*:\s*"([^"]+)"', prompt):
            sid = re.sub(r"_C\d+$", "", cid)
            grouped.setdefault(sid, [])
            if cid not in grouped[sid]:
                grouped[sid].append(cid)
        return {
            "axis_id": axis_id,
            "selection_scope": "axis_batch",
            "subtopic_selections": [
                {
                    "subtopic_id": sid,
                    "selection_decision": "selected" if ids else "insufficient_candidates",
                    "selected_candidate_ids": ids[:target],
                    "selected_papers": [
                        {"candidate_id": cid, "covered_branches": ["dry-run branch"], "selection_reason": "Dry-run preserves first available candidate per branch."}
                        for cid in ids[:target]
                    ],
                    "covered_branches": ["dry-run branch"],
                    "uncovered_branches": [],
                    "discarded_or_deprioritized": [],
                    "selection_notes": [{"issue": "dry_run", "details": "Mock axis-batch EvidenceSelector output."}],
                }
                for sid, ids in grouped.items()
            ],
            "axis_level_selection_notes": [{"issue": "dry_run", "details": "Mock axis-batch selection."}],
        }

    if "select a compact, diverse set of evidence papers from a retrieved candidate slate for one subtopic" in prompt:
        target = _target_int(prompt, "Target number of papers", default=3)
        ids = list(dict.fromkeys(re.findall(r'"candidate_id"\s*:\s*"([^"]+)"', prompt)))
        sid = re.sub(r"_C\d+$", "", ids[0]) if ids else (_axis_id(prompt) or "T01")
        return {
            "subtopic_id": sid,
            "selection_decision": "selected" if ids else "insufficient_candidates",
            "selected_candidate_ids": ids[:target],
            "selected_papers": [
                {"candidate_id": cid, "covered_branches": ["dry-run branch"], "selection_reason": "Dry-run selected candidate."}
                for cid in ids[:target]
            ],
            "covered_branches": ["dry-run branch"],
            "uncovered_branches": [],
            "discarded_or_deprioritized": [],
            "selection_notes": [{"issue": "dry_run", "details": "Mock subtopic EvidenceSelector output."}],
        }

    # v42 LLM supervisor config
    if "You are the Supervisor Config Agent" in prompt:
        return {
            "goal_summary": objective_short,
            "objective_type": "drug_repurposing" if "repurpos" in objective.lower() or "drug" in objective.lower() else "general_scientific_discovery",
            "target_context": "dry-run target context",
            "constraints": ["Preserve the user-stated constraints.", "Treat novelty claims as uncertain until reviewed."],
            "success_criteria": ["mechanistic specificity", "novelty potential", "testability", "clear falsifiable predictions"],
            "transferability_criteria": ["Transferable evidence is useful when the mechanism or process is plausibly conserved and testable in the target context."],
            "generation_guidance": {"diversity_dimensions": ["mechanism", "system", "method", "readout"], "mechanistic_depth": "specific but not over-constrained", "avoid_failure_modes": ["generic route labels", "duplicate mechanisms"]},
            "literature_guidance": {"direct_evidence": "evidence in the target context", "transferable_evidence": "evidence in related contexts", "evidence_to_preserve": ["mechanisms", "entities", "readouts"], "gap_types_to_preserve": ["unresolved mechanisms", "underexplored intervention points"]},
            "reflection_guidance": {"critical_review_dimensions": ["correctness", "specificity", "testability", "novelty risk"], "novelty_risk_handling": "mark uncertain instead of overclaiming", "specificity_expectations": "surface specific entities and readouts when broad labels are used"},
        }

    # v35+ axis-first prompts
    if "Create exactly 10 maximally diverse discovery axes" in prompt:
        return {"axes": [
            {"axis_id": f"A{i:02d}", "axis_name": f"Dry-run discovery axis {i}", "biological_vulnerability": f"Distinct route-level vulnerability {i}", "sub_branches": ["upstream driver", "core mechanism", "readout"], "why_relevant_to_goal": objective_short, "transfer_source_contexts": ["related context"], "diversity_rationale": "Dry-run non-overlap placeholder."}
            for i in range(1, 11)
        ], "diversity_audit": "Dry-run axes only; not scientific output."}

    if "convert one discovery axis into focused, diverse, researchable literature subtopics" in prompt:
        axis_id = _axis_id(prompt) or "A01"
        return {"axis_id": axis_id, "subtopics": [
            {"subtopic_id": f"{axis_id}_T{i:02d}", "question": f"What is subtopic {i} for {axis_id}?", "rationale": "Dry-run rationale.", "distinct_angle": "Dry-run distinct level of the axis.", "search_queries": [f"{axis_id} mechanism {i}", f"{axis_id} evidence {i}", f"{axis_id} readout {i}"]}
            for i in range(1, 4)
        ], "coverage_audit": "Dry-run subtopics cover different axis levels."}

    if "Use the retrieved abstracts in relation to the assigned discovery axis" in prompt:
        axis_id = _axis_id(prompt) or "A01"
        return {"axis_id": axis_id, "known_findings": ["Dry-run known finding."], "mechanisms_processes_entities": ["Dry-run mechanism", "Dry-run entity"], "direct_evidence": ["Dry-run direct evidence."], "transferable_evidence": ["Dry-run transferable evidence."], "gaps_contradictions_underexplored": ["Dry-run gap."], "additional_search_queries": [f"{axis_id} confidence query"], "evidence_summary": "Dry-run axis evidence synthesis."}

    if "global literature synthesis" in prompt.lower() and "cross-topic synthesis" in prompt.lower():
        return {"global_known_findings": ["Dry-run global finding."], "cross_topic_mechanisms_processes_entities": ["Dry-run shared mechanism."], "cross_topic_connections": ["Dry-run connection."], "global_gaps_contradictions_underexplored": ["Dry-run global gap."], "promising_hypothesis_directions": ["Dry-run hypothesis direction."], "additional_search_queries": [f"{objective_short} follow-up"], "global_evidence_summary": "Dry-run integrated synthesis."}

    if "creating initial testable hypotheses from axis-level literature syntheses" in prompt:
        return {"hypotheses": [
            {"hypothesis_id": f"S{i:03d}", "source_axis_ids": [f"A{((i-1)%10)+1:02d}"], "source_subtopic_ids": [f"A{((i-1)%10)+1:02d}_T01"], "title": f"Dry-run axis-literature hypothesis {i}", "hypothesis": f"A testable dry-run hypothesis {i} grounded in axis-level synthesis.", "candidate_intervention_or_focus": "candidate focus placeholder", "mechanistic_rationale": "Dry-run rationale placeholder.", "falsifiable_predictions": ["Prediction placeholder"], "key_assumptions": ["Assumption placeholder"], "first_validation_step": "Run a decisive validation assay.", "novelty_risk": "unknown", "must_verify_later": ["prior art", "evidence strength"]}
            for i in range(1, 12)
        ], "unused_or_weak_axes": []}

    if ("You are a specialist in scientific hypothesis critique, triage, and next-step routing" in prompt or ("You are an expert reviewer evaluating scientific hypotheses" in prompt and "hypothesis_reviews" in prompt)):
        return {
            "reflection_mode": "supervisor_guided_global_with_proximity",
            "hypothesis_reviews": [
                {
                    "hypothesis_id": "S001",
                    "decision": "keep_for_evolution",
                    "proximity_group_ids": ["G01"],
                    "merge_target_hypothesis_id": None,
                    "alignment_with_supervisor_goal": "Dry-run aligned.",
                    "hypothesis_logic_summary": "Dry-run logic summary.",
                    "supporting_evidence_ids_or_papers": [],
                    "strengths": ["Dry-run strength."],
                    "weaknesses": ["Dry-run weakness."],
                    "explicit_assumptions": ["Dry-run assumption."],
                    "evidence_gaps_or_contradictions": ["Dry-run evidence gap."],
                    "overclaim_or_scope_risk": "Dry-run scope risk.",
                    "specific_revision_or_evolution_instructions": ["Dry-run evolve with specific candidate and assay."],
                    "suggested_next_searches": ["dry-run next search"],
                    "focus_seeds": [],
                    "scores": {"scientific_soundness": 6, "alignment_with_goal": 6, "novelty_under_supervisor_definition": 5, "mechanistic_specificity": 5, "evidence_support": 4, "testability": 7, "potential_impact": 5, "rejection_pressure": 4},
                    "recommended_for_evolution": True,
                },
                {
                    "hypothesis_id": "S002",
                    "decision": "send_to_generation",
                    "proximity_group_ids": ["G01"],
                    "merge_target_hypothesis_id": None,
                    "alignment_with_supervisor_goal": "Dry-run partially aligned.",
                    "hypothesis_logic_summary": "Dry-run immature branch.",
                    "supporting_evidence_ids_or_papers": [],
                    "strengths": ["Dry-run specific fragment."],
                    "weaknesses": ["Not mature enough as a hypothesis."],
                    "explicit_assumptions": [],
                    "evidence_gaps_or_contradictions": ["Needs targeted evidence."],
                    "overclaim_or_scope_risk": "Immature branch.",
                    "specific_revision_or_evolution_instructions": [],
                    "suggested_next_searches": ["dry-run focused branch evidence"],
                    "focus_seeds": [{"seed_id": "RFS001", "source_stage": "reflection", "source_hypothesis_ids": ["S002"], "components": ["dry-run component"], "seed_summary": "Dry-run reflection seed.", "reason_for_generation": "Useful but immature branch.", "suggested_queries": ["dry-run focused branch evidence"], "max_selected_papers": 2}],
                    "scores": {"scientific_soundness": 4, "alignment_with_goal": 5, "novelty_under_supervisor_definition": 5, "mechanistic_specificity": 4, "evidence_support": 3, "testability": 5, "potential_impact": 4, "rejection_pressure": 7},
                    "recommended_for_evolution": False,
                },
            ],
            "keep_for_evolution": ["S001"],
            "revise_for_evolution": [],
            "merge_recommendations": [],
            "rejected_hypotheses": [],
            "needs_more_literature": [],
            "focus_seeds": [{"seed_id": "RFS001", "source_stage": "reflection", "source_hypothesis_ids": ["S002"], "components": ["dry-run component"], "seed_summary": "Dry-run reflection seed.", "reason_for_generation": "Useful but immature branch.", "suggested_queries": ["dry-run focused branch evidence"], "max_selected_papers": 2}],
            "batch_summary": {"strongest_hypotheses": ["S001"], "weakest_hypotheses": ["S002"], "main_relatedness_patterns_from_proximity": ["Dry-run relatedness group."], "common_weaknesses": ["Dry-run common weakness."], "recommended_next_step": "Dry-run proceed to Evolution and focused Generation."},
        }

    if "You are a specialist in hypothesis relatedness, redundancy detection, and semantic proximity analysis" in prompt:
        return {
            "proximity_mode": "relatedness_redundancy_focus_seed",
            "hypothesis_decisions": [
                {"hypothesis_id": "S001", "decision": "keep_distinct", "canonical_representative_id": "S001", "relationship_to_representative": "self", "reason": "Dry-run survivor.", "useful_components_if_absorbed": []},
                {"hypothesis_id": "S002", "decision": "absorb_duplicate", "canonical_representative_id": "S001", "relationship_to_representative": "near_duplicate", "reason": "Dry-run near duplicate.", "useful_components_if_absorbed": ["dry-run component"]},
            ],
            "proximity_groups": [
                {"group_id": "G01", "group_label": "Dry-run related group", "hypothesis_ids": ["S001", "S002"], "relationship": "near_duplicate_group", "canonical_representative_ids": ["S001"], "absorbed_hypothesis_ids": ["S002"], "related_but_distinct_ids": [], "semantic_basis": "Dry-run overlap."}
            ],
            "survivor_hypothesis_ids": ["S001", "S003", "S004", "S005"],
            "survivor_hypotheses": [
                {"hypothesis_id": f"S{i:03d}", "title": f"Dry-run original survivor {i}", "hypothesis": f"Dry-run specific survivor hypothesis {i}.", "source_hypothesis_ids": [f"S{i:03d}"], "retained_components": ["dry-run component"], "proximity_group_id": f"G{i:02d}", "survivor_type": "original"}
                for i in range(1, 5)
            ],
            "absorbed_or_duplicate_hypotheses": [
                {"hypothesis_id": "S002", "absorbed_into": "S001", "reason": "Dry-run duplicate/overlap.", "useful_components_preserved": ["dry-run component"]}
            ],
            "focus_seeds": [
                {"seed_id": "FS001", "source_stage": "proximity", "source_hypothesis_ids": ["S002"], "components": ["dry-run component A", "dry-run component B"], "seed_summary": "Dry-run lost branch seed.", "reason_for_generation": "Useful absorbed material not preserved by a survivor.", "suggested_queries": ["dry-run focused query"], "attached_evidence_ids_or_papers": [], "max_selected_papers": 2}
            ],
            "proximity_audit": {"input_hypothesis_count": 0, "survivor_count": 4, "absorbed_or_duplicate_count": 1, "focus_seed_count": 1, "coverage_notes": ["Dry-run proximity output."], "risks_or_uncertainties": ["Dry-run cannot assess scientific content."]}
        }
    if kind == "proximity":
        ids = _strategy_ids(prompt)
        return {"merged_outputs": [], "unchanged_output_ids": ids[:10]}

    if kind == "evolution":
        ids = _strategy_ids(prompt) or ["S001"]
        strategy = "feasibility"
        lp = prompt.lower()
        if "\"strategy\": \"simplify\"" in lp or "make it simpler" in lp:
            strategy = "simplify"
        elif "\"strategy\": \"combine\"" in lp or "combine the best parts" in lp:
            strategy = "combine"
        elif "\"strategy\": \"out_of_box\"" in lp or "out-of-box" in lp or "analogous elements" in lp:
            strategy = "out_of_box"
        return {
            "strategy": strategy,
            "parent_ids": ids[:2],
            "title": "Dry-run evolved hypothesis",
            "evolved_hypothesis": f"A dry-run evolved, more testable hypothesis for {objective_short}.",
            "simplified_claim": f"Dry-run simplified claim for {objective_short}.",
            "candidate_entities_or_interventions": ["dry-run candidate/probe"],
            "target_or_mechanistic_node": "dry-run mechanism node",
            "disease_context_or_subset": "dry-run context",
            "mechanism": "Use reflection and proximity context to tighten the mechanism.",
            "anticipated_outcomes": ["Objective-relevant readout changes in the predicted direction."],
            "experiments": ["Run a small validation assay or computational analysis."],
            "novelty_considerations": "Dry-run novelty is not scientific evidence.",
            "prior_art_risk": "unknown",
            "feasibility_notes": "Dry-run evolution is only structural.",
            "open_questions": ["direct prior art", "validation evidence"],
        }

    if kind == "intent":
        return {
            "intent_type": "general_scientific_discovery",
            "desired_output_type": "ranked_testable_hypotheses",
            "key_entities": _keywords(objective_short)[:5],
            "must_include_concepts": ["mechanism", "evidence", "validation"],
            "exclude_or_deprioritize": ["unsupported speculation"],
            "search_queries": [objective_short, f"{objective_short} mechanism evidence"],
            "ranking_preferences": ["novelty", "testability", "evidence grounding"],
        }

    if kind == "search_plan":
        return {
            "boolean_queries": [f'("{objective_short}") AND (mechanism OR evidence)'],
            "natural_language_queries": [objective_short, f"{objective_short} review mechanism evidence"],
            "focused_queries": [f"{objective_short} target pathway", f"{objective_short} validation assay"],
            "broader_context_queries": [f"{objective_short} broader scientific context"],
            "prior_art_queries": [f"{objective_short} prior art"],
            "source_preferences": ["PubMed", "EuropePMC", "OpenAlex", "Crossref", "SemanticScholar"],
            "notes": ["Dry-run search plan."],
        }

    if kind == "rerank":
        ids = _evidence_ids(prompt)
        return {"ranked_evidence_ids": ids, "relevance_notes": ["Dry-run reranking placeholder."]}

    if kind == "coverage":
        return {
            "categories": [
                {"category": "direct evidence", "status": "weak", "evidence_ids": _evidence_ids(prompt)[:2], "rationale": "Dry-run placeholder."},
                {"category": "validation feasibility", "status": "missing", "evidence_ids": [], "rationale": "Dry-run placeholder."},
            ],
            "follow_up_queries": [f"{objective_short} validation", f"{objective_short} prior art"],
        }

    return {"notes": ["Dry-run fallback JSON"], "result": "placeholder"}


def _response_kind(prompt: str) -> str:
    p = prompt.lower()
    if ("extract specific biological statements" in p or "extract a balanced set of discovery routes" in p) and "biological_statements" in p:
        return "route_planning"
    if ("draft hypothesis ideas" in p and "generation_debate_summary" in p) or "you are the generationagent. synthesize scientific outputs" in p:
        return "discovery_generation"
    if "you are the generationagent for one biological route" in p and "hypotheses" in p:
        return "route_generation"
    if (("focused retrieval plan" in p or "generate a focused search plan" in p) and "retrieval_goal" in p and "tool_requests" in p) or ("you are the generationagent" in p and "retrieval_goal" in p and "tool_requests" in p):
        return "discovery_search_plan"
    if "you are the generationagent for one biological route" in p and "search_queries" in p and "do not produce hypotheses" in p:
        return "route_search_plan"
    if ("hypothesis review agent - verification planning" in p or "decide what focused scientific retrieval checks" in p) and "retrieval_queries" in p:
        return "reflection_check_plan"
    if "# hypothesis review agent" in p and "overall_assessment" in p and "recommended_action" in p:
        return "reflection_critique"
    if "you are the rankingtournamentagent" in p:
        return "ranking"
    if "you are the proximityagent" in p:
        return "proximity"
    if "you are the evolutionagent" in p:
        return "evolution"
    if "refine the hypothesis below to make it simpler" in p:
        return "evolution"
    if "technological feasibility analysis" in p and "original conceptualization" in p:
        return "evolution"
    if "combine the best parts of the two hypotheses" in p:
        return "evolution"
    if "analogous elements from the provided concepts" in p:
        return "evolution"
    if "classify this" in p and "intent_type" in p:
        return "intent"
    if "source-specific search queries" in p or "retrieval query planner" in p:
        return "source_query_plan"
    if "create a general scientific literature search plan" in p:
        return "search_plan"
    if "rerank" in p or "rank evidence" in p:
        return "rerank"
    if "evidence coverage" in p or "coverage" in p:
        return "coverage"
    return "generic"




def _mock_axis_coverage_subtopics(axis_id: str, *, include_entity_enrichment: bool) -> dict[str, Any]:
    subtopics = []
    base = [
        ("T01", "Dry-run parent mechanism family", ["core process", "upstream branch"], ["dry run mechanism evidence", "dry run pathway branch"]),
        ("T02", "Dry-run complementary mechanism family", ["parallel process", "adaptive branch"], ["dry run complementary evidence", "dry run adaptive branch"]),
        ("T03", "Dry-run actionable biology family", ["candidate handle", "assay readout"], ["dry run perturbation", "dry run assay readout"]),
    ]
    if include_entity_enrichment:
        base.append(("T04", "Dry-run entity-informed branch family", ["retrieval-supported entity", "retrieval-supported pathway"], ["dry run retrieved entity", "dry run entity pathway"]))
    for suffix, name, branches, queries in base:
        sid = f"{axis_id}_{suffix}"
        subtopics.append({
            "subtopic_id": sid,
            "name": name,
            "question": f"What evidence supports {name.lower()} for {axis_id}?",
            "rationale": "Dry-run structured subtopic placeholder.",
            "covered_branches": branches,
            "search_queries": queries,
            "excluded_or_merged_concepts": [],
        })
    return {
        "axis_id": axis_id,
        "axis_explicit_concepts": ["dry-run explicit concept", "dry-run branch concept"],
        "subtopics": subtopics,
        "coverage_audit": [
            {"concept": "dry-run explicit concept", "status": "covered_as_parent", "where_represented": subtopics[0]["subtopic_id"], "reason": "Dry-run coverage."}
        ],
    }


def _mock_axis_synthesis(axis_id: str) -> dict[str, Any]:
    return {
        "axis_id": axis_id,
        "axis_name": "Dry-run axis",
        "current_state_of_knowledge": [
            {"summary": "Dry-run current knowledge summary.", "supporting_evidence_ids": []}
        ],
        "subtopic_coverage": [
            {
                "subtopic_id": f"{axis_id}_T01",
                "subtopic_name": "Dry-run parent mechanism family",
                "supported_branches": [
                    {"branch_name": "dry-run branch", "branch_type": "mechanism", "key_entities": ["ENTITY1"], "evidence_ids": [], "evidence_strength": "weak", "brief_rationale": "Dry-run branch preservation."}
                ],
                "weak_or_missing_branches": [],
                "actionable_handles": [
                    {"handle": "dry-run handle", "handle_type": "other", "relationship_to_evidence": "Dry-run testable handle.", "evidence_ids": []}
                ],
                "subtopic_summary": "Dry-run subtopic synthesis.",
            }
        ],
        "direct_evidence": [],
        "transferable_evidence": [],
        "gaps_contradictions_underexplored": [],
        "methodological_opportunities": [],
        "unexplored_intersections": [
            {"intersection": "dry-run intersection", "components": ["concept", "handle"], "why_interesting": "Dry-run opportunity.", "hypothesis_potential": "Could motivate a dry-run hypothesis.", "supporting_evidence_ids": []}
        ],
        "branch_preservation_audit": [
            {"input_subtopic_or_branch": "dry-run branch", "status": "preserved", "where_represented": f"{axis_id}_T01", "reason": "Dry-run audit."}
        ],
        "additional_search_queries": [],
        "axis_level_summary": "Dry-run evidence-preserving synthesis.",
    }


def _target_int(prompt: str, label: str, *, default: int) -> int:
    pattern = re.escape(label) + r"\s*:?\s*(\d+)"
    m = re.search(pattern, prompt, flags=re.IGNORECASE)
    if not m:
        return default
    try:
        return max(1, int(m.group(1)))
    except Exception:
        return default


def _axis_id(prompt: str) -> str:
    m = re.search(r'"axis_id"\s*:\s*"([A-Za-z0-9_:-]+)"', prompt)
    return m.group(1) if m else ""


def _output(i: int, objective_short: str, evidence_ids: list[str]) -> dict[str, Any]:
    return {
        "title": f"Dry-run hypothesis {i}",
        "hypothesis": f"A testable dry-run hypothesis {i} for {objective_short} links a plausible mechanism to an objective-relevant readout.",
        "answer_candidate_or_strategy": f"Dry-run candidate strategy {i} for {objective_short}",
        "mechanism_bridge": "Placeholder mechanism bridge for workflow simulation only.",
        "expected_outcome_or_readout": "Objective-relevant readout changes in the predicted direction.",
        "first_experiment_or_validation_step": "Run a small decisive validation experiment or computational check.",
        "main_risk": "Dry-run does not evaluate scientific truth.",
        "supporting_evidence_ids": evidence_ids,
        "status": "candidate",
        "evidence_status": "needs_more_evidence",
        "missing_evidence_type": "normal-mode verification required",
    }


def _maybe_tool_requests(prompt: str, objective_short: str) -> list[dict[str, Any]]:
    p = prompt.lower()
    if any(term in p for term in ["drug", "compound", "target", "therapeutic", "repurpos"]):
        return [
            {"tool_name": "opentargets", "query": objective_short, "purpose": "disease-target-drug association check", "parameters": {}},
            {"tool_name": "pubchem", "query": objective_short, "purpose": "compound normalization if a compound is named", "parameters": {}},
        ]
    return []


def _caller_name() -> str:
    for frame in inspect.stack()[2:10]:
        module = inspect.getmodule(frame.frame)
        mod_name = module.__name__ if module else Path(frame.filename).stem
        if mod_name not in {__name__, "llm.provider"}:
            return f"{mod_name}.{frame.function}"
    return "unknown"


def _extract_after(text: str, label: str) -> str:
    idx = text.find(label)
    if idx < 0:
        return ""
    tail = text[idx + len(label):].strip()
    return tail.split("\n\n", 1)[0].strip()


def _short(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    return text[:n].rstrip() or "the objective"


def _evidence_ids(prompt: str) -> list[str]:
    ids = re.findall(r"['\"]evidence_id['\"]\s*:\s*['\"]([^'\"]+)['\"]", prompt)
    ids += re.findall(r"\b(?:E|R|S|A)\d{1,4}-E\d{1,4}\b|\bE\d{1,4}\b", prompt)
    return list(dict.fromkeys(ids))[:12]


def _strategy_ids(prompt: str) -> list[str]:
    ids = re.findall(r"\bS\d{3}\b", prompt)
    return list(dict.fromkeys(ids))[:20]


def _strategy_id(prompt: str) -> str:
    ids = _strategy_ids(prompt)
    return ids[0] if ids else ""


def _keywords(text: str) -> list[str]:
    stop = {"the", "and", "for", "with", "that", "this", "from", "into", "about", "objective", "research"}
    out = []
    for token in re.split(r"[^A-Za-z0-9_+-]+", text):
        token = token.strip()
        if len(token) > 3 and token.lower() not in stop:
            out.append(token)
    return list(dict.fromkeys(out))


def _extract_source_names(prompt: str) -> list[str]:
    names = []
    for name in ["PubMed", "EuropePMC", "SemanticScholar", "OpenAlex", "Crossref", "WebSearch", "Scopus", "ScienceDirect"]:
        if f'"source_name": "{name}"' in prompt or f"source_name': '{name}'" in prompt:
            names.append(name)
    return names
