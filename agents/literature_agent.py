from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from llm.provider import ask_gemini_json
from utils.prompt_loader import render_prompt
from utils.json_compact import compact_json
from utils.config import load_config
from utils.run_logger import log_event
from retrieval.pubmed_api import PubMedAPI
from retrieval.europepmc_api import EuropePMCAPI
from retrieval.openalex_api import OpenAlexAPI
from retrieval.crossref_api import CrossrefAPI
from retrieval.semantic_scholar_api import SemanticScholarAPI
from retrieval.pubtator_api import PubTatorAPI, PubTatorAnnotation, pubtator_entity_terms
from schemas.evidence_packet import EvidencePacket
from schemas.paper_record import PaperRecord
from agents.evidence_selector_agent import EvidenceSelectorAgent


@dataclass
class SubtopicRetrievalResult:
    subtopic_id: str
    queries: list[str]
    records: list[PaperRecord] = field(default_factory=list)
    evidence_packets: list[EvidencePacket] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    raw_records_count: int = 0
    deduped_candidate_count: int = 0
    pubtator_annotated_count: int = 0
    candidate_pool_by_query: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    balanced_candidate_slate: list[dict[str, Any]] = field(default_factory=list)
    evidence_selector_payload: dict[str, Any] = field(default_factory=dict)
    raw_candidate_pool_by_query: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    deduped_candidate_pool: list[dict[str, Any]] = field(default_factory=list)
    resolved_evidence_selection: dict[str, Any] = field(default_factory=dict)
    filtered_candidate_pool_by_query: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    pmid_branch_filter_excluded: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    non_pmid_sanity_excluded: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    candidate_record_by_id: dict[str, PaperRecord] = field(default_factory=dict, repr=False)


@dataclass
class AxisLiteratureResult:
    axis_id: str
    axis: dict[str, Any]
    subtopics_payload: dict[str, Any]
    retrieval_results: list[SubtopicRetrievalResult] = field(default_factory=list)
    evidence_packets: list[EvidencePacket] = field(default_factory=list)
    synthesis: dict[str, Any] = field(default_factory=dict)
    subtopic_generation_context: dict[str, Any] = field(default_factory=dict)


@dataclass
class LiteratureAgentOutput:
    axes_payload: dict[str, Any]
    axis_results: list[AxisLiteratureResult] = field(default_factory=list)
    syntheses: list[dict[str, Any]] = field(default_factory=list)
    global_synthesis: dict[str, Any] = field(default_factory=dict)


class LiteratureAgent:
    """Literature helper: axis -> subtopics/queries -> retrieval -> axis evidence synthesis.

    Retrieval adapters are tools. The LLM parts are axis decomposition, axis synthesis, and global synthesis.
    """

    def __init__(self, model: str = "gemini-2.5-flash-lite", config_path: str = "configs/config.yaml") -> None:
        self.model = model
        allow_s2_unauthenticated = _semantic_scholar_allow_unauthenticated(config_path)
        self.adapters = {
            "PubMed": PubMedAPI(),
            "EuropePMC": EuropePMCAPI(),
            "OpenAlex": OpenAlexAPI(),
            "Crossref": CrossrefAPI(),
            "SemanticScholar": SemanticScholarAPI(allow_unauthenticated=allow_s2_unauthenticated),
        }
        self.pubtator = PubTatorAPI()

    def run_axis_first(
        self,
        *,
        objective: str,
        axes_payload: dict[str, Any],
        sources: list[str],
        max_subtopics_per_axis: int = 5,
        max_queries_per_subtopic: int = 5,
        raw_papers_per_source_query: int = 5,
        ai_papers_per_subtopic: int = 3,
        ai_papers_per_axis: int = 15,
        cutoff_year: int | None = None,
        enable_retrieval: bool = True,
        use_pubtator: bool = False,
        pubtator_raw_papers_per_source_query: int | None = None,
        pubtator_max_candidates_per_subtopic: int | None = None,
        max_axis_query_families: int = 6,
        entity_map_candidates_per_family: int = 10,
        use_query_reviewer: bool = True,
        query_reviewer_model: str | None = None,
        use_evidence_selector: bool = True,
        evidence_selector_model: str | None = None,
        evidence_selector_initial_depth_per_query: int = 3,
        evidence_selector_max_depth_per_query: int = 10,
        enable_pmid_branch_tag_filter: bool = True,
        subtopics_only: bool = False,
    ) -> LiteratureAgentOutput:
        """Axis-first literature workflow using only the fixed v2 route.

        This full-run path intentionally mirrors the single-axis route that was
        validated for branch preservation: query families -> QueryReviewer ->
        broad entity/concept inventory -> v2 entity-map-informed subtopics ->
        targeted retrieval -> EvidenceSelector -> synthesis. The older direct
        v1 axis decomposition is not used here.
        """
        axis_results: list[AxisLiteratureResult] = []
        syntheses: list[dict[str, Any]] = []
        query_reviewer = None
        if use_query_reviewer:
            from agents.query_reviewer_agent import QueryReviewerAgent
            query_reviewer = QueryReviewerAgent(model=query_reviewer_model or self.model)

        for axis in _axes(axes_payload):
            axis_id = str(axis.get("axis_id") or "").strip()
            if not axis_id:
                continue

            raw_query_families_payload = self.generate_axis_query_families(axis, max_query_families=max_axis_query_families)
            query_families_payload = raw_query_families_payload
            query_reviewer_payload: dict[str, Any]
            if query_reviewer is not None:
                query_families_payload = query_reviewer.review_axis_query_families(
                    axis=axis,
                    query_families_payload=raw_query_families_payload,
                    max_query_families=max_axis_query_families,
                )
                query_reviewer_payload = {
                    "enabled": True,
                    "stage": "axis_query_families",
                    "model": query_reviewer_model or self.model,
                    "raw_query_families_payload": raw_query_families_payload,
                    "reviewed_payload": query_families_payload,
                }
            else:
                query_reviewer_payload = {
                    "enabled": False,
                    "stage": "axis_query_families",
                    "raw_query_families_payload": raw_query_families_payload,
                    "reviewed_payload": query_families_payload,
                }

            if enable_retrieval:
                entity_concept_inventory = self.build_axis_entity_concept_inventory(
                    axis,
                    query_families_payload=query_families_payload,
                    sources=sources,
                    candidates_per_family=entity_map_candidates_per_family,
                    cutoff_year=cutoff_year,
                    use_pubtator=use_pubtator,
                )
            else:
                entity_concept_inventory = {
                    "retrieval_disabled": True,
                    "query_families_payload": query_families_payload,
                    "notes": ["Entity-map retrieval disabled; v2 decomposition used query-family plan only."],
                }

            subtopics_payload = self.decompose_axis_v2_entity_map(
                axis,
                concept_inventory=entity_concept_inventory,
                max_subtopics=max_subtopics_per_axis,
                max_queries_per_subtopic=max_queries_per_subtopic,
            )
            subtopic_generation_context = {
                "method": "v2_query_family_reviewed_entity_map_only",
                "raw_query_families_payload": raw_query_families_payload,
                "query_reviewer_payload": query_reviewer_payload,
                "query_families_payload": query_families_payload,
                "entity_concept_inventory": entity_concept_inventory,
                "v1_axis_decomposition_used": False,
            }

            if subtopics_only:
                axis_results.append(AxisLiteratureResult(
                    axis_id=axis_id,
                    axis=axis,
                    subtopics_payload=subtopics_payload,
                    retrieval_results=[],
                    evidence_packets=[],
                    synthesis={},
                    subtopic_generation_context=subtopic_generation_context,
                ))
                continue

            retrieval_results: list[SubtopicRetrievalResult] = []
            packets_by_subtopic: dict[str, list[EvidencePacket]] = {}

            for subtopic in _subtopics(subtopics_payload):
                sid = str(subtopic.get("subtopic_id") or "").strip()
                queries = [str(q).strip() for q in subtopic.get("search_queries", []) if str(q).strip()]
                if enable_retrieval:
                    result = self.retrieve_subtopic(
                        subtopic_id=sid,
                        queries=queries,
                        sources=sources,
                        max_queries=max_queries_per_subtopic,
                        raw_papers_per_source_query=raw_papers_per_source_query,
                        ai_papers_per_subtopic=ai_papers_per_subtopic,
                        cutoff_year=cutoff_year,
                        use_pubtator=use_pubtator,
                        pubtator_raw_papers_per_source_query=pubtator_raw_papers_per_source_query,
                        pubtator_max_candidates_per_subtopic=pubtator_max_candidates_per_subtopic,
                        use_evidence_selector=use_evidence_selector,
                        defer_evidence_selector=use_evidence_selector,
                        evidence_selector_model=evidence_selector_model,
                        evidence_selector_initial_depth_per_query=evidence_selector_initial_depth_per_query,
                        evidence_selector_max_depth_per_query=evidence_selector_max_depth_per_query,
                        enable_pmid_branch_tag_filter=enable_pmid_branch_tag_filter,
                        subtopic_payload=subtopic,
                    )
                else:
                    result = SubtopicRetrievalResult(subtopic_id=sid, queries=queries, warnings=["retrieval disabled"])
                retrieval_results.append(result)
                packets_by_subtopic[sid] = result.evidence_packets

            if enable_retrieval and use_evidence_selector:
                self.select_axis_evidence_batch(
                    axis=axis,
                    subtopics_payload=subtopics_payload,
                    retrieval_results=retrieval_results,
                    ai_papers_per_subtopic=ai_papers_per_subtopic,
                    evidence_selector_model=evidence_selector_model,
                )
                packets_by_subtopic = {r.subtopic_id: r.evidence_packets for r in retrieval_results}

            capped_packets = _select_balanced_packets_by_subtopic(
                packets_by_subtopic,
                max_total=max(1, int(ai_papers_per_axis)),
            )
            synthesis = self.synthesize_axis(axis, subtopics_payload, capped_packets)
            syntheses.append(synthesis)
            axis_results.append(AxisLiteratureResult(
                axis_id=axis_id,
                axis=axis,
                subtopics_payload=subtopics_payload,
                retrieval_results=retrieval_results,
                evidence_packets=capped_packets,
                synthesis=synthesis,
                subtopic_generation_context=subtopic_generation_context,
            ))

        if subtopics_only:
            return LiteratureAgentOutput(
                axes_payload=axes_payload,
                axis_results=axis_results,
                syntheses=[],
                global_synthesis={},
            )

        global_synthesis = self.synthesize_global(objective, syntheses)
        return LiteratureAgentOutput(
            axes_payload=axes_payload,
            axis_results=axis_results,
            syntheses=syntheses,
            global_synthesis=global_synthesis,
        )

    def generate_axis_anchor_queries(self, axis: dict[str, Any], *, max_anchor_queries: int = 3) -> dict[str, Any]:
        prompt = render_prompt(
            "v31/literature_axis_anchor_queries.md",
            axis_json=compact_json(axis),
            max_anchor_queries=str(max_anchor_queries),
        )
        return ask_gemini_json(prompt, model=self.model, agent="literature", purpose="axis_anchor_queries")

    def retrieve_axis_anchors(
        self,
        *,
        axis_id: str,
        queries: list[str],
        sources: list[str],
        max_queries: int = 3,
        raw_papers_per_source_query: int = 5,
        max_anchor_papers: int = 2,
        cutoff_year: int | None = None,
    ) -> SubtopicRetrievalResult:
        result = self.retrieve_subtopic(
            subtopic_id=f"{axis_id}_ANCHOR",
            queries=queries,
            sources=sources,
            max_queries=max_queries,
            raw_papers_per_source_query=raw_papers_per_source_query,
            ai_papers_per_subtopic=max_anchor_papers,
            cutoff_year=cutoff_year,
            use_pubtator=False,
        )
        for packet in result.evidence_packets:
            packet.evidence_type = "axis_anchor_review_or_overview"
            packet.metadata["anchor_for_axis_id"] = axis_id
        return result

    def generate_axis_query_families(self, axis: dict[str, Any], *, max_query_families: int = 6) -> dict[str, Any]:
        prompt = render_prompt(
            "v31/literature_axis_query_families.md",
            axis_json=compact_json(axis),
            max_query_families=str(max_query_families),
        )
        return ask_gemini_json(prompt, model=self.model, agent="literature", purpose="axis_query_families")

    def _search_source(self, source: str, query: str, *, limit: int, cutoff_year: int | None = None) -> list[PaperRecord]:
        """Search one source, passing date cutoffs to adapters that support them.

        PubMed, EuropePMC, and OpenAlex can apply the cutoff inside the API
        query. Other adapters keep the old behavior and are filtered after
        retrieval.
        """
        adapter = self.adapters[source]
        if cutoff_year is not None and source in {"PubMed", "EuropePMC", "OpenAlex"}:
            return adapter.search(query, limit=limit, cutoff_year=cutoff_year)
        return adapter.search(query, limit=limit)

    def build_axis_entity_concept_inventory(
        self,
        axis: dict[str, Any],
        *,
        query_families_payload: dict[str, Any],
        sources: list[str],
        candidates_per_family: int = 10,
        cutoff_year: int | None = None,
        use_pubtator: bool = True,
    ) -> dict[str, Any]:
        """Retrieve a stratified broad candidate pool and harvest entities/concepts.

        This is not final evidence selection. It avoids one flat top-N pool by
        preserving query-family provenance and summarizing concepts by family.
        """
        selected_sources = [s for s in sources if s in self.adapters]
        max_per_family = max(1, min(int(candidates_per_family), 25))
        families = query_families_payload.get("query_families", []) if isinstance(query_families_payload, dict) else []
        family_summaries: list[dict[str, Any]] = []
        all_records: list[PaperRecord] = []

        for idx, family in enumerate(families, start=1):
            if not isinstance(family, dict):
                continue
            fid = str(family.get("family_id") or f"QF{idx:02d}").strip()
            query = str(family.get("query") or "").strip()
            if not query:
                continue
            raw_records: list[PaperRecord] = []
            warnings: list[str] = []
            for source in selected_sources:
                try:
                    found = self._search_source(source, query, limit=max_per_family, cutoff_year=cutoff_year)
                    for record in found:
                        record.raw.setdefault("entity_map_query_families", [])
                        if fid not in record.raw["entity_map_query_families"]:
                            record.raw["entity_map_query_families"].append(fid)
                        record.raw.setdefault("entity_map_queries", [])
                        if query not in record.raw["entity_map_queries"]:
                            record.raw["entity_map_queries"].append(query)
                        record.raw.setdefault("retrieval_sources", [])
                        if source not in record.raw["retrieval_sources"]:
                            record.raw["retrieval_sources"].append(source)
                    raw_records.extend(found)
                    log_event("retrieval", "axis_entity_map_family_search", {"family_id": fid, "source": source, "query": query, "records": len(found)})
                except Exception as exc:
                    warnings.append(f"{source} failed for {fid}: {exc}")
                    log_event("retrieval", "axis_entity_map_family_search_failed", {"family_id": fid, "source": source, "query": query, "error": str(exc)}, status="error")
            deduped = _dedupe_records(raw_records)
            if cutoff_year is not None:
                deduped = [r for r in deduped if r.year is None or int(r.year) <= int(cutoff_year)]
            deduped = deduped[:max_per_family]
            all_records.extend(deduped)
            family_summaries.append({
                "family_id": fid,
                "name": family.get("name"),
                "query": query,
                "coverage_intent": family.get("coverage_intent"),
                "raw_records_count": len(raw_records),
                "deduped_after_cutoff_count": len(deduped),
                "candidate_titles": [_record_brief(r) for r in deduped[:max_per_family]],
                "warnings": warnings,
            })

        deduped_all = _dedupe_records(all_records)
        if cutoff_year is not None:
            deduped_all = [r for r in deduped_all if r.year is None or int(r.year) <= int(cutoff_year)]

        annotations: dict[str, PubTatorAnnotation] = {}
        if use_pubtator:
            annotations = self.pubtator.annotate_pmids([r.pmid for r in deduped_all if r.pmid])
            for r in deduped_all:
                ann = annotations.get(str(r.pmid)) if r.pmid else None
                if ann:
                    r.raw["pubtator"] = ann.compact()
                else:
                    r.raw.setdefault("pubtator", {})

        concept_inventory = _build_concept_inventory(axis, family_summaries, deduped_all)
        concept_inventory["query_families_payload"] = query_families_payload
        concept_inventory["use_pubtator"] = bool(use_pubtator)
        concept_inventory["pubtator_annotated_count"] = len(annotations)
        concept_inventory["candidate_pool_total_after_dedupe_cutoff"] = len(deduped_all)
        return concept_inventory

    def decompose_axis_v2_entity_map(
        self,
        axis: dict[str, Any],
        *,
        concept_inventory: dict[str, Any],
        max_subtopics: int,
        max_queries_per_subtopic: int,
    ) -> dict[str, Any]:
        prompt = render_prompt(
            "v31/literature_decompose_axis_v2_entity_map.md",
            axis_json=compact_json(axis),
            concept_inventory_json=compact_json(concept_inventory),
            max_subtopics=str(max_subtopics),
            max_queries_per_subtopic=str(max_queries_per_subtopic),
        )
        return ask_gemini_json(prompt, model=self.model, agent="literature", purpose="decompose_axis_v2_entity_map")

    def retrieve_subtopic(
        self,
        *,
        subtopic_id: str,
        queries: list[str],
        sources: list[str],
        max_queries: int,
        raw_papers_per_source_query: int,
        ai_papers_per_subtopic: int,
        cutoff_year: int | None = None,
        use_pubtator: bool = False,
        pubtator_raw_papers_per_source_query: int | None = None,
        pubtator_max_candidates_per_subtopic: int | None = None,
        use_evidence_selector: bool = False,
        defer_evidence_selector: bool = False,
        evidence_selector_model: str | None = None,
        evidence_selector_initial_depth_per_query: int = 3,
        evidence_selector_max_depth_per_query: int = 10,
        enable_pmid_branch_tag_filter: bool = True,
        subtopic_payload: dict[str, Any] | None = None,
    ) -> SubtopicRetrievalResult:
        selected_sources = [s for s in sources if s in self.adapters]
        clean_queries = queries[: max(1, int(max_queries))]
        branch_vocab = _derive_branch_vocab(clean_queries, subtopic_payload)
        warnings: list[str] = []
        records: list[PaperRecord] = []
        records_by_query_raw: dict[str, list[PaperRecord]] = {q: [] for q in clean_queries}
        normal_raw_limit = max(1, min(int(raw_papers_per_source_query), 20))
        # PubTator is an annotation/ranking layer, not a reason to inflate retrieval.
        raw_limit = normal_raw_limit

        for query in clean_queries:
            for source in selected_sources:
                try:
                    found = self._search_source(source, query, limit=raw_limit, cutoff_year=cutoff_year)
                    for rank_idx, record in enumerate(found, start=1):
                        record.raw.setdefault("retrieval_queries", [])
                        if query not in record.raw["retrieval_queries"]:
                            record.raw["retrieval_queries"].append(query)
                        record.raw.setdefault("retrieval_query_ranks", {})
                        existing_rank = record.raw["retrieval_query_ranks"].get(query)
                        if existing_rank is None or rank_idx < int(existing_rank):
                            record.raw["retrieval_query_ranks"][query] = rank_idx
                        record.raw.setdefault("retrieval_query_source_ranks", {})
                        record.raw["retrieval_query_source_ranks"].setdefault(query, {})
                        existing_source_rank = record.raw["retrieval_query_source_ranks"][query].get(source)
                        if existing_source_rank is None or rank_idx < int(existing_source_rank):
                            record.raw["retrieval_query_source_ranks"][query][source] = rank_idx
                        record.raw.setdefault("retrieval_sources", [])
                        if source not in record.raw["retrieval_sources"]:
                            record.raw["retrieval_sources"].append(source)
                    records_by_query_raw.setdefault(query, []).extend(found)
                    records.extend(found)
                    log_event("retrieval", "axis_subtopic_source_search", {"subtopic_id": subtopic_id, "source": source, "query": query, "records": len(found), "use_pubtator": use_pubtator})
                except Exception as exc:
                    warnings.append(f"{source} failed for {subtopic_id}: {exc}")
                    log_event("retrieval", "axis_subtopic_source_search_failed", {"subtopic_id": subtopic_id, "source": source, "query": query, "error": str(exc)}, status="error")

        raw_candidate_pool_by_query = {
            q: [_record_candidate_card(r, candidate_id=f"{subtopic_id}_RAW_{_safe_id(q)}_{idx:02d}", branch_vocab=branch_vocab, use_pubtator=use_pubtator) for idx, r in enumerate(recs[: max(1, int(evidence_selector_max_depth_per_query))], start=1)]
            for q, recs in records_by_query_raw.items()
        }

        deduped_all = _dedupe_records_merge_metadata(records)
        if cutoff_year is not None:
            deduped_all = [r for r in deduped_all if r.year is None or int(r.year) <= int(cutoff_year)]

        # Rebuild per-query ranked lists after global dedupe, preserving query membership.
        records_by_query = _records_by_query_after_dedupe(deduped_all, clean_queries)
        candidate_pool_by_query = {
            q: [_record_candidate_card(r, candidate_id=f"{subtopic_id}_{_safe_id(q)}_{idx:02d}", branch_vocab=branch_vocab, use_pubtator=use_pubtator) for idx, r in enumerate(recs[: max(1, int(evidence_selector_max_depth_per_query))], start=1)]
            for q, recs in records_by_query.items()
        }
        deduped_candidate_pool_cards = [
            _record_candidate_card(r, candidate_id=f"{subtopic_id}_DEDUP_{idx:03d}", branch_vocab=branch_vocab, use_pubtator=use_pubtator)
            for idx, r in enumerate(deduped_all, start=1)
        ]

        if use_pubtator and pubtator_max_candidates_per_subtopic:
            deduped_all = deduped_all[: max(1, int(pubtator_max_candidates_per_subtopic))]

        annotated_count = 0
        if use_pubtator:
            annotations = self.pubtator.annotate_pmids([r.pmid for r in deduped_all if r.pmid])
            for r in deduped_all:
                ann = annotations.get(str(r.pmid)) if r.pmid else None
                if ann:
                    r.raw["pubtator"] = ann.compact()
                    annotated_count += 1
                else:
                    r.raw.setdefault("pubtator", {})
            # Rebuild candidate cards after PubTator annotation so cards include entity terms.
            records_by_query = _records_by_query_after_dedupe(deduped_all, clean_queries)
            candidate_pool_by_query = {
                q: [_record_candidate_card(r, candidate_id=f"{subtopic_id}_{_safe_id(q)}_{idx:02d}", branch_vocab=branch_vocab, use_pubtator=use_pubtator) for idx, r in enumerate(recs[: max(1, int(evidence_selector_max_depth_per_query))], start=1)]
                for q, recs in records_by_query.items()
            }
            deduped_candidate_pool_cards = [
                _record_candidate_card(r, candidate_id=f"{subtopic_id}_DEDUP_{idx:03d}", branch_vocab=branch_vocab, use_pubtator=use_pubtator)
                for idx, r in enumerate(deduped_all, start=1)
            ]

        # Optional deterministic guardrails before the LLM curator:
        # 1) PMID-backed records with PubTator requested must have a branch signal
        #    from PubTator entities or lexical title/abstract matching.
        # 2) Non-PMID / unannotated records are NOT penalized for missing PubTator,
        #    but must have lexical support for both target context and the query branch.
        #    This removes metadata fragments/generic records while preserving non-PubMed recall.
        pmid_branch_filter_excluded_records_by_query: dict[str, list[PaperRecord]] = {q: [] for q in clean_queries}
        non_pmid_sanity_excluded_records_by_query: dict[str, list[PaperRecord]] = {q: [] for q in clean_queries}
        if enable_pmid_branch_tag_filter and use_pubtator:
            records_by_query, pmid_branch_filter_excluded_records_by_query = _apply_pmid_branch_tag_filter(
                records_by_query, clean_queries, branch_vocab
            )
        records_by_query, non_pmid_sanity_excluded_records_by_query = _apply_non_pmid_context_branch_sanity_filter(
            records_by_query, clean_queries
        )
        filtered_candidate_pool_by_query = {
            q: [_record_candidate_card(r, candidate_id=f"{subtopic_id}_FILTERED_{_safe_id(q)}_{idx:02d}", branch_vocab=branch_vocab, use_pubtator=use_pubtator) for idx, r in enumerate(recs[: max(1, int(evidence_selector_max_depth_per_query))], start=1)]
            for q, recs in records_by_query.items()
        }
        pmid_branch_filter_excluded = {
            q: [_record_candidate_card(r, candidate_id=f"{subtopic_id}_PMID_EXCLUDED_{_safe_id(q)}_{idx:02d}", branch_vocab=branch_vocab, use_pubtator=use_pubtator) for idx, r in enumerate(recs, start=1)]
            for q, recs in pmid_branch_filter_excluded_records_by_query.items() if recs
        }
        non_pmid_sanity_excluded = {
            q: [_record_candidate_card(r, candidate_id=f"{subtopic_id}_NONPMID_EXCLUDED_{_safe_id(q)}_{idx:02d}", branch_vocab=branch_vocab, use_pubtator=use_pubtator) for idx, r in enumerate(recs, start=1)]
            for q, recs in non_pmid_sanity_excluded_records_by_query.items() if recs
        }

        candidate_count = len(deduped_all)
        target_count = max(1, int(ai_papers_per_subtopic))
        selector_payload: dict[str, Any] = {"enabled": bool(use_evidence_selector)}
        resolved_selection: dict[str, Any] = {"enabled": bool(use_evidence_selector), "selection_resolution_notes": []}
        selected_records: list[PaperRecord]
        balanced_slate_cards = _build_round_robin_candidate_slate(
            records_by_query,
            clean_queries,
            depth_per_query=max(1, int(evidence_selector_initial_depth_per_query)),
            max_depth_per_query=max(1, int(evidence_selector_max_depth_per_query)),
            subtopic_id=subtopic_id,
            branch_vocab=branch_vocab,
            use_pubtator=use_pubtator,
        )

        if use_evidence_selector and balanced_slate_cards and not defer_evidence_selector:
            card_to_record = {str(card["candidate_id"]): card.get("_record") for card in balanced_slate_cards}
            llm_cards = [{k: v for k, v in card.items() if k != "_record"} for card in balanced_slate_cards]
            selector = EvidenceSelectorAgent(model=evidence_selector_model or self.model)
            selector_payload = selector.select_subtopic_evidence(
                subtopic=subtopic_payload or {"subtopic_id": subtopic_id, "search_queries": clean_queries},
                candidate_cards=llm_cards,
                target_papers=target_count,
            )
            selected_records = []
            seen_keys: set[str] = set()
            requested_ids = [str(cid) for cid in selector_payload.get("selected_candidate_ids", [])]
            invalid_ids: list[str] = []
            duplicate_ids: list[str] = []
            resolved_ids: list[str] = []
            for cid in requested_ids:
                rec = card_to_record.get(str(cid))
                if not isinstance(rec, PaperRecord):
                    invalid_ids.append(str(cid))
                    continue
                key = rec.stable_key()
                if key in seen_keys:
                    duplicate_ids.append(str(cid))
                    continue
                seen_keys.add(key)
                selected_records.append(rec)
                resolved_ids.append(str(cid))
                if len(selected_records) >= target_count:
                    break
            fallback_added: list[str] = []
            if len(selected_records) < target_count:
                for card in balanced_slate_cards:
                    rec = card.get("_record")
                    cid = str(card.get("candidate_id"))
                    if not isinstance(rec, PaperRecord):
                        continue
                    key = rec.stable_key()
                    if key in seen_keys:
                        continue
                    selected_records.append(rec)
                    seen_keys.add(key)
                    fallback_added.append(cid)
                    resolved_ids.append(cid)
                    if len(selected_records) >= target_count:
                        break
            resolved_selection = {
                "enabled": True,
                "requested_selected_candidate_ids": requested_ids,
                "resolved_selected_candidate_ids": resolved_ids,
                "invalid_selected_candidate_ids": invalid_ids,
                "duplicate_selected_candidate_ids": duplicate_ids,
                "fallback_added_candidate_ids": fallback_added,
                "resolved_selected_titles": [r.title for r in selected_records],
                "selection_resolution_notes": [],
            }
            if invalid_ids:
                resolved_selection["selection_resolution_notes"].append({"issue": "invalid_selector_ids", "details": invalid_ids})
            if duplicate_ids:
                resolved_selection["selection_resolution_notes"].append({"issue": "duplicate_selector_ids", "details": duplicate_ids})
            if fallback_added:
                resolved_selection["selection_resolution_notes"].append({"issue": "fallback_topup", "details": fallback_added})
        elif use_evidence_selector and defer_evidence_selector and balanced_slate_cards:
            selected_records = []
            selector_payload = {
                "enabled": True,
                "selection_scope": "axis_batch_pending",
                "subtopic_id": subtopic_id,
                "candidate_count": len(balanced_slate_cards),
            }
            resolved_selection = {
                "enabled": True,
                "selection_scope": "axis_batch_pending",
                "requested_selected_candidate_ids": [],
                "resolved_selected_candidate_ids": [],
                "resolved_selected_titles": [],
                "selection_resolution_notes": [
                    {"issue": "axis_batch_pending", "details": "EvidenceSelector selection is deferred to one batched call for the parent axis."}
                ],
            }
        else:
            selected_records = deduped_all[:target_count]
            resolved_selection = {
                "enabled": False,
                "requested_selected_candidate_ids": [],
                "resolved_selected_candidate_ids": [f"deterministic_{idx:03d}" for idx, _ in enumerate(selected_records, start=1)],
                "resolved_selected_titles": [r.title for r in selected_records],
                "selection_resolution_notes": [{"issue": "evidence_selector_disabled", "details": "Used first N deduped candidates."}],
            }

        packets = [_record_to_packet(subtopic_id, i, r) for i, r in enumerate(selected_records, start=1)]
        return SubtopicRetrievalResult(
            subtopic_id=subtopic_id,
            queries=clean_queries,
            records=selected_records,
            evidence_packets=packets,
            warnings=warnings,
            raw_records_count=len(records),
            deduped_candidate_count=candidate_count,
            pubtator_annotated_count=annotated_count,
            candidate_pool_by_query=candidate_pool_by_query,
            balanced_candidate_slate=[{k: v for k, v in card.items() if k != "_record"} for card in balanced_slate_cards],
            evidence_selector_payload=selector_payload,
            raw_candidate_pool_by_query=raw_candidate_pool_by_query,
            deduped_candidate_pool=deduped_candidate_pool_cards,
            resolved_evidence_selection=resolved_selection,
            filtered_candidate_pool_by_query=filtered_candidate_pool_by_query,
            pmid_branch_filter_excluded=pmid_branch_filter_excluded,
            non_pmid_sanity_excluded=non_pmid_sanity_excluded,
            candidate_record_by_id={str(card.get("candidate_id")): card.get("_record") for card in balanced_slate_cards if isinstance(card.get("_record"), PaperRecord)},
        )

    def select_axis_evidence_batch(
        self,
        *,
        axis: dict[str, Any],
        subtopics_payload: dict[str, Any],
        retrieval_results: list[SubtopicRetrievalResult],
        ai_papers_per_subtopic: int,
        evidence_selector_model: str | None = None,
    ) -> dict[str, Any]:
        """Resolve one EvidenceSelector call per axis into per-subtopic packets.

        This does not reduce retrieval depth, source coverage, query count,
        PubTator annotation, filtering, candidate-pool logging, or the final
        axis synthesis budget. It only batches the curator LLM call that chooses
        papers from each already-built per-subtopic slate.
        """
        target_count = max(1, int(ai_papers_per_subtopic))
        subtopics = _subtopics(subtopics_payload)
        subtopic_by_id = {str(st.get("subtopic_id") or "").strip(): st for st in subtopics}
        candidates_by_subtopic = {
            r.subtopic_id: list(r.balanced_candidate_slate)
            for r in retrieval_results
            if r.subtopic_id and r.balanced_candidate_slate
        }
        if not candidates_by_subtopic:
            return {
                "axis_id": str(axis.get("axis_id") or ""),
                "selection_scope": "axis_batch",
                "selection_decision": "no_candidates",
                "subtopic_selections": [],
            }

        selector = EvidenceSelectorAgent(model=evidence_selector_model or self.model)
        axis_payload = selector.select_axis_subtopic_evidence(
            axis=axis,
            subtopics=[subtopic_by_id.get(sid, {"subtopic_id": sid}) for sid in candidates_by_subtopic],
            candidates_by_subtopic=candidates_by_subtopic,
            target_papers_per_subtopic=target_count,
        )
        selections = axis_payload.get("subtopic_selections", []) if isinstance(axis_payload, dict) else []
        selection_by_subtopic: dict[str, dict[str, Any]] = {}
        for item in selections:
            if not isinstance(item, dict):
                continue
            sid = str(item.get("subtopic_id") or "").strip()
            if sid:
                selection_by_subtopic[sid] = item

        for result in retrieval_results:
            if not result.balanced_candidate_slate:
                continue
            selection_payload = selection_by_subtopic.get(result.subtopic_id)
            if not selection_payload:
                selection_payload = {
                    "subtopic_id": result.subtopic_id,
                    "selection_decision": "fallback_deterministic",
                    "selected_candidate_ids": [str(c.get("candidate_id")) for c in result.balanced_candidate_slate[:target_count]],
                    "selection_notes": [
                        {"issue": "missing_subtopic_selection", "details": "Axis-batch selector omitted this subtopic; used deterministic slate fallback."}
                    ],
                }
            selected_records, resolved_selection = _resolve_candidate_id_selection(
                selection_payload=selection_payload,
                balanced_slate_cards=result.balanced_candidate_slate,
                candidate_record_by_id=result.candidate_record_by_id,
                target_count=target_count,
                enabled=True,
                selection_scope="axis_batch",
            )
            result.records = selected_records
            result.evidence_packets = [_record_to_packet(result.subtopic_id, i, r) for i, r in enumerate(selected_records, start=1)]
            result.evidence_selector_payload = {
                **selection_payload,
                "enabled": True,
                "selection_scope": "axis_batch",
                "axis_id": axis_payload.get("axis_id") if isinstance(axis_payload, dict) else str(axis.get("axis_id") or ""),
                "axis_level_selection_notes": axis_payload.get("axis_level_selection_notes", []) if isinstance(axis_payload, dict) else [],
            }
            result.resolved_evidence_selection = resolved_selection

        return axis_payload if isinstance(axis_payload, dict) else {}

    def synthesize_axis(self, axis: dict[str, Any], subtopics_payload: dict[str, Any], packets: list[EvidencePacket]) -> dict[str, Any]:
        prompt = render_prompt(
            "v31/literature_synthesize_axis_evidence.md",
            axis_json=compact_json(axis),
            subtopics_json=compact_json(subtopics_payload),
            abstracts_context=_format_abstracts(packets),
        )
        return ask_gemini_json(prompt, model=self.model, agent="literature", purpose="synthesize_axis_evidence")

    def synthesize_global(self, objective: str, syntheses: list[dict[str, Any]]) -> dict[str, Any]:
        prompt = render_prompt(
            "v31/literature_global_synthesis.md",
            objective=objective,
            literature_syntheses_json=compact_json(syntheses),
        )
        return ask_gemini_json(prompt, model=self.model, agent="literature", purpose="global_synthesis")



def _resolve_candidate_id_selection(
    *,
    selection_payload: dict[str, Any],
    balanced_slate_cards: list[dict[str, Any]],
    candidate_record_by_id: dict[str, PaperRecord],
    target_count: int,
    enabled: bool,
    selection_scope: str,
) -> tuple[list[PaperRecord], dict[str, Any]]:
    requested_ids = [str(cid) for cid in selection_payload.get("selected_candidate_ids", [])]
    selected_records: list[PaperRecord] = []
    seen_keys: set[str] = set()
    invalid_ids: list[str] = []
    duplicate_ids: list[str] = []
    resolved_ids: list[str] = []

    for cid in requested_ids:
        rec = candidate_record_by_id.get(str(cid))
        if not isinstance(rec, PaperRecord):
            invalid_ids.append(str(cid))
            continue
        key = rec.stable_key()
        if key in seen_keys:
            duplicate_ids.append(str(cid))
            continue
        seen_keys.add(key)
        selected_records.append(rec)
        resolved_ids.append(str(cid))
        if len(selected_records) >= target_count:
            break

    fallback_added: list[str] = []
    if len(selected_records) < target_count:
        for card in balanced_slate_cards:
            cid = str(card.get("candidate_id"))
            rec = candidate_record_by_id.get(cid)
            if not isinstance(rec, PaperRecord):
                continue
            key = rec.stable_key()
            if key in seen_keys:
                continue
            selected_records.append(rec)
            seen_keys.add(key)
            fallback_added.append(cid)
            resolved_ids.append(cid)
            if len(selected_records) >= target_count:
                break

    resolved_selection = {
        "enabled": bool(enabled),
        "selection_scope": selection_scope,
        "requested_selected_candidate_ids": requested_ids,
        "resolved_selected_candidate_ids": resolved_ids,
        "invalid_selected_candidate_ids": invalid_ids,
        "duplicate_selected_candidate_ids": duplicate_ids,
        "fallback_added_candidate_ids": fallback_added,
        "resolved_selected_titles": [r.title for r in selected_records],
        "selection_resolution_notes": [],
    }
    if invalid_ids:
        resolved_selection["selection_resolution_notes"].append({"issue": "invalid_selector_ids", "details": invalid_ids})
    if duplicate_ids:
        resolved_selection["selection_resolution_notes"].append({"issue": "duplicate_selector_ids", "details": duplicate_ids})
    if fallback_added:
        resolved_selection["selection_resolution_notes"].append({"issue": "fallback_topup", "details": fallback_added})
    if not requested_ids:
        resolved_selection["selection_resolution_notes"].append({"issue": "no_selector_ids", "details": "No selected_candidate_ids were supplied; used deterministic slate fallback."})
    return selected_records, resolved_selection

def _axes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    items = payload.get("axes", []) if isinstance(payload, dict) else []
    return [x for x in items if isinstance(x, dict)]


def _subtopics(payload: dict[str, Any]) -> list[dict[str, Any]]:
    items = payload.get("subtopics", []) if isinstance(payload, dict) else []
    return [x for x in items if isinstance(x, dict)]


def _format_abstracts(packets: list[EvidencePacket]) -> str:
    if not packets:
        return "No retrieved abstracts were available."
    try:
        max_chars = max(500, min(int(os.getenv("AXIS_SYNTHESIS_ABSTRACT_CHARS", "1200")), 2200))
    except Exception:
        max_chars = 1200
    lines = []
    for packet in packets:
        meta = packet.metadata if isinstance(packet.metadata, dict) else {}
        source_bits = []
        if meta.get("year"):
            source_bits.append(f"year={meta.get('year')}")
        if meta.get("pmid"):
            source_bits.append(f"pmid={meta.get('pmid')}")
        if meta.get("doi"):
            source_bits.append(f"doi={meta.get('doi')}")
        if meta.get("selected_from_subtopic"):
            source_bits.append(f"selected_from_subtopic={meta.get('selected_from_subtopic')}")
        source_line = " | ".join(source_bits)
        if source_line:
            source_line = f"\n{source_line}"
        lines.append(f"[{packet.evidence_id}] {packet.title}{source_line}\n{packet.text[:max_chars]}")
    return "\n\n".join(lines)



def _dedupe_records_merge_metadata(records: list[PaperRecord]) -> list[PaperRecord]:
    """Alias-aware dedupe with PMID-preferred canonical records.

    Records are clustered if they share any stable identifier (PMID, DOI,
    PMCID, OpenAlex, Semantic Scholar, Scopus, or normalized title). The
    canonical record is chosen rather than fully merged: prefer a PMID-backed
    record, then richer abstract/title metadata. Query/source provenance is
    preserved on the canonical record so it can still appear under every query
    branch that retrieved any duplicate copy.
    """
    clusters: list[list[PaperRecord]] = []
    alias_to_cluster: dict[str, int] = {}

    for r in records:
        keys = [k for k in (r.stable_keys() or [r.paper_id]) if k] or [f"paper:{r.paper_id}"]
        hit_clusters = sorted({alias_to_cluster[k] for k in keys if k in alias_to_cluster})
        if not hit_clusters:
            idx = len(clusters)
            clusters.append([r])
            for k in keys:
                alias_to_cluster[k] = idx
            continue

        primary = hit_clusters[0]
        clusters[primary].append(r)
        for other in reversed(hit_clusters[1:]):
            if other == primary or other >= len(clusters):
                continue
            clusters[primary].extend(clusters[other])
            clusters[other] = []
        for member in clusters[primary]:
            for k in (member.stable_keys() or [member.paper_id]):
                alias_to_cluster[k] = primary

    out: list[PaperRecord] = []
    for members in clusters:
        members = [m for m in members if m]
        if not members:
            continue
        canonical = max(members, key=_canonical_record_priority)
        aliases: set[str] = set()
        for member in members:
            aliases.update(member.stable_keys() or [member.paper_id])
            if member is not canonical:
                _merge_record_metadata(canonical, member)
                _prefer_missing_record_fields_only(canonical, member)
        canonical.raw["dedupe_aliases"] = sorted(aliases)
        canonical.raw["dedupe_cluster_size"] = len(members)
        canonical.raw["dedupe_preferred_pmid_record"] = bool(canonical.pmid)
        out.append(canonical)

    out.sort(key=lambda r: ((r.year or 0), (r.citation_count or 0)), reverse=True)
    return out


def _canonical_record_priority(record: PaperRecord) -> tuple[int, int, int, int, int]:
    """Choose the record to keep for a duplicate cluster; prefer PMID records."""
    return (
        1 if record.pmid else 0,
        1 if record.abstract else 0,
        len(record.abstract or ""),
        len(record.title or ""),
        int(record.year or 0),
    )


def _prefer_missing_record_fields_only(existing: PaperRecord, incoming: PaperRecord) -> None:
    """Fill missing identifiers/metadata without replacing the PMID-preferred record's text."""
    if (not existing.abstract) and incoming.abstract:
        existing.abstract = incoming.abstract
    if (not existing.title) and incoming.title:
        existing.title = incoming.title
    if (not existing.doi) and incoming.doi:
        existing.doi = incoming.doi
    if (not existing.pmid) and incoming.pmid:
        existing.pmid = incoming.pmid
    if (not existing.pmcid) and incoming.pmcid:
        existing.pmcid = incoming.pmcid
    if (not existing.openalex_id) and incoming.openalex_id:
        existing.openalex_id = incoming.openalex_id
    if (not existing.semantic_scholar_id) and incoming.semantic_scholar_id:
        existing.semantic_scholar_id = incoming.semantic_scholar_id
    if (not existing.scopus_id) and incoming.scopus_id:
        existing.scopus_id = incoming.scopus_id
    if (existing.citation_count or 0) < (incoming.citation_count or 0):
        existing.citation_count = incoming.citation_count
    for source in incoming.source_apis:
        if source not in existing.source_apis:
            existing.source_apis.append(source)

def _prefer_richer_record_fields(existing: PaperRecord, incoming: PaperRecord) -> None:
    if (not existing.abstract) and incoming.abstract:
        existing.abstract = incoming.abstract
    elif incoming.abstract and existing.abstract and len(incoming.abstract) > len(existing.abstract):
        existing.abstract = incoming.abstract
    if (not existing.title) and incoming.title:
        existing.title = incoming.title
    if (not existing.pmid) and incoming.pmid:
        existing.pmid = incoming.pmid
    if (not existing.doi) and incoming.doi:
        existing.doi = incoming.doi
    if (not existing.pmcid) and incoming.pmcid:
        existing.pmcid = incoming.pmcid
    if (not existing.openalex_id) and incoming.openalex_id:
        existing.openalex_id = incoming.openalex_id
    if (not existing.semantic_scholar_id) and incoming.semantic_scholar_id:
        existing.semantic_scholar_id = incoming.semantic_scholar_id
    if (existing.citation_count or 0) < (incoming.citation_count or 0):
        existing.citation_count = incoming.citation_count
    for source in incoming.source_apis:
        if source not in existing.source_apis:
            existing.source_apis.append(source)


def _merge_record_metadata(target: PaperRecord, source: PaperRecord) -> None:
    for field_name in ["retrieval_queries", "retrieval_sources", "entity_map_queries", "entity_map_query_families"]:
        target.raw.setdefault(field_name, [])
        for value in source.raw.get(field_name, []) if isinstance(source.raw, dict) else []:
            if value not in target.raw[field_name]:
                target.raw[field_name].append(value)
    target.raw.setdefault("retrieval_query_ranks", {})
    for query, rank in (source.raw.get("retrieval_query_ranks", {}) or {}).items():
        existing = target.raw["retrieval_query_ranks"].get(query)
        try:
            rank_i = int(rank)
        except Exception:
            rank_i = 999999
        if existing is None or rank_i < int(existing):
            target.raw["retrieval_query_ranks"][query] = rank_i
    target.raw.setdefault("retrieval_query_source_ranks", {})
    for query, per_source in (source.raw.get("retrieval_query_source_ranks", {}) or {}).items():
        target.raw["retrieval_query_source_ranks"].setdefault(query, {})
        for src, rank in (per_source or {}).items():
            existing = target.raw["retrieval_query_source_ranks"][query].get(src)
            try:
                rank_i = int(rank)
            except Exception:
                rank_i = 999999
            if existing is None or rank_i < int(existing):
                target.raw["retrieval_query_source_ranks"][query][src] = rank_i


def _records_by_query_after_dedupe(records: list[PaperRecord], queries: list[str]) -> dict[str, list[PaperRecord]]:
    by_query: dict[str, list[PaperRecord]] = {q: [] for q in queries}
    for r in records:
        r_queries = r.raw.get("retrieval_queries", []) if isinstance(r.raw, dict) else []
        for q in queries:
            if q in r_queries:
                by_query.setdefault(q, []).append(r)

    def source_priority(rec: PaperRecord, query: str) -> int:
        per_source = (rec.raw.get("retrieval_query_source_ranks", {}) or {}).get(query, {}) if isinstance(rec.raw, dict) else {}
        sources = set(per_source) or set(rec.raw.get("retrieval_sources", []) if isinstance(rec.raw, dict) else []) or set(rec.source_apis)
        priority = {"PubMed": 0, "EuropePMC": 1, "SemanticScholar": 2, "OpenAlex": 3, "Crossref": 4}
        return min((priority.get(str(src), 9) for src in sources), default=9)

    def best_rank(rec: PaperRecord, query: str) -> int:
        per_source = (rec.raw.get("retrieval_query_source_ranks", {}) or {}).get(query, {}) if isinstance(rec.raw, dict) else {}
        ranks = []
        for rank in per_source.values():
            try:
                ranks.append(int(rank))
            except Exception:
                pass
        if ranks:
            return min(ranks)
        try:
            return int((rec.raw.get("retrieval_query_ranks", {}) or {}).get(query, 999999))
        except Exception:
            return 999999

    for q in list(by_query):
        by_query[q].sort(key=lambda r: (best_rank(r, q), -(r.year or 0), -(r.citation_count or 0)))
    return by_query


def _build_round_robin_candidate_slate(
    records_by_query: dict[str, list[PaperRecord]],
    queries: list[str],
    *,
    depth_per_query: int,
    max_depth_per_query: int,
    subtopic_id: str,
    branch_vocab: set[str] | None = None,
    use_pubtator: bool = False,
) -> list[dict[str, Any]]:
    """Build a query-balanced slate, backfilling duplicates within each query.

    For each query branch, try to contribute `depth_per_query` unique papers.
    If a top paper is already represented from another query, advance deeper in
    that query list until a unique record is found or max_depth_per_query is hit.
    """
    seen: set[str] = set()
    slate: list[dict[str, Any]] = []
    target_depth = max(1, min(int(depth_per_query), int(max_depth_per_query)))
    max_depth = max(1, int(max_depth_per_query))
    pointers: dict[str, int] = {q: 0 for q in queries}
    contributions: dict[str, int] = {q: 0 for q in queries}

    for round_idx in range(target_depth):
        for query in queries:
            candidates = records_by_query.get(query, [])
            while contributions[query] <= round_idx and pointers[query] < min(len(candidates), max_depth):
                rec = candidates[pointers[query]]
                pointers[query] += 1
                key = _record_identity_key(rec)
                if key in seen:
                    continue
                seen.add(key)
                contributions[query] += 1
                card = _record_candidate_card(rec, candidate_id=f"{subtopic_id}_C{len(slate)+1:03d}", branch_vocab=branch_vocab, use_pubtator=use_pubtator)
                card["round_robin_depth"] = round_idx + 1
                card["primary_round_robin_query"] = query
                card["query_contribution_index"] = contributions[query]
                card["_record"] = rec
                slate.append(card)
                break
    return slate


def _record_identity_key(record: PaperRecord) -> str:
    keys = record.stable_keys()
    aliases = record.raw.get("dedupe_aliases", []) if isinstance(record.raw, dict) else []
    if aliases:
        return "|".join(sorted(str(a) for a in aliases))
    return keys[0] if keys else f"paper:{record.paper_id}"



def _derive_branch_vocab(queries: list[str], subtopic_payload: dict[str, Any] | None = None) -> set[str]:
    """Derive candidate branch terms from the actual subtopic, not hardcoded biology."""
    texts: list[str] = []
    texts.extend([str(q) for q in queries if q])
    if isinstance(subtopic_payload, dict):
        for key in ["name", "subtopic_name", "rationale", "coverage_intent"]:
            if subtopic_payload.get(key):
                texts.append(str(subtopic_payload.get(key)))
        for key in ["covered_branches", "supporting_terms", "supporting_entities_or_terms", "search_queries"]:
            vals = subtopic_payload.get(key)
            if isinstance(vals, list):
                texts.extend(str(v) for v in vals if v)
    raw = " ; ".join(texts)
    stop = {
        "aml", "acute", "myeloid", "leukemia", "cancer", "therapy", "treatment", "mechanism", "mechanisms",
        "stress", "response", "protein", "proteins", "pathway", "pathways", "system", "systems", "cell", "cells",
        "disease", "model", "models", "branch", "branches", "evidence", "query", "queries", "and", "or", "the", "with",
        "from", "into", "role", "roles", "target", "targets", "targeting", "inhibitor", "inhibitors",
    }
    terms: set[str] = set()
    # Keep short uppercase biomedical acronyms from supplied context.
    for token in re.findall(r"\b[A-Za-z][A-Za-z0-9αβ\-/]{1,24}\b", raw):
        clean = token.strip(" -_/.,;:()[]{}")
        if not clean:
            continue
        low = clean.lower()
        if low in stop:
            continue
        if clean.isupper() or any(ch.isdigit() for ch in clean) or len(clean) >= 5:
            terms.add(clean)
    # Keep informative multi-word phrases from queries/branches.
    for phrase in re.findall(r"[A-Za-z0-9αβ]+(?:[ -][A-Za-z0-9αβ]+){1,3}", raw):
        words = [w for w in re.findall(r"[A-Za-z0-9αβ]+", phrase) if w.lower() not in stop]
        if len(words) >= 1 and any((w.isupper() or any(ch.isdigit() for ch in w) or len(w) >= 5) for w in words):
            terms.add(" ".join(words))
    return terms


def _term_in_text(term: str, text_lower: str) -> bool:
    term = str(term).strip()
    if not term:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(term.lower()).replace(r"\ ", r"[\s\-/]+") + r"(?![a-z0-9])"
    return bool(re.search(pattern, text_lower))


def _same_term(a: str, b: str) -> bool:
    aa = re.sub(r"[^a-z0-9]+", "", str(a).lower())
    bb = re.sub(r"[^a-z0-9]+", "", str(b).lower())
    return bool(aa and bb and (aa == bb or aa in bb or bb in aa))


def _pmid_branch_tag_rule(record: PaperRecord, pubtator_checked: bool, pubtator_branch_terms: list[str], branch_terms: list[str]) -> dict[str, Any]:
    if not pubtator_checked:
        return {"applies": False, "decision": "not_applicable_no_pmid_or_pubtator_not_requested"}
    if pubtator_branch_terms or branch_terms:
        return {"applies": True, "decision": "keep", "reason": "PMID-backed candidate has PubTator and/or lexical branch signal."}
    return {"applies": True, "decision": "exclude_from_precurator_slate", "reason": "PMID-backed candidate has no branch signal after PubTator/lexical check."}


def _apply_pmid_branch_tag_filter(
    records_by_query: dict[str, list[PaperRecord]],
    queries: list[str],
    branch_vocab: set[str],
) -> tuple[dict[str, list[PaperRecord]], dict[str, list[PaperRecord]]]:
    kept: dict[str, list[PaperRecord]] = {q: [] for q in queries}
    excluded: dict[str, list[PaperRecord]] = {q: [] for q in queries}
    for q in queries:
        for rec in records_by_query.get(q, []):
            ann = rec.raw.get("pubtator", {}) if isinstance(rec.raw, dict) else {}
            entity_terms = sorted(pubtator_entity_terms(ann)) if ann else []
            text = f"{rec.title or ''} {rec.abstract or ''}".lower()
            lexical_hits = [t for t in branch_vocab if _term_in_text(t, text)]
            pubtator_hits = [t for t in branch_vocab if any(_same_term(t, e) for e in entity_terms)]
            if rec.pmid and ann is not None and not lexical_hits and not pubtator_hits:
                excluded[q].append(rec)
            else:
                kept[q].append(rec)
        # If a query would become empty, fall back to unfiltered candidates for that branch.
        if not kept[q] and records_by_query.get(q):
            kept[q] = list(records_by_query.get(q, []))
            excluded[q] = []
    return kept, excluded


def _normalize_token_set(text: str) -> set[str]:
    return {t.lower() for t in re.findall(r"\b[A-Za-z][A-Za-z0-9αβ\-]{1,30}\b", text or "")}


def _query_target_context_terms(queries: list[str]) -> set[str]:
    """Infer the repeated target context from the actual query set.

    This is intentionally general: for an AML benchmark it will infer AML/leukemia;
    for other questions it can infer an organism, disease, model, tissue, etc.
    """
    token_counts: Counter[str] = Counter()
    phrase_counts: Counter[str] = Counter()
    for q in queries:
        q = str(q or "")
        toks = _normalize_token_set(q)
        for t in toks:
            token_counts[t] += 1
        for phrase in re.findall(r"[A-Za-z0-9αβ]+(?:[ -][A-Za-z0-9αβ]+){1,4}", q):
            norm = " ".join(re.findall(r"[A-Za-z0-9αβ]+", phrase.lower()))
            if norm:
                phrase_counts[norm] += 1
    n = max(1, len([q for q in queries if str(q).strip()]))
    generic = {
        "stress", "response", "protein", "proteins", "pathway", "pathways", "mechanism", "mechanisms",
        "therapy", "therapeutic", "target", "targets", "targeting", "inhibitor", "inhibitors",
        "role", "roles", "cell", "cells", "disease", "model", "models", "and", "or", "the", "with",
    }
    repeated = {t for t, c in token_counts.items() if c >= 2 and t not in generic and len(t) >= 3}
    repeated |= {p for p, c in phrase_counts.items() if c >= 2 and len(p) >= 5}
    # Biomedical abbreviation normalization for common query-context style; only fires if present in the query text.
    qtext = " ".join(str(q).lower() for q in queries)
    if "aml" in repeated or re.search(r"\baml\b", qtext):
        repeated.update({"aml", "acute myeloid leukemia", "leukemia", "leukaemia"})
    return repeated


def _query_branch_terms(query: str, target_terms: set[str]) -> set[str]:
    stop = {
        "aml", "acute", "myeloid", "leukemia", "leukaemia", "cancer", "disease", "model", "models",
        "and", "or", "the", "with", "in", "of", "for", "to", "by", "from",
        "role", "roles", "therapy", "therapeutic", "target", "targets", "targeting",
    }
    q = str(query or "")
    terms: set[str] = set()
    # Keep exact multi-word phrases except those dominated by inferred target context.
    for phrase in re.findall(r"[A-Za-z0-9αβ]+(?:[ -][A-Za-z0-9αβ]+){0,4}", q):
        words = [w.lower() for w in re.findall(r"[A-Za-z0-9αβ]+", phrase)]
        if not words:
            continue
        cleaned = [w for w in words if w not in stop and w not in target_terms]
        if cleaned:
            phrase_norm = " ".join(cleaned)
            if len(phrase_norm) >= 2:
                terms.add(phrase_norm)
            for w in cleaned:
                if len(w) >= 2:
                    terms.add(w)
    return terms


def _looks_like_metadata_fragment(record: PaperRecord) -> bool:
    title = (record.title or "").strip().lower()
    bad_prefixes = (
        "figure ", "fig. ", "table ", "supplementary ", "supplemental ",
        "decision letter", "author response", "editor's evaluation", "editors' evaluation",
        "peer review", "publisher correction", "correction:", "erratum", "retraction",
    )
    return title.startswith(bad_prefixes)


def _apply_non_pmid_context_branch_sanity_filter(
    records_by_query: dict[str, list[PaperRecord]],
    queries: list[str],
) -> tuple[dict[str, list[PaperRecord]], dict[str, list[PaperRecord]]]:
    """Filter obvious non-PMID noise without punishing missing PubTator.

    Only records without PMID are eligible for this filter. They must show lexical
    support for the repeated target context and the specific query branch. If a
    query would become empty, fallback to the unfiltered candidates for that query.
    """
    target_terms = _query_target_context_terms(queries)
    kept: dict[str, list[PaperRecord]] = {q: [] for q in queries}
    excluded: dict[str, list[PaperRecord]] = {q: [] for q in queries}
    for q in queries:
        branch_terms = _query_branch_terms(q, target_terms)
        for rec in records_by_query.get(q, []):
            if rec.pmid:
                kept[q].append(rec)
                continue
            text = f"{rec.title or ''} {rec.abstract or ''}".lower()
            context_hit = any(_term_in_text(t, text) for t in target_terms) if target_terms else True
            branch_hit = any(_term_in_text(t, text) for t in branch_terms) if branch_terms else True
            if _looks_like_metadata_fragment(rec) or not (context_hit and branch_hit):
                excluded[q].append(rec)
            else:
                kept[q].append(rec)
        if not kept[q] and records_by_query.get(q):
            kept[q] = list(records_by_query.get(q, []))
            excluded[q] = []
    return kept, excluded

def _record_candidate_card(record: PaperRecord, *, candidate_id: str, branch_vocab: set[str] | None = None, use_pubtator: bool = False) -> dict[str, Any]:
    ann = record.raw.get("pubtator", {}) if isinstance(record.raw, dict) else {}
    entity_terms = sorted(pubtator_entity_terms(ann)) if ann else []
    text = f"{record.title or ''} {record.abstract or ''}".lower()
    likely_review = bool(re.search(r"\b(review|systematic review|meta-analysis|overview)\b", text))
    explicit_aml = bool(re.search(r"\b(acute myeloid leukemia|aml)\b", text))
    source_queries = record.raw.get("retrieval_queries", []) if isinstance(record.raw, dict) else []
    query_ranks = record.raw.get("retrieval_query_ranks", {}) if isinstance(record.raw, dict) else {}
    retrieval_sources = record.raw.get("retrieval_sources", []) if isinstance(record.raw, dict) else []
    branch_vocab = branch_vocab or set()
    branch_terms = sorted([t for t in branch_vocab if _term_in_text(t, text)])
    pubtator_branch_terms = sorted([t for t in branch_vocab if any(_same_term(t, e) for e in entity_terms)])
    # Neutral entity pool for the selector: expose PubTator-discovered terms as
    # candidate context without promoting, scoring, or requiring them. This lets
    # the selector see named entities found by annotation even when they were not
    # already present in the subtopic/query-derived branch vocabulary.
    selector_entity_pool = sorted({str(t) for t in [*branch_terms, *pubtator_branch_terms, *entity_terms[:40]] if str(t).strip()})
    pubtator_checked = bool(use_pubtator and record.pmid)
    return {
        "candidate_id": candidate_id,
        "title": record.title,
        "year": record.year,
        "pmid": record.pmid,
        "doi": record.doi,
        "source": record.source_api,
        "journal": record.journal,
        "citation_count": record.citation_count,
        "source_queries": source_queries,
        "query_ranks": query_ranks,
        "retrieval_sources": retrieval_sources,
        "retrieval_query_source_ranks": record.raw.get("retrieval_query_source_ranks", {}) if isinstance(record.raw, dict) else {},
        "dedupe_aliases": record.raw.get("dedupe_aliases", []) if isinstance(record.raw, dict) else [],
        "pubtator_terms": entity_terms[:40],
        "pubtator_checked": pubtator_checked,
        "pubtator_branch_terms": pubtator_branch_terms,
        "selector_entity_pool": selector_entity_pool[:60],
        "pmid_branch_tag_rule": _pmid_branch_tag_rule(record, pubtator_checked, pubtator_branch_terms, branch_terms),
        "non_pmid_unannotated": bool(not record.pmid),
        "likely_review": likely_review,
        "explicit_aml_mention": explicit_aml,
        "branch_terms_detected": branch_terms,
        "directness_hint": _directness_hint(record, source_queries, branch_terms, explicit_aml, likely_review),
        "abstract_snippet": (record.abstract or "")[:900],
    }


def _directness_hint(record: PaperRecord, source_queries: list[str], branch_terms: list[str], explicit_aml: bool, likely_review: bool) -> str:
    title_text = (record.title or "").lower()
    query_hits = []
    for q in source_queries:
        important = [tok.lower() for tok in re.findall(r"[A-Za-z0-9αβ]+", q) if len(tok) >= 3]
        if important and all(tok in title_text for tok in important if tok not in {"aml", "and", "the"}):
            query_hits.append(q)
    if explicit_aml and branch_terms and not likely_review:
        return "direct_disease_mechanistic"
    if explicit_aml and branch_terms:
        return "disease_branch_background"
    if branch_terms:
        return "branch_background"
    if explicit_aml:
        return "disease_background"
    return "broad_background"


def _safe_id(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", str(text).strip().lower()).strip("_")
    return cleaned[:40] or "query"

def _dedupe_records(records: list[PaperRecord]) -> list[PaperRecord]:
    return _dedupe_records_merge_metadata(records)


def _select_balanced_packets_by_subtopic(
    packets_by_subtopic: dict[str, list[EvidencePacket]],
    *,
    max_total: int,
) -> list[EvidencePacket]:
    """Select evidence packets evenly across subtopics.

    This prevents the first generated subtopic from consuming the whole
    per-axis AI context budget. If a subtopic has fewer papers after dedupe,
    the remaining slots are filled from the other subtopics. No extra retrieval
    is triggered here; retrieval limits stay deterministic and bounded.
    """
    if max_total <= 0:
        return []

    deduped_by_subtopic = {
        sid: _dedupe_packets(packets)
        for sid, packets in packets_by_subtopic.items()
        if sid and packets
    }
    if not deduped_by_subtopic:
        return []

    subtopic_ids = list(deduped_by_subtopic.keys())
    base_quota = max(1, max_total // len(subtopic_ids))
    selected: list[EvidencePacket] = []
    seen: set[str] = set()

    def add_packet(packet: EvidencePacket, sid: str) -> bool:
        key = _packet_key(packet)
        if key in seen:
            return False
        seen.add(key)
        packet.metadata["selected_from_subtopic"] = sid
        selected.append(packet)
        return True

    # First pass: fair quota per subtopic.
    for sid in subtopic_ids:
        taken = 0
        for packet in deduped_by_subtopic[sid]:
            if add_packet(packet, sid):
                taken += 1
            if taken >= base_quota or len(selected) >= max_total:
                break

    # Second pass: fill remaining slots round-robin from available papers.
    while len(selected) < max_total:
        added_any = False
        for sid in subtopic_ids:
            for packet in deduped_by_subtopic[sid]:
                if add_packet(packet, sid):
                    added_any = True
                    break
            if len(selected) >= max_total:
                break
        if not added_any:
            break

    return selected[:max_total]


def _packet_key(packet: EvidencePacket) -> str:
    return str(
        packet.metadata.get("doi")
        or packet.metadata.get("pmid")
        or packet.paper_id
        or packet.title
    ).lower().strip()


def _dedupe_packets(packets: list[EvidencePacket]) -> list[EvidencePacket]:
    seen: set[str] = set()
    out: list[EvidencePacket] = []
    for p in packets:
        key = str(p.metadata.get("doi") or p.metadata.get("pmid") or p.paper_id or p.title).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _rank_records_with_pubtator_diversity(records: list[PaperRecord], queries: list[str], limit: int) -> list[PaperRecord]:
    """Rank a larger PubTator-enriched candidate pool for compact LLM reading.

    The first score keeps normal publication relevance heuristics, then a small
    greedy diversity pass avoids selecting only papers with identical PubTator
    entity sets. This uses entity/relation annotations to widen mechanism
    coverage while keeping the LLM paper budget unchanged.
    """
    if not records:
        return []
    query_terms = _query_terms(queries)

    def score(record: PaperRecord) -> float:
        text = f"{record.title} {record.abstract or ''}".lower()
        lexical = sum(1 for term in query_terms if term and term in text)
        ann = record.raw.get("pubtator", {}) if isinstance(record.raw, dict) else {}
        entities = ann.get("entities", {}) if isinstance(ann, dict) else {}
        entity_count = sum(len(v) for v in entities.values() if isinstance(v, list))
        relation_count = int(ann.get("relation_count") or len(ann.get("relations", []) or [])) if isinstance(ann, dict) else 0
        citation = min(float(record.citation_count or 0), 200.0) / 200.0
        year = float(record.year or 0) / 3000.0
        return lexical * 3.0 + min(entity_count, 20) * 0.15 + min(relation_count, 10) * 0.25 + citation + year

    ranked = sorted(records, key=score, reverse=True)
    target = max(1, int(limit))
    selected: list[PaperRecord] = []
    covered_terms: set[str] = set()
    remaining = ranked.copy()

    while remaining and len(selected) < target:
        best_idx = 0
        best_tuple = (-1.0, -1.0)
        for idx, rec in enumerate(remaining):
            terms = pubtator_entity_terms(rec.raw.get("pubtator", {}))
            novelty = len(terms - covered_terms)
            candidate = (float(novelty), score(rec))
            if candidate > best_tuple:
                best_tuple = candidate
                best_idx = idx
        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        covered_terms.update(pubtator_entity_terms(chosen.raw.get("pubtator", {})))

    selected_keys = {r.stable_key() for r in selected}
    selected.extend([r for r in ranked if r.stable_key() not in selected_keys])
    return selected


def _query_terms(queries: list[str]) -> set[str]:
    stop = {"aml", "acute", "myeloid", "leukemia", "pathway", "therapy", "cancer", "cells", "cell", "and", "or", "the", "of", "in"}
    terms: set[str] = set()
    for q in queries:
        for token in str(q).lower().replace("/", " ").replace("-", " ").split():
            token = token.strip(".,;:()[]{}")
            if len(token) >= 3 and token not in stop:
                terms.add(token)
    return terms



def _record_brief(record: PaperRecord) -> dict[str, Any]:
    return {
        "title": record.title,
        "year": record.year,
        "pmid": record.pmid,
        "doi": record.doi,
        "source": record.source_api,
        "journal": record.journal,
        "query_families": record.raw.get("entity_map_query_families", []),
        "abstract_snippet": (record.abstract or "")[:650],
    }


def _build_concept_inventory(axis: dict[str, Any], family_summaries: list[dict[str, Any]], records: list[PaperRecord]) -> dict[str, Any]:
    axis_text = json.dumps(axis, ensure_ascii=False)
    axis_explicit_terms = _extract_axis_terms(axis_text)

    entity_counter: Counter[str] = Counter()
    entity_types: dict[str, set[str]] = defaultdict(set)
    entity_families: dict[str, set[str]] = defaultdict(set)
    mechanism_counter: Counter[str] = Counter()
    mechanism_families: dict[str, set[str]] = defaultdict(set)

    for record in records:
        families = set(record.raw.get("entity_map_query_families", []) or [])
        pubtator = record.raw.get("pubtator", {}) if isinstance(record.raw, dict) else {}
        entities = pubtator.get("entities", {}) if isinstance(pubtator, dict) else {}
        for etype, items in entities.items():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text") or "").strip()
                ident = str(item.get("identifier") or "").strip()
                label = text or ident
                if not label:
                    continue
                norm = _normalize_concept(label)
                if len(norm) < 2:
                    continue
                entity_counter[norm] += 1
                entity_types[norm].add(str(etype))
                entity_families[norm].update(families)

        text = f"{record.title}. {record.abstract or ''}"
        for term in _extract_candidate_terms(text):
            mechanism_counter[term] += 1
            mechanism_families[term].update(families)

    top_entities = [
        {
            "term": term,
            "count": count,
            "entity_types": sorted(entity_types.get(term, [])),
            "query_families": sorted(entity_families.get(term, [])),
            "query_family_count": len(entity_families.get(term, [])),
        }
        for term, count in entity_counter.most_common(80)
    ]
    top_mechanism_terms = [
        {
            "term": term,
            "count": count,
            "query_families": sorted(mechanism_families.get(term, [])),
            "query_family_count": len(mechanism_families.get(term, [])),
        }
        for term, count in mechanism_counter.most_common(80)
    ]

    # Promote terms that occur across multiple query families even if not most frequent.
    cross_query_terms = []
    for term, families in {**entity_families, **mechanism_families}.items():
        if len(families) >= 2:
            cross_query_terms.append({"term": term, "query_families": sorted(families), "query_family_count": len(families)})
    cross_query_terms = sorted(cross_query_terms, key=lambda x: (-x["query_family_count"], x["term"]))[:80]

    return {
        "axis_explicit_terms": axis_explicit_terms,
        "family_summaries": family_summaries,
        "top_pubtator_entities": top_entities,
        "top_fallback_mechanism_terms": top_mechanism_terms,
        "cross_query_family_terms": cross_query_terms,
        "notes": [
            "Counts are for concept mapping only, not final evidence strength.",
            "Use query-family provenance to prevent one dominant literature area from swallowing rarer mechanisms.",
        ],
    }


def _normalize_concept(value: str) -> str:
    value = re.sub(r"\s+", " ", str(value)).strip()
    return value


def _extract_axis_terms(text: str) -> list[str]:
    # Lightweight, domain-neutral phrase extraction from the given axis text.
    cleaned = re.sub(r"[_{}\[\](),.;:\\/]", " ", text)
    words = [w.strip('"').strip() for w in cleaned.split()]
    candidates = []
    for w in words:
        if len(w) >= 4 and not w.isdigit():
            lw = w.lower()
            if lw not in {"axis", "name", "rationale", "distinct", "cells", "cell", "with", "have", "from", "that", "this", "mechanisms", "targeting"}:
                candidates.append(w)
    out = []
    seen = set()
    for c in candidates:
        key = c.lower()
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out[:80]


def _extract_candidate_terms(text: str) -> set[str]:
    """Fallback concepts when PubTator is absent.

    This intentionally avoids a hand-built domain taxonomy. It harvests short
    biomedical-looking phrases and uppercase/gene-like tokens from titles and
    abstracts for concept-map context only.
    """
    out: set[str] = set()
    raw = re.sub(r"\s+", " ", text or " ").strip()
    # Gene/protein-like tokens: XBP1, ATF4, HSPA5, p53, eIF2α-like text.
    for match in re.finditer(r"\b[A-Z][A-Z0-9]{2,}[A-Z0-9α-ω]*\b", raw):
        token = match.group(0).strip()
        if token not in {"AML", "DNA", "RNA", "THE", "AND", "FOR"}:
            out.add(token)
    # Mechanism phrases ending in common biomedical process words; generic, not disease-specific.
    phrase_pattern = r"\b([A-Za-z0-9α-ω\-/]+(?:\s+[A-Za-z0-9α-ω\-/]+){0,3}\s+(?:stress|response|signaling|pathway|homeostasis|proteostasis|autophagy|apoptosis|resistance|differentiation|inflammation|metabolism|degradation|translation|transcription|repair|activation|inhibition))\b"
    for match in re.finditer(phrase_pattern, raw, flags=re.IGNORECASE):
        phrase = re.sub(r"\s+", " ", match.group(1)).strip(" .,:;()[]{}")
        if 4 <= len(phrase) <= 80:
            out.add(phrase)
    return out

def _record_to_packet(subtopic_id: str, idx: int, record: PaperRecord) -> EvidencePacket:
    text_parts = [f"Title: {record.title}"]
    if record.year:
        text_parts.append(f"Year: {record.year}")
    if record.journal:
        text_parts.append(f"Journal: {record.journal}")
    if record.abstract:
        text_parts.append(f"Abstract: {record.abstract}")
    return EvidencePacket(
        evidence_id=f"{subtopic_id}_E{idx:03d}",
        paper_id=record.paper_id,
        title=record.title,
        source=record.source_api,
        text="\n".join(text_parts),
        evidence_type="axis_literature_abstract",
        metadata={
            "subtopic_id": subtopic_id,
            "year": record.year,
            "doi": record.doi,
            "pmid": record.pmid,
            "url": record.url,
            "citation_count": record.citation_count,
            "retrieval_queries": record.raw.get("retrieval_queries", []),
            "retrieval_sources": record.raw.get("retrieval_sources", []),
            "pubtator": record.raw.get("pubtator", {}),
            "literature_agent": "axis_first_literature_agent",
        },
    )


def _semantic_scholar_allow_unauthenticated(config_path: str) -> bool:
    try:
        config = load_config(config_path)
        return bool((((config.get("retrieval") or {}).get("semantic_scholar") or {}).get("allow_unauthenticated")))
    except Exception:
        return False
