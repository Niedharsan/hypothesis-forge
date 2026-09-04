from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from schemas.paper_record import PaperRecord
from schemas.evidence_packet import EvidencePacket


def stable_id_for_record(record: PaperRecord | dict[str, Any]) -> str:
    """Return a deterministic paper identity suitable for cross-agent reuse.

    Preference order deliberately favors PMID when present so PubTator/PubMed-backed
    records become the canonical memory entry for duplicate clusters. This is a
    storage/caching preference only, not a relevance ranking signal.
    """
    if isinstance(record, PaperRecord):
        pmid = record.pmid
        doi = record.doi
        pmcid = record.pmcid
        openalex_id = record.openalex_id
        s2 = record.semantic_scholar_id
        scopus = record.scopus_id
        title = record.title
        paper_id = record.paper_id
    else:
        pmid = record.get("pmid")
        doi = record.get("doi")
        pmcid = record.get("pmcid")
        openalex_id = record.get("openalex_id")
        s2 = record.get("semantic_scholar_id")
        scopus = record.get("scopus_id")
        title = record.get("title") or ""
        paper_id = record.get("paper_id") or record.get("id") or ""

    if pmid:
        return f"PMID:{str(pmid).strip()}"
    norm_doi = _normalize_doi(doi)
    if norm_doi:
        return f"DOI:{norm_doi}"
    if pmcid:
        return f"PMCID:{str(pmcid).strip()}"
    if openalex_id:
        return f"OPENALEX:{str(openalex_id).strip()}"
    if s2:
        return f"S2:{str(s2).strip()}"
    if scopus:
        return f"SCOPUS:{str(scopus).strip()}"
    norm_title = _normalize_title(title)
    if norm_title:
        return f"TITLE:{norm_title}"
    return f"PAPER:{paper_id}"


def build_paper_memory(
    *,
    retrieval_results: list[Any],
    selected_packets: list[EvidencePacket] | None = None,
    axis_id: str | None = None,
    axis_name: str | None = None,
    cutoff_year: int | None = None,
) -> dict[str, Any]:
    """Build a run-level paper memory from retrieval artifacts without extra LLM calls.

    This is not a memory agent. It is a deterministic evidence index that later
    agents can consult before reading/summarizing a paper again.
    """
    selected_packets = selected_packets or []
    selected_ids = {str(p.paper_id) for p in selected_packets}
    selected_titles = {_normalize_title(p.title) for p in selected_packets}
    entries: dict[str, dict[str, Any]] = {}
    duplicate_events: list[dict[str, Any]] = []

    def ensure_entry(record: PaperRecord, *, source_stage: str, subtopic_id: str, query: str | None = None, status: str = "candidate") -> dict[str, Any]:
        sid = stable_id_for_record(record)
        entry = entries.get(sid)
        if entry is None:
            raw = record.raw if isinstance(record.raw, dict) else {}
            pubtator = raw.get("pubtator") if isinstance(raw.get("pubtator"), dict) else {}
            entry = {
                "stable_id": sid,
                "canonical_preference": "pmid_first",
                "title": record.title,
                "abstract": record.abstract or "",
                "abstract_available": bool(record.abstract),
                "year": record.year,
                "journal": record.journal,
                "doi": record.doi,
                "pmid": record.pmid,
                "pmcid": record.pmcid,
                "openalex_id": record.openalex_id,
                "semantic_scholar_id": record.semantic_scholar_id,
                "scopus_id": record.scopus_id,
                "url": record.url,
                "source_apis": list(record.source_apis or []),
                "all_sources_seen": sorted(set(raw.get("retrieval_sources", []) or record.source_apis or [])),
                "source_queries_seen": sorted(set(raw.get("retrieval_queries", []) or [])),
                "retrieval_query_ranks": raw.get("retrieval_query_ranks", {}),
                "retrieval_query_source_ranks": raw.get("retrieval_query_source_ranks", {}),
                "branch_terms_detected": sorted(set(raw.get("branch_terms_detected", []) or [])),
                "filter_hitsets": sorted(set(raw.get("filter_hitsets", []) or [])),
                "pubtator_checked": bool(record.pmid),
                "pubtator_terms": _extract_pubtator_terms(pubtator),
                "mesh_terms": list(record.mesh_terms or []),
                "keywords": list(record.keywords or []),
                "concepts": list(record.concepts or []),
                "seen_in_subtopics": [],
                "seen_in_queries": [],
                "seen_in_stages": [],
                "statuses": [],
                "used_in_synthesis": False,
                "selected_evidence_ids": [],
                "notes": [],
            }
            entries[sid] = entry
        else:
            duplicate_events.append({
                "stable_id": sid,
                "existing_title": entry.get("title"),
                "new_title": record.title,
                "source_stage": source_stage,
                "subtopic_id": subtopic_id,
                "query": query,
            })
            # Prefer PMID-backed fields if existing entry was non-PMID and new record has PMID.
            if not entry.get("pmid") and record.pmid:
                entry.update({
                    "title": record.title,
                    "abstract": record.abstract or entry.get("abstract", ""),
                    "abstract_available": bool(record.abstract or entry.get("abstract")),
                    "year": record.year,
                    "journal": record.journal,
                    "doi": record.doi or entry.get("doi"),
                    "pmid": record.pmid,
                    "pmcid": record.pmcid or entry.get("pmcid"),
                    "url": record.url or entry.get("url"),
                    "canonical_preference": "replaced_with_pmid_record",
                })
        _append_unique(entry["seen_in_subtopics"], subtopic_id)
        if query:
            _append_unique(entry["seen_in_queries"], query)
        _append_unique(entry["seen_in_stages"], source_stage)
        _append_unique(entry["statuses"], status)
        return entry

    for result in retrieval_results:
        subtopic_id = str(getattr(result, "subtopic_id", "") or "")
        # selected records / evidence packets
        for rec in getattr(result, "records", []) or []:
            if isinstance(rec, PaperRecord):
                ensure_entry(rec, source_stage="selected_records", subtopic_id=subtopic_id, status="selected_by_curator_or_fallback")
        # candidate cards can be dict-only; add limited info under stable id when no PaperRecord object exists.
        for stage_name in [
            "deduped_candidate_pool",
            "balanced_candidate_slate",
            "pmid_branch_filter_excluded",
            "non_pmid_sanity_excluded",
        ]:
            data = getattr(result, stage_name, None)
            if not data:
                continue
            if isinstance(data, dict):
                iterable = []
                for q, cards in data.items():
                    for c in cards:
                        iterable.append((c, q))
            else:
                iterable = [(c, None) for c in data]
            for card, query in iterable:
                if not isinstance(card, dict):
                    continue
                _add_card_entry(entries, card, source_stage=stage_name, subtopic_id=subtopic_id, query=query)
        # per-query pools
        for stage_name in ["raw_candidate_pool_by_query", "candidate_pool_by_query", "filtered_candidate_pool_by_query"]:
            data = getattr(result, stage_name, None)
            if not isinstance(data, dict):
                continue
            for query, cards in data.items():
                for card in cards or []:
                    if isinstance(card, dict):
                        _add_card_entry(entries, card, source_stage=stage_name, subtopic_id=subtopic_id, query=query)

    for packet in selected_packets:
        sid = _stable_id_for_packet(packet)
        # Try title fallback if packet paper_id differs from card stable ID.
        entry = entries.get(sid)
        if entry is None:
            tkey = f"TITLE:{_normalize_title(packet.title)}"
            entry = entries.get(tkey)
        if entry is None:
            entry = {
                "stable_id": sid,
                "canonical_preference": "packet_only",
                "title": packet.title,
                "abstract": packet.text or "",
                "abstract_available": bool(packet.text),
                "year": None,
                "journal": None,
                "doi": None,
                "pmid": _extract_prefixed(packet.paper_id, "PMID"),
                "pmcid": None,
                "url": None,
                "source_apis": [packet.source] if packet.source else [],
                "all_sources_seen": [packet.source] if packet.source else [],
                "source_queries_seen": [],
                "retrieval_query_ranks": {},
                "retrieval_query_source_ranks": {},
                "branch_terms_detected": [],
                "filter_hitsets": [],
                "pubtator_checked": bool(_extract_prefixed(packet.paper_id, "PMID")),
                "pubtator_terms": [],
                "mesh_terms": [],
                "keywords": [],
                "concepts": [],
                "seen_in_subtopics": [],
                "seen_in_queries": [],
                "seen_in_stages": [],
                "statuses": [],
                "used_in_synthesis": False,
                "selected_evidence_ids": [],
                "notes": [],
            }
            entries[sid] = entry
        entry["used_in_synthesis"] = True
        _append_unique(entry["selected_evidence_ids"], packet.evidence_id)
        if packet.metadata.get("selected_from_subtopic"):
            _append_unique(entry["seen_in_subtopics"], str(packet.metadata.get("selected_from_subtopic")))
        _append_unique(entry["seen_in_stages"], "axis_synthesis_input")
        _append_unique(entry["statuses"], "used_in_synthesis")

    entries_list = sorted(entries.values(), key=lambda e: (not bool(e.get("used_in_synthesis")), e.get("stable_id", "")))
    return {
        "schema_version": "v1",
        "purpose": "Deterministic run evidence index for reuse by Reflection/Evolution/Verification. It avoids rereading identical papers but does not suppress future searches or new papers.",
        "axis_id": axis_id,
        "axis_name": axis_name,
        "cutoff_year": cutoff_year,
        "counts": {
            "memory_entries": len(entries_list),
            "used_in_synthesis": sum(1 for e in entries_list if e.get("used_in_synthesis")),
            "with_pmid": sum(1 for e in entries_list if e.get("pmid")),
            "with_abstract": sum(1 for e in entries_list if e.get("abstract_available")),
            "duplicate_events_logged": len(duplicate_events),
        },
        "entries": entries_list,
        "duplicate_events": duplicate_events[:500],
    }


def compact_memory_for_reflection(memory: dict[str, Any], *, max_entries: int = 40) -> dict[str, Any]:
    entries = memory.get("entries", []) if isinstance(memory, dict) else []
    selected = [e for e in entries if e.get("used_in_synthesis")]
    # Add a limited number of selected/curated candidates first; this keeps reflection cheap.
    others = [e for e in entries if not e.get("used_in_synthesis") and ("selected_by_curator_or_fallback" in e.get("statuses", []) or "balanced_candidate_slate" in e.get("seen_in_stages", []))]
    chosen = (selected + others)[:max_entries]
    return {
        "source_memory_counts": memory.get("counts", {}),
        "entries_shown": len(chosen),
        "entries": [
            {
                "stable_id": e.get("stable_id"),
                "title": e.get("title"),
                "year": e.get("year"),
                "pmid": e.get("pmid"),
                "doi": e.get("doi"),
                "used_in_synthesis": e.get("used_in_synthesis"),
                "selected_evidence_ids": e.get("selected_evidence_ids", []),
                "seen_in_subtopics": e.get("seen_in_subtopics", []),
                "source_queries_seen": e.get("source_queries_seen", []),
                "branch_terms_detected": e.get("branch_terms_detected", []),
                "filter_hitsets": e.get("filter_hitsets", []),
                "pubtator_terms": e.get("pubtator_terms", [])[:30],
                "abstract_snippet": (e.get("abstract") or "")[:600],
            }
            for e in chosen
        ],
    }


def _add_card_entry(entries: dict[str, dict[str, Any]], card: dict[str, Any], *, source_stage: str, subtopic_id: str, query: str | None) -> None:
    sid = stable_id_for_record(card)
    entry = entries.get(sid)
    if entry is None:
        entry = {
            "stable_id": sid,
            "canonical_preference": "card_only_pmid_first",
            "title": card.get("title") or "",
            "abstract": card.get("abstract") or card.get("abstract_snippet") or "",
            "abstract_available": bool(card.get("abstract") or card.get("abstract_snippet")),
            "year": card.get("year"),
            "journal": card.get("journal"),
            "doi": card.get("doi"),
            "pmid": card.get("pmid"),
            "pmcid": card.get("pmcid"),
            "openalex_id": card.get("openalex_id"),
            "semantic_scholar_id": card.get("semantic_scholar_id"),
            "scopus_id": card.get("scopus_id"),
            "url": card.get("url"),
            "source_apis": card.get("source_apis", []) or [],
            "all_sources_seen": card.get("retrieval_sources", []) or card.get("sources", []) or [],
            "source_queries_seen": card.get("source_queries", []) or card.get("retrieval_queries", []) or [],
            "retrieval_query_ranks": card.get("query_ranks", {}) or card.get("retrieval_query_ranks", {}),
            "retrieval_query_source_ranks": card.get("retrieval_query_source_ranks", {}),
            "branch_terms_detected": card.get("branch_terms_detected", []) or [],
            "filter_hitsets": card.get("filter_hitsets", []) or [],
            "pubtator_checked": bool(card.get("pmid")),
            "pubtator_terms": card.get("pubtator_terms", []) or card.get("pubtator_entity_terms", []) or [],
            "mesh_terms": card.get("mesh_terms", []) or [],
            "keywords": card.get("keywords", []) or [],
            "concepts": card.get("concepts", []) or [],
            "seen_in_subtopics": [],
            "seen_in_queries": [],
            "seen_in_stages": [],
            "statuses": [],
            "used_in_synthesis": False,
            "selected_evidence_ids": [],
            "notes": [],
        }
        entries[sid] = entry
    _append_unique(entry["seen_in_subtopics"], subtopic_id)
    if query:
        _append_unique(entry["seen_in_queries"], query)
    _append_unique(entry["seen_in_stages"], source_stage)
    if source_stage.endswith("excluded") or "excluded" in source_stage:
        _append_unique(entry["statuses"], "deterministically_excluded")
    else:
        _append_unique(entry["statuses"], "candidate")


def _stable_id_for_packet(packet: EvidencePacket) -> str:
    raw = str(packet.paper_id or "")
    if raw.startswith(("PMID:", "DOI:", "PMCID:", "TITLE:")):
        return raw
    if raw.isdigit():
        return f"PMID:{raw}"
    title_key = _normalize_title(packet.title)
    if title_key:
        return f"TITLE:{title_key}"
    return f"PACKET:{packet.evidence_id}"


def _extract_pubtator_terms(pubtator: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    for key in ("genes", "diseases", "chemicals", "species", "mutations", "cell_lines"):
        values = pubtator.get(key, [])
        if isinstance(values, list):
            for v in values:
                if isinstance(v, str):
                    terms.append(v)
                elif isinstance(v, dict):
                    txt = v.get("text") or v.get("name") or v.get("identifier")
                    if txt:
                        terms.append(str(txt))
    # Some cards use compact entity list.
    for key in ("entities", "terms"):
        values = pubtator.get(key, [])
        if isinstance(values, list):
            for v in values:
                if isinstance(v, str):
                    terms.append(v)
                elif isinstance(v, dict):
                    txt = v.get("text") or v.get("name") or v.get("type")
                    if txt:
                        terms.append(str(txt))
    return sorted(set(t.strip() for t in terms if str(t).strip()))


def _normalize_doi(value: Any) -> str:
    if not value:
        return ""
    doi = str(value).strip().lower()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    doi = re.sub(r"^doi:\s*", "", doi)
    return doi.strip().rstrip(".")


def _normalize_title(value: Any) -> str:
    title = str(value or "").lower()
    title = re.sub(r"[^a-z0-9]+", " ", title)
    return " ".join(title.split())


def _append_unique(items: list[Any], value: Any) -> None:
    if value is None or value == "":
        return
    if value not in items:
        items.append(value)


def _extract_prefixed(value: Any, prefix: str) -> str | None:
    raw = str(value or "")
    marker = f"{prefix}:"
    if raw.upper().startswith(marker):
        return raw.split(":", 1)[1]
    return None
