from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retrieval.pubmed_api import PubMedAPI
from retrieval.europepmc_api import EuropePMCAPI
from retrieval.openalex_api import OpenAlexAPI
from retrieval.crossref_api import CrossrefAPI
from retrieval.semantic_scholar_api import SemanticScholarAPI
from schemas.evidence_packet import EvidencePacket
from schemas.paper_record import PaperRecord
from utils.run_logger import log_event
from utils.config import load_config


@dataclass
class RouteLiteratureResult:
    axis_id: str
    queries: list[str]
    evidence_packets: list[EvidencePacket] = field(default_factory=list)
    records: list[PaperRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class MultiSourceLiteratureAgent:
    """Direct multi-source literature adapter wrapper.

    This replaces the old global retrieval workflow. It does only route-specific
    search through selected APIs, deduplicates records, builds compact evidence
    packets, and performs no Gemini reranking or generic pre-generation retrieval.
    """

    def __init__(self, config_path: str = 'configs/config.yaml') -> None:
        self.config_path = config_path
        self.adapters = {
            'PubMed': PubMedAPI(),
            'EuropePMC': EuropePMCAPI(),
            'OpenAlex': OpenAlexAPI(),
            'Crossref': CrossrefAPI(),
            'SemanticScholar': SemanticScholarAPI(allow_unauthenticated=_semantic_scholar_allow_unauthenticated(config_path)),
        }

    def investigate_route(self, *, axis_id: str, objective: str, queries: list[str], sources: list[str], max_queries: int = 2, papers_per_axis: int = 5) -> RouteLiteratureResult:
        clean_queries = [q.strip() for q in queries if str(q).strip()][:max(1, int(max_queries))]
        if not clean_queries:
            return RouteLiteratureResult(axis_id=axis_id, queries=[], warnings=[f'No queries for {axis_id}'])

        selected = [s for s in sources if s in self.adapters]
        warnings: list[str] = []
        records: list[PaperRecord] = []
        per_call_limit = max(1, min(int(papers_per_axis), 5))

        for query in clean_queries:
            for source in selected:
                try:
                    found = self.adapters[source].search(query, limit=per_call_limit)
                    records.extend(found)
                    log_event('retrieval', 'route_source_search', {'axis_id': axis_id, 'source': source, 'query': query, 'records': len(found)})
                except Exception as exc:
                    warnings.append(f'{source} failed for {axis_id}: {exc}')
                    log_event('retrieval', 'route_source_search_failed', {'axis_id': axis_id, 'source': source, 'query': query, 'error': str(exc)}, status='error')

        deduped = _dedupe_records(records)[:max(1, int(papers_per_axis))]
        packets = [_record_to_packet(axis_id, i, r) for i, r in enumerate(deduped, start=1)]
        return RouteLiteratureResult(axis_id=axis_id, queries=clean_queries, evidence_packets=packets, records=deduped, warnings=warnings)


def _dedupe_records(records: list[PaperRecord]) -> list[PaperRecord]:
    seen: set[str] = set()
    out: list[PaperRecord] = []
    for r in records:
        keys = r.stable_keys() or [r.paper_id]
        key = next((k for k in keys if k), r.paper_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    out.sort(key=lambda r: ((r.year or 0), (r.citation_count or 0)), reverse=True)
    return out


def _record_to_packet(axis_id: str, idx: int, record: PaperRecord) -> EvidencePacket:
    source = record.source_api
    text_parts = [f'Title: {record.title}']
    if record.year: text_parts.append(f'Year: {record.year}')
    if record.journal: text_parts.append(f'Journal: {record.journal}')
    if record.abstract: text_parts.append(f'Abstract: {record.abstract}')
    text = '\n'.join(text_parts)
    return EvidencePacket(
        evidence_id=f'{axis_id}_E{idx:03d}',
        paper_id=record.paper_id,
        title=record.title,
        source=source,
        text=text,
        evidence_type='route_literature_abstract',
        metadata={
            'axis_id': axis_id,
            'year': record.year,
            'doi': record.doi,
            'pmid': record.pmid,
            'url': record.url,
            'citation_count': record.citation_count,
            'literature_agent': 'multi_source_literature_agent',
        },
    )


def _semantic_scholar_allow_unauthenticated(config_path: str) -> bool:
    try:
        config = load_config(config_path)
        return bool((((config.get("retrieval") or {}).get("semantic_scholar") or {}).get("allow_unauthenticated")))
    except Exception:
        return False
