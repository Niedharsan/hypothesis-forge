from __future__ import annotations

import os
from urllib.parse import quote
from typing import Any

from dotenv import load_dotenv

from retrieval.api_client import CachedAPIClient, clean_text, safe_int
from retrieval.query_variants import build_literature_query_variants
from schemas.paper_record import PaperRecord
from utils.run_logger import log_event

load_dotenv()

SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1"
SEMANTIC_SCHOLAR_RECOMMENDATIONS_URL = "https://api.semanticscholar.org/recommendations/v1/papers"
SEMANTIC_SCHOLAR_FIELDS = (
    "paperId,title,abstract,authors,year,externalIds,url,venue,publicationTypes,"
    "publicationDate,citationCount,influentialCitationCount,referenceCount,"
    "openAccessPdf,isOpenAccess,fieldsOfStudy,s2FieldsOfStudy,tldr,embedding,journal"
)
SEMANTIC_SCHOLAR_LIGHT_FIELDS = (
    "paperId,title,abstract,authors,year,externalIds,url,venue,publicationTypes,"
    "citationCount,influentialCitationCount,referenceCount,openAccessPdf,isOpenAccess,"
    "fieldsOfStudy,s2FieldsOfStudy,tldr,journal"
)

_SEMANTIC_SCHOLAR_CLIENT = CachedAPIClient(
    cache_dir="data/cache/api",
    min_interval_seconds=1.2,
)


class SemanticScholarAPI:
    """Semantic Scholar Academic Graph + Recommendations adapter.

    The adapter intentionally exposes more than keyword search so the shared
    retrieval layer can use Semantic Scholar as a scholarly graph/AI retrieval
    source: search, enrichment, citation/reference expansion, and related-paper
    recommendations.
    """

    source_name = "SemanticScholar"

    def __init__(
        self,
        client: CachedAPIClient | None = None,
        allow_unauthenticated: bool = False,
        include_embeddings: bool = False,
    ):
        self.client = client or _SEMANTIC_SCHOLAR_CLIENT
        self.allow_unauthenticated = allow_unauthenticated
        self.include_embeddings = include_embeddings

    @property
    def api_key(self) -> str | None:
        return os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None

    @property
    def enabled(self) -> bool:
        return bool(self.api_key) or self.allow_unauthenticated

    @property
    def fields(self) -> str:
        return SEMANTIC_SCHOLAR_FIELDS if self.include_embeddings else SEMANTIC_SCHOLAR_LIGHT_FIELDS

    def search(self, query: str, limit: int = 10, year: str | None = None, offset: int | None = None) -> list[PaperRecord]:
        if not self.enabled:
            return []
        query_variants = build_literature_query_variants(query)
        records: list[PaperRecord] = []
        seen: set[str] = set()
        variant_result_counts: dict[str, int] = {}
        for executed_query in query_variants:
            params: dict[str, Any] = {"query": executed_query, "limit": limit, "fields": self.fields}
            if year:
                params["year"] = year
            if offset:
                params["offset"] = offset
            try:
                data = self.client.get_json(
                    f"{SEMANTIC_SCHOLAR_BASE_URL}/paper/search",
                    params=params,
                    headers=self._headers(),
                    cache_namespace="semantic_scholar",
                    max_retries=4,
                )
            except Exception as exc:
                log_event("retrieval", "semantic_scholar_search_adapter_error", {"query": query, "executed_query": executed_query, "error": str(exc)}, status="error")
                raise
            items = data.get("data", [])
            variant_result_counts[executed_query] = len(items)
            for item in items:
                rec = self.record_from_item(item, retrieval_mode="semantic_search")
                key = rec.paper_id or rec.doi or rec.title
                if key and key not in seen:
                    seen.add(key)
                    rec.raw["original_query"] = query
                    rec.raw["executed_query_variants"] = query_variants
                    rec.raw["matched_query_variant"] = executed_query
                    rec.raw["variant_result_counts"] = variant_result_counts
                    records.append(rec)
                    if len(records) >= limit:
                        return records
        return records

    def details(self, paper_id: str, include_embeddings: bool | None = None) -> PaperRecord | None:
        if not self.enabled:
            return None
        fields = SEMANTIC_SCHOLAR_FIELDS if include_embeddings is True else self.fields
        try:
            data = self.client.get_json(
                f"{SEMANTIC_SCHOLAR_BASE_URL}/paper/{quote(paper_id, safe='')}",
                params={"fields": fields},
                headers=self._headers(),
                cache_namespace="semantic_scholar",
                max_retries=4,
            )
        except Exception as exc:
            log_event("retrieval", "semantic_scholar_details_adapter_error", {"paper_id": paper_id, "error": str(exc)}, status="error")
            raise
        return self.record_from_item(data, retrieval_mode="paper_details")

    def batch_details(self, paper_ids: list[str], include_embeddings: bool | None = None) -> list[PaperRecord]:
        if not self.enabled or not paper_ids:
            return []
        fields = SEMANTIC_SCHOLAR_FIELDS if include_embeddings is True else self.fields
        try:
            data = self.client.post_json(
                f"{SEMANTIC_SCHOLAR_BASE_URL}/paper/batch",
                params={"fields": fields},
                payload={"ids": paper_ids[:500]},
                headers=self._headers(),
                cache_namespace="semantic_scholar",
                max_retries=4,
            )
        except Exception as exc:
            log_event("retrieval", "semantic_scholar_batch_details_adapter_error", {"paper_ids": paper_ids[:10], "error": str(exc)}, status="error")
            raise
        if isinstance(data, list):
            items = data
        else:
            items = data.get("data", []) if isinstance(data, dict) else []
        return [self.record_from_item(item, retrieval_mode="batch_details") for item in items if item]

    def citations(self, paper_id: str, limit: int = 10) -> list[PaperRecord]:
        return self._edge_records(paper_id, "citations", limit, child_key="citingPaper", retrieval_mode="citation_expansion")

    def references(self, paper_id: str, limit: int = 10) -> list[PaperRecord]:
        return self._edge_records(paper_id, "references", limit, child_key="citedPaper", retrieval_mode="reference_expansion")

    def recommendations(
        self,
        positive_paper_ids: list[str],
        negative_paper_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[PaperRecord]:
        if not self.enabled or not positive_paper_ids:
            return []
        try:
            data = self.client.post_json(
                SEMANTIC_SCHOLAR_RECOMMENDATIONS_URL,
                params={"fields": self.fields, "limit": limit},
                payload={
                    "positivePaperIds": positive_paper_ids[:20],
                    "negativePaperIds": (negative_paper_ids or [])[:20],
                },
                headers=self._headers(),
                cache_namespace="semantic_scholar_recommendations",
                max_retries=4,
            )
        except Exception as exc:
            log_event("retrieval", "semantic_scholar_recommendations_adapter_error", {"positive_paper_ids": positive_paper_ids[:10], "error": str(exc)}, status="error")
            raise
        return [self.record_from_item(item, retrieval_mode="recommendation") for item in data.get("recommendedPapers", data.get("data", []))]

    def expand_from_seed_papers(
        self,
        seed_paper_ids: list[str],
        max_seed_papers: int = 5,
        max_results_per_paper: int = 5,
        use_citations: bool = True,
        use_references: bool = True,
        use_recommendations: bool = True,
    ) -> list[PaperRecord]:
        if not self.enabled:
            return []
        seeds = [paper_id for paper_id in seed_paper_ids if paper_id][:max_seed_papers]
        expanded: list[PaperRecord] = []
        if use_recommendations and seeds:
            expanded.extend(self.recommendations(seeds, limit=max_seed_papers * max_results_per_paper))
        for paper_id in seeds:
            if use_citations:
                expanded.extend(self.citations(paper_id, limit=max_results_per_paper))
            if use_references:
                expanded.extend(self.references(paper_id, limit=max_results_per_paper))
        return expanded

    def _edge_records(self, paper_id: str, edge: str, limit: int, child_key: str, retrieval_mode: str) -> list[PaperRecord]:
        if not self.enabled or not paper_id:
            return []
        try:
            data = self.client.get_json(
                f"{SEMANTIC_SCHOLAR_BASE_URL}/paper/{quote(paper_id, safe='')}/{edge}",
                params={"limit": limit, "fields": self.fields},
                headers=self._headers(),
                cache_namespace="semantic_scholar",
                max_retries=4,
            )
        except Exception as exc:
            log_event("retrieval", "semantic_scholar_edge_adapter_error", {"paper_id": paper_id, "edge": edge, "error": str(exc)}, status="error")
            raise
        records: list[PaperRecord] = []
        for item in data.get("data", []):
            child = item.get(child_key) or item.get("paper") or item
            if child:
                records.append(self.record_from_item(child, retrieval_mode=retrieval_mode, parent_paper_id=paper_id))
        return records

    def _headers(self) -> dict[str, str] | None:
        if not self.api_key:
            return None
        return {"x-api-key": self.api_key}

    def record_from_item(self, item: dict, retrieval_mode: str | None = None, parent_paper_id: str | None = None) -> PaperRecord:
        external_ids = item.get("externalIds") or {}
        journal = item.get("journal") if isinstance(item.get("journal"), dict) else {}
        venue = item.get("venue")
        open_access_pdf = item.get("openAccessPdf") if isinstance(item.get("openAccessPdf"), dict) else {}
        tldr = item.get("tldr") if isinstance(item.get("tldr"), dict) else item.get("tldr")
        tldr_text = tldr.get("text") if isinstance(tldr, dict) else None
        fields = item.get("fieldsOfStudy") or []
        s2_fields = item.get("s2FieldsOfStudy") or []
        s2_field_names = [field.get("category") for field in s2_fields if isinstance(field, dict) and field.get("category")]
        concepts = list(dict.fromkeys([*fields, *s2_field_names]))
        raw = {"semantic_scholar": item}
        raw["semantic_scholar_metadata"] = {
            "publication_types": item.get("publicationTypes") or [],
            "publication_date": item.get("publicationDate"),
            "influential_citation_count": safe_int(item.get("influentialCitationCount")),
            "reference_count": safe_int(item.get("referenceCount")),
            "open_access_pdf": open_access_pdf,
            "tldr": tldr_text,
            "embedding": item.get("embedding"),
            "retrieval_mode": retrieval_mode,
            "parent_paper_id": parent_paper_id,
        }
        return PaperRecord(
            paper_id=f"s2:{item.get('paperId')}",
            title=clean_text(item.get("title")) or "Untitled Semantic Scholar paper",
            abstract=clean_text(item.get("abstract")) or clean_text(tldr_text),
            authors=[a.get("name") for a in item.get("authors", []) if isinstance(a, dict) and a.get("name")],
            year=safe_int(item.get("year")),
            journal=journal.get("name") or venue,
            doi=external_ids.get("DOI"),
            pmid=external_ids.get("PubMed"),
            pmcid=external_ids.get("PubMedCentral"),
            semantic_scholar_id=item.get("paperId"),
            url=item.get("url") or open_access_pdf.get("url"),
            source_apis=[self.source_name],
            citation_count=safe_int(item.get("citationCount")),
            is_open_access=item.get("isOpenAccess") if item.get("isOpenAccess") is not None else bool(open_access_pdf.get("url")),
            concepts=concepts,
            raw=raw,
        )
