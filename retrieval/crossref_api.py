from __future__ import annotations

import os
from html import unescape

from retrieval.api_client import CachedAPIClient, clean_text, safe_int
from retrieval.query_variants import build_literature_query_variants
from schemas.paper_record import PaperRecord


class CrossrefAPI:
    source_name = "Crossref"

    def __init__(self, client: CachedAPIClient | None = None):
        self.client = client or CachedAPIClient()

    def search(self, query: str, limit: int = 10) -> list[PaperRecord]:
        query_variants = build_literature_query_variants(query)
        records: list[PaperRecord] = []
        seen: set[str] = set()
        variant_result_counts: dict[str, int] = {}
        for executed_query in query_variants:
            params = {"query": executed_query, "rows": limit}
            if os.getenv("CROSSREF_MAILTO"):
                params["mailto"] = os.getenv("CROSSREF_MAILTO")
            data = self.client.get_json(
                "https://api.crossref.org/works",
                params=params,
                cache_namespace="crossref",
            )
            items = data.get("message", {}).get("items", [])
            variant_result_counts[executed_query] = len(items)
            for item in items:
                rec = self.record_from_item(item)
                key = rec.doi or rec.title
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

    def record_from_item(self, item: dict) -> PaperRecord:
        title = (item.get("title") or ["Untitled Crossref work"])[0]
        abstract = item.get("abstract")
        return PaperRecord(
            paper_id=f"doi:{item.get('DOI')}" if item.get("DOI") else f"crossref:{title}",
            title=clean_text(title) or "Untitled Crossref work",
            abstract=clean_text(unescape(abstract)) if abstract else None,
            authors=_author_names(item.get("author", [])),
            year=_date_year(item.get("published-print") or item.get("published-online") or item.get("created")),
            journal=(item.get("container-title") or [None])[0],
            doi=item.get("DOI"),
            url=item.get("URL"),
            source_apis=[self.source_name],
            citation_count=safe_int(item.get("is-referenced-by-count")),
            raw={"crossref": item},
        )


def _author_names(items: list[dict]) -> list[str]:
    names = []
    for item in items:
        name = item.get("name") or " ".join(filter(None, [item.get("given"), item.get("family")]))
        if name:
            names.append(name)
    return names


def _date_year(date_obj: dict | None) -> int | None:
    parts = (date_obj or {}).get("date-parts") or []
    if parts and parts[0]:
        return safe_int(parts[0][0])
    return None
