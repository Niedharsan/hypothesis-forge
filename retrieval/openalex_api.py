from __future__ import annotations

import os

from retrieval.api_client import CachedAPIClient, clean_text, safe_int
from retrieval.query_variants import build_literature_query_variants
from schemas.paper_record import PaperRecord


class OpenAlexAPI:
    source_name = "OpenAlex"

    def __init__(self, client: CachedAPIClient | None = None):
        self.client = client or CachedAPIClient()

    def search(self, query: str, limit: int = 10) -> list[PaperRecord]:
        query_variants = build_literature_query_variants(query)
        records: list[PaperRecord] = []
        seen: set[str] = set()
        variant_result_counts: dict[str, int] = {}
        for executed_query in query_variants:
            params = {"search": executed_query, "per-page": limit}
            if os.getenv("OPENALEX_MAILTO"):
                params["mailto"] = os.getenv("OPENALEX_MAILTO")
            if os.getenv("OPENALEX_API_KEY"):
                params["api_key"] = os.getenv("OPENALEX_API_KEY")
            data = self.client.get_json(
                "https://api.openalex.org/works",
                params=params,
                headers=None,
                cache_namespace="openalex",
            )
            items = data.get("results", [])
            variant_result_counts[executed_query] = len(items)
            for item in items:
                rec = self.record_from_item(item)
                key = rec.openalex_id or rec.doi or rec.title
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
        location = item.get("primary_location") or {}
        source = location.get("source") or {}
        open_access = item.get("open_access") or {}
        return PaperRecord(
            paper_id=item.get("id") or f"openalex:{item.get('doi') or item.get('title')}",
            title=clean_text(item.get("title")) or "Untitled OpenAlex work",
            abstract=_abstract_from_inverted_index(item.get("abstract_inverted_index")),
            authors=[a.get("author", {}).get("display_name") for a in item.get("authorships", []) if a.get("author", {}).get("display_name")],
            year=safe_int(item.get("publication_year")),
            journal=source.get("display_name"),
            doi=item.get("doi"),
            openalex_id=item.get("id"),
            url=location.get("landing_page_url"),
            source_apis=[self.source_name],
            citation_count=safe_int(item.get("cited_by_count")),
            is_open_access=bool(open_access.get("is_oa")) if open_access else None,
            full_text_available=bool(open_access.get("oa_url")) if open_access else False,
            concepts=[c.get("display_name") for c in item.get("concepts", []) if c.get("display_name")],
            raw={"openalex": item},
        )


def _abstract_from_inverted_index(index: dict | None) -> str | None:
    if not index:
        return None
    positions: dict[int, str] = {}
    for word, offsets in index.items():
        for offset in offsets:
            positions[int(offset)] = word
    return " ".join(positions[i] for i in sorted(positions))
