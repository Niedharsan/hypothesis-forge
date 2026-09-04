from __future__ import annotations

from retrieval.api_client import CachedAPIClient, clean_text, safe_int
from retrieval.query_variants import build_literature_query_variants
from schemas.paper_record import PaperRecord


class EuropePMCAPI:
    source_name = "EuropePMC"

    def __init__(self, client: CachedAPIClient | None = None):
        self.client = client or CachedAPIClient()

    def search(self, query: str, limit: int = 10, cutoff_year: int | None = None) -> list[PaperRecord]:
        records: list[PaperRecord] = []
        seen: set[str] = set()
        query_variants = build_literature_query_variants(query)
        variant_result_counts: dict[str, int] = {}
        executed_search_queries: dict[str, str] = {}
        for executed_query in query_variants:
            search_query = executed_query
            # Apply the cutoff inside Europe PMC retrieval where possible, rather
            # than fetching unrestricted recent records and filtering afterward.
            # Parentheses preserve query logic when adding the year clause.
            if cutoff_year is not None:
                search_query = f"({executed_query}) AND PUB_YEAR:[1800 TO {int(cutoff_year)}]"
            executed_search_queries[executed_query] = search_query
            data = self.client.get_json(
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                params={"query": search_query, "format": "json", "pageSize": limit, "resultType": "core"},
                cache_namespace="europepmc",
            )
            items = data.get("resultList", {}).get("result", [])
            variant_result_counts[executed_query] = len(items)
            for item in items:
                rec = self.record_from_item(item)
                key = rec.pmid or rec.doi or rec.title
                if key and key not in seen:
                    seen.add(key)
                    rec.raw["original_query"] = query
                    rec.raw["executed_query_variants"] = query_variants
                    rec.raw["matched_query_variant"] = executed_query
                    rec.raw["variant_result_counts"] = variant_result_counts
                    rec.raw["executed_search_queries"] = executed_search_queries
                    rec.raw["date_filter"] = {"PUB_YEAR": f"1800 TO {int(cutoff_year)}"} if cutoff_year is not None else None
                    records.append(rec)
                    if len(records) >= limit:
                        return records
        return records

    def record_from_item(self, item: dict) -> PaperRecord:
        full_text_urls = item.get("fullTextUrlList", {}).get("fullTextUrl", []) if item.get("fullTextUrlList") else []
        return PaperRecord(
            paper_id=f"europepmc:{item.get('id')}",
            title=clean_text(item.get("title")) or "Untitled Europe PMC record",
            abstract=clean_text(item.get("abstractText")),
            authors=[a.strip() for a in (item.get("authorString") or "").split(",") if a.strip()],
            year=safe_int(item.get("pubYear")),
            journal=item.get("journalTitle"),
            doi=item.get("doi"),
            pmid=item.get("pmid"),
            pmcid=item.get("pmcid"),
            url=full_text_urls[0].get("url") if full_text_urls else None,
            source_apis=[self.source_name],
            citation_count=safe_int(item.get("citedByCount")),
            is_open_access=item.get("isOpenAccess") == "Y",
            full_text_available=bool(full_text_urls),
            full_text=None,
            raw={"europepmc": item},
        )
