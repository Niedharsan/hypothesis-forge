from __future__ import annotations

import os
import xml.etree.ElementTree as ET

from retrieval.api_client import CachedAPIClient, clean_text, year_from_text
from retrieval.query_variants import build_literature_query_variants
from schemas.paper_record import PaperRecord


class PubMedAPI:
    source_name = "PubMed"

    def __init__(self, client: CachedAPIClient | None = None):
        # NCBI E-utilities recommend no more than 3 requests/sec without an API key,
        # and allow higher throughput with an API key. PubMed uses esearch + esummary
        # + efetch, so use a safer default interval than the generic client.
        interval = 0.11 if os.getenv("NCBI_API_KEY") else 0.34
        self.client = client or CachedAPIClient(min_interval_seconds=interval)

    def search(self, query: str, limit: int = 10, cutoff_year: int | None = None) -> list[PaperRecord]:
        query_variants = build_literature_query_variants(query)
        pmids: list[str] = []
        variant_by_pmid: dict[str, str] = {}
        variant_result_counts: dict[str, int] = {}
        variant_query_translations: dict[str, str] = {}
        variant_sort_modes: dict[str, str] = {}
        for executed_query in query_variants:
            params = {
                "db": "pubmed",
                "term": executed_query,
                "retmode": "json",
                "retmax": limit,
                "sort": "relevance",
            }
            variant_sort_modes[executed_query] = "relevance"
            # When a cutoff is requested, ask PubMed for records within the date
            # window directly instead of retrieving unrestricted recent records and
            # filtering afterward. This mirrors manual PubMed use with a publication
            # date cap and keeps PMID-backed pre-cutoff papers available for PubTator.
            if cutoff_year is not None:
                params.update({
                    "datetype": "pdat",
                    "mindate": "1800/01/01",
                    "maxdate": f"{int(cutoff_year)}/12/31",
                })
            _add_ncbi_identity(params)

            search_data = self.client.get_json(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params=params,
                cache_namespace="pubmed_esearch",
            )
            esearch_result = search_data.get("esearchresult", {})
            ids_for_variant = [str(pmid) for pmid in esearch_result.get("idlist", [])]
            variant_result_counts[executed_query] = len(ids_for_variant)
            if esearch_result.get("querytranslation"):
                variant_query_translations[executed_query] = str(esearch_result.get("querytranslation"))
            for pmid in ids_for_variant:
                if pmid not in variant_by_pmid:
                    variant_by_pmid[pmid] = executed_query
                    pmids.append(pmid)

        pmids = pmids[:limit]
        if not pmids:
            return []

        summary_params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "json"}
        _add_ncbi_identity(summary_params)
        summary = self.client.get_json(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params=summary_params,
            cache_namespace="pubmed_esummary",
        )
        result = summary.get("result", {})

        parsed_by_pmid: dict[str, PaperRecord] = {}
        try:
            fetch_params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
            _add_ncbi_identity(fetch_params)
            xml_text = self.client.get_text(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params=fetch_params,
                cache_namespace="pubmed_efetch",
            )
            for parsed in self.parse_xml_records(xml_text):
                if parsed.pmid:
                    parsed_by_pmid[str(parsed.pmid)] = parsed
        except Exception:
            # PubMed summaries are still useful if abstract fetch fails. The
            # API client already logs the request error.
            parsed_by_pmid = {}

        records = []
        for pmid in pmids:
            item = result.get(pmid, {})
            parsed = parsed_by_pmid.get(str(pmid))
            article_ids = item.get("articleids", [])
            doi = (parsed.doi if parsed else None) or _article_id(article_ids, "doi")
            pmcid = (parsed.pmcid if parsed else None) or _article_id(article_ids, "pmc")
            records.append(PaperRecord(
                paper_id=f"pmid:{pmid}",
                title=(parsed.title if parsed else None) or clean_text(item.get("title")) or "Untitled PubMed record",
                abstract=(parsed.abstract if parsed else None),
                authors=(parsed.authors if parsed and parsed.authors else [a.get("name") for a in item.get("authors", []) if a.get("name")]),
                year=(parsed.year if parsed else None) or year_from_text(item.get("pubdate")),
                journal=(parsed.journal if parsed else None) or item.get("fulljournalname"),
                doi=doi,
                pmid=pmid,
                pmcid=pmcid,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                source_apis=[self.source_name],
                mesh_terms=(parsed.mesh_terms if parsed else []),
                raw={
                    "esummary": item,
                    "efetch_abstract_found": bool(parsed and parsed.abstract),
                    "original_query": query,
                    "executed_query_variants": query_variants,
                    "matched_query_variant": variant_by_pmid.get(str(pmid)),
                    "variant_result_counts": variant_result_counts,
                    "variant_query_translations": variant_query_translations,
                    "variant_sort_modes": variant_sort_modes,
                    "date_filter": {"datetype": "pdat", "mindate": "1800/01/01", "maxdate": f"{int(cutoff_year)}/12/31"} if cutoff_year is not None else None,
                },
            ))
        return records

    def parse_xml_records(self, xml_text: str) -> list[PaperRecord]:
        root = ET.fromstring(xml_text)
        records = []
        for article in root.findall(".//PubmedArticle"):
            pmid = _node_text(article, ".//PMID")
            title = clean_text(_node_text(article, ".//ArticleTitle")) or "Untitled PubMed record"
            abstract_parts = [node.text or "" for node in article.findall(".//AbstractText")]
            doi = _article_id_xml(article, "doi")
            pmcid = _article_id_xml(article, "pmc")
            records.append(PaperRecord(
                paper_id=f"pmid:{pmid}" if pmid else title,
                title=title,
                abstract=clean_text(" ".join(abstract_parts)),
                authors=[
                    " ".join(filter(None, [_node_text(author, "ForeName"), _node_text(author, "LastName")]))
                    for author in article.findall(".//Author")
                ],
                year=year_from_text(_node_text(article, ".//PubDate/Year")),
                journal=_node_text(article, ".//Journal/Title"),
                doi=doi,
                pmid=pmid,
                pmcid=pmcid,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                source_apis=[self.source_name],
                mesh_terms=[_node_text(mesh, "DescriptorName") or "" for mesh in article.findall(".//MeshHeading")],
                raw={"pubmed_xml": xml_text},
            ))
        return records


def _article_id(article_ids: list[dict], id_type: str) -> str | None:
    for item in article_ids:
        if item.get("idtype") == id_type:
            return item.get("value")
    return None


def _node_text(node: ET.Element, path: str) -> str | None:
    found = node.find(path)
    return found.text if found is not None else None


def _article_id_xml(article: ET.Element, id_type: str) -> str | None:
    for node in article.findall(".//ArticleId"):
        if node.attrib.get("IdType") == id_type:
            return node.text
    return None


def _add_ncbi_identity(params: dict) -> None:
    params["tool"] = os.getenv("NCBI_TOOL", "hypothesisforge")
    if os.getenv("NCBI_EMAIL"):
        params["email"] = os.getenv("NCBI_EMAIL")
    if os.getenv("NCBI_API_KEY"):
        params["api_key"] = os.getenv("NCBI_API_KEY")
