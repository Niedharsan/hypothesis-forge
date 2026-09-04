from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retrieval.api_client import CachedAPIClient
from utils.run_logger import log_event


PUBTATOR3_EXPORT_BIOCJSON = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson"
PUBTATOR3_EXPORT_PUBTATOR = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/pubtator"


@dataclass
class PubTatorAnnotation:
    pmid: str
    entities: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    relations: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def compact(self) -> dict[str, Any]:
        return {
            "pmid": self.pmid,
            "entities": self.entities,
            "relations": self.relations,
            "entity_counts": {k: len(v) for k, v in self.entities.items()},
            "relation_count": len(self.relations),
        }


class PubTatorAPI:
    """Small PubTator3 annotation client.

    PubTator3 is used as an annotation/enrichment layer over already retrieved
    papers. NCBI's PubTator3 API asks users to post no more than three requests
    per second. Large PMID batches can also time out, so this client uses small
    batches, a polite request interval, and a PubTator text-format fallback when
    BioC JSON is unavailable or empty.
    """

    def __init__(self, client: CachedAPIClient | None = None, batch_size: int = 10):
        # PubTator3 docs ask for <=3 requests/sec; 0.40 sec/request is safer.
        self.client = client or CachedAPIClient(min_interval_seconds=0.40)
        # Small batches avoid PubTator export timeout/empty-response failures.
        self.batch_size = max(1, min(int(batch_size), 20))

    def annotate_pmids(self, pmids: list[str]) -> dict[str, PubTatorAnnotation]:
        clean_pmids: list[str] = []
        seen = set()
        for pmid in pmids:
            value = str(pmid or "").strip()
            if not value or not value.isdigit() or value in seen:
                continue
            seen.add(value)
            clean_pmids.append(value)
        if not clean_pmids:
            log_event("pubtator", "annotate_pmids_skipped", {"reason": "no_valid_pmids"})
            return {}

        out: dict[str, PubTatorAnnotation] = {}
        failed_batches = 0
        empty_batches = 0
        for i in range(0, len(clean_pmids), self.batch_size):
            batch = clean_pmids[i:i + self.batch_size]
            batch_key = ",".join(batch)
            parsed: dict[str, PubTatorAnnotation] = {}

            # Primary path: BioC JSON export.
            try:
                data = self.client.get_json(
                    PUBTATOR3_EXPORT_BIOCJSON,
                    params={"pmids": batch_key},
                    cache_namespace="pubtator3_biocjson_v59",
                    max_retries=3,
                )
                parsed = _parse_biocjson(data)
                log_event(
                    "pubtator",
                    "annotate_pmids_biocjson",
                    {"batch_pmids": len(batch), "parsed_docs": len(parsed)},
                )
            except Exception as exc:
                log_event(
                    "pubtator",
                    "annotate_pmids_biocjson_failed",
                    {"batch_pmids": len(batch), "error": str(exc)},
                    status="error",
                )

            # Fallback path: PubTator plain text export. This is often more
            # robust and easier to parse for entity inventory purposes.
            if not parsed:
                try:
                    text = self.client.get_text(
                        PUBTATOR3_EXPORT_PUBTATOR,
                        params={"pmids": batch_key},
                        cache_namespace="pubtator3_pubtator_text_v59",
                        max_retries=3,
                    )
                    parsed = _parse_pubtator_text(text)
                    log_event(
                        "pubtator",
                        "annotate_pmids_pubtator_text",
                        {"batch_pmids": len(batch), "parsed_docs": len(parsed), "chars": len(text or "")},
                    )
                except Exception as exc:
                    failed_batches += 1
                    log_event(
                        "pubtator",
                        "annotate_pmids_pubtator_text_failed",
                        {"batch_pmids": len(batch), "error": str(exc)},
                        status="error",
                    )

            if not parsed:
                empty_batches += 1
            out.update(parsed)

        log_event(
            "pubtator",
            "annotate_pmids_complete",
            {
                "requested_pmids": len(clean_pmids),
                "annotated_pmids": len(out),
                "batch_size": self.batch_size,
                "failed_batches": failed_batches,
                "empty_batches": empty_batches,
            },
        )
        return out


def _parse_biocjson(data: Any) -> dict[str, PubTatorAnnotation]:
    """Parse BioC JSON robustly across common collection wrappers."""
    documents: list[Any] = []
    if isinstance(data, dict):
        if isinstance(data.get("documents"), list):
            documents = data.get("documents", [])
        elif isinstance(data.get("collection"), dict) and isinstance(data["collection"].get("documents"), list):
            documents = data["collection"].get("documents", [])
    elif isinstance(data, list):
        documents = data

    parsed: dict[str, PubTatorAnnotation] = {}
    for doc in documents:
        if not isinstance(doc, dict):
            continue
        pmid = str(doc.get("id") or doc.get("pmid") or "").strip()
        if not pmid:
            continue
        entities: dict[str, list[dict[str, Any]]] = {}
        relations: list[dict[str, Any]] = []

        for passage in doc.get("passages", []) or []:
            if not isinstance(passage, dict):
                continue
            for ann in passage.get("annotations", []) or []:
                if not isinstance(ann, dict):
                    continue
                infons = ann.get("infons", {}) or {}
                entity_type = str(
                    infons.get("type")
                    or infons.get("biotype")
                    or infons.get("category")
                    or "Unknown"
                )
                entity = {
                    "text": ann.get("text"),
                    "type": entity_type,
                    "identifier": infons.get("identifier") or infons.get("id") or infons.get("Identifier"),
                    "infons": infons,
                    "locations": ann.get("locations", []),
                }
                entities.setdefault(entity_type, []).append(entity)
            for rel in passage.get("relations", []) or []:
                if isinstance(rel, dict):
                    relations.append(rel)

        for rel in doc.get("relations", []) or []:
            if isinstance(rel, dict):
                relations.append(rel)

        parsed[pmid] = PubTatorAnnotation(pmid=pmid, entities=entities, relations=relations, raw=doc)
    return parsed


def _parse_pubtator_text(text: str) -> dict[str, PubTatorAnnotation]:
    """Parse PubTator tab-delimited export.

    Common lines:
      PMID|t|Title
      PMID|a|Abstract
      PMID<TAB>start<TAB>end<TAB>mention<TAB>type<TAB>identifier
    """
    parsed: dict[str, PubTatorAnnotation] = {}
    if not text:
        return parsed
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "|t|" in line or "|a|" in line:
            pmid = line.split("|", 1)[0].strip()
            if pmid:
                parsed.setdefault(pmid, PubTatorAnnotation(pmid=pmid, raw={"format": "pubtator"}))
            continue
        parts = line.split("\t")
        if len(parts) < 6:
            continue
        pmid, start, end, mention, entity_type, identifier = parts[:6]
        pmid = str(pmid or "").strip()
        if not pmid:
            continue
        ann = parsed.setdefault(pmid, PubTatorAnnotation(pmid=pmid, raw={"format": "pubtator"}))
        entity = {
            "text": mention,
            "type": entity_type,
            "identifier": identifier,
            "infons": {"type": entity_type, "identifier": identifier},
            "locations": [{"offset": _safe_int(start), "length": max(0, (_safe_int(end) or 0) - (_safe_int(start) or 0))}],
        }
        ann.entities.setdefault(entity_type, []).append(entity)
    # Keep only PMIDs that had actual entity annotations, not title-only docs.
    return {pmid: ann for pmid, ann in parsed.items() if any(ann.entities.values())}


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def pubtator_entity_terms(annotation: PubTatorAnnotation | dict[str, Any] | None) -> set[str]:
    """Return normalized entity texts/ids for scoring/diversity."""
    if annotation is None:
        return set()
    if isinstance(annotation, PubTatorAnnotation):
        entities = annotation.entities
    else:
        entities = annotation.get("entities", {}) if isinstance(annotation, dict) else {}
    terms: set[str] = set()
    for items in entities.values():
        for item in items:
            if not isinstance(item, dict):
                continue
            for key in ("text", "identifier"):
                value = item.get(key)
                if value:
                    terms.add(str(value).lower().strip())
    return {t for t in terms if t}
