from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any


@dataclass
class PaperRecord:
    paper_id: str
    title: str
    abstract: str | None = None
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    journal: str | None = None
    doi: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    scopus_id: str | None = None
    semantic_scholar_id: str | None = None
    openalex_id: str | None = None
    url: str | None = None
    source_apis: list[str] = field(default_factory=list)
    citation_count: int | None = None
    is_open_access: bool | None = None
    full_text_available: bool = False
    full_text: str | None = None
    keywords: list[str] = field(default_factory=list)
    mesh_terms: list[str] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def source_api(self) -> str:
        return self.source_apis[0] if self.source_apis else "Unknown"

    @property
    def metadata(self) -> dict[str, Any]:
        return self.raw

    def stable_keys(self) -> list[str]:
        keys = []
        for prefix, value in [
            ("doi", _normalize_doi(self.doi)),
            ("pmid", self.pmid),
            ("pmcid", self.pmcid),
            ("s2", self.semantic_scholar_id),
            ("openalex", self.openalex_id),
            ("scopus", self.scopus_id),
        ]:
            if value:
                keys.append(f"{prefix}:{str(value).lower().strip()}")
        title_key = " ".join(self.title.lower().split())
        if title_key:
            keys.append(f"title:{title_key}")
        return keys

    def stable_key(self) -> str:
        keys = self.stable_keys()
        return keys[0] if keys else f"paper:{self.paper_id}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    doi = str(value).strip().lower()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    doi = re.sub(r"^doi:\s*", "", doi)
    return doi.strip().rstrip(".") or None
