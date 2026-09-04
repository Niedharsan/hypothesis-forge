from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from schemas.paper_record import PaperRecord


class PaperRecordPayload(BaseModel):
    paper_id: str
    title: str
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    journal: str | None = None
    doi: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    scopus_id: str | None = None
    semantic_scholar_id: str | None = None
    openalex_id: str | None = None
    url: str | None = None
    source_apis: list[str] = Field(default_factory=list)
    citation_count: int | None = None
    is_open_access: bool | None = None
    full_text_available: bool = False
    full_text: str | None = None
    keywords: list[str] = Field(default_factory=list)
    mesh_terms: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_record(cls, record: PaperRecord) -> "PaperRecordPayload":
        return cls.model_validate(record.to_dict())
