from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field, field_validator

StageName = Literal[
    "axis_generation",
    "subtopic_generation",
    "literature_retrieval",
    "synthesis",
    "hypothesis_generation",
    "proximity",
    "reflection",
    "evolution",
    "candidate_ranking",
]


class StartRunRequest(BaseModel):
    research_objective: str = Field(min_length=3, max_length=20_000)
    cutoff_year: int = Field(default=2023, ge=1900, le=2100)
    model: str = Field(default="gemini-2.5-flash-lite", min_length=3, max_length=100)
    output_count: int = Field(default=10, ge=1, le=50)
    runtime_mode: Literal["normal", "dry_run"] = "normal"
    literature_sources: list[str] = Field(default_factory=lambda: ["PubMed", "EuropePMC", "OpenAlex", "Crossref"])
    use_pubtator: bool = False
    enable_evolution_retrieval: bool = False

    @field_validator("research_objective")
    @classmethod
    def clean_objective(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 3:
            raise ValueError("Research objective must contain text")
        return value


class StageRequest(BaseModel):
    stage: StageName
    source_stage: StageName | None = None
    selected_ids: list[str] = Field(default_factory=list, max_length=200)
    include_all: bool = False
    output_count: int = Field(default=10, ge=1, le=50)
    stage_guidance: str = Field(default="", max_length=20_000)
    selection_source: Literal["auto", "user", "supervisor"] = "user"


class SelectionRequest(BaseModel):
    stage: StageName
    selected_ids: list[str] = Field(default_factory=list, max_length=200)
    rejected_ids: list[str] = Field(default_factory=list, max_length=200)
    saved_ids: list[str] = Field(default_factory=list, max_length=200)
    selection_source: Literal["auto", "user", "supervisor"] = "user"


class FocusSeedRequest(BaseModel):
    source_card_id: str = Field(min_length=1, max_length=200)
    title: str = Field(default="", max_length=2_000)
    summary: str = Field(default="", max_length=20_000)
    guidance: str = Field(default="", max_length=20_000)
