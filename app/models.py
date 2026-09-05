from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

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
_STAGE_ORDER = [
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
_STAGE_INDEX = {stage: index for index, stage in enumerate(_STAGE_ORDER)}


def _current_year() -> int:
    return datetime.now(timezone.utc).year


class StartRunRequest(BaseModel):
    research_objective: str = Field(min_length=3, max_length=20_000)
    cutoff_year: int = Field(default_factory=_current_year, ge=1900, le=2100)
    model: str = Field(default="gemini-2.5-flash-lite", min_length=3, max_length=100)
    output_count: Literal[10] = 10
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

    @model_validator(mode="after")
    def adjacent_forward_only(self) -> "StageRequest":
        if self.source_stage is not None and _STAGE_INDEX[self.source_stage] + 1 != _STAGE_INDEX[self.stage]:
            raise ValueError("Stage transitions must advance exactly one checkpoint")
        return self


class SelectionRequest(BaseModel):
    stage: StageName
    selected_ids: list[str] = Field(default_factory=list, max_length=200)
    rejected_ids: list[str] = Field(default_factory=list, max_length=200)
    saved_ids: list[str] = Field(default_factory=list, max_length=200)
    selection_source: Literal["auto", "user", "supervisor"] = "user"

    @model_validator(mode="after")
    def disjoint_statuses(self) -> "SelectionRequest":
        combined = [*self.selected_ids, *self.rejected_ids, *self.saved_ids]
        if len(combined) != len(set(combined)):
            raise ValueError("A card cannot be selected, saved, and rejected in the same update")
        return self


class FocusSeedRequest(BaseModel):
    source_card_id: str = Field(min_length=1, max_length=200)
    source_stage: StageName | None = None
    title: str = Field(default="", max_length=2_000)
    summary: str = Field(default="", max_length=20_000)
    guidance: str = Field(default="", max_length=20_000)
