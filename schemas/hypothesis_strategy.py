from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class HypothesisStrategy:
    strategy_id: str
    title: str
    strategy_type: str
    proposed_intervention: str
    target_or_pathway: str
    mechanism: str
    why_selective_for_pathogen_or_disease_state: str
    why_host_sparing: str
    supporting_evidence_ids: list[str] = field(default_factory=list)
    prior_art_overlap: str = "unknown"
    novelty_rationale: str = ""
    risks: list[str] = field(default_factory=list)
    falsification_test: str = ""
    first_experiment: str = ""
    family: str | None = None
    parent_strategy_ids: list[str] = field(default_factory=list)
    generation_round: int = 0
    novelty_score: float = 0.0
    evidence_score: float = 0.0
    feasibility_score: float = 0.0
    host_safety_score: float = 0.0
    prior_art_risk_score: float = 0.0
    overall_score: float = 0.0
    elo_score: float = 1000.0
    status: str = "candidate"
    metadata: dict[str, Any] = field(default_factory=dict)


    @property
    def answer_candidate_or_strategy(self) -> str:
        return self.proposed_intervention

    @property
    def objective_relevance_rationale(self) -> str:
        return self.why_selective_for_pathogen_or_disease_state

    @property
    def feasibility_or_selectivity_rationale(self) -> str:
        return self.why_host_sparing

    @property
    def feasibility_or_selectivity_score(self) -> float:
        return self.host_safety_score

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["answer_candidate_or_strategy"] = self.answer_candidate_or_strategy
        data["objective_relevance_rationale"] = self.objective_relevance_rationale
        data["feasibility_or_selectivity_rationale"] = self.feasibility_or_selectivity_rationale
        data["feasibility_or_selectivity_score"] = self.feasibility_or_selectivity_score
        return data
