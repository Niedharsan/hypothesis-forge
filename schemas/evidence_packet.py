from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EvidencePacket:
    evidence_id: str
    paper_id: str
    title: str
    source: str
    text: str
    evidence_type: str = "abstract_metadata"
    relevance_score: float = 0.0
    supports_topics: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source_api(self) -> str:
        return self.source

    @property
    def section_type(self) -> str:
        return self.evidence_type

    @property
    def citation_count(self) -> int | None:
        value = self.metadata.get("citation_count")
        return value if isinstance(value, int) else None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
