from __future__ import annotations

import asyncio
from dataclasses import dataclass, fields

from mcp_client.literature import LiteratureMCPClient, build_mcp_source_adapters
from mcp_server.models import PaperRecordPayload
from mcp_server.server import create_server
from schemas.paper_record import PaperRecord


class StubAdapter:
    def __init__(self, records: list[PaperRecord]):
        self.records = records
        self.calls: list[dict] = []

    def search(self, query: str, **kwargs) -> list[PaperRecord]:
        self.calls.append({"query": query, **kwargs})
        return list(self.records)


@dataclass
class StubAdapters:
    pubmed: StubAdapter
    europepmc: StubAdapter
    openalex: StubAdapter
    crossref: StubAdapter
    semantic_scholar: StubAdapter


def test_mcp_client_round_trips_paper_records_and_cutoff():
    adapters = _stub_adapters()
    client = LiteratureMCPClient(server=create_server(adapters=adapters))

    records = client.search("PubMed", "stress response", limit=3, cutoff_year=2022)

    assert adapters.pubmed.calls == [{"query": "stress response", "limit": 3, "cutoff_year": 2022}]
    assert len(records) == 1
    assert records[0].paper_id == "pmid:123"
    assert records[0].source_apis == ["PubMed"]


def test_mcp_source_adapters_preserve_existing_search_shape():
    direct = _stub_adapters()
    server = create_server(adapters=direct)
    adapters = build_mcp_source_adapters(server=server)

    records = adapters["OpenAlex"].search("ER stress", limit=2, cutoff_year=2020)
    adapters["Crossref"].search("autophagy", limit=4, cutoff_year=2020)

    assert records[0].paper_id == "pmid:123"
    assert direct.openalex.calls == [{"query": "ER stress", "limit": 2, "cutoff_year": 2020}]
    # Crossref has no server-side cutoff tool argument; LiteratureAgent retains
    # its existing post-retrieval cutoff filtering for this source.
    assert direct.crossref.calls == [{"query": "autophagy", "limit": 4}]


def test_mcp_client_can_be_called_from_an_existing_event_loop():
    adapters = _stub_adapters()
    client = LiteratureMCPClient(server=create_server(adapters=adapters))

    async def inside_loop():
        return client.search("EuropePMC", "proteostasis", limit=1, cutoff_year=2021)

    records = asyncio.run(inside_loop())
    assert records[0].paper_id == "pmid:123"
    assert adapters.europepmc.calls == [{"query": "proteostasis", "limit": 1, "cutoff_year": 2021}]


def test_mcp_payload_schema_stays_in_lockstep_with_paper_record():
    dataclass_fields = {field.name for field in fields(PaperRecord)}
    payload_fields = set(PaperRecordPayload.model_fields)
    assert payload_fields == dataclass_fields


def _stub_adapters() -> StubAdapters:
    record = PaperRecord(
        paper_id="pmid:123",
        title="Stress response signaling",
        abstract="Synthetic abstract",
        authors=["Ada Lovelace"],
        year=2021,
        journal="Journal of Testing",
        doi="10.1000/test",
        pmid="123",
        url="https://example.org/paper",
        source_apis=["PubMed"],
        citation_count=12,
        raw={"stub": True},
    )
    return StubAdapters(
        pubmed=StubAdapter([record]),
        europepmc=StubAdapter([record]),
        openalex=StubAdapter([record]),
        crossref=StubAdapter([record]),
        semantic_scholar=StubAdapter([record]),
    )
