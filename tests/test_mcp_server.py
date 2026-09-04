from __future__ import annotations

import asyncio
from dataclasses import dataclass

from mcp.client import Client
import mcp_server.server as mcp_server_module
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


def test_mcp_server_lists_expected_tools():
    server = create_server(adapters=_stub_adapters())

    async def run():
        async with Client(server) as client:
            response = await client.list_tools()
            return sorted(tool.name for tool in response.tools)

    assert asyncio.run(run()) == [
        "search_crossref",
        "search_europepmc",
        "search_openalex",
        "search_pubmed",
        "search_semantic_scholar",
    ]


def test_create_server_builds_default_adapters(monkeypatch):
    stub_adapters = _stub_adapters()
    monkeypatch.setattr(mcp_server_module, "build_adapters", lambda config_path: stub_adapters)

    server = create_server()

    result = _call_tool(server, "search_crossref", {"query": "stress", "limit": 1})

    assert stub_adapters.crossref.calls == [{"query": "stress", "limit": 1}]
    assert _first_record(result).paper_id == "pmid:123"


def test_search_pubmed_tool_returns_normalized_paper_records():
    adapters = _stub_adapters()
    server = create_server(adapters=adapters)

    result = _call_tool(
        server,
        "search_pubmed",
        {"query": "stress response", "limit": 3, "cutoff_year": 2021},
    )

    assert adapters.pubmed.calls == [{"query": "stress response", "limit": 3, "cutoff_year": 2021}]
    record = PaperRecord(**result.structured_content["result"][0])
    assert record.paper_id == "pmid:123"
    assert record.source_apis == ["PubMed"]


def test_search_europepmc_tool_passes_cutoff_year():
    adapters = _stub_adapters()
    server = create_server(adapters=adapters)

    result = _call_tool(
        server,
        "search_europepmc",
        {"query": "proteostasis", "limit": 4, "cutoff_year": 2020},
    )

    assert adapters.europepmc.calls == [{"query": "proteostasis", "limit": 4, "cutoff_year": 2020}]
    assert _first_record(result).paper_id == "pmid:123"


def test_search_openalex_tool_uses_existing_adapter():
    adapters = _stub_adapters()
    server = create_server(adapters=adapters)

    result = _call_tool(server, "search_openalex", {"query": "ER stress", "limit": 2, "cutoff_year": 2022})

    assert adapters.openalex.calls == [{"query": "ER stress", "limit": 2, "cutoff_year": 2022}]
    assert _first_record(result).paper_id == "pmid:123"


def test_search_crossref_tool_uses_existing_adapter():
    adapters = _stub_adapters()
    server = create_server(adapters=adapters)

    result = _call_tool(server, "search_crossref", {"query": "autophagy", "limit": 5})

    assert adapters.crossref.calls == [{"query": "autophagy", "limit": 5}]
    assert _first_record(result).paper_id == "pmid:123"


def test_search_semantic_scholar_tool_passes_optional_filters():
    adapters = _stub_adapters()
    server = create_server(adapters=adapters)

    result = _call_tool(
        server,
        "search_semantic_scholar",
        {"query": "translation control", "limit": 6, "year": "2021-", "offset": 3},
    )

    assert adapters.semantic_scholar.calls == [
        {"query": "translation control", "limit": 6, "year": "2021-", "offset": 3}
    ]
    assert _first_record(result).paper_id == "pmid:123"


def test_mcp_server_supports_multiple_tool_calls_in_one_session():
    adapters = _stub_adapters()
    server = create_server(adapters=adapters)

    async def run():
        async with Client(server) as client:
            first = await client.call_tool("search_pubmed", {"query": "stress", "limit": 1})
            second = await client.call_tool("search_crossref", {"query": "stress", "limit": 1})
            return first, second

    first, second = asyncio.run(run())

    assert first.structured_content["result"][0]["paper_id"] == "pmid:123"
    assert second.structured_content["result"][0]["paper_id"] == "pmid:123"
    assert adapters.pubmed.calls == [{"query": "stress", "limit": 1, "cutoff_year": None}]
    assert adapters.crossref.calls == [{"query": "stress", "limit": 1}]


def test_main_starts_stdio_server(monkeypatch):
    calls: list[str] = []

    class FakeServer:
        def run(self, transport: str):
            calls.append(transport)

    monkeypatch.setattr(mcp_server_module, "create_server", lambda: FakeServer())

    mcp_server_module.main([])

    assert calls == ["stdio"]


def _call_tool(server, name: str, arguments: dict):
    async def run():
        async with Client(server) as client:
            return await client.call_tool(name, arguments)

    return asyncio.run(run())


def _first_record(result) -> PaperRecord:
    return PaperRecord(**result.structured_content["result"][0])


def _stub_adapters() -> StubAdapters:
    sample_record = PaperRecord(
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
        pubmed=StubAdapter([sample_record]),
        europepmc=StubAdapter([sample_record]),
        openalex=StubAdapter([sample_record]),
        crossref=StubAdapter([sample_record]),
        semantic_scholar=StubAdapter([sample_record]),
    )
