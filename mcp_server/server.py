from __future__ import annotations

from typing import Annotated

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from mcp_server.adapters import LiteratureAdapters, build_adapters
from mcp_server.models import PaperRecordPayload

QueryArg = Annotated[str, Field(min_length=1, max_length=2_000, description="Literature search query.")]
LimitArg = Annotated[int, Field(ge=1, le=50, description="Maximum number of normalized paper records to return.")]
CutoffYearArg = Annotated[int | None, Field(default=None, ge=1800, description="Optional inclusive publication-year cutoff.")]
SemanticScholarYearArg = Annotated[str | None, Field(default=None, max_length=32, description="Optional Semantic Scholar year filter expression.")]
OffsetArg = Annotated[int | None, Field(default=None, ge=0, description="Optional Semantic Scholar result offset.")]


def create_server(
    *,
    adapters: LiteratureAdapters | None = None,
    config_path: str = "configs/config.yaml",
) -> MCPServer:
    adapter_set = adapters if adapters is not None else build_adapters(config_path)
    mcp = MCPServer(
        "hypothesisforge-literature",
        title="HypothesisForge Literature MCP Server",
        description="Phase 1 adapter-only MCP server exposing the existing literature search adapters.",
        instructions=(
            "This server exposes the repository's existing literature adapters as MCP tools. "
            "It does not change the main HypothesisForge workflow, which still uses the direct Python retrieval path."
        ),
    )

    @mcp.tool(description="Search PubMed and return normalized PaperRecord objects.")
    def search_pubmed(
        query: QueryArg,
        limit: LimitArg = 10,
        cutoff_year: CutoffYearArg = None,
    ) -> list[PaperRecordPayload]:
        return _serialize(adapter_set.pubmed.search(query, limit=limit, cutoff_year=cutoff_year))

    @mcp.tool(description="Search Europe PMC and return normalized PaperRecord objects.")
    def search_europepmc(
        query: QueryArg,
        limit: LimitArg = 10,
        cutoff_year: CutoffYearArg = None,
    ) -> list[PaperRecordPayload]:
        return _serialize(adapter_set.europepmc.search(query, limit=limit, cutoff_year=cutoff_year))

    @mcp.tool(description="Search OpenAlex and return normalized PaperRecord objects.")
    def search_openalex(
        query: QueryArg,
        limit: LimitArg = 10,
        cutoff_year: CutoffYearArg = None,
    ) -> list[PaperRecordPayload]:
        return _serialize(adapter_set.openalex.search(query, limit=limit, cutoff_year=cutoff_year))

    @mcp.tool(description="Search Crossref and return normalized PaperRecord objects.")
    def search_crossref(query: QueryArg, limit: LimitArg = 10) -> list[PaperRecordPayload]:
        return _serialize(adapter_set.crossref.search(query, limit=limit))

    @mcp.tool(description="Search Semantic Scholar and return normalized PaperRecord objects.")
    def search_semantic_scholar(
        query: QueryArg,
        limit: LimitArg = 10,
        year: SemanticScholarYearArg = None,
        offset: OffsetArg = None,
    ) -> list[PaperRecordPayload]:
        return _serialize(adapter_set.semantic_scholar.search(query, limit=limit, year=year, offset=offset))

    return mcp


def _serialize(records: list) -> list[PaperRecordPayload]:
    return [PaperRecordPayload.from_record(record) for record in records]


def main() -> None:
    create_server().run("stdio")
