from __future__ import annotations

import argparse
import ipaddress
from typing import Annotated, Any

from mcp.server import MCPServer
from pydantic import Field

from mcp_server.adapters import LiteratureAdapters, build_adapters
from mcp_server.models import PaperRecordPayload
from mcp_server.resources import (
    list_run_catalog,
    read_artifact_catalog,
    read_compact_paper_memory,
    read_run_artifact,
    read_run_summary,
)

QueryArg = Annotated[str, Field(min_length=1, max_length=2_000, description="Literature search query.")]
LimitArg = Annotated[int, Field(ge=1, le=50, description="Maximum number of normalized paper records to return.")]
CutoffYearArg = Annotated[int | None, Field(default=None, description="Optional inclusive publication-year cutoff.")]
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
        title="HypothesisForge MCP Server",
        description=(
            "Reusable MCP boundary for normalized literature search plus read-only persisted "
            "HypothesisForge run artifacts."
        ),
        instructions=(
            "Use the literature tools for normalized scholarly search. Use resources to inspect persisted "
            "run summaries, artifacts, and compact paper memory. HypothesisForge keeps query review, "
            "evidence selection, deduplication, synthesis, PubTator processing, reflection, evolution, "
            "and ranking inside the application rather than exposing them as MCP tools."
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

    @mcp.resource(
        "hypothesisforge://runs",
        title="Persisted HypothesisForge runs",
        description="Compact catalog of persisted HypothesisForge runs, newest first.",
        mime_type="application/json",
    )
    def persisted_runs() -> dict[str, Any]:
        return list_run_catalog()

    @mcp.resource(
        "hypothesisforge://runs/{run_id}",
        title="HypothesisForge run summary",
        description="Compact persisted run state with stage counts, usage, and artifact metadata.",
        mime_type="application/json",
    )
    def persisted_run_summary(run_id: str) -> dict[str, Any]:
        return read_run_summary(run_id)

    @mcp.resource(
        "hypothesisforge://runs/{run_id}/artifacts",
        title="HypothesisForge run artifact catalog",
        description="Metadata for JSON artifacts persisted by a HypothesisForge run.",
        mime_type="application/json",
    )
    def persisted_artifact_catalog(run_id: str) -> dict[str, Any]:
        return read_artifact_catalog(run_id)

    @mcp.resource(
        "hypothesisforge://runs/{run_id}/artifacts/{filename}",
        title="HypothesisForge persisted artifact",
        description="One declared read-only JSON artifact from a persisted HypothesisForge run.",
        mime_type="application/json",
    )
    def persisted_artifact(run_id: str, filename: str) -> Any:
        return read_run_artifact(run_id, filename)

    @mcp.resource(
        "hypothesisforge://runs/{run_id}/paper-memory",
        title="HypothesisForge compact paper memory",
        description="Compact paper-memory artifact persisted for downstream scientific reasoning.",
        mime_type="application/json",
    )
    def persisted_paper_memory(run_id: str) -> dict[str, Any]:
        return read_compact_paper_memory(run_id)

    return mcp


def _serialize(records: list) -> list[PaperRecordPayload]:
    return [PaperRecordPayload.from_record(record) for record in records]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the HypothesisForge MCP server.")
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default="stdio",
        help="MCP transport. Defaults to stdio for local desktop/IDE clients.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Streamable HTTP bind host. Unauthenticated Phase 3 mode is restricted to loopback hosts.",
    )
    parser.add_argument("--port", type=int, default=8000, help="Streamable HTTP port.")
    args = parser.parse_args(argv)

    server = create_server()
    if args.transport == "stdio":
        server.run("stdio")
        return

    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if not _is_loopback_host(args.host):
        parser.error(
            "Unauthenticated Streamable HTTP is restricted to loopback in Phase 3. "
            "Add MCP authorization before binding a public/non-loopback interface."
        )

    server.run(
        "streamable-http",
        host=args.host,
        port=args.port,
        stateless_http=True,
        json_response=True,
    )


def _is_loopback_host(host: str) -> bool:
    if host.strip().lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip()).is_loopback
    except ValueError:
        return False
