from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from mcp.client import Client

from mcp_server.server import create_server
from schemas.paper_record import PaperRecord
from utils.run_logger import log_event

_TOOL_BY_SOURCE = {
    "PubMed": "search_pubmed",
    "EuropePMC": "search_europepmc",
    "OpenAlex": "search_openalex",
    "Crossref": "search_crossref",
    "SemanticScholar": "search_semantic_scholar",
}
_CUTOFF_AWARE_SOURCES = {"PubMed", "EuropePMC", "OpenAlex"}


class LiteratureMCPClient:
    """Synchronous facade over the real MCP literature tool surface.

    The scientific pipeline is synchronous, while the MCP SDK client is async.
    Each search therefore opens a short-lived MCP protocol session against the
    in-process literature MCP server. The same client contract can later be
    pointed at a stdio or Streamable HTTP transport without changing the agents.
    """

    def __init__(self, *, server: Any | None = None, config_path: str = "configs/config.yaml") -> None:
        self.server = server if server is not None else create_server(config_path=config_path)

    def search(
        self,
        source: str,
        query: str,
        *,
        limit: int = 10,
        cutoff_year: int | None = None,
        year: str | None = None,
        offset: int | None = None,
    ) -> list[PaperRecord]:
        tool = _TOOL_BY_SOURCE.get(source)
        if not tool:
            raise ValueError(f"Unsupported MCP literature source: {source}")

        arguments: dict[str, Any] = {"query": query, "limit": int(limit)}
        if cutoff_year is not None and source in _CUTOFF_AWARE_SOURCES:
            arguments["cutoff_year"] = int(cutoff_year)
        if source == "SemanticScholar":
            if year is not None:
                arguments["year"] = year
            if offset is not None:
                arguments["offset"] = int(offset)

        try:
            result = _run_sync(self._call_tool(tool, arguments))
            records = _records_from_result(result)
            log_event(
                "mcp",
                "literature_tool_call",
                {"source": source, "tool": tool, "query": query, "limit": int(limit), "records": len(records)},
            )
            return records
        except Exception as exc:
            log_event(
                "mcp",
                "literature_tool_call_failed",
                {"source": source, "tool": tool, "query": query, "error_type": type(exc).__name__},
                status="error",
            )
            raise

    async def _call_tool(self, tool: str, arguments: dict[str, Any]):
        async with Client(self.server) as client:
            return await client.call_tool(tool, arguments)


class MCPSourceAdapter:
    """Adapter-shaped MCP facade so existing LiteratureAgent logic stays unchanged."""

    def __init__(self, source: str, client: LiteratureMCPClient) -> None:
        if source not in _TOOL_BY_SOURCE:
            raise ValueError(f"Unsupported MCP literature source: {source}")
        self.source_name = source
        self.client = client

    def search(
        self,
        query: str,
        limit: int = 10,
        cutoff_year: int | None = None,
        year: str | None = None,
        offset: int | None = None,
    ) -> list[PaperRecord]:
        return self.client.search(
            self.source_name,
            query,
            limit=limit,
            cutoff_year=cutoff_year,
            year=year,
            offset=offset,
        )


def build_mcp_source_adapters(
    config_path: str = "configs/config.yaml",
    *,
    server: Any | None = None,
) -> dict[str, MCPSourceAdapter]:
    client = LiteratureMCPClient(server=server, config_path=config_path)
    return {source: MCPSourceAdapter(source, client) for source in _TOOL_BY_SOURCE}


def _records_from_result(result: Any) -> list[PaperRecord]:
    if bool(getattr(result, "is_error", False)):
        raise RuntimeError("Literature MCP tool returned an error")
    structured = getattr(result, "structured_content", None)
    if not isinstance(structured, dict):
        raise RuntimeError("Literature MCP tool returned no structured content")
    payload = structured.get("result")
    if not isinstance(payload, list):
        raise RuntimeError("Literature MCP tool returned an invalid result payload")

    records: list[PaperRecord] = []
    for item in payload:
        if not isinstance(item, dict):
            raise RuntimeError("Literature MCP tool returned a non-object paper record")
        records.append(PaperRecord(**item))
    return records


def _run_sync(awaitable):
    """Run one MCP coroutine from ordinary sync code, including inside an active event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    # A running loop (for example Jupyter/ASGI async caller code) cannot nest
    # asyncio.run(). Execute this isolated MCP call in a worker thread instead.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="hypothesisforge-mcp") as pool:
        return pool.submit(asyncio.run, awaitable).result()
