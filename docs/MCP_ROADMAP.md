# MCP integration roadmap

HypothesisForge now uses MCP as the reusable literature-search capability boundary while keeping scientific orchestration inside the application.

## Phase 1 — literature MCP server — complete

The reusable source adapters are exposed as typed MCP tools:

- `search_pubmed`
- `search_europepmc`
- `search_openalex`
- `search_crossref`
- `search_semantic_scholar`

The MCP server reuses the existing retrieval adapters rather than reimplementing source clients. It can be launched independently over stdio with:

```bash
python -m mcp_server
```

PubTator remains a direct annotation/enrichment layer because it is internal evidence processing rather than the primary literature-search capability boundary.

## Phase 2 — canonical LiteratureAgent path uses MCP — complete

The canonical FastAPI workflow now composes the existing `LiteratureAgent` scientific logic with MCP-backed source adapters:

```text
LiteratureAgent planning / filtering / evidence selection / synthesis
        -> MCP client
        -> HypothesisForge literature MCP server
        -> existing source adapters
        -> literature APIs
```

The application uses an in-process MCP protocol transport for normal local execution. This avoids requiring a second subprocess for every search while still exercising the actual MCP client/tool contract. The separately launchable stdio server remains available for external MCP clients.

`MultiSourceLiteratureAgent`, used by the route-specific generation helper, also uses the shared MCP source-adapter facade. Query Reviewer, Evidence Selector, deduplication, PubTator filtering, synthesis, paper memory, Proximity, Reflection, Evolution, and ranking remain application logic and are not MCP tools.

## Phase 3 — resources and remote deployment — pending

- expose useful persisted run artifacts and compact paper memory as read-only MCP resources
- add an explicit Streamable HTTP server mode for remote clients
- add authorization before exposing a remote MCP endpoint outside localhost/trusted environments
- add MCP Inspector/transport-level integration coverage for stdio and Streamable HTTP
- document external-client examples and deployment boundaries

Do not expose internal reflection/evolution/ranking steps as MCP tools unless another application genuinely needs them. MCP is the reusable capability boundary, not a wrapper around every internal function.
