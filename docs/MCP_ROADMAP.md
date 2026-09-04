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

## Phase 3 — resources and Streamable HTTP — complete

Read-only MCP resources expose persisted run data without turning internal reasoning stages into tools:

- `hypothesisforge://runs` — compact persisted-run catalog
- `hypothesisforge://runs/{run_id}` — compact run summary with stage counts, usage, and artifact metadata
- `hypothesisforge://runs/{run_id}/artifacts` — artifact catalog
- `hypothesisforge://runs/{run_id}/artifacts/{filename}` — one declared persisted JSON artifact
- `hypothesisforge://runs/{run_id}/paper-memory` — persisted compact paper memory

The resource layer reads the existing run store; it does not create a second persistence model. Individual artifact reads are restricted to filenames already declared in the run's artifact provenance, and resource access is read-only.

Streamable HTTP is available with:

```bash
python -m mcp_server --transport streamable-http --host 127.0.0.1 --port 8000
```

The endpoint is `http://127.0.0.1:8000/mcp`. Phase 3 deliberately permits only loopback HTTP binds because this standalone mode has no authentication. Do not bind it to a public/non-loopback interface until MCP authorization is configured. A trusted reverse proxy or tunnel may terminate authenticated remote access while the MCP process remains loopback-bound.

Transport-level tests exercise the real stdio subprocess and a real Streamable HTTP server/client connection, including resource discovery and reads. MCP Inspector can also be pointed at the stdio server or the local HTTP endpoint for manual inspection.

Query Reviewer, Evidence Selector, deduplication, PubTator, synthesis, paper-memory construction, Proximity, Reflection, Evolution, and ranking remain internal scientific application logic.

## Future deployment work

- add MCP authorization when a public/non-loopback deployment is actually required
- add production reverse-proxy/TLS configuration as deployment concerns
- keep internal reasoning stages private unless another application demonstrates a real reusable capability need

MCP is the reusable capability boundary, not a wrapper around every internal function.
