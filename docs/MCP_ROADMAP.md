# MCP conversion roadmap

HypothesisForge is currently a standalone multi-agent application with ordinary Python literature adapters.

## Phase 1 — real literature MCP server
Expose only reusable external capabilities as MCP tools:
- `search_pubmed`
- `search_europepmc`
- `search_openalex`
- `search_crossref`
- `search_semantic_scholar`
- `fetch_paper_metadata`
- optional `annotate_pubtator`

Keep the existing retrieval adapters as the implementation behind those tool contracts.

## Phase 2 — make LiteratureAgent an MCP client
Replace direct adapter calls inside LiteratureAgent with an MCP client. Preserve the existing query-review, evidence-selection, deduplication, filtering, synthesis, and paper-memory logic unchanged.

## Phase 3 — resources + deployment
Expose persisted run artifacts/paper memory as read-only MCP resources, support stdio for local/Codex-style use and Streamable HTTP for remote clients, then add MCP Inspector/integration tests.

Do not expose internal reflection/evolution steps as MCP tools unless another application genuinely needs them. MCP is the reusable capability boundary, not a wrapper around every function.
