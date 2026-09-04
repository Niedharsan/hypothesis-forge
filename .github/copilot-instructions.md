# HypothesisForge Copilot Instructions

## Project structure
- `app/`: FastAPI entrypoints, request models, orchestration, storage, provenance-safe focus seeds, and static UI assets.
- `agents/`: stage-specific agents for supervisor setup, literature work, generation, proximity, reflection, and evolution.
- `mcp_client/`: synchronous facade used by the application to call the literature MCP tool surface.
- `mcp_server/`: MCP server exposing the existing literature adapters as typed tools; independently launchable with `python -m mcp_server`.
- `retrieval/`: direct source adapters used behind the MCP server plus PubTator and shared API helpers.
- `llm/`: provider abstraction, Gemini client, and dry-run mock client.
- `runtime/`: runtime mode and per-run limits.
- `schemas/`: typed payload and record models shared across stages.
- `utils/`: config loading, prompt rendering, run logging, evidence-memory helpers, and credential redaction.
- `prompts/v31/`: prompt templates used by the current stage agents.
- `tests/`: API, MCP server/client, retrieval, security, and full dry-run workflow regressions.

## Architecture
- FastAPI orchestrates a fixed 9-stage workflow and persists checkpointed run state plus JSON artifacts under `data/runs/` unless `HYPOTHESIS_FORGE_RUNS_DIR` overrides it.
- The workflow stages are `axis_generation`, `subtopic_generation`, `literature_retrieval`, `synthesis`, `hypothesis_generation`, `proximity`, `reflection`, `evolution`, and `candidate_ranking`.
- The canonical FastAPI runtime uses `MCPLiteratureAgent`, which inherits the established `LiteratureAgent` scientific behavior and changes only the external search transport to MCP-backed adapter facades.
- `MultiSourceLiteratureAgent` also uses the shared MCP source-adapter facade for its route-specific search path.
- The MCP server must continue to reuse the existing PubMed, Europe PMC, OpenAlex, Crossref, and Semantic Scholar adapters. Do not reimplement those clients inside MCP code.
- Keep Query Reviewer, Evidence Selector, deduplication, filtering, PubTator annotation, synthesis, paper memory, Proximity, Reflection, Evolution, and ranking inside HypothesisForge; they are not MCP tools.
- PubTator remains direct/local unless there is a concrete interoperability reason to expose it.
- `dry_run` mode must continue to use the bundled mock LLM path and skip live network retrieval while preserving the stage flow and persisted checkpoints.
- Preserve the current v78-derived orchestration model; make targeted fixes instead of redesigning stage boundaries or agent responsibilities.
- Card IDs are not globally unique across stages. UI/API actions that refer to an existing card must preserve stage provenance.
- Never return or log API keys. Sanitize upstream errors and keep `.env`, `data/runs/`, and `data/cache/` out of git.

## Test and verification commands
- `python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt`
- `. .venv/bin/activate && pytest -q`
- For manual end-to-end verification, exercise `POST /runs` and repeated `POST /runs/{run_id}/stage` calls in `dry_run` mode until `candidate_ranking`.
- For external MCP testing, start the stdio server with `python -m mcp_server`; the canonical FastAPI app itself uses the in-process MCP protocol transport.
