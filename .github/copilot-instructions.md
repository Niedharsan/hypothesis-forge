# HypothesisForge Copilot Instructions

## Project structure
- `app/`: FastAPI entrypoints, request models, orchestration, storage, and static UI assets.
- `agents/`: stage-specific agents for supervisor setup, literature work, generation, proximity, reflection, and evolution.
- `mcp_server/`: Phase 1 MCP server package that exposes the existing literature adapters as typed MCP tools.
- `retrieval/`: direct Python adapters for external literature sources plus shared API client helpers.
- `llm/`: provider abstraction, Gemini client, and dry-run mock client.
- `runtime/`: runtime mode and per-run limits.
- `schemas/`: typed payload and record models shared across stages.
- `utils/`: config loading, prompt rendering, run logging, and evidence-memory helpers.
- `prompts/v31/`: prompt templates used by the current stage agents.
- `tests/`: API-level regression coverage for health, persistence, selection handling, and full dry-run stage traversal.

## Architecture
- Keep the existing standalone architecture: FastAPI orchestrates a fixed 9-stage workflow and persists checkpointed run state plus JSON artifacts under `data/runs/` unless `HYPOTHESIS_FORGE_RUNS_DIR` overrides it.
- The workflow stages are `axis_generation`, `subtopic_generation`, `literature_retrieval`, `synthesis`, `hypothesis_generation`, `proximity`, `reflection`, `evolution`, and `candidate_ranking`.
- `MultiSourceLiteratureAgent` is a direct Python wrapper around the repository's literature adapters. Do not describe it as MCP-based unless the implementation actually changes.
- The Phase 1 MCP server is adapter-only. Do not switch `LiteratureAgent` or the main workflow to MCP unless that integration work is explicitly requested.
- `dry_run` mode should continue to use the bundled mock LLM path and skip live network retrieval while preserving the stage flow and persisted checkpoints.
- Preserve the current v78-derived orchestration model; make targeted fixes instead of redesigning stage boundaries or agent responsibilities.

## Test and verification commands
- `python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt`
- `. .venv/bin/activate && pytest -q`
- For manual end-to-end verification, exercise `POST /runs` and repeated `POST /runs/{run_id}/stage` calls in `dry_run` mode until `candidate_ranking`.
