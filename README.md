# HypothesisForge

**Standalone multi-agent software for literature-grounded biological and biomedical hypothesis discovery.**

HypothesisForge turns a research objective into a checkpointed discovery workflow: it generates diverse research axes, builds literature-informed subtopics, retrieves and curates evidence, synthesizes findings, proposes testable hypotheses, critiques and evolves them, and ranks the resulting candidates.

## Current capabilities

### Scientific workflow
- supervisor-guided research configuration
- exactly 10 initial discovery axes
- query-family generation and Query Reviewer refinement
- literature-informed subtopic generation
- MCP-backed literature search
- optional PubTator annotation/filtering for PMID-backed records
- alias-aware paper deduplication and evidence selection
- axis-level and global synthesis
- hypothesis generation
- proximity clustering / merge / salvage
- reflection
- evolution, with optional focused literature retrieval
- final candidate ranking
- persisted evidence and compact paper-memory artifacts

### Runtime and UI
- FastAPI service with a browser UI
- explicit human checkpoints between all nine stages
- select / save / reject routing without deleting prior outputs
- persistent research-run archive
- automatic restoration of the last opened run after refresh or server restart
- reopening an archived run does not make new LLM calls
- provenance-preserving focus seeds
- per-stage token/cost accounting and logs
- a run-wide cumulative LLM-call budget
- current-year literature cutoff by default; historical cutoffs can be supplied explicitly

## 9-stage workflow

1. `axis_generation`
2. `subtopic_generation`
3. `literature_retrieval`
4. `synthesis`
5. `hypothesis_generation`
6. `proximity`
7. `reflection`
8. `evolution`
9. `candidate_ranking`

Checkpoint transitions are intentionally sequential. The browser UI lets a researcher choose which outputs continue at each stage, while unselected branches can be saved for later.

## Architecture

In normal mode, reusable scholarly search crosses a real MCP client/server boundary while scientific reasoning stays inside HypothesisForge:

```text
FastAPI / 9-stage orchestrator
        -> LiteratureAgent scientific logic
        -> MCP client
        -> HypothesisForge MCP server
        -> existing source adapters
        -> PubMed / Europe PMC / OpenAlex / Crossref / Semantic Scholar
```

The canonical FastAPI application uses the MCP protocol in-process, so a second MCP process is not required for normal local use. Query Review, deduplication, Evidence Selection, PubTator processing, synthesis, paper memory, Proximity, Reflection, Evolution, and ranking remain internal application logic rather than being exposed as artificial MCP wrappers.

Default FastAPI research runs use **PubMed, Europe PMC, OpenAlex, and Crossref**. Semantic Scholar is also exposed by the MCP server and can be enabled when configured.

In `dry_run` mode, the bundled mock LLM path is used and live network literature retrieval is skipped while preserving the complete checkpointed workflow.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# add GEMINI_API_KEY for live runs
uvicorn app.main:app --reload --port 8010
```

Open `http://127.0.0.1:8010`.

Useful optional environment variables are documented in `.env.example`, including NCBI, OpenAlex, Crossref, and Semantic Scholar settings. Set `HYPOTHESIS_FORGE_RUNS_DIR` to store persisted research runs somewhere other than `data/runs/`.

## Research archive and resume

Each question is stored as a separate run under `data/runs/` by default. The **Archive** tab lists prior questions and lets the user reopen any persisted run, including its completed stages, hypotheses, selections, usage, and artifacts, without repeating earlier LLM or literature work.

The browser also remembers the last opened run and restores it automatically after a page refresh or local server restart.

## MCP

The literature MCP server is independently launchable over stdio:

```bash
python -m mcp_server
```

Current MCP tools:

- `search_pubmed`
- `search_europepmc`
- `search_openalex`
- `search_crossref`
- `search_semantic_scholar`

Read-only MCP resources expose persisted research state without exposing internal reasoning agents:

- `hypothesisforge://runs`
- `hypothesisforge://runs/{run_id}`
- `hypothesisforge://runs/{run_id}/artifacts`
- `hypothesisforge://runs/{run_id}/artifacts/{filename}`
- `hypothesisforge://runs/{run_id}/paper-memory`

For local Streamable HTTP testing:

```bash
python -m mcp_server --transport streamable-http --host 127.0.0.1 --port 8000
```

The endpoint is `http://127.0.0.1:8000/mcp`. Unauthenticated HTTP is deliberately restricted to loopback. Public/non-loopback deployment requires an authentication and TLS/reverse-proxy boundary.

Automated tests exercise both a real stdio MCP subprocess and a real local Streamable HTTP server/client connection. See [`docs/MCP_ROADMAP.md`](docs/MCP_ROADMAP.md) for the completed MCP phases and the remaining production-deployment considerations.

## API

- `GET /health` — runtime health and literature-transport status
- `GET /runs` — compact archive of persisted research runs
- `POST /runs` — create a run and execute the fixed 10-axis initial checkpoint
- `GET /runs/{run_id}` — load persisted run state
- `POST /runs/{run_id}/stage` — advance selected outputs to the next checkpoint
- `POST /runs/{run_id}/selection` — select, save, or reject cards without deleting them
- `POST /runs/{run_id}/focus-seed` — persist a card as a focused generation seed
- `GET /runs/{run_id}/artifacts` — inspect persisted JSON artifacts

## Verification

```bash
pytest -q
```

The regression suite covers the FastAPI workflow, full nine-stage dry runs, MCP client/server contracts, real stdio and Streamable HTTP transports, retrieval behavior, persistence/archive behavior, provenance/lineage, security redaction, cumulative LLM budgets, and domain-general generation behavior.

GitHub Actions runs the suite on pushes and pull requests using Python 3.12.

## Data and security

HypothesisForge is currently **local-first research/portfolio software**, not an authenticated public SaaS service. Research runs can contain objectives, retrieved evidence, prompts, model outputs, and usage information.

`.env`, `data/runs/`, `data/cache/`, local virtual environments, and generated run directories are excluded from git. API credentials should never be committed or exposed in logs.

Do not expose the FastAPI or unauthenticated MCP service directly to an untrusted network. Outputs require scientific review and experimental validation; HypothesisForge is a research-assistance system, not a substitute for domain expertise.
