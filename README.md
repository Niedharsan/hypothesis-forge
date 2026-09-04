# HypothesisForge

**Standalone multi-agent software for literature-grounded scientific hypothesis discovery.**

## What is included

### Scientific workflow
- supervisor-guided research configuration
- axis generation
- subtopic generation with query-family review
- MCP-based search across PubMed, Europe PMC, OpenAlex, Crossref, and Semantic Scholar
- optional direct PubTator annotation/filtering
- evidence selection
- axis and global literature synthesis
- hypothesis generation
- proximity clustering/merge/salvage
- reflection
- evolution, including optional focused literature retrieval
- final candidate ranking
- persisted evidence and paper-memory artifacts

### Runtime and UI
- FastAPI service with a browser UI
- persisted run state under `data/runs/` by default
- explicit stage checkpoints with select / save / reject routing
- provenance-preserving focus-seed persistence
- resumable runs
- per-stage logs plus token/cost accounting

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

In normal mode, the canonical FastAPI workflow routes reusable literature search through an MCP client and the HypothesisForge literature MCP server. The MCP tools reuse the existing Python source adapters; query review, deduplication, evidence selection, filtering, synthesis, paper memory, reflection, and evolution remain internal scientific logic. PubTator remains a direct annotation layer.

In `dry_run` mode, the LLM path uses the bundled mock client and live network literature retrieval is skipped while preserving the full checkpointed workflow.

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

To verify the install:

```bash
pytest -q
```

Set `HYPOTHESIS_FORGE_RUNS_DIR` if you want run data somewhere other than `data/runs/`.

## MCP

The canonical FastAPI workflow uses an in-process MCP client/server transport, so no second process is required when running the application normally:

```text
LiteratureAgent scientific logic
        -> MCP client
        -> HypothesisForge literature MCP server
        -> existing source adapters
        -> PubMed / Europe PMC / OpenAlex / Crossref / Semantic Scholar
```

The same literature server is independently launchable over stdio for external MCP clients:

```bash
python -m mcp_server
```

Current MCP tools:

- `search_pubmed`
- `search_europepmc`
- `search_openalex`
- `search_crossref`
- `search_semantic_scholar`

See [`docs/MCP_ROADMAP.md`](docs/MCP_ROADMAP.md) for the remaining resource/deployment work.

## API

- `POST /runs` — create a run and execute `axis_generation`
- `GET /runs/{run_id}` — load persisted run state
- `POST /runs/{run_id}/stage` — advance selected outputs to a later stage
- `POST /runs/{run_id}/selection` — select, save, or reject cards without deleting them
- `POST /runs/{run_id}/focus-seed` — persist a card as a focused generation seed; `source_stage` disambiguates repeated card IDs
- `GET /runs/{run_id}/artifacts` — inspect persisted JSON artifacts

## Deployment and data notes

HypothesisForge is local-first research/portfolio software. Run artifacts can contain research objectives, retrieved evidence, prompts, and model outputs. `data/runs/`, `data/cache/`, `.env`, and local virtual environments are excluded from git. Do not expose the FastAPI service directly to an untrusted network without adding an authentication/reverse-proxy boundary.

Outputs require scientific review; the system does not replace domain expertise or experimental validation.
