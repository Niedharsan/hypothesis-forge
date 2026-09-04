# HypothesisForge

**Standalone multi-agent software for literature-grounded scientific hypothesis discovery.**

## What is included

### Scientific workflow
- supervisor-guided research configuration
- axis generation
- subtopic generation with query-family review
- direct Python literature adapters for PubMed, Europe PMC, OpenAlex, Crossref, and Semantic Scholar
- optional PubTator annotation/filtering
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
- focus-seed creation
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

In `dry_run` mode, the LLM path uses the bundled mock client and the `literature_retrieval` stage skips live network retrieval while still preserving the full checkpointed workflow.

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

## API

- `POST /runs` — create a run and execute `axis_generation`
- `GET /runs/{run_id}` — load persisted run state
- `POST /runs/{run_id}/stage` — advance selected outputs to a later stage
- `POST /runs/{run_id}/selection` — select, save, or reject cards without deleting them
- `POST /runs/{run_id}/focus-seed` — persist a card as a focused generation seed
- `GET /runs/{run_id}/artifacts` — inspect persisted JSON artifacts

## MCP roadmap

`MultiSourceLiteratureAgent` currently calls the repository's Python literature adapters directly. The standalone Phase 1 adapter server can be launched with `python -m mcp_server`, but the main HypothesisForge workflow still uses the direct Python retrieval path until Phase 2. See [`docs/MCP_ROADMAP.md`](docs/MCP_ROADMAP.md).

## Status

Research/portfolio software. Outputs require scientific review; the system does not replace domain expertise or experimental validation.
