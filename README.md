# HypothesisForge

**Multi-Agent AI for Literature-Grounded Scientific Discovery**

## What is included

### v78 scientific core
- Supervisor-guided research configuration
- research-axis generation
- query-family generation + Query Reviewer
- PubMed, Europe PMC, OpenAlex, Crossref and Semantic Scholar adapters
- optional PubTator annotation/filtering
- Evidence Selector
- axis and global literature synthesis
- literature-grounded hypothesis generation
- Proximity clustering/merge/salvage
- Reflection
- Evolution, including optional focused literature retrieval
- persistent evidence/paper memory

### Runtime/UI layer
- FastAPI service
- persisted run state and JSON artifacts
- explicit stage checkpoints
- select / save / reject routing without deleting outputs
- focus-seed creation
- resumeable runs
- stage logs
- token/cost accounting from the v78 run logger
- standalone browser UI served by FastAPI
- final candidate-ranking checkpoint


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

Use `dry_run` in the UI to exercise the LLM pipeline with the bundled mock provider. Network literature retrieval is deliberately skipped in dry-run mode.

## API

- `POST /runs` — create a run and generate axes
- `GET /runs/{run_id}` — load persisted state
- `POST /runs/{run_id}/stage` — advance selected outputs to a later stage
- `POST /runs/{run_id}/selection` — select/save/reject cards
- `POST /runs/{run_id}/focus-seed` — retain a card as a focused Generation seed
- `GET /runs/{run_id}/artifacts` — inspect persisted artifacts

## MCP

The next architectural step is intentionally narrow: expose the literature adapters as a real MCP server, then make `LiteratureAgent` consume those tools as an MCP client. See [`docs/MCP_ROADMAP.md`](docs/MCP_ROADMAP.md).

## Status

Research/portfolio software. Outputs require scientific review; the system does not replace domain expertise or experimental validation.
