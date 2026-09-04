from __future__ import annotations

from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.models import FocusSeedRequest, SelectionRequest, StageRequest, StartRunRequest
from app.orchestrator import STAGES, create_focus_seed, run_stage, start_run, update_selection
from app.storage import RUNS_DIR, list_artifacts, read_run, write_run

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)

app = FastAPI(
    title="HypothesisForge API",
    version="0.1.0",
    description="Multi-agent, literature-grounded scientific hypothesis discovery with human checkpoints.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8010", "http://127.0.0.1:8010"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
STATIC = ROOT / "app" / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "hypothesis-forge", "stages": STAGES, "runs_dir": str(RUNS_DIR)}


@app.post("/runs")
def create_run(payload: StartRunRequest) -> dict[str, Any]:
    try:
        return start_run(payload)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    try:
        return read_run(run_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc


@app.post("/runs/{run_id}/stage")
def advance_stage(run_id: str, payload: StageRequest) -> dict[str, Any]:
    try:
        run = read_run(run_id)
        return run_stage(run, payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/runs/{run_id}/selection")
def selection(run_id: str, payload: SelectionRequest) -> dict[str, Any]:
    try:
        run = update_selection(read_run(run_id), payload)
        write_run(run)
        return run
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Card not found: {exc}") from exc


@app.post("/runs/{run_id}/focus-seed")
def focus_seed(run_id: str, payload: FocusSeedRequest) -> dict[str, Any]:
    try:
        return create_focus_seed(read_run(run_id), payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Card not found: {exc}") from exc


@app.get("/runs/{run_id}/artifacts")
def artifacts(run_id: str) -> list[dict[str, Any]]:
    try:
        read_run(run_id)
        return list_artifacts(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc
