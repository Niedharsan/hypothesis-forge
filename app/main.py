from __future__ import annotations

from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import app.orchestrator as orchestrator
from agents.mcp_literature_agent import MCPLiteratureAgent
from app.focus_seeds import create_focus_seed
from app.models import FocusSeedRequest, SelectionRequest, StageRequest, StartRunRequest
from app.storage import RUNS_DIR, list_artifacts, read_run
from utils.security import redact_sensitive_text

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)

# Composition root: the canonical FastAPI runtime uses the MCP-backed
# LiteratureAgent while preserving the v78 scientific implementation itself.
# The orchestrator functions resolve this global at execution time.
orchestrator.LiteratureAgent = MCPLiteratureAgent
STAGES = orchestrator.STAGES

app = FastAPI(
    title="HypothesisForge API",
    version="0.2.0",
    description="Multi-agent, literature-grounded scientific hypothesis discovery with human checkpoints and MCP literature retrieval.",
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
    return {
        "status": "ok",
        "service": "hypothesis-forge",
        "stages": STAGES,
        "runs_dir": str(RUNS_DIR),
        "literature_transport": "mcp-in-process",
    }


@app.post("/runs")
def create_run(payload: StartRunRequest) -> dict[str, Any]:
    try:
        return orchestrator.start_run(payload)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=redact_sensitive_text(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=redact_sensitive_text(exc)) from exc


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
        return orchestrator.run_stage(run, payload)
    except (FileNotFoundError, ValueError) as exc:
        if isinstance(exc, FileNotFoundError):
            raise HTTPException(status_code=404, detail="Run not found") from exc
        raise HTTPException(status_code=400, detail=redact_sensitive_text(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=redact_sensitive_text(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=redact_sensitive_text(exc)) from exc


@app.post("/runs/{run_id}/selection")
def selection(run_id: str, payload: SelectionRequest) -> dict[str, Any]:
    try:
        return orchestrator.update_selection(read_run(run_id), payload)
    except (FileNotFoundError, ValueError) as exc:
        if isinstance(exc, FileNotFoundError):
            raise HTTPException(status_code=404, detail="Run not found") from exc
        raise HTTPException(status_code=400, detail=redact_sensitive_text(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Card not found: {redact_sensitive_text(exc)}") from exc


@app.post("/runs/{run_id}/focus-seed")
def focus_seed(run_id: str, payload: FocusSeedRequest) -> dict[str, Any]:
    try:
        return create_focus_seed(read_run(run_id), payload)
    except (FileNotFoundError, ValueError) as exc:
        if isinstance(exc, FileNotFoundError):
            raise HTTPException(status_code=404, detail="Run not found") from exc
        raise HTTPException(status_code=400, detail=redact_sensitive_text(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Card not found: {redact_sensitive_text(exc)}") from exc


@app.get("/runs/{run_id}/artifacts")
def artifacts(run_id: str) -> list[dict[str, Any]]:
    try:
        read_run(run_id)
        return list_artifacts(run_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc
