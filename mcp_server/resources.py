from __future__ import annotations

import re
from typing import Any

from mcp.server.mcpserver.exceptions import ResourceNotFoundError

import app.storage as storage

PAPER_MEMORY_FILENAME = "07b_paper_memory_compact.json"
_ARTIFACT_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}\.json")
_MISSING = object()


def list_run_catalog(*, limit: int = 100) -> dict[str, Any]:
    """Return a compact catalog of persisted HypothesisForge runs."""
    runs: list[dict[str, Any]] = []
    if not storage.RUNS_DIR.is_dir():
        return {"runs": runs, "count": 0}

    candidates = sorted(
        (path for path in storage.RUNS_DIR.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            run = storage.read_run(path.name)
        except (FileNotFoundError, ValueError, OSError):
            continue
        runs.append(_summary(run, include_artifacts=False))
        if len(runs) >= limit:
            break
    return {"runs": runs, "count": len(runs)}


def read_run_summary(run_id: str) -> dict[str, Any]:
    """Return compact persisted run state without duplicating full card payloads."""
    try:
        run = storage.read_run(run_id)
    except (FileNotFoundError, ValueError) as exc:
        raise ResourceNotFoundError(f"Run not found: {run_id}") from exc
    return _summary(run, include_artifacts=True)


def read_artifact_catalog(run_id: str) -> dict[str, Any]:
    """Return metadata for JSON artifacts persisted for one run."""
    try:
        run = storage.read_run(run_id)
    except (FileNotFoundError, ValueError) as exc:
        raise ResourceNotFoundError(f"Run not found: {run_id}") from exc
    artifacts = _artifact_metadata(run)
    return {"run_id": run_id, "artifacts": artifacts, "count": len(artifacts)}


def read_run_artifact(run_id: str, filename: str) -> Any:
    """Read one declared persisted JSON artifact without permitting path traversal."""
    if not _ARTIFACT_NAME_RE.fullmatch(filename):
        raise ResourceNotFoundError("Artifact not found")
    try:
        run = storage.read_run(run_id)
    except (FileNotFoundError, ValueError) as exc:
        raise ResourceNotFoundError(f"Run not found: {run_id}") from exc

    allowed = {item["filename"] for item in _artifact_metadata(run)}
    if filename not in allowed:
        raise ResourceNotFoundError(f"Artifact not found: {filename}")

    payload = storage.read_json(storage.run_dir(run_id) / filename, _MISSING)
    if payload is _MISSING:
        raise ResourceNotFoundError(f"Artifact not found: {filename}")
    return payload


def read_compact_paper_memory(run_id: str) -> dict[str, Any]:
    """Read the compact paper-memory artifact persisted by the Proximity stage."""
    payload = read_run_artifact(run_id, PAPER_MEMORY_FILENAME)
    if not isinstance(payload, dict):
        raise ResourceNotFoundError(f"Paper memory unavailable for run: {run_id}")
    return payload


def _summary(run: dict[str, Any], *, include_artifacts: bool) -> dict[str, Any]:
    stages = run.get("stages") if isinstance(run.get("stages"), dict) else {}
    summary: dict[str, Any] = {
        "run_id": run.get("run_id"),
        "status": run.get("status"),
        "current_stage": run.get("current_stage"),
        "created_at": run.get("created_at"),
        "objective": run.get("objective"),
        "cutoff_year": run.get("cutoff_year"),
        "runtime_mode": run.get("runtime_mode"),
        "literature_sources": run.get("literature_sources", []),
        "usage": run.get("usage", {}),
        "stage_counts": {
            stage: len(cards) if isinstance(cards, list) else 0
            for stage, cards in stages.items()
        },
    }
    if include_artifacts:
        summary["artifacts"] = _artifact_metadata(run)
    return summary


def _artifact_metadata(run: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in run.get("artifacts", []) or []:
        if not isinstance(item, dict):
            continue
        filename = str(item.get("filename") or "")
        if not _ARTIFACT_NAME_RE.fullmatch(filename):
            continue
        out.append(
            {
                "id": item.get("id"),
                "label": item.get("label"),
                "filename": filename,
                "stage": item.get("stage"),
            }
        )
    return out
