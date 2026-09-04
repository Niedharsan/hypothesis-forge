from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any

RUN_LOCK = threading.RLock()
ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = Path(os.getenv("HYPOTHESIS_FORGE_RUNS_DIR", ROOT / "data" / "runs"))
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]{6,160}", run_id):
        raise ValueError("Invalid run id")
    return run_id


def run_dir(run_id: str) -> Path:
    return RUNS_DIR / _safe_run_id(run_id)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_run(run: dict[str, Any], artifacts: dict[str, Any] | None = None) -> None:
    with RUN_LOCK:
        rdir = run_dir(str(run["run_id"]))
        rdir.mkdir(parents=True, exist_ok=True)
        write_json(rdir / "run_state.json", run)
        for filename, payload in (artifacts or {}).items():
            write_json(rdir / filename, payload)


def read_run(run_id: str) -> dict[str, Any]:
    with RUN_LOCK:
        payload = read_json(run_dir(run_id) / "run_state.json")
        if not isinstance(payload, dict):
            raise FileNotFoundError(run_id)
        return payload


def list_artifacts(run_id: str) -> list[dict[str, Any]]:
    rdir = run_dir(run_id)
    if not rdir.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(rdir.glob("*.json")):
        if path.name == "run_state.json":
            continue
        out.append({"filename": path.name, "data": read_json(path)})
    return out
