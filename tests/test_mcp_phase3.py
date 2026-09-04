from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from mcp import StdioServerParameters
from mcp.client import Client


def _seed_run(root: Path) -> str:
    run_id = "hf-test-phase3-123456"
    rdir = root / run_id
    rdir.mkdir(parents=True, exist_ok=True)
    memory = {
        "source_memory_counts": {"memory_entries": 1, "used_in_synthesis": 1},
        "entries_shown": 1,
        "entries": [{"stable_id": "PMID:123", "title": "Synthetic evidence", "used_in_synthesis": True}],
    }
    ranking = {"ranked_candidates": [{"candidate_id": "EVO-H1", "rank": 1, "title": "Candidate"}]}
    run = {
        "run_id": run_id,
        "status": "completed",
        "current_stage": "candidate_ranking",
        "created_at": "2026-09-04T20:00:00+00:00",
        "objective": "Synthetic Phase 3 transport validation",
        "cutoff_year": 2026,
        "runtime_mode": "dry_run",
        "literature_sources": ["PubMed"],
        "usage": {"calls": 12},
        "stages": {"axis_generation": [{"id": f"A{i:02d}"} for i in range(1, 11)], "candidate_ranking": [{"id": "RANK-01"}]},
        "artifacts": [
            {"id": "paper-memory", "label": "Compact paper memory", "filename": "07b_paper_memory_compact.json", "stage": "proximity", "data": memory},
            {"id": "ranking", "label": "Candidate ranking", "filename": "11_candidate_ranking.json", "stage": "candidate_ranking", "data": ranking},
        ],
    }
    (rdir / "run_state.json").write_text(json.dumps(run), encoding="utf-8")
    (rdir / "07b_paper_memory_compact.json").write_text(json.dumps(memory), encoding="utf-8")
    (rdir / "11_candidate_ranking.json").write_text(json.dumps(ranking), encoding="utf-8")
    return run_id


def _assert_phase3_surface(client: Client, run_id: str):
    async def run():
        tools = await client.list_tools()
        resources = await client.list_resources()
        templates = await client.list_resource_templates()
        catalog = await client.read_resource("hypothesisforge://runs")
        summary = await client.read_resource(f"hypothesisforge://runs/{run_id}")
        artifacts = await client.read_resource(f"hypothesisforge://runs/{run_id}/artifacts")
        memory = await client.read_resource(f"hypothesisforge://runs/{run_id}/paper-memory")
        ranking = await client.read_resource(
            f"hypothesisforge://runs/{run_id}/artifacts/11_candidate_ranking.json"
        )
        return tools, resources, templates, catalog, summary, artifacts, memory, ranking

    return run()


def _json_resource(result):
    assert len(result.contents) == 1
    return json.loads(result.contents[0].text)


def _validate_results(results, run_id: str) -> None:
    tools, resources, templates, catalog, summary, artifacts, memory, ranking = results
    assert sorted(tool.name for tool in tools.tools) == [
        "search_crossref",
        "search_europepmc",
        "search_openalex",
        "search_pubmed",
        "search_semantic_scholar",
    ]
    assert [str(resource.uri) for resource in resources.resources] == ["hypothesisforge://runs"]
    uris = {str(template.uri_template) for template in templates.resource_templates}
    assert uris == {
        "hypothesisforge://runs/{run_id}",
        "hypothesisforge://runs/{run_id}/artifacts",
        "hypothesisforge://runs/{run_id}/artifacts/{filename}",
        "hypothesisforge://runs/{run_id}/paper-memory",
    }
    assert _json_resource(catalog)["runs"][0]["run_id"] == run_id
    assert _json_resource(summary)["stage_counts"]["axis_generation"] == 10
    assert _json_resource(artifacts)["count"] == 2
    assert _json_resource(memory)["entries"][0]["stable_id"] == "PMID:123"
    assert _json_resource(ranking)["ranked_candidates"][0]["rank"] == 1


def test_real_stdio_transport_exposes_phase3_resources(tmp_path):
    run_id = _seed_run(tmp_path)
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server"],
        env={"HYPOTHESIS_FORGE_RUNS_DIR": str(tmp_path)},
    )

    async def run():
        async with Client(params) as client:
            return await _assert_phase3_surface(client, run_id)

    _validate_results(asyncio.run(run()), run_id)


def test_real_streamable_http_transport_exposes_phase3_resources(tmp_path):
    run_id = _seed_run(tmp_path)
    port = _free_port()
    env = os.environ.copy()
    env["HYPOTHESIS_FORGE_RUNS_DIR"] = str(tmp_path)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mcp_server",
            "--transport",
            "streamable-http",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_port(port, process)

        async def run():
            async with Client(f"http://127.0.0.1:{port}/mcp") as client:
                return await _assert_phase3_surface(client, run_id)

        _validate_results(asyncio.run(run()), run_id)
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_port(port: int, process: subprocess.Popen, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(f"MCP HTTP server exited early. stdout={stdout!r} stderr={stderr!r}")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.05)
    raise AssertionError("Timed out waiting for MCP Streamable HTTP server")
