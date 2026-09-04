import app.main as app_main
from agents.mcp_literature_agent import MCPLiteratureAgent


def test_canonical_fastapi_runtime_uses_mcp_literature_agent():
    assert app_main.orchestrator.LiteratureAgent is MCPLiteratureAgent
    health = app_main.health()
    assert health["literature_transport"] == "mcp-in-process"
