from __future__ import annotations

from agents.literature_agent import LiteratureAgent as DirectLiteratureAgent
from mcp_client.literature import build_mcp_source_adapters
from retrieval.pubtator_api import PubTatorAPI


class MCPLiteratureAgent(DirectLiteratureAgent):
    """LiteratureAgent with search transport routed through MCP.

    All query planning, evidence selection, deduplication, filtering, synthesis,
    and paper-memory behavior is inherited unchanged. Only the five reusable
    literature-search adapters are replaced by MCP-backed adapter facades.
    PubTator remains a local/direct annotation layer by design.
    """

    def __init__(self, model: str = "gemini-2.5-flash-lite", config_path: str = "configs/config.yaml") -> None:
        self.model = model
        self.adapters = build_mcp_source_adapters(config_path)
        self.pubtator = PubTatorAPI()
