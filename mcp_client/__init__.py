"""MCP client boundary used by HypothesisForge's literature workflow."""

from mcp_client.literature import LiteratureMCPClient, MCPSourceAdapter, build_mcp_source_adapters

__all__ = ["LiteratureMCPClient", "MCPSourceAdapter", "build_mcp_source_adapters"]
