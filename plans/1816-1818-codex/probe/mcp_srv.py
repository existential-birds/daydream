from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent, CallToolResult
m = FastMCP("probe", host="127.0.0.1", port=18767)
@m.tool()
def refuse(reason: str = "") -> CallToolResult:
    """Return an error result with two text blocks."""
    return CallToolResult(content=[TextContent(type="text", text="refused"), TextContent(type="text", text="second")], isError=True)
@m.tool()
def ok() -> str:
    """Return plain text."""
    return "fine"
m.run(transport="streamable-http")
