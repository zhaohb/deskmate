"""MCP server for local activity data.

Implemented using the official `mcp` SDK. Talks to the local HTTP API; if the
SDK isn't installed, importing this module is a no-op and `deskmate mcp`
will fail with a clear message."""

from .server import run_stdio

__all__ = ["run_stdio"]
