"""Minimal MCP stdio server. Exposes local activity tools:
  * search           — full-text search frames + transcripts
  * recent_frames    — list latest captured frames
  * recent_events    — list latest UI events
  * capture_once     — trigger a paired capture
  * health           — daemon liveness probe

All tools call the local HTTP API (default 127.0.0.1:3030) so the MCP server
can run as a separate process from the recorder."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

API = os.environ.get("DESKMATE_API", "http://127.0.0.1:3030")


def _http_get(path: str, params: dict[str, Any] | None = None) -> Any:
    return httpx.get(f"{API}{path}", params=params, timeout=20).json()


def _http_post(path: str) -> Any:
    return httpx.post(f"{API}{path}", timeout=20).json()


def run_stdio() -> None:
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import TextContent, Tool
    except ImportError as exc:
        raise SystemExit(
            "MCP SDK not installed. Install with: pip install 'deskmate[mcp]' "
            f"(missing: {exc})"
        ) from exc

    import anyio

    server = Server("deskmate")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="search",
                description="Full-text search over captured frames (OCR + accessibility text), UI events and audio transcripts.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "q": {"type": "string"},
                        "app_name": {"type": "string"},
                        "window_name": {"type": "string"},
                        "start_time": {"type": "string", "description": "ISO 8601"},
                        "end_time": {"type": "string", "description": "ISO 8601"},
                        "content_type": {
                            "type": "string",
                            "enum": ["all", "frames", "ocr", "audio", "ui", "element"],
                            "default": "all",
                        },
                        "role": {"type": "string", "description": "With content_type=element, filter by UIA role"},
                        "min_length": {"type": "integer"},
                        "max_length": {"type": "integer"},
                        "speaker_ids": {"type": "string", "description": "comma-separated speaker ids"},
                        "include_frames": {"type": "boolean", "default": False},
                        "limit": {"type": "integer", "default": 20},
                        "offset": {"type": "integer", "default": 0},
                    },
                    "required": ["q"],
                },
            ),
            Tool(name="recent_frames", description="List latest captured frames.",
                 inputSchema={"type": "object", "properties": {"limit": {"type": "integer", "default": 20}}}),
            Tool(name="recent_events", description="List latest UI events (clicks, focus, key_text, clipboard).",
                 inputSchema={"type": "object", "properties": {"limit": {"type": "integer", "default": 50}}}),
            Tool(name="capture_once", description="Trigger a paired capture now.",
                 inputSchema={"type": "object", "properties": {}}),
            Tool(name="health", description="Daemon liveness + counters.",
                 inputSchema={"type": "object", "properties": {}}),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        if name == "search":
            data = _http_get("/search", params={k: v for k, v in arguments.items() if v is not None})
        elif name == "recent_frames":
            data = _http_get("/frames", params={"limit": arguments.get("limit", 20)})
        elif name == "recent_events":
            data = _http_get("/events/recent", params={"limit": arguments.get("limit", 50)})
        elif name == "capture_once":
            data = _http_post("/capture")
        elif name == "health":
            data = _http_get("/health")
        else:
            data = {"error": f"unknown tool: {name}"}
        return [TextContent(type="text", text=json.dumps(data, ensure_ascii=False, indent=2))]

    async def _main() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    anyio.run(_main)
