"""Minimal MCP stdio server. Exposes local activity tools:
  * search           — full-text search frames + transcripts
  * recent_frames    — list latest captured frames
  * recent_events    — list latest UI events
  * capture_once     — trigger a paired capture
  * health           — daemon liveness probe
  * ask              — LLM Q&A over local activity (searches context + runs tools)
  * list_apps        — list available report-generating apps
  * run_app          — run an app (report generator) and return its result
  * list_app_outputs — list an app's past run outputs
  * get_app_output   — fetch one output file of a past app run

All tools call the local HTTP API (default 127.0.0.1:3030) so the MCP server
can run as a separate process from the recorder."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

API = os.environ.get("DESKMATE_API", "http://127.0.0.1:3030")

# ask runs an LLM (+ tool rounds); run_app spawns a report subprocess that itself
# calls the LLM — both are far slower than the read-only queries, so they get
# their own generous timeouts (10 min for the LLM-backed tools).
DEFAULT_TIMEOUT = 20
ASK_TIMEOUT = int(os.environ.get("DESKMATE_MCP_ASK_TIMEOUT", "600"))
RUN_APP_TIMEOUT = int(os.environ.get("DESKMATE_MCP_RUN_APP_TIMEOUT", "600"))
PROGRESS_INTERVAL_S = 15.0

# trust_env=False: never route the local API through an HTTP(S)_PROXY. On a
# machine behind a corporate proxy httpx would otherwise try to send even
# 127.0.0.1 traffic through the proxy (which refuses it / returns an empty
# body), breaking every tool. We only ever talk to the local daemon.
def _client(timeout: float) -> httpx.Client:
    return httpx.Client(trust_env=False, timeout=timeout)


def _http_get(path: str, params: dict[str, Any] | None = None, timeout: float = DEFAULT_TIMEOUT) -> Any:
    with _client(timeout) as c:
        return c.get(f"{API}{path}", params=params).json()


def _http_get_text(path: str, timeout: float = DEFAULT_TIMEOUT) -> str:
    # Some endpoints (app output files) return markdown/json text, not a JSON body.
    with _client(timeout) as c:
        return c.get(f"{API}{path}").text


def _http_post(path: str, body: dict[str, Any] | None = None, timeout: float = DEFAULT_TIMEOUT) -> Any:
    with _client(timeout) as c:
        return c.post(f"{API}{path}", json=body).json()


async def _http_post_with_progress(
    path: str,
    body: dict[str, Any] | None,
    *,
    timeout: float,
    label: str,
) -> Any:
    """Run a blocking POST in a worker thread and emit MCP progress while waiting."""
    import asyncio

    from mcp.server.lowlevel.server import request_ctx

    task = asyncio.create_task(asyncio.to_thread(_http_post, path, body, timeout))
    elapsed = 0.0
    while not task.done():
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout=PROGRESS_INTERVAL_S)
        except TimeoutError:
            elapsed += PROGRESS_INTERVAL_S
            try:
                ctx = request_ctx.get()
            except LookupError:
                continue
            token = ctx.meta.progressToken if ctx.meta else None
            if token is None:
                continue
            await ctx.session.send_progress_notification(
                token,
                min(elapsed, timeout),
                total=timeout,
                message=f"{label}: waiting for DeskMate API ({int(elapsed)}s)",
            )
    return await task


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
                name="deskmate_search",
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
            Tool(name="deskmate_recent_frames", description="List latest captured frames.",
                 inputSchema={"type": "object", "properties": {"limit": {"type": "integer", "default": 20}}}),
            Tool(name="deskmate_recent_events", description="List latest UI events (clicks, focus, key_text, clipboard).",
                 inputSchema={"type": "object", "properties": {"limit": {"type": "integer", "default": 50}}}),
            Tool(name="deskmate_capture_once", description="Trigger a paired capture now.",
                 inputSchema={"type": "object", "properties": {}}),
            Tool(name="deskmate_health", description="Daemon liveness + counters.",
                 inputSchema={"type": "object", "properties": {}}),
            Tool(
                name="deskmate_ask",
                description=(
                    "Ask a natural-language question about the user's recent activity. "
                    "An LLM agent searches the local captured context (screen OCR, UI events, "
                    "audio transcripts, meetings, todos) and runs tools to answer. Returns the "
                    "grounded answer plus a summary of the tool calls it made. Can take up to "
                    "10 minutes."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "The question to answer from local activity."},
                    },
                    "required": ["question"],
                },
            ),
            Tool(
                name="deskmate_list_apps",
                description=(
                    "List the available DeskMate apps (LLM report generators over local activity, "
                    "e.g. day-recap, standup-update, meeting-summary, time-breakdown, todo-list, "
                    "email-compose). Returns each app's name, title, description and recent outputs."
                ),
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="deskmate_run_app",
                description=(
                    "Run a DeskMate app (report generator) and return its result. Most apps take a "
                    "look-back window (hours, or start_time/end_time). App-specific params: "
                    "video-export uses minutes or start/end; email-compose needs provider "
                    "(gmail|outlook), to, intent, optional send; meeting-summary uses meeting_id. "
                    "Use list_apps to see available names. Can take minutes."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "app_name": {"type": "string", "description": "App to run (see list_apps)."},
                        "hours": {"type": "string", "description": "Look-back window in hours (most apps)."},
                        "start_time": {"type": "string", "description": "ISO 8601 window start (with end_time)."},
                        "end_time": {"type": "string", "description": "ISO 8601 window end (with start_time)."},
                        "minutes": {"type": "string", "description": "video-export: minutes of recent activity."},
                        "provider": {"type": "string", "enum": ["gmail", "outlook"], "description": "email-compose."},
                        "to": {"type": "string", "description": "email-compose: recipient address."},
                        "intent": {"type": "string", "description": "email-compose: what to write."},
                        "account": {"type": "string", "description": "email-compose: which connected account."},
                        "reply_to": {"type": "string", "description": "email-compose: message id to reply to."},
                        "send": {"type": "boolean", "description": "email-compose: actually send (else draft)."},
                        "meeting_id": {"type": "string", "description": "meeting-summary: target meeting."},
                    },
                    "required": ["app_name"],
                },
            ),
            Tool(
                name="deskmate_list_app_outputs",
                description="List the past run outputs (run ids + files) of a given app.",
                inputSchema={
                    "type": "object",
                    "properties": {"app_name": {"type": "string"}},
                    "required": ["app_name"],
                },
            ),
            Tool(
                name="deskmate_get_app_output",
                description="Fetch one output file (markdown or json) of a past app run.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "app_name": {"type": "string"},
                        "run_id": {"type": "string"},
                        "filename": {"type": "string"},
                    },
                    "required": ["app_name", "run_id", "filename"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        if name == "deskmate_search":
            data = _http_get("/search", params={k: v for k, v in arguments.items() if v is not None})
        elif name == "deskmate_recent_frames":
            data = _http_get("/frames", params={"limit": arguments.get("limit", 20)})
        elif name == "deskmate_recent_events":
            data = _http_get("/events/recent", params={"limit": arguments.get("limit", 50)})
        elif name == "deskmate_capture_once":
            data = _http_post("/capture")
        elif name == "deskmate_health":
            data = _http_get("/health")
        elif name == "deskmate_ask":
            data = await _http_post_with_progress(
                "/ask",
                {"question": arguments.get("question", "")},
                timeout=ASK_TIMEOUT,
                label="deskmate_ask",
            )
        elif name == "deskmate_list_apps":
            data = _http_get("/apps")
        elif name == "deskmate_run_app":
            app_name = arguments.get("app_name")
            if not app_name:
                data = {"error": "app_name is required"}
            else:
                body = {k: v for k, v in arguments.items() if k != "app_name" and v is not None}
                data = await _http_post_with_progress(
                    f"/apps/{app_name}/run",
                    body,
                    timeout=RUN_APP_TIMEOUT,
                    label=f"deskmate_run_app:{app_name}",
                )
        elif name == "deskmate_list_app_outputs":
            app_name = arguments.get("app_name")
            data = _http_get(f"/apps/{app_name}/outputs") if app_name else {"error": "app_name is required"}
        elif name == "deskmate_get_app_output":
            app_name = arguments.get("app_name")
            run_id = arguments.get("run_id")
            filename = arguments.get("filename")
            if not (app_name and run_id and filename):
                data = {"error": "app_name, run_id and filename are required"}
            else:
                text = _http_get_text(f"/apps/{app_name}/outputs/{run_id}/{filename}")
                return [TextContent(type="text", text=text)]
        else:
            data = {"error": f"unknown tool: {name}"}
        return [TextContent(type="text", text=json.dumps(data, ensure_ascii=False, indent=2))]

    async def _main() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    anyio.run(_main)
