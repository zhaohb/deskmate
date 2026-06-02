"""Shared LLM + HTTP transport engine for Ask and pipe apps.

Both the Ask agent (``engine/ask.py``) and the pipe-app runner (``apps/agent.py``)
talk to the local DeskMate API and to Ollama through the exact same
proxy-bypassing ``http.client`` transport and the same ``/api/chat`` request
shape. Those primitives are centralized here so the two agents run on one
engine, while each keeps its own orchestration logic and its own module-level
``OLLAMA_BASE`` / ``OLLAMA_MODEL`` settings (the apps override the model at
runtime via ``agent.OLLAMA_MODEL``).

This module is intentionally stateless: callers pass the base URL and model on
every call, so moving the helpers here does not change any behavior.
"""

from __future__ import annotations

import http.client
import json
import os
import re
from typing import Any
from urllib.parse import urlparse

DEFAULT_OLLAMA_BASE = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen3_8b_ov:v1"
DEFAULT_CHAT_TIMEOUT = 600


def resolve_ollama_settings() -> tuple[str, str, int]:
    """Resolve Ollama base URL, model name, and chat timeout.

    Priority (highest first): ``OLLAMA_*`` env vars, then ``~/.deskmate/config.toml``
    ``[ollama]`` (and ``DESKMATE_ollama__*`` env), then module defaults.
    """
    try:
        from ..config import load as load_config

        ollama = load_config().ollama
        base = ollama.base
        model = ollama.model
        timeout = ollama.chat_timeout
    except Exception:
        base = DEFAULT_OLLAMA_BASE
        model = DEFAULT_OLLAMA_MODEL
        timeout = DEFAULT_CHAT_TIMEOUT

    if v := os.environ.get("OLLAMA_BASE"):
        base = v
    if v := os.environ.get("OLLAMA_MODEL"):
        model = v
    if v := os.environ.get("OLLAMA_CHAT_TIMEOUT"):
        timeout = int(v)
    return base, model, timeout


_OLLAMA_BASE, _OLLAMA_MODEL, _CHAT_TIMEOUT = resolve_ollama_settings()

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
_FUNCTION_BLOCK_RE = re.compile(
    r"<function=([^>\s]+)\s*>(.*?)(?:</function>|(?=<tool_call>)|$)",
    re.DOTALL | re.IGNORECASE,
)
_PARAMETER_RE = re.compile(
    r"<parameter=([^>\s]+)\s*>(.*?)</parameter>",
    re.DOTALL | re.IGNORECASE,
)
_MISTRAL_TOOL_CALLS_RE = re.compile(
    r"\[TOOL_CALLS\]\s*(\[.*?\])\s*\[/TOOL_CALLS\]",
    re.DOTALL,
)


def strip_thinking(text: str) -> str:
    """Remove ``<think>...</think>`` reasoning blocks and trim whitespace."""
    return THINK_RE.sub("", text or "").strip()


def strip_tool_call_markup(text: str) -> str:
    """Remove tool-call XML/JSON blocks from assistant text."""
    cleaned = _TOOL_CALL_BLOCK_RE.sub("", text or "")
    cleaned = _MISTRAL_TOOL_CALLS_RE.sub("", cleaned)
    return strip_thinking(cleaned).strip()


def parse_tool_calls_from_text(content: str) -> list[dict[str, Any]]:
    """Parse tool calls embedded in model text (Qwen3.5 XML, Qwen JSON, Mistral)."""
    calls: list[dict[str, Any]] = []
    if not content:
        return calls

    for block in _TOOL_CALL_BLOCK_RE.finditer(content):
        inner = block.group(1).strip()
        if not inner:
            continue
        if inner.startswith("{"):
            try:
                obj = json.loads(inner)
            except json.JSONDecodeError:
                continue
            name = obj.get("name")
            if name:
                call_idx = len(calls)
                calls.append({
                    "id": f"call_{call_idx}",
                    "type": "function",
                    "function": {
                        "index": call_idx,
                        "name": name,
                        "arguments": obj.get("arguments") or {},
                    },
                })
            continue

        for fn_match in _FUNCTION_BLOCK_RE.finditer(inner):
            name = fn_match.group(1).strip()
            fn_body = fn_match.group(2)
            args: dict[str, Any] = {}
            for param in _PARAMETER_RE.finditer(fn_body):
                args[param.group(1).strip()] = param.group(2).strip()
            call_idx = len(calls)
            calls.append({
                "id": f"call_{call_idx}",
                "type": "function",
                "function": {"index": call_idx, "name": name, "arguments": args},
            })

    mistral = _MISTRAL_TOOL_CALLS_RE.search(content)
    if mistral and not calls:
        try:
            items = json.loads(mistral.group(1))
        except json.JSONDecodeError:
            items = []
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and item.get("name"):
                    call_idx = len(calls)
                    calls.append({
                        "id": f"call_{call_idx}",
                        "type": "function",
                        "function": {
                            "index": call_idx,
                            "name": item["name"],
                            "arguments": item.get("arguments") or {},
                        },
                    })
    return calls


def extract_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Return structured tool calls from ``tool_calls`` or parsed assistant content."""
    existing = message.get("tool_calls")
    if existing:
        return list(existing)
    return parse_tool_calls_from_text(message.get("content") or "")


def normalize_assistant_message(message: dict[str, Any]) -> dict[str, Any]:
    """Fill ``tool_calls`` when the model returned Qwen-style XML in ``content``."""
    msg = dict(message)
    calls = extract_tool_calls(msg)
    if calls:
        msg["tool_calls"] = normalize_tool_calls(calls)
        msg["content"] = strip_tool_call_markup(msg.get("content") or "")
    return msg


def normalize_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure tool calls have IDs and the shape Ollama accepts in chat history."""
    normalized: list[dict[str, Any]] = []
    for idx, tc in enumerate(tool_calls):
        fn = dict(tc.get("function") or {})
        fn.setdefault("index", idx)
        fn.setdefault("arguments", {})
        normalized.append({
            "id": tc.get("id") or f"call_{idx}",
            "type": tc.get("type") or "function",
            "function": fn,
        })
    return normalized


def raw_request(
    method: str,
    url: str,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 60,
) -> Any:
    """Issue an HTTP request via ``http.client`` (bypasses env proxies)."""
    parsed = urlparse(url)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=timeout)
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    try:
        conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        data = resp.read().decode("utf-8")
    except TimeoutError:
        conn.close()
        raise
    except OSError as exc:
        conn.close()
        if "timed out" in str(exc).lower():
            raise TimeoutError(str(exc)) from exc
        raise
    conn.close()
    if resp.status >= 400:
        raise RuntimeError(f"HTTP {resp.status}: {data[:500]}")
    return json.loads(data)


def http_get(url: str, timeout: int = 15) -> Any:
    return raw_request("GET", url, timeout=timeout)


def http_post(url: str, body: dict, timeout: int = 120) -> Any:
    return raw_request(
        "POST", url,
        body=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )


def http_patch(url: str, body: dict, timeout: int = 60) -> Any:
    return raw_request(
        "PATCH", url,
        body=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )


def chat_ollama(
    messages: list[dict],
    tools: list[dict] | None = None,
    *,
    base: str,
    model: str,
    num_predict: int = 4096,
    temperature: float = 0.3,
    timeout: int | None = None,
) -> dict:
    """Call Ollama ``/api/chat`` (non-streaming) and return the message dict."""
    if timeout is None:
        timeout = _CHAT_TIMEOUT
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    if tools:
        body["tools"] = tools
    result = http_post(f"{base}/api/chat", body, timeout=timeout)
    return normalize_assistant_message(result.get("message", {}))
