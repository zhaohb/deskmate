"""Shared LLM + HTTP transport engine for Ask and pipe apps.

Both the Ask agent (``engine/ask.py``) and the pipe-app runner (``apps/agent.py``)
talk to the local pc_assistant API and to Ollama through the exact same
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
# Local OpenVINO / small models can exceed 3 min on large single-shot prompts.
DEFAULT_CHAT_TIMEOUT = int(os.environ.get("OLLAMA_CHAT_TIMEOUT", "600"))

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove ``<think>...</think>`` reasoning blocks and trim whitespace."""
    return THINK_RE.sub("", text or "").strip()


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
        timeout = DEFAULT_CHAT_TIMEOUT
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
    return result.get("message", {})
