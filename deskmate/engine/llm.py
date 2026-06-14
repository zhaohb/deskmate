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
import socket
from collections.abc import Callable
from typing import Any
from urllib.parse import ParseResult, urlparse


class FriendlyError(Exception):
    """An error that prints a plain-language cause and a short fix hint.

    Use this for failures a user can act on (service not running, model not
    pulled, timeouts). ``str(err)`` renders as::

        <summary>
          Cause: <why it happened>
          Fix:   <what to do>
    """

    def __init__(self, summary: str, *, cause: str | None = None, fix: str | None = None) -> None:
        self.summary = summary
        self.cause = cause
        self.fix = fix
        super().__init__(summary)

    def __str__(self) -> str:
        lines = [self.summary]
        if self.cause:
            lines.append(f"  Cause: {self.cause}")
        if self.fix:
            lines.append(f"  Fix:   {self.fix}")
        return "\n".join(lines)


# Friendly variants that also keep their builtin base type, so existing
# ``except TimeoutError`` / ``except ConnectionError`` / ``except RuntimeError``
# handlers keep working unchanged while the message becomes actionable.
class FriendlyConnectionError(FriendlyError, ConnectionError):
    pass


class FriendlyTimeoutError(FriendlyError, TimeoutError):
    pass


class FriendlyHTTPError(FriendlyError, RuntimeError):
    pass


def _describe_endpoint(parsed: ParseResult) -> tuple[str, str, str]:
    """Return ``(endpoint, service_name, fix_hint)`` for a known DeskMate dependency."""
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    endpoint = f"{parsed.scheme or 'http'}://{host}:{port}"
    if port == 11434:
        return (
            endpoint,
            "Ollama",
            f"start it with `ollama serve` (or open the Ollama app), then verify with "
            f"`curl {endpoint}/api/tags`.",
        )
    if port == 3030:
        return (
            endpoint,
            "the DeskMate API",
            "start it with `python -m deskmate.engine.cli serve` (or `... ui`).",
        )
    return (
        endpoint,
        "the server",
        f"make sure the service at {endpoint} is running and reachable.",
    )


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


def _resolve_keep_alive() -> str | int:
    """How long Ollama keeps a model resident between calls.

    Defaults to ``-1`` (never unload) so the OpenVINO backend's ~30s cold load
    is paid once, not on every Ask round. Override via
    ``DESKMATE_OLLAMA_KEEP_ALIVE`` — an int (seconds, ``-1`` = forever, ``0`` =
    unload immediately) or a duration string Ollama understands (e.g. ``"10m"``).
    """
    raw = os.environ.get("DESKMATE_OLLAMA_KEEP_ALIVE", "-1").strip()
    try:
        return int(raw)
    except ValueError:
        return raw or -1


_KEEP_ALIVE: str | int = _resolve_keep_alive()


def _resolve_think() -> bool:
    """Whether to ask Ollama to run the model's thinking/reasoning pass.

    Resolved fresh on each call (not cached) so the Settings-page toggle takes
    effect without restarting the daemon. Priority: the ``DESKMATE_OLLAMA_THINK``
    env var (``0``/``false``/``no``/``off`` = off) overrides everything; else the
    ``[ollama] think`` config value; else ``True`` (reason before answering,
    which improves answer and tool-call quality — the reasoning comes back in a
    separate field so it never leaks into the answer).
    """
    raw = os.environ.get("DESKMATE_OLLAMA_THINK")
    if raw is not None:
        return raw.strip().lower() not in ("0", "false", "no", "off", "")
    try:
        from ..config import load as load_config

        return bool(load_config().ollama.think)
    except Exception:
        return True


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
    """Issue an HTTP request via ``http.client`` (bypasses env proxies).

    Transport and HTTP failures are re-raised as :class:`FriendlyError`
    subclasses that explain the cause and how to fix it, while preserving the
    original builtin type (``TimeoutError`` / ``ConnectionError`` /
    ``RuntimeError``) for existing handlers.
    """
    parsed = urlparse(url)
    endpoint, service, fix = _describe_endpoint(parsed)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=timeout)
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    try:
        conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        data = resp.read().decode("utf-8")
    except (TimeoutError, socket.timeout) as exc:
        conn.close()
        raise FriendlyTimeoutError(
            f"Timed out after {timeout}s waiting for {service} at {endpoint}.",
            cause="the server accepted the connection but did not respond in time "
            "(the model may still be loading, or the request is too large).",
            fix="retry, raise the timeout (config [ollama] chat_timeout or "
            "OLLAMA_CHAT_TIMEOUT), or switch to a smaller/faster model.",
        ) from exc
    except ConnectionRefusedError as exc:
        conn.close()
        raise FriendlyConnectionError(
            f"Cannot reach {service} at {endpoint} (connection refused).",
            cause=f"{service} is not running, or it is not listening on {endpoint}.",
            fix=fix,
        ) from exc
    except OSError as exc:
        conn.close()
        if "timed out" in str(exc).lower():
            raise FriendlyTimeoutError(
                f"Timed out after {timeout}s waiting for {service} at {endpoint}.",
                cause="no response from the server in time.",
                fix="retry, raise the timeout, or use a smaller/faster model.",
            ) from exc
        raise FriendlyConnectionError(
            f"Cannot reach {service} at {endpoint} ({exc.__class__.__name__}: {exc}).",
            cause=f"a network error occurred while connecting to {endpoint}.",
            fix=fix,
        ) from exc
    conn.close()
    if resp.status >= 400:
        snippet = data[:500].strip()
        cause, http_fix = _http_failure_hint(resp.status, snippet, parsed, service, fix)
        raise FriendlyHTTPError(
            f"{service} returned HTTP {resp.status} for {path.split('?')[0]}.",
            cause=cause,
            fix=http_fix,
        )
    return json.loads(data)


def _http_failure_hint(
    status: int, body: str, parsed: ParseResult, service: str, default_fix: str
) -> tuple[str, str]:
    """Map an HTTP error status + body to a plain cause and fix hint."""
    low = body.lower()
    if status == 404 and ("model" in low and ("not found" in low or "try pulling" in low)):
        model = ""
        m = re.search(r"model '([^']+)'", body) or re.search(r'model "([^"]+)"', body)
        if m:
            model = m.group(1)
        target = f"`{model}`" if model else "the configured model"
        return (
            f"{service} does not have {target} installed.",
            f"pull it with `ollama pull {model or '<model>'}`, or set an installed "
            "model via config [ollama] model (or the OLLAMA_MODEL env var). "
            "List installed models with `ollama list`.",
        )
    if status in (404, 405):
        return (
            f"the endpoint {parsed.path} is not available on {service}.",
            "check the URL/route, or update DeskMate if the API has changed.",
        )
    if status in (502, 503, 504):
        return (
            f"{service} is reachable but not ready to serve requests.",
            default_fix,
        )
    detail = f" Response: {body}" if body else ""
    return (f"{service} rejected the request (HTTP {status}).{detail}", default_fix)



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
    keep_alive: str | int | None = _KEEP_ALIVE,
) -> dict:
    """Call Ollama ``/api/chat`` (non-streaming) and return the message dict.

    ``keep_alive`` controls how long Ollama keeps the model resident after the
    call. It matters a lot for the OpenVINO backend, whose cold load can take
    ~30s: with the default 5-minute idle unload, a multi-round Ask reloads the
    model on nearly every turn and can blow past the chat timeout. Pinning it
    (``-1`` = never unload) makes every turn after the first a warm ~1s call.
    Set ``DESKMATE_OLLAMA_KEEP_ALIVE`` to override (e.g. ``"10m"``, ``0``).
    """
    if timeout is None:
        timeout = _CHAT_TIMEOUT
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": _resolve_think(),
        # Fixed generation params (DeskMate default): temperature/top_p/
        # repeat_penalty pinned to 1.0 so behavior matches the model's intended
        # defaults regardless of any temperature a caller passes. num_predict
        # stays caller-controlled (it's a length cap, not a sampling knob).
        "options": {
            "temperature": 1.0,
            "top_p": 1.0,
            "repeat_penalty": 1.0,
            "num_predict": num_predict,
        },
    }
    if keep_alive is not None:
        body["keep_alive"] = keep_alive
    if tools:
        body["tools"] = tools
    result = http_post(f"{base}/api/chat", body, timeout=timeout)
    return normalize_assistant_message(result.get("message", {}))


def chat_ollama_stream(
    messages: list[dict],
    tools: list[dict] | None = None,
    *,
    base: str,
    model: str,
    on_token: Callable[[str], None],
    on_thinking: Callable[[str], None] | None = None,
    num_predict: int = 4096,
    temperature: float = 0.3,
    timeout: int | None = None,
    keep_alive: str | int | None = _KEEP_ALIVE,
) -> dict:
    """Stream Ollama ``/api/chat`` (``stream:true``), calling ``on_token`` per chunk.

    Same request shape as :func:`chat_ollama` but reads the NDJSON token stream
    so callers can surface text as it is produced (the OpenVINO build emits one
    JSON object per token). ``on_token(text)`` is invoked for each non-empty
    content delta; the accumulated, normalized assistant message is returned at
    the end (so existing post-processing — including tool-call extraction — still
    works). ``tools`` may be passed so a tool-calling turn can also stream: any
    structured ``tool_calls`` deltas are accumulated and returned in the final
    message (callers gate what they actually show the user — see ask.py).

    Uses a raw proxy-bypassing ``http.client`` connection (like
    :func:`raw_request`) read line-by-line; transport errors raise the same
    :class:`FriendlyError` subtypes as the non-streaming path.
    """
    if timeout is None:
        timeout = _CHAT_TIMEOUT
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "think": _resolve_think(),
        # Fixed generation params (DeskMate default): temperature/top_p/
        # repeat_penalty pinned to 1.0 so behavior matches the model's intended
        # defaults regardless of any temperature a caller passes. num_predict
        # stays caller-controlled (it's a length cap, not a sampling knob).
        "options": {
            "temperature": 1.0,
            "top_p": 1.0,
            "repeat_penalty": 1.0,
            "num_predict": num_predict,
        },
    }
    if tools:
        body["tools"] = tools
    if keep_alive is not None:
        body["keep_alive"] = keep_alive

    parsed = urlparse(base)
    endpoint, service, fix = _describe_endpoint(parsed)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=timeout)
    pieces: list[str] = []
    think_pieces: list[str] = []
    stream_tool_calls: list[dict] = []
    try:
        conn.request(
            "POST", "/api/chat",
            body=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        if resp.status >= 400:
            snippet = resp.read().decode("utf-8", "replace")[:500].strip()
            cause, http_fix = _http_failure_hint(resp.status, snippet, parsed, service, fix)
            raise FriendlyHTTPError(
                f"{service} returned HTTP {resp.status} for /api/chat.",
                cause=cause, fix=http_fix,
            )
        for raw_line in resp:
            line = raw_line.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = obj.get("message") or {}
            # Thinking/reasoning arrives in its own field (OpenVINO build splits
            # it out); stream it separately so the UI can show the reasoning
            # without it polluting the answer content.
            think_delta = message.get("thinking") or ""
            if think_delta:
                think_pieces.append(think_delta)
                if on_thinking is not None:
                    try:
                        on_thinking(think_delta)
                    except Exception:  # noqa: BLE001
                        pass
            delta = message.get("content") or ""
            if delta:
                pieces.append(delta)
                try:
                    on_token(delta)
                except Exception:  # noqa: BLE001  (a UI sink must never kill the stream)
                    pass
            # Some builds emit structured tool_calls as their own delta object.
            tcs = message.get("tool_calls")
            if tcs:
                stream_tool_calls.extend(tcs)
            if obj.get("done"):
                break
    except TimeoutError as exc:
        conn.close()
        raise FriendlyTimeoutError(
            f"Timed out after {timeout}s waiting for {service} at {endpoint}.",
            cause="the model did not finish generating in time.",
            fix="retry, raise the timeout, or use a smaller/faster model.",
        ) from exc
    except ConnectionRefusedError as exc:
        conn.close()
        raise FriendlyConnectionError(
            f"Cannot reach {service} at {endpoint} (connection refused).",
            cause=f"{service} is not running, or not listening on {endpoint}.",
            fix=fix,
        ) from exc
    finally:
        conn.close()
    msg: dict[str, Any] = {"role": "assistant", "content": "".join(pieces)}
    if stream_tool_calls:
        msg["tool_calls"] = stream_tool_calls
    out = normalize_assistant_message(msg)
    if think_pieces:
        out["thinking"] = "".join(think_pieces)
    return out
