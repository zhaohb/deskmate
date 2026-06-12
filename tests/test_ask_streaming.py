"""Tests for the streaming Ask path (chat_ollama_stream + run_ask on_token)."""

from __future__ import annotations

import json

from deskmate.engine import ask, llm


class _FakeResp:
    """Minimal http.client-style response yielding NDJSON token lines."""

    def __init__(self, chunks: list[str]) -> None:
        self.status = 200
        lines = [
            json.dumps({"message": {"role": "assistant", "content": c}, "done": False})
            for c in chunks
        ]
        lines.append(json.dumps({"message": {"role": "assistant", "content": ""}, "done": True}))
        self._data = ("\n".join(lines) + "\n").encode("utf-8")

    def __iter__(self):
        return iter(self._data.splitlines(keepends=True))

    def read(self):
        return self._data


class _FakeConn:
    def __init__(self, resp: _FakeResp) -> None:
        self._resp = resp

    def request(self, *a, **k) -> None:  # noqa: ANN002, ANN003
        pass

    def getresponse(self):
        return self._resp

    def close(self) -> None:
        pass


def test_chat_ollama_stream_emits_tokens_and_returns_full(monkeypatch) -> None:
    chunks = ["Hello", ", ", "world"]
    monkeypatch.setattr(
        llm.http.client, "HTTPConnection", lambda *a, **k: _FakeConn(_FakeResp(chunks))
    )
    seen: list[str] = []
    msg = llm.chat_ollama_stream(
        [{"role": "user", "content": "hi"}],
        base="http://127.0.0.1:11434",
        model="m",
        on_token=seen.append,
    )
    assert seen == chunks                       # each token delivered live
    assert msg["content"] == "Hello, world"     # accumulated full message


def test_chat_ollama_stream_survives_bad_sink(monkeypatch) -> None:
    """A throwing on_token must not break the stream (UI sink isolation)."""
    monkeypatch.setattr(
        llm.http.client, "HTTPConnection", lambda *a, **k: _FakeConn(_FakeResp(["a", "b"]))
    )

    def boom(_):
        raise RuntimeError("sink failed")

    msg = llm.chat_ollama_stream(
        [{"role": "user", "content": "hi"}],
        base="http://127.0.0.1:11434", model="m", on_token=boom,
    )
    assert msg["content"] == "ab"


def test_chat_ollama_routes_to_stream_when_on_token(monkeypatch) -> None:
    """ask._chat_ollama streams iff on_token is given — even with tools attached.

    The agent loop passes ``tools=ASK_TOOLS`` on EVERY round (including the round
    where the model writes its final answer), so streaming must NOT be gated on
    the absence of tools — otherwise the answer would never stream. The prose
    gate (not tool-presence) is what hides tool-call markup from the UI.
    """
    called = {"stream": 0, "plain": 0}
    monkeypatch.setattr(
        ask.llm, "chat_ollama_stream",
        lambda *a, **k: called.__setitem__("stream", called["stream"] + 1) or {"content": "x"},
    )
    monkeypatch.setattr(
        ask.llm, "chat_ollama",
        lambda *a, **k: called.__setitem__("plain", called["plain"] + 1) or {"content": "y"},
    )
    # No sink → no streaming, regardless of tools.
    ask._chat_ollama([{"role": "user", "content": "q"}], tools=[{"x": 1}])
    assert called == {"stream": 0, "plain": 1}
    # With a sink → stream, even though tools are attached (the agent-loop case).
    ask._chat_ollama([{"role": "user", "content": "q"}], tools=[{"x": 1}], on_token=lambda t: None)
    assert called == {"stream": 1, "plain": 1}
