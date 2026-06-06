"""Tests for the live-translation feature:

- the streaming endpoint detector that slices audio per utterance (Phase 1),
- the TranscriptTranslator's context window, same-language skip and fail-soft
  behavior (Phase 2),
- the audio_transcriptions translation columns + setter and the
  TRANSCRIPT_TRANSLATED event type.

These avoid real audio devices and a real Ollama server: the endpoint detector
is fed synthetic PCM, and the translator's LLM call is monkeypatched.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from deskmate import events as bus
from deskmate.audio.capture import _EndpointBuffer
from deskmate.audio.translator import TranscriptTranslator, _lang_name
from deskmate.db.manager import DatabaseManager


# ── Phase 1: endpoint detector ───────────────────────────────────────────────


def _speech(n_samples: int, rate: int = 16000) -> np.ndarray:
    """Loud-ish white noise that reads as speech under the RMS gate."""
    return (np.random.randn(n_samples) * 4000).astype(np.int16)


def _silence(n_samples: int) -> np.ndarray:
    return np.zeros(n_samples, dtype=np.int16)


def _feed(eb: _EndpointBuffer, arr: np.ndarray, frame: int = 4000):
    """Feed an array frame-by-frame; return the first emitted utterance, if any."""
    for i in range(0, len(arr), frame):
        out = eb.add(arr[i : i + frame])
        if out is not None:
            return out
    return None


def test_endpoint_emits_after_pause() -> None:
    eb = _EndpointBuffer(sample_rate=16000, silence_ms=700, max_chunk_s=8.0, min_chunk_s=1.0)
    # 2s speech, no emit yet
    assert _feed(eb, _speech(16000 * 2)) is None
    # then 0.75s silence (> 700ms) → emit the utterance
    out = _feed(eb, _silence(int(16000 * 0.75)))
    assert out is not None
    # Emitted chunk should hold the speech (and the trailing pause).
    assert len(out) >= 16000 * 2


def test_endpoint_no_emit_before_silence_threshold() -> None:
    eb = _EndpointBuffer(sample_rate=16000, silence_ms=700, max_chunk_s=8.0, min_chunk_s=1.0)
    assert _feed(eb, _speech(16000 * 2)) is None
    # Only 0.25s of silence — below the 700ms endpoint threshold.
    assert _feed(eb, _silence(int(16000 * 0.25))) is None


def test_endpoint_max_chunk_cap_forces_emit() -> None:
    eb = _EndpointBuffer(sample_rate=16000, silence_ms=700, max_chunk_s=3.0, min_chunk_s=1.0)
    # 4s of continuous speech with no pause must still emit at the 3s cap.
    out = _feed(eb, _speech(16000 * 4))
    assert out is not None
    assert len(out) >= 16000 * 3


def test_endpoint_drops_pure_silence() -> None:
    eb = _EndpointBuffer(sample_rate=16000, silence_ms=700, max_chunk_s=2.0, min_chunk_s=1.0)
    # All silence up to the cap → nothing emitted (no speech heard).
    assert _feed(eb, _silence(16000 * 3)) is None


def test_endpoint_min_chunk_carries_fragment_forward() -> None:
    eb = _EndpointBuffer(sample_rate=16000, silence_ms=400, max_chunk_s=8.0, min_chunk_s=1.5)
    # A 0.5s blip + pause is below min_chunk_s (1.5s): should NOT emit on its own.
    assert _feed(eb, _speech(int(16000 * 0.5)), frame=2000) is None
    out = _feed(eb, _silence(int(16000 * 0.6)), frame=2000)
    assert out is None  # carried forward, not emitted as a fragment


# ── Phase 2: translator ──────────────────────────────────────────────────────


def test_translator_skips_same_language() -> None:
    tr = TranscriptTranslator(target_lang="en", skip_if_same=True)
    # Source already English → skip (returns None), no LLM call needed.
    assert tr.translate("hello there", source_lang="en") is None


def test_translator_calls_llm_and_returns_text(monkeypatch) -> None:
    captured = {}

    def fake_chat(messages, **kwargs):
        captured["messages"] = messages
        return {"content": "你好世界"}

    tr = TranscriptTranslator(target_lang="zh", skip_if_same=True, context_window=2)
    monkeypatch.setattr("deskmate.audio.translator.chat_ollama", fake_chat)
    out = tr.translate("hello world", source_lang="en", device="mic")
    assert out == "你好世界"
    # The current line must appear in the user prompt.
    user = captured["messages"][-1]["content"]
    assert "hello world" in user


def test_translator_context_window_accumulates(monkeypatch) -> None:
    seen_prompts = []

    def fake_chat(messages, **kwargs):
        seen_prompts.append(messages[-1]["content"])
        return {"content": "X"}

    tr = TranscriptTranslator(target_lang="zh", skip_if_same=True, context_window=2)
    monkeypatch.setattr("deskmate.audio.translator.chat_ollama", fake_chat)
    tr.translate("first line", source_lang="en", device="mic")
    tr.translate("second line", source_lang="en", device="mic")
    # The 2nd call's prompt should carry the 1st line as context.
    assert "first line" in seen_prompts[1]


def test_translator_context_is_per_device(monkeypatch) -> None:
    seen = []

    def fake_chat(messages, **kwargs):
        seen.append(messages[-1]["content"])
        return {"content": "X"}

    tr = TranscriptTranslator(target_lang="zh", context_window=3)
    monkeypatch.setattr("deskmate.audio.translator.chat_ollama", fake_chat)
    tr.translate("mic line", source_lang="en", device="mic")
    tr.translate("loopback line", source_lang="en", device="loopback")
    # The loopback prompt must NOT contain the mic device's context.
    assert "mic line" not in seen[1]


def test_translator_failsoft_on_llm_error(monkeypatch) -> None:
    def boom(messages, **kwargs):
        raise RuntimeError("ollama down")

    tr = TranscriptTranslator(target_lang="zh")
    monkeypatch.setattr("deskmate.audio.translator.chat_ollama", boom)
    # Must swallow the error and return None rather than raise into the audio loop.
    assert tr.translate("hello", source_lang="en") is None


def test_translator_empty_input_returns_none() -> None:
    tr = TranscriptTranslator(target_lang="zh")
    assert tr.translate("   ", source_lang="en") is None


def test_lang_name_maps_known_codes() -> None:
    assert _lang_name("zh") == "Simplified Chinese"
    assert _lang_name("en") == "English"
    assert _lang_name("xx") == "xx"  # unknown code passes through
    assert _lang_name(None) == "the source language"


# ── DB + event ───────────────────────────────────────────────────────────────


def test_db_translation_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = DatabaseManager(Path(tmp) / "t.db")
        try:
            tid = db.insert_transcript(device="mic", text="hello", language="en")
            db.set_transcript_translation(tid, "你好", "zh")
            row = [r for r in db.recent_transcripts() if r["id"] == tid][0]
            assert row["translation"] == "你好"
            assert row["translation_lang"] == "zh"
        finally:
            db.close()


def test_transcript_translated_event_exists() -> None:
    assert hasattr(bus.EventType, "TRANSCRIPT_TRANSLATED")
    assert bus.EventType.TRANSCRIPT_TRANSLATED.value == "transcript_translated"


# ── runtime toggle: config persistence ───────────────────────────────────────


def test_set_audio_value_persists_scalar(tmp_path, monkeypatch) -> None:
    import deskmate.config as config

    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text("[audio]\nenabled = true\n", encoding="utf-8")
    monkeypatch.setattr(config.paths, "config_path", lambda: cfg_file)

    config.set_audio_value("translate_enabled", True)
    config.set_audio_value("translate_target_lang", "en")
    config.set_audio_value("translate_latency_mode", "quality")

    text = cfg_file.read_text(encoding="utf-8")
    assert "translate_enabled = true" in text
    assert 'translate_target_lang = "en"' in text
    assert 'translate_latency_mode = "quality"' in text


def test_set_audio_value_updates_existing_line(tmp_path, monkeypatch) -> None:
    import deskmate.config as config

    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        "[audio]\ntranslate_enabled = false  # keep comment\n", encoding="utf-8"
    )
    monkeypatch.setattr(config.paths, "config_path", lambda: cfg_file)

    config.set_audio_value("translate_enabled", True)
    text = cfg_file.read_text(encoding="utf-8")
    # Updated in place, not duplicated, and the trailing comment is preserved.
    assert text.count("translate_enabled") == 1
    assert "translate_enabled = true" in text
    assert "# keep comment" in text


def test_render_toml_value() -> None:
    from deskmate.config import _render_toml_value as r

    assert r(True) == "true"
    assert r(False) == "false"
    assert r("zh") == '"zh"'
    assert r(700) == "700"


# ── runtime toggle: API endpoint ──────────────────────────────────────────────


class _StubDaemon:
    """Records set_translation calls so the API hot-apply path can be asserted."""

    def __init__(self) -> None:
        self.calls = []

    def set_translation(self, *, enabled=None, target_lang=None, latency_mode=None):
        self.calls.append({"enabled": enabled, "target_lang": target_lang, "latency_mode": latency_mode})
        return {}


def _translate_client(tmp_path, monkeypatch):
    import deskmate.config as config
    from deskmate.config import Config
    from deskmate.db.manager import DatabaseManager
    from deskmate.engine.api import create_app
    from starlette.testclient import TestClient

    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text("[audio]\nenabled = true\n", encoding="utf-8")
    monkeypatch.setattr(config.paths, "config_path", lambda: cfg_file)

    cfg = Config()
    db = DatabaseManager(tmp_path / "t.db")
    daemon = _StubDaemon()
    app = create_app(cfg=cfg, db=db, daemon=daemon)
    return TestClient(app), cfg, daemon, db


def test_translate_get_returns_settings(tmp_path, monkeypatch) -> None:
    client, _cfg, _daemon, db = _translate_client(tmp_path, monkeypatch)
    try:
        r = client.get("/config/audio/translate")
        assert r.status_code == 200
        body = r.json()
        assert "translate_enabled" in body
        assert "translate_target_lang" in body
        assert "translate_latency_mode" in body
    finally:
        db.close()


def test_translate_post_persists_and_hot_applies(tmp_path, monkeypatch) -> None:
    client, cfg, daemon, db = _translate_client(tmp_path, monkeypatch)
    try:
        r = client.post(
            "/config/audio/translate",
            json={"enabled": True, "target_lang": "en", "latency_mode": "quality"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["translate_enabled"] is True
        assert body["translate_target_lang"] == "en"
        assert body["translate_latency_mode"] == "quality"
        assert body["hot_applied"] is True
        # In-memory cfg updated and the daemon hot-apply path was hit.
        assert cfg.audio.translate_enabled is True
        assert daemon.calls and daemon.calls[-1]["target_lang"] == "en"
        # Persisted to the config file.
        text = (tmp_path / "config.toml").read_text(encoding="utf-8")
        assert "translate_enabled = true" in text
    finally:
        db.close()


def test_translate_post_rejects_bad_latency(tmp_path, monkeypatch) -> None:
    client, _cfg, _daemon, db = _translate_client(tmp_path, monkeypatch)
    try:
        r = client.post("/config/audio/translate", json={"latency_mode": "instant"})
        assert r.status_code == 400
    finally:
        db.close()
