"""Tests for the DeskMate self-diagnostics ("doctor") checks.

Focused on the newer health probes that don't need a live Ollama: background
worker liveness, capture freshness, disk space, DB integrity, and model
auto-start consistency. Network/process-dependent checks (Ollama, GenAI build)
are exercised only via the full report's never-raise contract.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from deskmate import paths
from deskmate.config import load as load_config
from deskmate.db import DatabaseManager
from deskmate.engine import doctor


# ── helpers ────────────────────────────────────────────────────────────────
def _finished_thread() -> threading.Thread:
    """A thread that has run to completion — is_alive() is False."""
    t = threading.Thread(target=lambda: None)
    t.start()
    t.join()
    return t


def _live_thread(stop: threading.Event) -> threading.Thread:
    t = threading.Thread(target=stop.wait, daemon=True)
    t.start()
    return t


class _Worker:
    def __init__(self, thread: threading.Thread | None) -> None:
        self._thread = thread


class _FakeDaemon:
    pass


# ── background workers ───────────────────────────────────────────────────────
def test_workers_no_daemon_degrades_to_ok() -> None:
    r = doctor._check_workers(None, None)
    assert r.status == doctor.OK
    assert "not introspectable" in r.message


def test_workers_all_alive() -> None:
    stop = threading.Event()
    try:
        d = _FakeDaemon()
        t = _live_thread(stop)
        t.name = "daemon-audio"
        d._threads = [t]
        d._translate_thread = None
        r = doctor._check_workers(None, d)
        assert r.status == doctor.OK
        assert "daemon-audio" in r.message
    finally:
        stop.set()


def test_workers_detects_dead() -> None:
    stop = threading.Event()
    try:
        d = _FakeDaemon()
        alive = _live_thread(stop)
        alive.name = "daemon-audio"
        dead = _finished_thread()
        dead.name = "daemon-retention"
        d._threads = [alive, dead]
        d._translate_thread = None          # feature off → skipped, not a failure
        d.app_scheduler = _Worker(_finished_thread())  # crashed sub-worker
        d.habit_watcher = _Worker(None)     # disabled → skipped
        r = doctor._check_workers(None, d)
        assert r.status == doctor.FAIL
        assert "daemon-retention" in r.message
        assert "AppScheduler" in r.message
        assert "daemon-audio" in r.message  # listed as alive
    finally:
        stop.set()


# ── a11y capture watchers ────────────────────────────────────────────────────
class _Recorder:
    def __init__(self, win_events=None, inp=None, clipboard=None, enabled=True):
        self.win_events = win_events
        self.input = inp
        self.clipboard = clipboard
        self.cfg = type("C", (), {"enabled": enabled})()


class _IsAliveOnly:
    """Watcher exposing is_alive() but no _thread handle (like WinEventWatcher)."""
    def __init__(self, alive): self._alive = alive
    def is_alive(self): return self._alive


def test_a11y_no_daemon_degrades_ok() -> None:
    r = doctor._check_a11y_watchers(None, None)
    assert r.status == doctor.OK
    assert "not introspectable" in r.message


def test_a11y_all_alive() -> None:
    stop = threading.Event()
    try:
        d = _FakeDaemon()
        d.ui = _Recorder(win_events=_IsAliveOnly(True))
        d._ui_pipeline = _Worker(_live_thread(stop))  # uses ._thread? no — flush
        # pipeline uses _flush_thread; emulate it
        d._ui_pipeline = type("P", (), {"_flush_thread": _live_thread(stop)})()
        d._linker = _Worker(_live_thread(stop))
        r = doctor._check_a11y_watchers(None, d)
        assert r.status == doctor.OK
        assert "WinEventWatcher" in r.message
        assert "FrameLinkerActor" in r.message
    finally:
        stop.set()


def test_a11y_detects_dead_watcher() -> None:
    stop = threading.Event()
    try:
        d = _FakeDaemon()
        d.ui = _Recorder(win_events=_IsAliveOnly(False))  # crashed
        d._linker = _Worker(_live_thread(stop))
        r = doctor._check_a11y_watchers(None, d)
        assert r.status == doctor.FAIL
        assert "WinEventWatcher" in r.message
    finally:
        stop.set()


def test_a11y_disabled_recorder_skipped() -> None:
    stop = threading.Event()
    try:
        d = _FakeDaemon()
        # recorder disabled → its watchers skipped, but linker still counts.
        d.ui = _Recorder(win_events=_IsAliveOnly(False), enabled=False)
        d._linker = _Worker(_live_thread(stop))
        r = doctor._check_a11y_watchers(None, d)
        assert r.status == doctor.OK   # dead watcher ignored because disabled
        assert "FrameLinkerActor" in r.message
    finally:
        stop.set()


# ── OCR engine ───────────────────────────────────────────────────────────────
def _ocr_cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, engine: str):
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    cfg = load_config()
    cfg.ocr.engine = engine
    return cfg


def test_ocr_off_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r = doctor._check_ocr(_ocr_cfg(tmp_path, monkeypatch, "off"))
    assert r.status == doctor.OK
    assert "disabled" in r.message


def test_ocr_rapidocr_available(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from deskmate.screen import ocr as ocr_mod
    monkeypatch.setattr(ocr_mod, "_rapidocr_engine", lambda: object())
    r = doctor._check_ocr(_ocr_cfg(tmp_path, monkeypatch, "rapidocr"))
    assert r.status == doctor.OK
    assert "rapidocr" in r.message


def test_ocr_rapidocr_unavailable_warns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from deskmate.screen import ocr as ocr_mod
    monkeypatch.setattr(ocr_mod, "_rapidocr_engine", lambda: None)
    monkeypatch.setattr(ocr_mod, "_winrt_available", lambda: True)
    r = doctor._check_ocr(_ocr_cfg(tmp_path, monkeypatch, "rapidocr"))
    assert r.status == doctor.WARN
    assert "falling back to winrt" in r.message


def test_ocr_winrt_unavailable_warns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from deskmate.screen import ocr as ocr_mod
    monkeypatch.setattr(ocr_mod, "_winrt_available", lambda: False)
    r = doctor._check_ocr(_ocr_cfg(tmp_path, monkeypatch, "winrt"))
    assert r.status == doctor.WARN
    assert "tesseract" in r.message


# ── capture freshness ────────────────────────────────────────────────────────
class _FakeDB:
    def __init__(self, health: dict) -> None:
        self._health = health

    def health(self) -> dict:
        return self._health


def _cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    return load_config()


def test_capture_freshness_recent_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from deskmate.db.manager import _now_iso

    cfg = _cfg(tmp_path, monkeypatch)
    db = _FakeDB({"last_frame_timestamp": _now_iso()})
    r = doctor._check_capture_freshness(cfg, db)
    assert r.status == doctor.OK


def test_capture_freshness_stale_warns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(tmp_path, monkeypatch)
    db = _FakeDB({"last_frame_timestamp": "2000-01-01T00:00:00+00:00"})
    r = doctor._check_capture_freshness(cfg, db)
    assert r.status == doctor.WARN
    assert "stalled" in r.message


def test_capture_freshness_disabled_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(tmp_path, monkeypatch)
    cfg.capture.enabled = False
    r = doctor._check_capture_freshness(cfg, db=None)
    assert r.status == doctor.OK
    assert "disabled" in r.message


def test_age_seconds_parsing() -> None:
    assert doctor._age_seconds(None) is None
    assert doctor._age_seconds("not-a-date") is None
    assert doctor._age_seconds("2000-01-01T00:00:00+00:00") > 0


# ── disk ─────────────────────────────────────────────────────────────────────
def test_disk_reports_volume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(tmp_path, monkeypatch)
    r = doctor._check_disk(cfg)
    assert r.status in (doctor.OK, doctor.WARN, doctor.FAIL)
    assert "free" in r.message


# ── DB integrity (real sqlite via DatabaseManager) ───────────────────────────
def test_db_integrity_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    paths.ensure_dirs()
    db = DatabaseManager(str(paths.db_path()))
    try:
        r = doctor._check_db_integrity(None, db)
        assert r.status == doctor.OK
        assert "integrity ok" in r.message
    finally:
        db.close()


def test_db_integrity_no_handle() -> None:
    r = doctor._check_db_integrity(None, None)
    assert r.status == doctor.WARN


# ── managed model process ────────────────────────────────────────────────────
def _patch_status(monkeypatch: pytest.MonkeyPatch, st: dict) -> None:
    from deskmate.modelsvc import service
    monkeypatch.setattr(service, "status", lambda cfg: st)


def test_managed_process_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_status(monkeypatch, {
        "pid": 1234, "pid_alive": True, "managed_by_deskmate": True,
        "running": True, "external": False, "running_backend": "openvino",
    })
    r = doctor._check_managed_process(None)
    assert r.status == doctor.OK
    assert "pid 1234" in r.message and "openvino" in r.message


def test_managed_process_stale_pid_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    # We launched it (PID file exists) but the process is gone — the key case.
    _patch_status(monkeypatch, {
        "pid": 4321, "pid_alive": False, "managed_by_deskmate": False,
        "running": False, "external": False, "running_backend": "",
        "backend": "official",
    })
    r = doctor._check_managed_process(None)
    assert r.status == doctor.WARN
    assert "no longer running" in r.message


def test_managed_process_external_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_status(monkeypatch, {
        "pid": None, "pid_alive": False, "managed_by_deskmate": False,
        "running": True, "external": True, "running_backend": "official",
    })
    r = doctor._check_managed_process(None)
    assert r.status == doctor.OK
    assert "external" in r.message


def test_managed_process_none_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_status(monkeypatch, {
        "pid": None, "pid_alive": False, "managed_by_deskmate": False,
        "running": False, "external": False,
    })
    r = doctor._check_managed_process(None)
    assert r.status == doctor.OK


def test_managed_process_status_error_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    from deskmate.modelsvc import service

    def _boom(cfg):
        raise RuntimeError("boom")
    monkeypatch.setattr(service, "status", _boom)
    r = doctor._check_managed_process(None)
    assert r.status == doctor.WARN


# ── model auto-start ─────────────────────────────────────────────────────────
def test_model_autostart_disabled_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(tmp_path, monkeypatch)
    cfg.model_service.auto_start = False
    r = doctor._check_model_autostart(cfg)
    assert r.status == doctor.OK
    assert "disabled" in r.message


# ── never-raise contract for the whole report ────────────────────────────────
def test_report_never_raises_and_is_serialisable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    paths.ensure_dirs()
    cfg = load_config()
    db = DatabaseManager(str(paths.db_path()))
    try:
        rep = doctor.report(cfg, db, None)
    finally:
        db.close()
    assert rep["overall"] in (doctor.OK, doctor.WARN, doctor.FAIL)
    assert set(rep["summary"]) == {"ok", "warn", "fail"}
    names = {c["name"] for c in rep["checks"]}
    # The new checks are all present in the report.
    for expected in ("Background workers", "A11y capture watchers",
                     "Managed model process", "OCR engine",
                     "Capture freshness", "Disk space",
                     "Database", "Model auto-start"):
        assert expected in names
    for c in rep["checks"]:
        assert set(c) == {"name", "status", "message", "fix"}


# ── localization (中英文) ─────────────────────────────────────────────────────
def test_report_chinese_localizes_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    paths.ensure_dirs()
    cfg = load_config()
    db = DatabaseManager(str(paths.db_path()))
    try:
        en = doctor.report(cfg, db, None, lang="en")
        zh = doctor.report(cfg, db, None, lang="zh")
    finally:
        db.close()
    en_names = {c["name"] for c in en["checks"]}
    zh_names = {c["name"] for c in zh["checks"]}
    # Status verdicts must be identical across languages (same machine state).
    assert en["summary"] == zh["summary"]
    # The Chinese report uses localized names, not the English ones.
    assert "Disk space" in en_names and "磁盘空间" in zh_names
    assert "OCR engine" in en_names and "OCR 引擎" in zh_names
    # zh report should contain at least one CJK character somewhere.
    blob = "".join(c["name"] + c["message"] for c in zh["checks"])
    assert any("一" <= ch <= "鿿" for ch in blob)


def test_report_lang_variants_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # "zh-CN" / unknown values must not crash; zh* → Chinese, else English.
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    paths.ensure_dirs()
    cfg = load_config()
    db = DatabaseManager(str(paths.db_path()))
    try:
        zh_cn = doctor.report(cfg, db, None, lang="zh-CN")
        bogus = doctor.report(cfg, db, None, lang="fr")
    finally:
        db.close()
    assert any(c["name"] == "磁盘空间" for c in zh_cn["checks"])
    assert any(c["name"] == "Disk space" for c in bogus["checks"])


def test_lang_contextvar_resets_after_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # After a zh report, a bare check (no report wrapper) defaults to English.
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    cfg = load_config()
    doctor.report(cfg, None, None, lang="zh")
    r = doctor._check_disk(cfg)
    assert r.name == "Disk space"  # context reset, back to default "en"
