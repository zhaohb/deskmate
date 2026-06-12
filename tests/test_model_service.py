"""Tests for the Model Service feature (modelsvc + /models/* endpoints).

Pure functions only — env building, exe validation, status parsing, Zip-Slip
guard, config defaults, and the config endpoint. Real downloads / process
launches / taskkill are never exercised; subprocess and network are mocked.
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from deskmate import paths
from deskmate.config import load as load_config
from deskmate.db import DatabaseManager
from deskmate.engine.api import create_app
from deskmate.modelsvc import service


# ── launch env ───────────────────────────────────────────────────────────────
def test_build_launch_env_official(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    cfg = load_config()
    cfg.ollama.base = "http://127.0.0.1:11434"
    env = service.build_launch_env(cfg, service.BACKEND_OFFICIAL, tmp_path / "ollama.exe")
    assert env["OLLAMA_HOST"] == "127.0.0.1:11434"
    assert "GODEBUG" not in env            # official build needs no cgocheck hack
    assert "OLLAMA_REGISTRY" not in env    # no registry configured


def test_build_launch_env_openvino_sets_path_and_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    cfg = load_config()
    cfg.ollama.base = "http://localhost:9999"
    cfg.model_service.registry = "http://my-registry:5000"
    cfg.model_service.genai_runtime_dir = str(tmp_path / "rt" / "bin")
    # No setupvars.bat under that fake dir → manual-PATH fallback.
    env = service.build_launch_env(cfg, service.BACKEND_OPENVINO, tmp_path / "ollama.exe")
    assert env["OLLAMA_HOST"] == "localhost:9999"
    assert "GODEBUG" not in env            # not required for this build
    assert env["OLLAMA_REGISTRY"] == "http://my-registry:5000"
    assert env["PATH"].startswith(str(tmp_path / "rt" / "bin"))


def test_build_launch_env_uses_setupvars_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When setupvars.bat is found, its captured env is used (+ our OLLAMA_*)."""
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    monkeypatch.setattr(service.sys, "platform", "win32")
    cfg = load_config()
    cfg.model_service.backend = service.BACKEND_OPENVINO
    cfg.model_service.registry = "http://reg:5000"
    fake_setup = tmp_path / "pkg" / "setupvars.bat"
    monkeypatch.setattr(service, "find_setupvars", lambda c: fake_setup)
    monkeypatch.setattr(
        service, "capture_setupvars_env",
        lambda p: {"PATH": "C:\\ov\\bin;C:\\ov\\tbb", "OpenVINO_DIR": "C:\\ov\\cmake"},
    )
    env = service.build_launch_env(cfg, service.BACKEND_OPENVINO, tmp_path / "ollama.exe")
    # captured vars are present...
    assert env["OpenVINO_DIR"] == "C:\\ov\\cmake"
    assert env["PATH"] == "C:\\ov\\bin;C:\\ov\\tbb"
    # ...and our OLLAMA_* are layered on top.
    assert env["OLLAMA_HOST"] == "127.0.0.1:11434"
    assert env["OLLAMA_REGISTRY"] == "http://reg:5000"


def test_capture_setupvars_env_parses_set_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """capture_setupvars_env parses `set` KEY=VALUE lines into a dict.

    Also pins the invocation that was the actual bug: ``shell=True`` with a
    doubled-quote ``cmd /c ""<bat>" && set"`` command (single-quoting mangled
    backslash paths into "not recognized", yielding an empty env).
    """
    monkeypatch.setattr(service.sys, "platform", "win32")

    class _Done:
        returncode = 0
        # status echo + cmd pseudo var + a value containing '=' must all be handled
        stdout = (
            b"FOO=bar\r\nPATH=C:\\a;C:\\b\r\n=::=::\\\r\nBAZ=q=ux\r\n"
            b"[setupvars.bat] OpenVINO environment initialized\r\n"
        )
        stderr = b""

    captured = {}

    def _fake_run(cmd, **kw):
        captured["cmd"] = cmd
        captured["shell"] = kw.get("shell")
        return _Done()

    monkeypatch.setattr(service.subprocess, "run", _fake_run)
    bat = tmp_path / "setupvars.bat"
    out = service.capture_setupvars_env(bat)
    assert out["FOO"] == "bar"
    assert out["PATH"] == "C:\\a;C:\\b"
    assert out["BAZ"] == "q=ux"          # only the first '=' splits
    assert "" not in out                  # the `=::` cmd pseudo-var is dropped
    # the echoed status line ("... initialized", has a space in the key) dropped
    assert all(" " not in k for k in out)
    # invocation shape: shell=True + doubled outer quotes around the quoted path
    assert captured["shell"] is True
    assert captured["cmd"] == f'cmd /c ""{bat}" && set"'


def test_capture_setupvars_env_returns_empty_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-zero exit (e.g. the old quoting bug) yields {} so we fall back."""
    monkeypatch.setattr(service.sys, "platform", "win32")

    class _Fail:
        returncode = 1
        stdout = b""
        stderr = b"'...' is not recognized as an internal or external command"

    monkeypatch.setattr(service.subprocess, "run", lambda *a, **k: _Fail())
    assert service.capture_setupvars_env(tmp_path / "setupvars.bat") == {}


# ── exe validation ─────────────────────────────────────────────────────────--
def test_validate_exe_path_accepts_existing(tmp_path: Path) -> None:
    exe = tmp_path / "ollama.exe"
    exe.write_text("stub", encoding="utf-8")
    resolved = service.validate_exe_path(str(exe))
    assert resolved == exe.resolve()


def test_validate_exe_path_rejects_missing() -> None:
    with pytest.raises(ValueError):
        service.validate_exe_path(r"C:\nope\does-not-exist.exe")


def test_validate_exe_path_rejects_empty() -> None:
    with pytest.raises(ValueError):
        service.validate_exe_path("   ")


def test_validate_exe_path_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        service.validate_exe_path(str(tmp_path))


# ── status parsing ──────────────────────────────────────────────────────────-
def test_status_running_managed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    cfg = load_config()
    monkeypatch.setattr(service, "probe_running", lambda base, timeout=2: {"models": [{"name": "qwen3"}]})
    monkeypatch.setattr(service, "_probe_version", lambda base, timeout=2: "0.30.7")
    monkeypatch.setattr(service, "pid_alive", lambda pid: True)
    service.write_pid_file({"pid": 4321, "backend": "official", "exe": "x", "started_at": 0})

    s = service.status(cfg)
    assert s["running"] is True
    assert s["models"] == ["qwen3"]
    assert s["managed_by_deskmate"] is True
    assert s["external"] is False
    assert s["version"] == "0.30.7"


def test_status_external_when_no_pid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    cfg = load_config()
    monkeypatch.setattr(service, "probe_running", lambda base, timeout=2: {"models": []})
    monkeypatch.setattr(service, "_probe_version", lambda base, timeout=2: "")
    service.clear_pid_file()

    s = service.status(cfg)
    assert s["running"] is True
    assert s["managed_by_deskmate"] is False
    assert s["external"] is True       # running but not ours -> Stop should be disabled


def test_status_not_running(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    cfg = load_config()
    monkeypatch.setattr(service, "probe_running", lambda base, timeout=2: None)

    s = service.status(cfg)
    assert s["running"] is False
    assert s["external"] is False
    assert s["models"] == []


# ── Zip-Slip guard ──────────────────────────────────────────────────────────-
def test_extract_zip_rejects_traversal(tmp_path: Path) -> None:
    bad = tmp_path / "bad.zip"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../escape.txt", "pwned")
    bad.write_bytes(buf.getvalue())
    with pytest.raises(ValueError):
        service.extract_zip(bad, tmp_path / "out")


def test_extract_zip_ok(tmp_path: Path) -> None:
    good = tmp_path / "good.zip"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("sub/ollama.exe", "stub")
    good.write_bytes(buf.getvalue())
    out = tmp_path / "out"
    service.extract_zip(good, out)
    assert (out / "sub" / "ollama.exe").is_file()
    assert service.find_exe_in_dir(out) == (out / "sub" / "ollama.exe")


# ── OpenVINO downloads (separate exe + runtime, configurable dir) ─────────────
def test_resolve_download_dir_default_and_custom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    cfg = load_config()
    assert service.resolve_download_dir(cfg) == paths.ollama_openvino_dir()
    cfg.model_service.download_dir = str(tmp_path / "my-ov")
    assert service.resolve_download_dir(cfg) == (tmp_path / "my-ov")


def test_obtain_openvino_exe_copies_local_path(tmp_path: Path) -> None:
    src = tmp_path / "src" / "ollama.exe"
    src.parent.mkdir()
    src.write_text("stub", encoding="utf-8")
    dest_dir = tmp_path / "dl"
    out = service.obtain_openvino_exe(str(src), dest_dir)
    assert out == dest_dir / "ollama.exe"
    assert out.is_file()


def test_obtain_openvino_exe_downloads_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Mock the raw-file download so no network is touched.
    def fake_download_file(url, dest_path, *, progress=None, chunk=1 << 20):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text("downloaded", encoding="utf-8")
        if progress:
            progress(10, 10)
        return dest_path

    monkeypatch.setattr(service, "download_file", fake_download_file)
    dest_dir = tmp_path / "dl"
    out = service.obtain_openvino_exe("https://example/ollama.exe", dest_dir)
    assert out == dest_dir / "ollama.exe"
    assert out.read_text(encoding="utf-8") == "downloaded"


def test_obtain_openvino_exe_rejects_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        service.obtain_openvino_exe("  ", tmp_path)


def test_download_genai_uses_runtime_subdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    def fake_install(progress=None, dest_dir=None, url=None):
        captured["dest"] = dest_dir
        return dest_dir / "runtime" / "bin"

    monkeypatch.setattr(service, "install_genai_runtime", fake_install)
    dl_dir = tmp_path / "dl"
    service.download_genai(dl_dir)
    assert captured["dest"] == dl_dir / "runtime"


def test_download_genai_passes_custom_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    def fake_install(progress=None, dest_dir=None, url=None):
        captured["url"] = url
        return dest_dir / "bin"

    monkeypatch.setattr(service, "install_genai_runtime", fake_install)
    service.download_genai(tmp_path / "dl", url="https://example/genai.zip")
    assert captured["url"] == "https://example/genai.zip"


def _make_genai_version(root: Path, name: str) -> None:
    """Create a fake extracted GenAI version folder (genai DLL + TBB dll)."""
    pkg = root / "runtime" / name / "runtime"
    bindir = pkg / "bin"
    bindir.mkdir(parents=True)
    (bindir / "openvino_genai.dll").write_text("x", encoding="utf-8")
    # TBB lives in its own folder, like the real package.
    tbb = pkg / "3rdparty" / "tbb" / "bin"
    tbb.mkdir(parents=True)
    (tbb / "tbb12.dll").write_text("x", encoding="utf-8")


def test_genai_bin_dir_prefers_release_over_debug(tmp_path: Path) -> None:
    """The real package ships both Release and Debug DLLs — pick Release."""
    base = tmp_path / "pkg" / "runtime" / "bin" / "intel64"
    for variant in ("Release", "Debug"):
        d = base / variant
        d.mkdir(parents=True)
        (d / ("openvino_genai.dll" if variant == "Release" else "openvino_genaid.dll")).write_text(
            "x", encoding="utf-8"
        )
    got = service.genai_bin_dir(tmp_path / "pkg")
    assert got.name == "Release"


def test_list_genai_versions_sorted_newest_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    cfg = load_config()
    cfg.model_service.download_dir = str(tmp_path / "ov")
    _make_genai_version(tmp_path / "ov", "openvino_genai_2026.1.0.0")
    _make_genai_version(tmp_path / "ov", "openvino_genai_2026.3.0.0")
    versions = service.list_genai_versions(cfg)
    names = [v["name"] for v in versions]
    assert names == ["openvino_genai_2026.3.0.0", "openvino_genai_2026.1.0.0"]
    # newest version's bin dir is what build_launch_env puts on PATH
    cfg.model_service.backend = service.BACKEND_OPENVINO
    env = service.build_launch_env(cfg, service.BACKEND_OPENVINO, tmp_path / "ollama.exe")
    assert env["PATH"].startswith(versions[0]["bin"])


def test_build_launch_env_includes_tbb_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OpenVINO needs TBB (its own folder) on PATH besides the genai DLL dir."""
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    cfg = load_config()
    cfg.model_service.download_dir = str(tmp_path / "ov")
    cfg.model_service.backend = service.BACKEND_OPENVINO
    _make_genai_version(tmp_path / "ov", "openvino_genai_2026.3.0.0")
    env = service.build_launch_env(cfg, service.BACKEND_OPENVINO, tmp_path / "ollama.exe")
    path = env["PATH"]
    assert "bin" in path  # main DLL dir
    # the TBB dir (…/3rdparty/tbb/bin) must also be present
    assert any("tbb" in seg.lower() for seg in path.split(os.pathsep))


def test_genai_selected_version_overrides_newest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    cfg = load_config()
    cfg.model_service.download_dir = str(tmp_path / "ov")
    cfg.model_service.backend = service.BACKEND_OPENVINO
    _make_genai_version(tmp_path / "ov", "openvino_genai_2026.1.0.0")
    _make_genai_version(tmp_path / "ov", "openvino_genai_2026.3.0.0")
    # Explicitly pick the OLDER one.
    older = [v for v in service.list_genai_versions(cfg) if "2026.1" in v["name"]][0]
    cfg.model_service.genai_runtime_dir = older["bin"]
    env = service.build_launch_env(cfg, service.BACKEND_OPENVINO, tmp_path / "ollama.exe")
    assert env["PATH"].startswith(older["bin"])


# ── service log ───────────────────────────────────────────────────────────--
def test_read_service_log_tails_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    from deskmate import paths

    paths.logs_dir().mkdir(parents=True, exist_ok=True)
    log = paths.modelsvc_log_file()
    log.write_text("\n".join(f"line {i}" for i in range(50)), encoding="utf-8")
    out = service.read_service_log(max_lines=10)
    lines = out.splitlines()
    assert len(lines) == 10
    assert lines[-1] == "line 49"


def test_read_service_log_empty_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    assert service.read_service_log() == ""


def test_models_log_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    cfg = load_config()
    db = DatabaseManager()
    monkeypatch.setattr(service, "probe_running", lambda base, timeout=2: None)
    from deskmate import paths

    paths.logs_dir().mkdir(parents=True, exist_ok=True)
    paths.modelsvc_log_file().write_text("hello from ollama", encoding="utf-8")
    client = TestClient(create_app(cfg=cfg, db=db, daemon=None))
    resp = client.get("/models/log")
    assert resp.status_code == 200
    assert "hello from ollama" in resp.json()["log"]


# ── config ──────────────────────────────────────────────────────────────────-
def test_config_model_service_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    cfg = load_config()
    assert cfg.model_service.backend == "official"
    assert cfg.model_service.auto_start is False
    assert cfg.model_service.registry == ""
    # default toml written on first load includes the section
    text = paths.config_path().read_text(encoding="utf-8")
    assert "[model_service]" in text


def test_set_config_value_model_service_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    load_config()  # write defaults
    from deskmate.config import set_config_value

    set_config_value("model_service", "registry", "http://reg:5000")
    text = paths.config_path().read_text(encoding="utf-8")
    assert 'registry = "http://reg:5000"' in text
    assert load_config().model_service.registry == "http://reg:5000"


def test_set_config_value_escapes_windows_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Windows exe path (backslashes) must persist as valid TOML and reload.

    Regression: an unescaped backslash path like ``C:\\Users\\...`` produced
    ``\\U`` which TOML reads as an escape → "Invalid hex value" on next load.
    """
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    load_config()
    from deskmate.config import set_config_value

    win = r"C:\Users\Win11\.deskmate\bin\ollama-openvino\ollama.exe"
    set_config_value("model_service", "ollama_exe_path", win)
    # The file must still parse (this is what crashed before the fix)...
    reloaded = load_config()
    # ...and the value must round-trip byte-for-byte.
    assert reloaded.model_service.ollama_exe_path == win


# ── /models/config endpoint ──────────────────────────────────────────────────
def test_models_config_endpoint_persists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    cfg = load_config()
    db = DatabaseManager()
    # Avoid touching subprocess/network when the endpoint reports status.
    monkeypatch.setattr(service, "probe_running", lambda base, timeout=2: None)
    app = create_app(cfg=cfg, db=db, daemon=None)
    client = TestClient(app)

    resp = client.post("/models/config", json={"registry": "http://reg:5000", "backend": "openvino"})
    assert resp.status_code == 200
    body = resp.json()
    assert "registry" in body["saved"]
    assert "backend" in body["saved"]
    assert cfg.model_service.registry == "http://reg:5000"
    assert cfg.model_service.backend == "openvino"
    text = paths.config_path().read_text(encoding="utf-8")
    assert 'registry = "http://reg:5000"' in text


def test_models_config_endpoint_rejects_bad_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    cfg = load_config()
    db = DatabaseManager()
    monkeypatch.setattr(service, "probe_running", lambda base, timeout=2: None)
    client = TestClient(create_app(cfg=cfg, db=db, daemon=None))

    resp = client.post("/models/config", json={"backend": "bogus"})
    assert resp.status_code == 200
    assert "backend" in resp.json()["errors"]


def test_models_status_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    cfg = load_config()
    db = DatabaseManager()
    monkeypatch.setattr(service, "probe_running", lambda base, timeout=2: None)
    client = TestClient(create_app(cfg=cfg, db=db, daemon=None))

    resp = client.get("/models/status")
    assert resp.status_code == 200
    assert resp.json()["running"] is False


def test_models_active_sets_ollama_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    cfg = load_config()
    db = DatabaseManager()
    monkeypatch.setattr(service, "probe_running", lambda base, timeout=2: None)
    client = TestClient(create_app(cfg=cfg, db=db, daemon=None))

    resp = client.post("/models/active", json={"model": "qwen3:8b"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["active_model"] == "qwen3:8b"
    # Persisted to [ollama] model — what Ask / apps read.
    assert cfg.ollama.model == "qwen3:8b"
    text = paths.config_path().read_text(encoding="utf-8")
    assert 'model = "qwen3:8b"' in text


def test_models_active_rejects_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    cfg = load_config()
    db = DatabaseManager()
    monkeypatch.setattr(service, "probe_running", lambda base, timeout=2: None)
    client = TestClient(create_app(cfg=cfg, db=db, daemon=None))

    resp = client.post("/models/active", json={"model": "  "})
    assert resp.status_code == 400
