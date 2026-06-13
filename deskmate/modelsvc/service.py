"""Provision and run the local Ollama service for DeskMate.

DeskMate's Ask + LLM apps need a local Ollama server. This module is the
machinery behind the **Model Service** UI page: it downloads the Ollama
executable (official build) or accepts a user-supplied OpenVINO ``ollama.exe``,
downloads the OpenVINO GenAI runtime, pulls models, and launches the service in
the background so it survives a DeskMate restart.

Design notes:

* **Two backends.** The official build auto-downloads from a GitHub release zip.
  The OpenVINO build (``zhaohb/ollama_openvino``) has no GitHub releases — its
  prebuilt exe lives on a Google Drive folder that isn't scriptable — so the
  user supplies a local path; we only auto-download its GenAI runtime DLLs
  (a real direct URL) and wire the launch env (``GODEBUG=cgocheck=0`` + the
  runtime dir on ``PATH``).
* **Detached + persistent.** The service is launched as a detached process with
  its own group so it outlives the DeskMate UI process. A JSON PID file under
  ``~/.deskmate`` records what we started; status is derived from an HTTP probe
  of ``/api/tags`` plus PID liveness. Stop is explicit (Windows ``taskkill``).
* **Windows-first.** ``sys.platform == "win32"`` is the primary target; other
  OSes are handled best-effort and never hard-crash.

Everything here is pure stdlib so it ships in the wheel without extra deps.
"""

from __future__ import annotations

import http.client
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .. import paths
from ..logger import get

logger = get("modelsvc")

# ── Backends ─────────────────────────────────────────────────────────────────
BACKEND_OFFICIAL = "official"
BACKEND_OPENVINO = "openvino"

# Official Ollama Windows build (clean GitHub release asset).
OFFICIAL_RELEASE_TAG = "v0.30.7"
OFFICIAL_ASSET = "ollama-windows-amd64.zip"
OFFICIAL_URL = (
    f"https://github.com/ollama/ollama/releases/download/"
    f"{OFFICIAL_RELEASE_TAG}/{OFFICIAL_ASSET}"
)

# OpenVINO GenAI runtime (DLLs) the OV build needs to launch. Direct download.
# Pinned to dev20260530: the earlier dev20260516 nightly has an output-degradation
# bug on the Intel GPU plugin under long context (garbage tokens / repetition loops /
# the runner hanging), which surfaced as "garbled" Ask answers — its long system
# prompt reliably tripped it. dev20260530 fixes it. Bump this (not below 20260530)
# when adopting a newer runtime, and re-verify long-context generation on GPU.
GENAI_RUNTIME_URL = (
    "https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/"
    "nightly/2026.3.0.0.dev20260530/"
    "openvino_genai_windows_2026.3.0.0.dev20260530_x86_64.zip"
)

DEFAULT_BASE = "http://127.0.0.1:11434"

# Progress callback: (bytes_done, bytes_total) where total is -1 if unknown.
ProgressCb = Callable[[int, int], None]


# ── Download + extract ───────────────────────────────────────────────────────
def download_zip(
    url: str,
    dest_dir: Path,
    *,
    progress: ProgressCb | None = None,
    chunk: int = 1 << 20,
) -> Path:
    """Stream ``url`` into ``dest_dir`` as a ``.zip``, reporting progress.

    Downloads to a ``<name>.part`` temp file and only ``os.replace``s it to the
    final ``.zip`` on full success, so an interrupted download never leaves a
    truncated zip that looks complete. The partial file is removed on error.
    ``progress(downloaded, total)`` is called per chunk; ``total`` is ``-1``
    when the server sends no ``Content-Length``.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = Path(urlparse(url).path).name or "download.zip"
    if not name.endswith(".zip"):
        name += ".zip"
    final = dest_dir / name
    part = dest_dir / (name + ".part")

    req = urllib.request.Request(url, headers={"User-Agent": "deskmate-modelsvc"})
    try:
        # Connect timeout only; large downloads must not hit an overall read cap.
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            total = int(resp.headers.get("Content-Length") or -1)
            done = 0
            with part.open("wb") as fh:
                while True:
                    block = resp.read(chunk)
                    if not block:
                        break
                    fh.write(block)
                    done += len(block)
                    if progress is not None:
                        progress(done, total)
        os.replace(part, final)
        logger.info("downloaded %s -> %s (%d bytes)", url, final, done)
        return final
    except Exception:
        try:
            part.unlink()
        except OSError:
            pass
        raise


def download_file(
    url: str,
    dest_path: Path,
    *,
    progress: ProgressCb | None = None,
    chunk: int = 1 << 20,
) -> Path:
    """Stream ``url`` to the exact file ``dest_path`` (for non-zip files, e.g. the exe).

    Same ``.part`` + ``os.replace`` safety and ``progress`` reporting as
    :func:`download_zip`, but writes to a caller-chosen path instead of deriving
    a ``.zip`` name. The parent dir is created if missing.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    part = dest_path.with_name(dest_path.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "deskmate-modelsvc"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            total = int(resp.headers.get("Content-Length") or -1)
            done = 0
            with part.open("wb") as fh:
                while True:
                    block = resp.read(chunk)
                    if not block:
                        break
                    fh.write(block)
                    done += len(block)
                    if progress is not None:
                        progress(done, total)
        os.replace(part, dest_path)
        logger.info("downloaded %s -> %s (%d bytes)", url, dest_path, done)
        return dest_path
    except Exception:
        try:
            part.unlink()
        except OSError:
            pass
        raise


def _looks_like_url(text: str) -> bool:
    """True if ``text`` is an http(s) URL (vs. a local filesystem path)."""
    return text.strip().lower().startswith(("http://", "https://"))


def extract_zip(zip_path: Path, dest_dir: Path) -> Path:
    """Extract ``zip_path`` into ``dest_dir``, guarding against Zip-Slip.

    Any member whose resolved destination escapes ``dest_dir`` (via ``..`` or an
    absolute path) raises ``ValueError`` rather than being written outside the
    target — defense-in-depth even for our own server-controlled URLs.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_root = dest_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            target = (dest_root / member).resolve()
            if target != dest_root and dest_root not in target.parents:
                raise ValueError(f"unsafe path in zip: {member!r}")
        zf.extractall(dest_dir)
    return dest_dir


def find_exe_in_dir(directory: Path, name: str = "ollama.exe") -> Path | None:
    """Return the first ``name`` found anywhere under ``directory`` (rglob)."""
    if not directory.is_dir():
        return None
    for candidate in directory.rglob(name):
        if candidate.is_file():
            return candidate
    return None


def install_official(progress: ProgressCb | None = None) -> Path:
    """Download + extract the official Ollama build; return the ``ollama.exe``."""
    target = paths.ollama_official_dir()
    zip_path = download_zip(OFFICIAL_URL, target, progress=progress)
    extract_zip(zip_path, target)
    try:
        zip_path.unlink()
    except OSError:
        pass
    exe = find_exe_in_dir(target)
    if exe is None:
        raise RuntimeError(
            f"downloaded official Ollama but no ollama.exe found under {target}"
        )
    return exe


def install_genai_runtime(
    progress: ProgressCb | None = None,
    dest_dir: Path | None = None,
    url: str | None = None,
) -> Path:
    """Download + extract the OpenVINO GenAI runtime; return the dir for PATH.

    The package extracts into a versioned subdir whose ``runtime/bin`` holds the
    DLLs. We return the deepest directory that actually contains the runtime
    DLLs so it can be prepended to ``PATH`` at launch; falls back to the extract
    root if the expected layout isn't found. ``dest_dir`` overrides the default
    ``genai_runtime_dir``; ``url`` overrides :data:`GENAI_RUNTIME_URL` so a user
    can pull a different version (each version extracts to its own subfolder, so
    multiple versions coexist and can be picked later).
    """
    target = dest_dir or paths.genai_runtime_dir()
    zip_path = download_zip(url or GENAI_RUNTIME_URL, target, progress=progress)
    extract_zip(zip_path, target)
    try:
        zip_path.unlink()
    except OSError:
        pass
    return genai_bin_dir(target)


def list_genai_versions(cfg: Any) -> list[dict[str, str]]:
    """List installed GenAI runtime versions, newest first.

    Each GenAI zip unpacks to a versioned top-level folder (e.g.
    ``openvino_genai_windows_2026.3.0.0..._x86_64``). We scan both the download
    dir's ``runtime/`` (current layout) **and** the legacy standalone
    ``paths.genai_runtime_dir()`` (where the GenAI download landed in earlier
    builds) so a runtime installed either way is found. Returns ``[{name, bin}]``
    where ``bin`` is the DLL dir to put on PATH; de-duplicated by ``bin``.
    """
    roots = [resolve_download_dir(cfg) / "runtime", paths.genai_runtime_dir()]
    versions: list[dict[str, str]] = []
    seen: set[str] = set()
    for runtime_root in roots:
        if not runtime_root.is_dir():
            continue
        children = sorted(runtime_root.iterdir(), reverse=True)
        found_child = False
        for child in children:
            if child.is_dir() and _has_genai_dll(child):
                found_child = True
                bin_dir = str(genai_bin_dir(child))
                if bin_dir not in seen:
                    seen.add(bin_dir)
                    versions.append({"name": child.name, "bin": bin_dir})
        # Some packages may drop DLLs directly under the root (no version folder).
        if not found_child and _has_genai_dll(runtime_root):
            bin_dir = str(genai_bin_dir(runtime_root))
            if bin_dir not in seen:
                seen.add(bin_dir)
                versions.append({"name": runtime_root.name, "bin": bin_dir})
    return versions


def resolve_download_dir(cfg: Any) -> Path:
    """Where OpenVINO downloads (exe + runtime) land.

    Honors ``[model_service] download_dir`` when the user set one; otherwise the
    default bundle dir ``paths.ollama_openvino_dir()``. Both the exe and the
    GenAI runtime go under here so they stay together.
    """
    configured = (cfg.model_service.download_dir or "").strip()
    return Path(configured).expanduser() if configured else paths.ollama_openvino_dir()


def obtain_openvino_exe(
    exe_src: str, dest_dir: Path, progress: ProgressCb | None = None
) -> Path:
    """Place an OpenVINO ``ollama.exe`` into ``dest_dir`` and return its path.

    ``exe_src`` may be an **http(s) URL** (downloaded with progress) or a **local
    path** (copied in). Either way the result is ``dest_dir/ollama.exe`` so the
    rest of the flow has one stable location. Idempotent: copying a file onto
    itself is skipped.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_exe = dest_dir / "ollama.exe"
    src = (exe_src or "").strip()
    if not src:
        raise ValueError("no ollama.exe path or URL provided")
    if _looks_like_url(src):
        download_file(src, dest_exe, progress=progress)
    else:
        exe = validate_exe_path(src)
        if exe.resolve() != dest_exe.resolve():
            shutil.copy2(exe, dest_exe)
    return dest_exe


def download_genai(
    dest_dir: Path, progress: ProgressCb | None = None, url: str | None = None
) -> Path:
    """Download + extract the GenAI runtime into ``dest_dir/runtime``; return its bin dir.

    ``url`` overrides the default GenAI URL so the user can fetch a specific
    version; each version extracts to its own subfolder under ``runtime/``.
    """
    return install_genai_runtime(progress=progress, dest_dir=dest_dir / "runtime", url=url)


def _has_genai_dll(root: Path) -> bool:
    """True if an ``openvino_genai*.dll`` already exists under ``root``."""
    return root.is_dir() and next(root.rglob("openvino_genai*.dll"), None) is not None


def genai_bin_dir(runtime_root: Path) -> Path:
    """Locate the GenAI runtime DLL dir under ``runtime_root`` (``runtime/bin``).

    The Windows package nests the DLLs in ``<pkg>/runtime/bin/intel64/Release``
    (and a parallel ``…/Debug``). Return the dir holding ``openvino_genai*.dll``,
    **preferring Release over Debug** (Debug DLLs are slower and need the debug
    CRT); else the first ``runtime/bin`` ancestor; else ``runtime_root``.
    """
    if not runtime_root.is_dir():
        return runtime_root
    dll_dirs = {dll.parent for dll in runtime_root.rglob("openvino_genai*.dll")}
    if dll_dirs:
        # Prefer a Release dir; fall back to any (sorted for determinism).
        release = [d for d in dll_dirs if d.name.lower() == "release"]
        return release[0] if release else sorted(dll_dirs)[0]
    for bin_dir in runtime_root.rglob("bin"):
        if bin_dir.is_dir() and bin_dir.parent.name == "runtime":
            return bin_dir
    return runtime_root


# ── Locate / validate an exe ─────────────────────────────────────────────────
def validate_exe_path(raw: str) -> Path:
    """Validate a user-supplied ``ollama.exe`` path; return the resolved Path.

    Expands ``~``, resolves the path, and requires an existing regular file
    (and a ``.exe`` suffix on Windows) so a typo or a directory can't be
    launched. Raises ``ValueError`` with an actionable message otherwise. This
    is a local user-chosen file, not a sandbox boundary.
    """
    text = (raw or "").strip().strip('"')
    if not text:
        raise ValueError("no executable path provided")
    candidate = Path(text).expanduser()
    try:
        candidate = candidate.resolve()
    except OSError as exc:
        raise ValueError(f"invalid path: {exc}") from exc
    if not candidate.is_file():
        raise ValueError(f"not a file: {candidate}")
    if sys.platform == "win32" and candidate.suffix.lower() != ".exe":
        raise ValueError(f"expected a .exe file, got: {candidate.name}")
    return candidate


def resolve_exe(cfg: Any) -> tuple[str, Path | None]:
    """Return ``(backend, exe_path)`` for the configured backend.

    Official: the auto-downloaded exe under ``ollama_official_dir`` (or None if
    not installed yet). OpenVINO: the user-supplied path validated, or None if
    unset/invalid (callers surface the friendly error on launch).
    """
    ms = cfg.model_service
    backend = ms.backend
    if backend == BACKEND_OPENVINO:
        if ms.ollama_exe_path:
            try:
                return backend, validate_exe_path(ms.ollama_exe_path)
            except ValueError:
                pass
        # Fall back to the downloaded exe in the download dir if config is stale.
        downloaded = resolve_download_dir(cfg) / "ollama.exe"
        return backend, downloaded if downloaded.is_file() else None
    return BACKEND_OFFICIAL, find_exe_in_dir(paths.ollama_official_dir())


# ── Launch env ───────────────────────────────────────────────────────────────
def _host_from_base(base: str) -> str:
    """Derive an ``OLLAMA_HOST`` (``host:port``) from a base URL."""
    parsed = urlparse(base or DEFAULT_BASE)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 11434
    return f"{host}:{port}"


def build_launch_env(cfg: Any, backend: str, exe: Path) -> dict[str, str]:
    """Build the environment for ``ollama serve`` for the given backend.

    Always sets ``OLLAMA_HOST`` from ``[ollama] base`` and ``OLLAMA_REGISTRY``
    when a custom source is configured.

    For the OpenVINO backend the build needs its GenAI/OpenVINO + TBB DLLs on the
    environment. We get that env the most reliable way: **actually run the
    package's ``setupvars.bat`` and capture the environment it produces**
    (:func:`capture_setupvars_env`), then layer our ``OLLAMA_*`` on top. If
    ``setupvars.bat`` can't be found/run (or non-Windows), we fall back to
    prepending the runtime DLL dir **and** the separate TBB dir to ``PATH``
    manually (:func:`_ov_runtime_path_dirs`).
    """
    registry = (cfg.model_service.registry or "").strip()

    def _with_ours(env: dict[str, str]) -> dict[str, str]:
        env["OLLAMA_HOST"] = _host_from_base(cfg.ollama.base)
        if registry:
            env["OLLAMA_REGISTRY"] = registry
        return env

    if backend == BACKEND_OPENVINO and sys.platform == "win32":
        setupvars = find_setupvars(cfg)
        if setupvars is not None:
            captured = capture_setupvars_env(setupvars)
            if captured:
                logger.info("captured OpenVINO env from %s", setupvars)
                return _with_ours(captured)
            logger.warning("setupvars.bat at %s produced no env; falling back", setupvars)

    env = os.environ.copy()
    if backend == BACKEND_OPENVINO:
        rt = _openvino_runtime_bin(cfg)
        if rt:
            dirs = _ov_runtime_path_dirs(Path(rt))
            env["PATH"] = os.pathsep.join(dirs) + os.pathsep + env.get("PATH", "")
    return _with_ours(env)


def find_setupvars(cfg: Any) -> Path | None:
    """Locate the active OpenVINO GenAI runtime's ``setupvars.bat``, if present.

    The package ships ``setupvars.bat`` at its archive root; we ascend from the
    DLL dir to find it, then fall back to scanning the download ``runtime/`` tree.
    """
    rt = _openvino_runtime_bin(cfg)
    if rt:
        cur = Path(rt)
        for _ in range(8):
            cand = cur / "setupvars.bat"
            if cand.is_file():
                return cand
            if cur.parent == cur:
                break
            cur = cur.parent
    runtime_root = resolve_download_dir(cfg) / "runtime"
    if runtime_root.is_dir():
        try:
            return next(runtime_root.rglob("setupvars.bat"), None)
        except OSError:
            return None
    return None


def capture_setupvars_env(setupvars: Path) -> dict[str, str]:
    """Run ``setupvars.bat`` in a cmd shell and capture the environment it sets.

    "Sources" the batch file then dumps the resulting environment via ``set``,
    parsing every ``KEY=VALUE`` line back into a dict (this is how you carry a
    Windows .bat's environment into a child process). ``setupvars.bat`` puts the
    OpenVINO + TBB lib dirs on ``PATH`` (via ``OPENVINO_LIB_PATHS``) and sets
    ``OpenVINO_DIR`` etc., so the captured dict is exactly the env ``ollama
    serve`` needs. Returns ``{}`` on any failure so the caller can fall back.

    Invocation detail (load-bearing): we use ``shell=True`` with the command
    ``cmd /c ""<bat>" && set"`` — the **doubled** outer quotes are required.
    ``cmd /c`` strips one layer of quotes, so a singly-quoted path with
    backslashes (or spaces) is otherwise mangled into "'...' is not recognized".
    """
    if sys.platform != "win32":
        return {}
    # Double-quote the path, then wrap the whole "<quoted-bat> && set" in another
    # pair so cmd /c's quote-stripping leaves a valid command. Verified to be the
    # only reliable form for paths with backslashes/dots/spaces.
    command = f'cmd /c ""{setupvars}" && set"'
    try:
        proc = subprocess.run(  # noqa: S602
            command,
            shell=True,
            capture_output=True,
            timeout=60,
            cwd=str(setupvars.parent),
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("running setupvars.bat failed: %s", exc)
        return {}
    if proc.returncode != 0:
        logger.warning(
            "setupvars.bat exited %s: %s",
            proc.returncode,
            (proc.stderr or b"").decode("utf-8", "replace")[:200].strip(),
        )
        return {}
    # `set` output is in the console codepage; keys/paths are ASCII, so decode
    # leniently. Keep only real ``KEY=VALUE`` lines (skip cmd's ``=C:`` pseudo
    # vars and any echo noise like "[setupvars.bat] ... initialized").
    text = (proc.stdout or b"").decode("utf-8", "replace")
    env: dict[str, str] = {}
    for line in text.splitlines():
        if not line or line.startswith("="):
            continue
        key, sep, val = line.partition("=")
        # A valid env var name has no spaces; this drops echoed status lines.
        if sep and key and " " not in key:
            env[key] = val
    return env


def _ov_runtime_path_dirs(bin_dir: Path) -> list[str]:
    """All dirs that must be on ``PATH`` for the OpenVINO GenAI runtime to load.

    ``setupvars.bat`` adds more than the main DLL folder. We mirror the
    runtime-critical ones: the OpenVINO/GenAI DLL dir (``bin_dir``, where
    ``openvino_genai.dll`` lives) **plus** the Intel TBB dir
    (``runtime/3rdparty/tbb/bin``, where ``tbb12.dll`` lives) — a separate
    dependency in its own folder that OpenVINO needs at load time. ``bin_dir`` is
    always first (so it stays the primary DLL source); the TBB dir is appended
    when found by walking up to the runtime root and locating ``tbb*.dll``.
    """
    dirs = [str(bin_dir)]
    # Ascend from the DLL dir to the package's runtime root (the folder holding
    # ``runtime/`` or ``3rdparty/``), then locate the TBB dir under it.
    root: Path | None = None
    cur = bin_dir
    for _ in range(6):
        if (cur / "3rdparty").is_dir() or (cur / "runtime").is_dir():
            root = cur
            break
        if cur.parent == cur:
            break
        cur = cur.parent
    if root is not None:
        for name in ("tbb12.dll", "tbbmalloc.dll", "tbb.dll"):
            try:
                hit = next(root.rglob(name), None)
            except OSError:
                hit = None
            if hit is not None:
                tbb = str(hit.parent)
                if tbb not in dirs:
                    dirs.append(tbb)
                break
    return dirs


def _openvino_runtime_bin(cfg: Any) -> str:
    """Resolve the GenAI runtime DLL dir to put on PATH for the OV build.

    Prefers an explicit ``genai_runtime_dir`` config (the version the user
    selected); otherwise the newest installed version under the download dir's
    ``runtime/``, then the legacy standalone download location.
    """
    configured = (cfg.model_service.genai_runtime_dir or "").strip()
    if configured:
        return configured
    versions = list_genai_versions(cfg)
    if versions:
        return versions[0]["bin"]  # newest first
    return str(genai_bin_dir(paths.genai_runtime_dir()))


# ── PID file + liveness ──────────────────────────────────────────────────────
def read_pid_file() -> dict | None:
    """Load the PID/metadata JSON, tolerating a missing or corrupt file."""
    path = paths.modelsvc_pid_file()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def write_pid_file(data: dict) -> None:
    paths.modelsvc_pid_file().write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )


def clear_pid_file() -> None:
    try:
        paths.modelsvc_pid_file().unlink()
    except OSError:
        pass


def pid_alive(pid: int) -> bool:
    """Best-effort liveness check for ``pid`` (True if the process exists)."""
    if not pid or pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            out = subprocess.run(  # noqa: S603
                ["tasklist", "/FI", f"PID eq {int(pid)}", "/NH"],
                capture_output=True, text=True, timeout=5, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return str(int(pid)) in (out.stdout or "")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


# ── Status probe ─────────────────────────────────────────────────────────────
def probe_running(base: str, timeout: int = 2) -> dict | None:
    """Return ``/api/tags`` JSON if Ollama answers at ``base``, else ``None``."""
    from ..engine import llm  # noqa: PLC0415  (avoid import cycle at module load)

    try:
        return llm.http_get(f"{base}/api/tags", timeout=timeout)
    except Exception:  # noqa: BLE001  (any transport/HTTP failure => not running)
        return None


def _probe_version(base: str, timeout: int = 2) -> str:
    from ..engine import llm  # noqa: PLC0415

    try:
        data = llm.http_get(f"{base}/api/version", timeout=timeout)
        return str(data.get("version") or "") if isinstance(data, dict) else ""
    except Exception:  # noqa: BLE001
        return ""


def _classify_ollama_process(proc: Any, cfg: Any = None) -> str:
    """Classify a running ollama process as "openvino" / "official" / "".

    Identifies the OpenVINO build (which can live at any user-chosen path) by
    several signals, strongest first:
      1. Exe path is the configured OpenVINO exe, or under DeskMate's OpenVINO
         bundle dir, or the path/name contains "openvino".
      2. Exe path is under DeskMate's official bundle dir.
      3. The process environment carries OpenVINO markers
         (INTEL_OPENVINO_DIR / OPENVINO_LIB_PATHS) — the OV build always loads
         the GenAI runtime, so these are present; the official build has none.
      4. A loaded module named like the OpenVINO GenAI runtime.
    Anything that answers /api/tags but shows no OV markers is treated as the
    official build.
    """
    try:
        exe = (proc.exe() or "")
    except Exception:  # noqa: BLE001
        exe = ""
    low = exe.replace("\\", "/").lower()

    # 1. Explicit OpenVINO signals on the path.
    if "openvino" in low:
        return BACKEND_OPENVINO
    try:
        ovp = Path(exe).resolve()
        if cfg is not None and cfg.model_service.ollama_exe_path:
            if ovp == Path(cfg.model_service.ollama_exe_path).resolve():
                return BACKEND_OPENVINO
        ov_dir = paths.ollama_openvino_dir().resolve()
        if ov_dir in ovp.parents or ovp == ov_dir / "ollama.exe":
            return BACKEND_OPENVINO
    except OSError:
        pass

    # 2. DeskMate's official bundle dir.
    try:
        off_dir = paths.ollama_official_dir().resolve()
        op = Path(exe).resolve()
        if off_dir in op.parents or op == off_dir / "ollama.exe":
            return BACKEND_OFFICIAL
    except OSError:
        pass

    # 3. OpenVINO environment markers on the process.
    try:
        env = proc.environ() or {}
        if any(k in env for k in ("INTEL_OPENVINO_DIR", "OPENVINO_LIB_PATHS", "OpenVINO_DIR")):
            return BACKEND_OPENVINO
    except Exception:  # noqa: BLE001
        pass

    # 4. Loaded OpenVINO GenAI runtime DLL (best effort).
    try:
        for m in proc.memory_maps():
            mpath = (getattr(m, "path", "") or "").lower()
            if "openvino_genai" in mpath or "openvino" in mpath:
                return BACKEND_OPENVINO
    except Exception:  # noqa: BLE001
        pass

    # An ollama process with no OpenVINO signal is the official build.
    return BACKEND_OFFICIAL


def detect_running_backend(base: str, pid: int | None = None, cfg: Any = None) -> str:
    """Best-effort identify which backend's process is actually serving.

    Strategy, in order:
      1. If we have the launched PID, classify that process.
      2. Otherwise classify whoever is listening on the ollama port. This
         catches services we didn't start (no PID file) and OpenVINO exes at
         arbitrary user paths.

    Returns "openvino" / "official" / "" (no process found / psutil missing).
    """
    try:
        import psutil  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return ""

    # 1. Known PID → classify it.
    if pid:
        try:
            return _classify_ollama_process(psutil.Process(int(pid)), cfg)
        except Exception:  # noqa: BLE001
            pass

    # 2. Whoever is listening on the port.
    port = urlparse(base or DEFAULT_BASE).port or 11434
    try:
        for c in psutil.net_connections(kind="inet"):
            if c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port == port and c.pid:
                try:
                    return _classify_ollama_process(psutil.Process(c.pid), cfg)
                except Exception:  # noqa: BLE001
                    continue
    except Exception:  # noqa: BLE001
        return ""
    return ""


def status(cfg: Any) -> dict[str, Any]:
    """Combined service status: HTTP probe + PID file + install state."""
    base = cfg.ollama.base or DEFAULT_BASE
    tags = probe_running(base)
    running = tags is not None
    models: list[str] = []
    if running and isinstance(tags, dict):
        models = [m.get("name") for m in (tags.get("models") or []) if m.get("name")]

    pid_info = read_pid_file()
    pid = int(pid_info.get("pid")) if pid_info and pid_info.get("pid") else None
    alive = pid_alive(pid) if pid else False
    managed = bool(pid_info) and alive
    # Running but not started/owned by us => an external Ollama (don't kill it).
    external = running and not managed

    # Which backend is the *currently running* service? Identify it from the
    # actual process on disk (its exe path), not the configured backend — that's
    # the only reliable signal and works even for a service we didn't start.
    # Fall back to the PID file's recorded backend if process inspection fails.
    running_backend = ""
    if running:
        running_backend = detect_running_backend(base, pid, cfg)
        if not running_backend and managed and pid_info:
            running_backend = str(pid_info.get("backend") or "")

    backend, exe = resolve_exe(cfg)
    dl_dir = resolve_download_dir(cfg)
    ov_exe_ready = (dl_dir / "ollama.exe").is_file() or bool(
        cfg.model_service.ollama_exe_path
        and Path(cfg.model_service.ollama_exe_path).is_file()
    )
    ov_runtime_ready = _runtime_installed(cfg)
    return {
        "running": running,
        "backend": cfg.model_service.backend,
        # The backend of the service currently running under DeskMate's PID
        # ("openvino" | "official" | ""). The UI uses this to mark which panel
        # owns the running service and its log.
        "running_backend": running_backend,
        "exe": str(exe) if exe else "",
        "base": base,
        "pid": pid,
        "pid_alive": alive,
        "managed_by_deskmate": managed,
        "external": external,
        "registry": cfg.model_service.registry or "",
        "download_dir": str(dl_dir),
        "official_installed": find_exe_in_dir(paths.ollama_official_dir()) is not None,
        # OpenVINO: the exe and the GenAI runtime are tracked independently
        # (two separate downloads); "ready" means both are present.
        "openvino_exe_ready": ov_exe_ready,
        "genai_runtime_installed": ov_runtime_ready,
        "openvino_ready": ov_exe_ready and ov_runtime_ready,
        # Installed GenAI versions (newest first) + which one is on PATH, so the
        # UI can offer a version picker when more than one is present.
        "genai_versions": list_genai_versions(cfg),
        "genai_selected": _openvino_runtime_bin(cfg) if ov_runtime_ready else "",
        "genai_url": cfg.model_service.genai_url or "",
        "models": models,
        # The model Ask / apps actually use ([ollama] model) — what the user
        # "chose to use", surfaced here so the Model step can show + switch it.
        "active_model": cfg.ollama.model or "",
        "version": _probe_version(base) if running else "",
    }


def _runtime_installed(cfg: Any) -> bool:
    """True if a GenAI runtime is present — in the download dir or legacy dir."""
    return (
        _has_genai_dll(resolve_download_dir(cfg) / "runtime")
        or (
            bool(cfg.model_service.genai_runtime_dir)
            and _has_genai_dll(Path(cfg.model_service.genai_runtime_dir))
        )
        or _has_genai_dll(paths.genai_runtime_dir())
    )


# ── Start / stop ─────────────────────────────────────────────────────────────
def start_service(cfg: Any) -> dict[str, Any]:
    """Launch ``ollama serve`` detached; return :func:`status`.

    Idempotent: if something is already answering at the endpoint we just
    report status. Raises ``ValueError`` (mapped to HTTP 400 by the API) when
    the backend's executable can't be resolved.
    """
    base = cfg.ollama.base or DEFAULT_BASE
    if probe_running(base) is not None:
        logger.info("ollama already running at %s; start is a no-op", base)
        return status(cfg)

    backend, exe = resolve_exe(cfg)
    if exe is None:
        if backend == BACKEND_OPENVINO:
            raise ValueError(
                "No OpenVINO ollama.exe configured. Set its path on the Model "
                "Service page first."
            )
        raise ValueError(
            "Official Ollama is not installed yet. Click Download first."
        )

    env = build_launch_env(cfg, backend, exe)
    paths.logs_dir().mkdir(parents=True, exist_ok=True)
    # Truncate (mode "w") so each launch shows a fresh log for this backend,
    # written to that backend's own file.
    log_fh = open(paths.modelsvc_log_file(backend), "w", encoding="utf-8")  # noqa: SIM115
    try:
        if sys.platform == "win32":
            flags = (
                subprocess.DETACHED_PROCESS
                | subprocess.CREATE_NEW_PROCESS_GROUP
                | subprocess.CREATE_NO_WINDOW
            )
            proc = subprocess.Popen(  # noqa: S603
                [str(exe), "serve"],
                env=env, stdout=log_fh, stderr=log_fh,
                cwd=str(exe.parent), creationflags=flags, close_fds=True,
            )
        else:
            proc = subprocess.Popen(  # noqa: S603
                [str(exe), "serve"],
                env=env, stdout=log_fh, stderr=log_fh,
                cwd=str(exe.parent), start_new_session=True, close_fds=True,
            )
    finally:
        log_fh.close()

    write_pid_file({
        "pid": proc.pid,
        "backend": backend,
        "exe": str(exe),
        "started_at": time.time(),
    })
    logger.info("launched ollama serve (backend=%s pid=%s)", backend, proc.pid)

    # Give it a moment to bind the port before reporting back.
    deadline = time.time() + 12
    while time.time() < deadline:
        if probe_running(base) is not None:
            break
        # If the process already died, stop waiting — it failed to launch.
        if proc.poll() is not None:
            break
        time.sleep(0.5)

    result = status(cfg)
    if not result.get("running"):
        # Surface WHY: the child's exit code + the tail of its log, so the UI
        # can show an actionable message instead of a silent "still stopped".
        result["start_error"] = _startup_failure_detail(proc)
    return result


def _startup_failure_detail(proc: subprocess.Popen) -> str:
    """Build a short reason the just-launched service isn't answering yet."""
    rc = proc.poll()
    tail = ""
    try:
        log = paths.modelsvc_log_file()
        if log.is_file():
            lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = " ".join(ln.strip() for ln in lines[-8:] if ln.strip())[-600:]
    except OSError:
        pass
    if rc is not None and rc != 0:
        base = f"ollama exited with code {rc}."
    elif rc is None:
        base = "ollama is still starting (not answering yet)."
    else:
        base = "ollama started but the API did not come up."
    return f"{base} {('Log: ' + tail) if tail else ''}".strip()


def stop_service(cfg: Any) -> dict[str, Any]:
    """Stop a DeskMate-launched service; return :func:`status`.

    Only acts on a process we started (PID file) and that is alive — we never
    kill an external Ollama. The PID is an int we wrote and is passed as a fixed
    argv list (no shell), so there is no injection surface.
    """
    pid_info = read_pid_file()
    pid = int(pid_info.get("pid")) if pid_info and pid_info.get("pid") else None
    if pid and pid_alive(pid):
        if sys.platform == "win32":
            subprocess.run(  # noqa: S603
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True, text=True, timeout=10, check=False,
            )
        else:
            import signal  # noqa: PLC0415

            try:
                os.kill(pid, signal.SIGTERM)
                for _ in range(10):
                    if not pid_alive(pid):
                        break
                    time.sleep(0.3)
                if pid_alive(pid):
                    os.kill(pid, signal.SIGKILL)
            except OSError as exc:
                logger.warning("stop_service: kill %s failed: %s", pid, exc)
        logger.info("stopped ollama serve (pid=%s)", pid)
    clear_pid_file()
    return status(cfg)


# ── Service log ──────────────────────────────────────────────────────────────
def read_service_log(
    max_lines: int = 400, max_bytes: int = 200_000, backend: str | None = None
) -> str:
    """Return the tail of a launched Ollama service's stdout/stderr log.

    This is ``ollama serve``'s own output (model type, inference device, request
    handling, errors) — the file we redirect the detached process into — so the
    UI can surface what the service is actually doing. Each backend has its own
    log; pass ``backend`` ("openvino"/"official") to read that one. Reads only
    the last ``max_bytes`` and keeps the last ``max_lines`` to stay cheap on a
    long log. Returns ``""`` when there is no log yet.
    """
    log = paths.modelsvc_log_file(backend)
    if not log.is_file():
        return ""
    try:
        size = log.stat().st_size
        with log.open("rb") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
                fh.readline()  # drop the partial first line after seeking
            data = fh.read()
    except OSError:
        return ""
    text = data.decode("utf-8", "replace")
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


# ── Pull a model (streaming) ─────────────────────────────────────────────────
def pull_model_stream(cfg: Any, model: str) -> Iterator[dict[str, Any]]:
    """Yield Ollama ``/api/pull`` NDJSON progress objects for ``model``.

    Uses a raw ``http.client`` connection (proxy-bypassing, like
    :mod:`deskmate.engine.llm`) and reads the response line-by-line so the UI
    sees ``{status, digest, total, completed}`` updates live — unlike
    ``llm.raw_request`` which buffers the whole body. Errors are surfaced as a
    final ``{"status": "error", "error": ...}`` object instead of raising.
    """
    base = cfg.ollama.base or DEFAULT_BASE
    parsed = urlparse(base)
    # Self-hosted registries are usually plain HTTP; without insecure=true Ollama
    # assumes HTTPS for /api/pull and the pull fails (often silently) against an
    # http:// registry. Default on via [model_service] pull_insecure.
    insecure = True
    try:
        insecure = bool(cfg.model_service.pull_insecure)
    except Exception:
        insecure = True
    body = json.dumps(
        {"model": model, "stream": True, "insecure": insecure}
    ).encode("utf-8")
    conn = http.client.HTTPConnection(
        parsed.hostname or "127.0.0.1", parsed.port or 11434, timeout=600
    )
    try:
        conn.request(
            "POST", "/api/pull", body=body,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        if resp.status >= 400:
            detail = resp.read().decode("utf-8", "replace")[:300].strip()
            yield {"status": "error", "error": f"HTTP {resp.status}: {detail}"}
            return
        for raw_line in resp:
            line = raw_line.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
    except Exception as exc:  # noqa: BLE001
        yield {"status": "error", "error": str(exc)}
    finally:
        conn.close()
