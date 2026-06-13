"""DeskMate self-diagnostics ("doctor").

A small, dependency-light health check that surfaces the environment problems
DeskMate users actually hit: which Ollama backend is running and whether its
GenAI runtime is a known-bad version, whether the active model is pullable,
whether winrt (toast) is importable, whether an HTTP proxy is hijacking
localhost, plus DB/recording/connector state.

Design mirrors a classic `doctor` command: every check returns a uniform
:class:`CheckResult` (name / status / message / fix-hint), so the API and any
CLI can render them the same way. Checks never raise — a failing check is
reported as a ``fail`` result, not an exception, so one broken probe can't take
down the whole report.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable

# ── Result type ──────────────────────────────────────────────────────────────

OK = "ok"
WARN = "warn"
FAIL = "fail"


@dataclass
class CheckResult:
    """One diagnostic line. ``status`` is "ok" | "warn" | "fail"; ``fix`` is an
    optional actionable hint shown to the user when not ok."""

    name: str
    status: str
    message: str
    fix: str | None = None


# ── GenAI runtime version policy ──────────────────────────────────────────────

# Minimum acceptable OpenVINO GenAI nightly build. Builds before this date had a
# long-context output-degradation bug on the Intel GPU plugin (garbled Ask
# answers); dev20260530 was the first to resolve it. We compare the actually
# loaded build's date against this threshold, so ANY older nightly is flagged —
# not just one hard-coded bad version.
_MIN_GENAI_BUILD = "dev20260530"

# Matches the "devYYYYMMDD" stamp inside a GenAI runtime path/version string,
# e.g. "openvino_genai_windows_2026.3.0.0.dev20260530_x86_64".
_GENAI_BUILD_RE = re.compile(r"dev(\d{8})")


def _genai_build_date(text: str) -> int | None:
    """Extract the devYYYYMMDD build date as an int (e.g. 20260530) from a GenAI
    runtime path/version string, or None if absent."""
    if not text:
        return None
    m = _GENAI_BUILD_RE.search(text)
    return int(m.group(1)) if m else None


# ── Individual checks ─────────────────────────────────────────────────────────
#
# Each check takes the loaded cfg and returns a CheckResult (or list of them).
# They must not raise: wrap risky work in try/except and degrade to a result.


def _check_api(cfg: Any) -> CheckResult:
    """Is the DeskMate API itself reachable? (We're usually running inside it,
    but this confirms the configured host:port answers.)"""
    base = f"http://{cfg.server.host}:{cfg.server.port}"
    try:
        from . import llm  # noqa: PLC0415

        llm.http_get(f"{base}/health", timeout=3)
        return CheckResult("DeskMate API", OK, f"reachable at {base}")
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            "DeskMate API", WARN, f"not reachable at {base} ({type(exc).__name__})",
            fix="Start the daemon (python -m deskmate.engine.cli serve).",
        )


def _check_ollama(cfg: Any) -> CheckResult:
    """Is an Ollama service answering at the configured base?"""
    from ..modelsvc import service as modelsvc  # noqa: PLC0415

    base = cfg.ollama.base
    tags = modelsvc.probe_running(base)
    if tags is None:
        return CheckResult(
            "Ollama service", FAIL, f"no service answering at {base}",
            fix="Open Model Service and start a backend (OpenVINO or official).",
        )
    n = len((tags.get("models") or [])) if isinstance(tags, dict) else 0
    return CheckResult("Ollama service", OK, f"reachable at {base} ({n} model(s))")


def _check_running_backend(cfg: Any) -> CheckResult:
    """Which backend is actually running, and is its GenAI runtime a known-bad
    build? This is the check that catches the 'garbled Ask output' bug."""
    from ..modelsvc import service as modelsvc  # noqa: PLC0415

    base = cfg.ollama.base
    if modelsvc.probe_running(base) is None:
        return CheckResult("Running backend", WARN, "no service running", fix=None)

    backend = modelsvc.detect_running_backend(base, None, cfg)
    if not backend:
        return CheckResult(
            "Running backend", WARN,
            "a service is running but its backend couldn't be identified",
        )
    if backend != modelsvc.BACKEND_OPENVINO:
        return CheckResult("Running backend", OK, f"{backend}")

    # OpenVINO: compare the ACTUALLY LOADED GenAI runtime build against the
    # minimum acceptable date. Read it from the running process env
    # (INTEL_OPENVINO_DIR) — that's the truth of what's loaded, which can differ
    # from the config-selected runtime on disk. Fall back to the configured
    # runtime path only if the process env couldn't be read.
    runtime = ""
    try:
        runtime = modelsvc.running_openvino_runtime_dir(base) or ""
    except Exception:  # noqa: BLE001
        runtime = ""
    if not runtime:
        try:
            runtime = modelsvc._openvino_runtime_bin(cfg) or ""
        except Exception:  # noqa: BLE001
            runtime = ""

    have = _genai_build_date(runtime)
    want = _genai_build_date(_MIN_GENAI_BUILD)
    if have is None:
        return CheckResult(
            "Running backend", OK,
            "openvino (GenAI runtime build not detected)",
        )
    if want is not None and have < want:
        return CheckResult(
            "Running backend", WARN,
            f"OpenVINO GenAI runtime dev{have} is older than dev{want} — builds "
            "before that garble long-context output on the Intel GPU plugin",
            fix=f"Update the GenAI runtime to {_MIN_GENAI_BUILD} or newer in "
            "Model Service (or set [model_service] genai_runtime_dir).",
        )
    return CheckResult("Running backend", OK, f"openvino (GenAI dev{have})")


def _check_active_model(cfg: Any) -> CheckResult:
    """Is the configured active model actually installed on the running service?"""
    from ..modelsvc import service as modelsvc  # noqa: PLC0415

    model = (cfg.ollama.model or "").strip()
    if not model:
        return CheckResult(
            "Active model", WARN, "no [ollama] model configured",
            fix="Pick an active model in Model Service.",
        )
    tags = modelsvc.probe_running(cfg.ollama.base)
    if tags is None:
        return CheckResult("Active model", WARN, f"configured: {model} (service not running)")
    names = [m.get("name") for m in (tags.get("models") or []) if isinstance(m, dict)]

    # Validate the name against /api/chat, not just /api/tags. Some Ollama builds
    # (the OpenVINO fork) reject a short name like "qwen3.5_4b_ov:v1" with
    # {"error":"invalid model name"} on /api/chat even though it shows in the tag
    # list — they require a fully-qualified "host/namespace/model:tag". This is
    # exactly the failure that breaks Ask, so probe for it here.
    try:
        from . import llm  # noqa: PLC0415

        body = {"model": model, "messages": [{"role": "user", "content": "ok"}],
                "stream": False, "options": {"num_predict": 1}}
        llm.http_post(f"{cfg.ollama.base}/api/chat", body, timeout=8)
        return CheckResult("Active model", OK, model)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "invalid model name" in msg.lower():
            suggestion = ""
            for n in names:
                if n.endswith("/" + model) or n.split("/")[-1] == model:
                    suggestion = n
                    break
            fix = (f'Use the fully-qualified name'
                   + (f' "{suggestion}"' if suggestion else
                      ' (host/namespace/model:tag, e.g. registry.ollama.ai/library/<model>:<tag>)')
                   + " — this Ollama build rejects bare short names on /api/chat.")
            return CheckResult(
                "Active model", FAIL,
                f'"{model}" is rejected by /api/chat ("invalid model name")', fix=fix)
        # A timeout or other error means the name was ACCEPTED (model is just
        # loading/slow) — that's not a name problem.
        if model in names:
            return CheckResult("Active model", OK, f"{model} (accepted; service busy/loading)")
        return CheckResult(
            "Active model", WARN, f"{model} not in the installed list",
            fix=f"Pull it (e.g. ollama pull {model}) or choose an installed model.",
        )


def _check_genai_runtime(cfg: Any) -> CheckResult:
    """Is an OpenVINO GenAI runtime installed (needed by the OpenVINO backend)?"""
    from ..modelsvc import service as modelsvc  # noqa: PLC0415

    try:
        installed = modelsvc._runtime_installed(cfg)
    except Exception as exc:  # noqa: BLE001
        return CheckResult("GenAI runtime", WARN, f"could not determine ({type(exc).__name__})")
    if installed:
        return CheckResult("GenAI runtime", OK, "installed")
    return CheckResult(
        "GenAI runtime", WARN, "not installed (only needed for the OpenVINO backend)",
        fix="Download the GenAI runtime in Model Service if you use OpenVINO.",
    )


def _check_winrt(cfg: Any) -> CheckResult:
    """Is winrt importable? Without it, toast reminders silently fall back to the
    in-app inbox instead of native Windows notifications."""
    import sys  # noqa: PLC0415

    if sys.platform != "win32":
        return CheckResult("Windows toasts (winrt)", OK, "not applicable (non-Windows)")
    try:
        import winrt.windows.ui.notifications  # noqa: F401, PLC0415
        return CheckResult("Windows toasts (winrt)", OK, "available")
    except Exception:  # noqa: BLE001
        return CheckResult(
            "Windows toasts (winrt)", WARN,
            "winrt not importable — reminders fall back to the in-app inbox",
            fix="Install winrt in the daemon's environment (it's a core dependency).",
        )


def _check_proxy(cfg: Any) -> CheckResult:
    """Is an HTTP proxy set that does NOT bypass localhost? On corporate machines
    a proxy can hijack 127.0.0.1 requests (we've seen it return IE error pages),
    breaking local Ollama/registry calls made with plain clients."""
    proxies = {k: os.environ.get(k) for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")}
    active = {k: v for k, v in proxies.items() if v}
    if not active:
        return CheckResult("HTTP proxy", OK, "no proxy set")
    no_proxy = (os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or "").lower()
    bypasses_local = any(t in no_proxy for t in ("127.0.0.1", "localhost"))
    if bypasses_local:
        return CheckResult("HTTP proxy", OK, "proxy set but localhost is bypassed (NO_PROXY)")
    sample = next(iter(active.values()))
    return CheckResult(
        "HTTP proxy", WARN,
        f"proxy set ({sample}) without localhost in NO_PROXY — may hijack local calls",
        fix="Add 127.0.0.1,localhost to NO_PROXY. (DeskMate's own calls bypass the "
        "proxy, but external tools/curl may not.)",
    )


def _check_recording(cfg: Any, db: Any = None) -> CheckResult:
    """Is capture actually producing data?"""
    if db is None:
        return CheckResult("Recording", WARN, "database handle unavailable")
    try:
        stats = db.health()
    except Exception as exc:  # noqa: BLE001
        return CheckResult("Recording", WARN, f"could not read capture stats ({type(exc).__name__})")
    frames = int(stats.get("frames") or 0)
    if frames > 0:
        return CheckResult("Recording", OK, f"{frames} frame(s) captured")
    return CheckResult(
        "Recording", WARN, "no frames captured yet",
        fix="Keep DeskMate running in the background; check capture isn't paused.",
    )


def _check_schema(cfg: Any, db: Any = None) -> CheckResult:
    """Does the DB schema match the code's expected version?"""
    if db is None:
        return CheckResult("DB schema", WARN, "database handle unavailable")
    try:
        from ..db import schema as schema_mod  # noqa: PLC0415

        have = str(db.schema_version())
        want = str(schema_mod.SCHEMA_VERSION)
    except Exception as exc:  # noqa: BLE001
        return CheckResult("DB schema", WARN, f"could not read schema ({type(exc).__name__})")
    if have == want:
        return CheckResult("DB schema", OK, f"version {have}")
    return CheckResult(
        "DB schema", WARN, f"DB at {have}, code expects {want}",
        fix="Restart the daemon to apply pending migrations.",
    )


def _check_connectors(cfg: Any) -> CheckResult:
    """Are any mailbox connectors configured? (Informational — absence is fine.)
    A connector counts as configured when it has OAuth client credentials."""
    try:
        gmail = bool(getattr(getattr(cfg, "gmail", None), "client_id", ""))
        outlook = bool(getattr(getattr(cfg, "outlook", None), "client_id", ""))
    except Exception:  # noqa: BLE001
        gmail = outlook = False
    connected = [n for n, on in (("Gmail", gmail), ("Outlook", outlook)) if on]
    if connected:
        return CheckResult("Mail connectors", OK, ", ".join(connected) + " configured")
    return CheckResult(
        "Mail connectors", OK, "none configured (optional)",
        fix="Connect Gmail/Outlook on the Email page for mail-aware answers.",
    )


# Order matters: most fundamental first. Checks that need the DB take (cfg, db);
# the rest take (cfg,). run_all() dispatches by signature.
_CHECKS: list[Callable[..., CheckResult]] = [
    _check_api,
    _check_ollama,
    _check_running_backend,
    _check_active_model,
    _check_genai_runtime,
    _check_winrt,
    _check_proxy,
    _check_recording,
    _check_schema,
    _check_connectors,
]

# Checks that also need a DB handle.
_DB_CHECKS = {_check_recording, _check_schema}


# ── Orchestration ─────────────────────────────────────────────────────────────


def run_all(cfg: Any, db: Any = None) -> list[CheckResult]:
    """Run every check against ``cfg`` (and ``db`` for the ones that need it). A
    check that itself blows up becomes a ``fail`` line rather than propagating."""
    out: list[CheckResult] = []
    for check in _CHECKS:
        try:
            out.append(check(cfg, db) if check in _DB_CHECKS else check(cfg))
        except Exception as exc:  # noqa: BLE001
            out.append(CheckResult(getattr(check, "__name__", "check"), FAIL,
                                   f"check raised {type(exc).__name__}: {exc}"))
    return out


def report(cfg: Any, db: Any = None) -> dict[str, Any]:
    """Run all checks and return a JSON-serialisable report with a summary."""
    results = run_all(cfg, db)
    counts = {OK: 0, WARN: 0, FAIL: 0}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    overall = FAIL if counts[FAIL] else (WARN if counts[WARN] else OK)
    return {
        "overall": overall,
        "summary": {"ok": counts[OK], "warn": counts[WARN], "fail": counts[FAIL]},
        "checks": [asdict(r) for r in results],
    }
