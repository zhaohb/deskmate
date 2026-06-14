"""DeskMate self-diagnostics ("doctor").

A small, dependency-light health check that surfaces the environment problems
DeskMate users actually hit: which Ollama backend is running and whether its
GenAI runtime is a known-bad version, whether the active model is pullable,
whether winrt (toast) is importable, whether an HTTP proxy is hijacking
localhost, the background worker/watcher/process fleet, OCR engine availability,
plus DB/recording/connector state.

Design mirrors a classic `doctor` command: every check returns a uniform
:class:`CheckResult` (name / status / message / fix-hint), so the API and any
CLI can render them the same way. Checks never raise — a failing check is
reported as a ``fail`` result, not an exception, so one broken probe can't take
down the whole report.

Localization: all user-facing text is bilingual (English / Chinese). The active
language is held in a context variable set by :func:`report`; every string is
built with the :func:`_t` helper, which keeps both languages adjacent in the
source. The status codes ("ok"/"warn"/"fail") are language-independent.
"""

from __future__ import annotations

import contextvars
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


# ── Localization ──────────────────────────────────────────────────────────────

# Active report language ("en" | "zh"). Set per-report by report(); a context
# variable keeps it isolated across concurrent requests/threads.
_LANG: contextvars.ContextVar[str] = contextvars.ContextVar("doctor_lang", default="en")


def _t(en: str, zh: str) -> str:
    """Pick the English or Chinese string for the active report language."""
    return zh if _LANG.get() == "zh" else en


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


def _age_seconds(iso_ts: Any) -> float | None:
    """Seconds elapsed since an ISO-8601 timestamp (as written by the DB's
    _now_iso, e.g. "2026-06-14T09:30:00+08:00"), or None if it can't be parsed.

    Uses the wall clock at call time. Returns max(0, age) so minor clock skew
    can't produce a negative age that reads as 'fresh'."""
    if not iso_ts:
        return None
    from datetime import datetime, timezone  # noqa: PLC0415

    try:
        dt = datetime.fromisoformat(str(iso_ts).replace("Z", "+00:00"))
    except Exception:  # noqa: BLE001
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return max(0.0, (now - dt).total_seconds())


# ── Individual checks ─────────────────────────────────────────────────────────
#
# Each check takes the loaded cfg and returns a CheckResult (or list of them).
# They must not raise: wrap risky work in try/except and degrade to a result.


def _check_api(cfg: Any) -> CheckResult:
    """Is the DeskMate API itself reachable? (We're usually running inside it,
    but this confirms the configured host:port answers.)"""
    name = _t("DeskMate API", "DeskMate API 服务")
    base = f"http://{cfg.server.host}:{cfg.server.port}"
    try:
        from . import llm  # noqa: PLC0415

        llm.http_get(f"{base}/health", timeout=3)
        return CheckResult(name, OK, _t(f"reachable at {base}", f"可访问:{base}"))
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name, WARN,
            _t(f"not reachable at {base} ({type(exc).__name__})",
               f"无法访问 {base}({type(exc).__name__})"),
            fix=_t("Start the daemon (python -m deskmate.engine.cli serve).",
                   "启动守护进程(python -m deskmate.engine.cli serve)。"),
        )


def _check_ollama(cfg: Any) -> CheckResult:
    """Is an Ollama service answering at the configured base?"""
    from ..modelsvc import service as modelsvc  # noqa: PLC0415

    name = _t("Ollama service", "Ollama 服务")
    base = cfg.ollama.base
    tags = modelsvc.probe_running(base)
    if tags is None:
        return CheckResult(
            name, FAIL, _t(f"no service answering at {base}", f"{base} 无服务应答"),
            fix=_t("Open Model Service and start a backend (OpenVINO or official).",
                   "打开「模型服务」并启动一个后端(OpenVINO 或官方版)。"),
        )
    n = len((tags.get("models") or [])) if isinstance(tags, dict) else 0
    return CheckResult(name, OK, _t(f"reachable at {base} ({n} model(s))",
                                    f"可访问:{base}(共 {n} 个模型)"))


def _check_running_backend(cfg: Any) -> CheckResult:
    """Which backend is actually running, and is its GenAI runtime a known-bad
    build? This is the check that catches the 'garbled Ask output' bug."""
    from ..modelsvc import service as modelsvc  # noqa: PLC0415

    name = _t("Running backend", "运行中的后端")
    base = cfg.ollama.base
    if modelsvc.probe_running(base) is None:
        return CheckResult(name, WARN, _t("no service running", "没有服务在运行"), fix=None)

    backend = modelsvc.detect_running_backend(base, None, cfg)
    if not backend:
        return CheckResult(
            name, WARN,
            _t("a service is running but its backend couldn't be identified",
               "有服务在运行,但无法识别其后端类型"),
        )
    if backend != modelsvc.BACKEND_OPENVINO:
        return CheckResult(name, OK, f"{backend}")

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
            name, OK,
            _t("openvino (GenAI runtime build not detected)",
               "openvino(未检测到 GenAI 运行时构建号)"),
        )
    if want is not None and have < want:
        return CheckResult(
            name, WARN,
            _t(f"OpenVINO GenAI runtime dev{have} is older than dev{want} — builds "
               "before that garble long-context output on the Intel GPU plugin",
               f"OpenVINO GenAI 运行时 dev{have} 早于 dev{want} —— 更早的构建在 "
               "Intel GPU 插件上会把长上下文输出弄乱"),
            fix=_t(f"Update the GenAI runtime to {_MIN_GENAI_BUILD} or newer in "
                   "Model Service (or set [model_service] genai_runtime_dir).",
                   f"在「模型服务」中把 GenAI 运行时升级到 {_MIN_GENAI_BUILD} 或更新版本"
                   "(或设置 [model_service] genai_runtime_dir)。"),
        )
    return CheckResult(name, OK, f"openvino (GenAI dev{have})")


def _check_active_model(cfg: Any) -> CheckResult:
    """Is the configured active model actually installed on the running service?"""
    from ..modelsvc import service as modelsvc  # noqa: PLC0415

    name = _t("Active model", "当前模型")
    model = (cfg.ollama.model or "").strip()
    if not model:
        return CheckResult(
            name, WARN, _t("no [ollama] model configured", "未配置 [ollama] 模型"),
            fix=_t("Pick an active model in Model Service.", "在「模型服务」中选择一个当前模型。"),
        )
    tags = modelsvc.probe_running(cfg.ollama.base)
    if tags is None:
        return CheckResult(name, WARN, _t(f"configured: {model} (service not running)",
                                          f"已配置:{model}(服务未运行)"))
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
        return CheckResult(name, OK, model)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "invalid model name" in msg.lower():
            suggestion = ""
            for n in names:
                if n.endswith("/" + model) or n.split("/")[-1] == model:
                    suggestion = n
                    break
            fix = _t(
                'Use the fully-qualified name'
                + (f' "{suggestion}"' if suggestion else
                   ' (host/namespace/model:tag, e.g. registry.ollama.ai/library/<model>:<tag>)')
                + " — this Ollama build rejects bare short names on /api/chat.",
                '请使用完整模型名'
                + (f' "{suggestion}"' if suggestion else
                   '(host/namespace/model:tag,例如 registry.ollama.ai/library/<model>:<tag>)')
                + " —— 此 Ollama 构建在 /api/chat 上拒绝短名。",
            )
            return CheckResult(
                name, FAIL,
                _t(f'"{model}" is rejected by /api/chat ("invalid model name")',
                   f'"{model}" 被 /api/chat 拒绝("invalid model name")'), fix=fix)
        # A timeout or other error means the name was ACCEPTED (model is just
        # loading/slow) — that's not a name problem.
        if model in names:
            return CheckResult(name, OK, _t(f"{model} (accepted; service busy/loading)",
                                            f"{model}(名称已接受;服务繁忙/加载中)"))
        return CheckResult(
            name, WARN, _t(f"{model} not in the installed list",
                           f"{model} 不在已安装列表中"),
            fix=_t(f"Pull it (e.g. ollama pull {model}) or choose an installed model.",
                   f"拉取它(例如 ollama pull {model})或选择一个已安装的模型。"),
        )


def _check_genai_runtime(cfg: Any) -> CheckResult:
    """Is an OpenVINO GenAI runtime installed (needed by the OpenVINO backend)?"""
    from ..modelsvc import service as modelsvc  # noqa: PLC0415

    name = _t("GenAI runtime", "GenAI 运行时")
    try:
        installed = modelsvc._runtime_installed(cfg)
    except Exception as exc:  # noqa: BLE001
        return CheckResult(name, WARN, _t(f"could not determine ({type(exc).__name__})",
                                          f"无法确定({type(exc).__name__})"))
    if installed:
        return CheckResult(name, OK, _t("installed", "已安装"))
    return CheckResult(
        name, WARN, _t("not installed (only needed for the OpenVINO backend)",
                       "未安装(仅 OpenVINO 后端需要)"),
        fix=_t("Download the GenAI runtime in Model Service if you use OpenVINO.",
               "若使用 OpenVINO,请在「模型服务」中下载 GenAI 运行时。"),
    )


def _check_winrt(cfg: Any) -> CheckResult:
    """Is winrt importable? Without it, toast reminders silently fall back to the
    in-app inbox instead of native Windows notifications."""
    import sys  # noqa: PLC0415

    name = _t("Windows toasts (winrt)", "Windows 通知 (winrt)")
    if sys.platform != "win32":
        return CheckResult(name, OK, _t("not applicable (non-Windows)", "不适用(非 Windows)"))
    try:
        import winrt.windows.ui.notifications  # noqa: F401, PLC0415
        return CheckResult(name, OK, _t("available", "可用"))
    except Exception:  # noqa: BLE001
        return CheckResult(
            name, WARN,
            _t("winrt not importable — reminders fall back to the in-app inbox",
               "winrt 无法导入 —— 提醒将退回到应用内收件箱"),
            fix=_t("Install winrt in the daemon's environment (it's a core dependency).",
                   "在守护进程环境中安装 winrt(它是核心依赖)。"),
        )


def _check_proxy(cfg: Any) -> CheckResult:
    """Is an HTTP proxy set that does NOT bypass localhost? On corporate machines
    a proxy can hijack 127.0.0.1 requests (we've seen it return IE error pages),
    breaking local Ollama/registry calls made with plain clients."""
    name = _t("HTTP proxy", "HTTP 代理")
    proxies = {k: os.environ.get(k) for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")}
    active = {k: v for k, v in proxies.items() if v}
    if not active:
        return CheckResult(name, OK, _t("no proxy set", "未设置代理"))
    no_proxy = (os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or "").lower()
    bypasses_local = any(t in no_proxy for t in ("127.0.0.1", "localhost"))
    if bypasses_local:
        return CheckResult(name, OK, _t("proxy set but localhost is bypassed (NO_PROXY)",
                                        "已设代理,但 localhost 已被排除(NO_PROXY)"))
    sample = next(iter(active.values()))
    return CheckResult(
        name, WARN,
        _t(f"proxy set ({sample}) without localhost in NO_PROXY — may hijack local calls",
           f"已设代理({sample}),但 NO_PROXY 未包含 localhost —— 可能劫持本地请求"),
        fix=_t("Add 127.0.0.1,localhost to NO_PROXY. (DeskMate's own calls bypass the "
               "proxy, but external tools/curl may not.)",
               "把 127.0.0.1,localhost 加入 NO_PROXY。(DeskMate 自身的请求会绕过代理,"
               "但外部工具/curl 不一定会。)"),
    )


def _check_recording(cfg: Any, db: Any = None) -> CheckResult:
    """Is capture actually producing data?"""
    name = _t("Recording", "录制")
    if db is None:
        return CheckResult(name, WARN, _t("database handle unavailable", "数据库句柄不可用"))
    try:
        stats = db.health()
    except Exception as exc:  # noqa: BLE001
        return CheckResult(name, WARN, _t(f"could not read capture stats ({type(exc).__name__})",
                                          f"无法读取捕获统计({type(exc).__name__})"))
    frames = int(stats.get("frames") or 0)
    if frames > 0:
        return CheckResult(name, OK, _t(f"{frames} frame(s) captured", f"已捕获 {frames} 帧"))
    return CheckResult(
        name, WARN, _t("no frames captured yet", "尚未捕获任何帧"),
        fix=_t("Keep DeskMate running in the background; check capture isn't paused.",
               "让 DeskMate 在后台持续运行;确认捕获未被暂停。"),
    )


def _check_schema(cfg: Any, db: Any = None) -> CheckResult:
    """Does the DB schema match the code's expected version?"""
    name = _t("DB schema", "数据库结构版本")
    if db is None:
        return CheckResult(name, WARN, _t("database handle unavailable", "数据库句柄不可用"))
    try:
        from ..db import schema as schema_mod  # noqa: PLC0415

        have = str(db.schema_version())
        want = str(schema_mod.SCHEMA_VERSION)
    except Exception as exc:  # noqa: BLE001
        return CheckResult(name, WARN, _t(f"could not read schema ({type(exc).__name__})",
                                          f"无法读取结构版本({type(exc).__name__})"))
    if have == want:
        return CheckResult(name, OK, _t(f"version {have}", f"版本 {have}"))
    return CheckResult(
        name, WARN, _t(f"DB at {have}, code expects {want}",
                       f"数据库为 {have},代码期望 {want}"),
        fix=_t("Restart the daemon to apply pending migrations.",
               "重启守护进程以应用待执行的迁移。"),
    )


def _check_workers(cfg: Any, daemon: Any = None) -> CheckResult:
    """Are the daemon's background workers actually alive?

    The daemon runs a fleet of daemon-threads (capture, audio, retention,
    semantic-index, translate) plus sub-workers that each own a thread
    (AppScheduler, HabitWatcher, ContextFusionBus, PipeScheduler,
    RedactReconciler). A worker that crashes dies *silently* — capture or
    summarization just stops with nothing in the UI. This check enumerates the
    threads that SHOULD be running given the current config and reports any that
    aren't alive.

    Only meaningful in the in-process deployment where the API can see the
    daemon object; in a split API/daemon setup we degrade to an informational
    line rather than a false alarm.
    """
    name = _t("Background workers", "后台工作线程")
    if daemon is None:
        return CheckResult(
            name, OK,
            _t("not introspectable (API running without an in-process daemon)",
               "无法检视(API 未与守护进程同进程运行)"),
        )

    dead: list[str] = []
    alive: list[str] = []

    def _note(label: str, thread: Any) -> None:
        # A None thread means that worker isn't supposed to be running (feature
        # disabled) — skip it. A non-None thread that isn't alive is a crash.
        if thread is None:
            return
        if getattr(thread, "is_alive", lambda: False)():
            alive.append(label)
        else:
            dead.append(label)

    # The daemon's own loop threads (already filtered by config at start()).
    try:
        for t in (getattr(daemon, "_threads", None) or []):
            _note(getattr(t, "name", "thread"), t)
    except Exception:  # noqa: BLE001
        pass
    _note("daemon-translate", getattr(daemon, "_translate_thread", None))

    # Sub-workers: present on the daemon only when their feature is enabled;
    # each exposes a single ._thread. getattr keeps us safe if one is absent.
    for attr, label in (
        ("app_scheduler", "AppScheduler"),
        ("pipe_scheduler", "PipeScheduler"),
        ("habit_watcher", "HabitWatcher"),
        ("fusion_bus", "ContextFusionBus"),
        ("reconciler", "RedactReconciler"),
    ):
        worker = getattr(daemon, attr, None)
        if worker is not None:
            _note(label, getattr(worker, "_thread", None))

    if dead:
        return CheckResult(
            name, FAIL,
            _t(f"{len(dead)} worker(s) not running: {', '.join(dead)}",
               f"{len(dead)} 个工作线程未运行:{', '.join(dead)}")
            + (_t(f" (alive: {', '.join(alive)})", f"(存活:{', '.join(alive)})") if alive else ""),
            fix=_t("Restart the daemon to respawn crashed workers; check logs for "
                   "the traceback that killed them.",
                   "重启守护进程以重建崩溃的工作线程;查看日志中导致崩溃的堆栈。"),
        )
    if not alive:
        return CheckResult(
            name, WARN, _t("no background workers found running", "未发现任何后台工作线程在运行"),
            fix=_t("The daemon may still be starting — re-run shortly.",
                   "守护进程可能仍在启动 —— 稍后重试。"),
        )
    return CheckResult(name, OK, _t(f"{len(alive)} running ({', '.join(alive)})",
                                    f"{len(alive)} 个在运行({', '.join(alive)})"))


def _check_a11y_watchers(cfg: Any, daemon: Any = None) -> CheckResult:
    """Are the accessibility (a11y) capture watchers alive?

    The UI/activity capture path runs several long-lived threads that
    `_check_workers` doesn't see (it only inspects the daemon's own loop threads
    and the ._thread sub-workers): the UiRecorder's WinEvent / Input / Clipboard
    watchers, the UI-event flush pipeline, and the frame-linker actor. If one of
    these dies, UI events or frame linking silently stop while the rest of
    DeskMate looks healthy.

    Like _check_workers, this is only introspectable in the in-process
    deployment; degrade to an informational OK otherwise.
    """
    name = _t("A11y capture watchers", "无障碍捕获监视器")
    if daemon is None:
        return CheckResult(
            name, OK,
            _t("not introspectable (API running without an in-process daemon)",
               "无法检视(API 未与守护进程同进程运行)"),
        )

    dead: list[str] = []
    alive: list[str] = []

    def _note(label: str, obj: Any, attr: str = "_thread") -> None:
        # obj absent → feature/component not constructed: skip silently.
        if obj is None:
            return
        thread = getattr(obj, attr, None)
        # Some watchers expose is_alive() directly (e.g. WinEventWatcher);
        # prefer the thread handle when present, else fall back to is_alive().
        if thread is not None:
            (alive if getattr(thread, "is_alive", lambda: False)() else dead).append(label)
        elif hasattr(obj, "is_alive"):
            (alive if obj.is_alive() else dead).append(label)
        # else: no introspectable thread (e.g. recorder disabled) → skip.

    recorder = getattr(daemon, "ui", None)
    if recorder is not None and getattr(getattr(recorder, "cfg", None), "enabled", True):
        _note("WinEventWatcher", getattr(recorder, "win_events", None))
        # InputHooks owns both a pump thread and a worker thread.
        _note("InputHooks", getattr(recorder, "input", None), attr="_thread")
        _note("InputHooks-worker", getattr(recorder, "input", None), attr="_worker")
        _note("ClipboardWatcher", getattr(recorder, "clipboard", None))

    _note("UiEventPipeline", getattr(daemon, "_ui_pipeline", None), attr="_flush_thread")
    _note("FrameLinkerActor", getattr(daemon, "_linker", None))

    if dead:
        return CheckResult(
            name, FAIL,
            _t(f"{len(dead)} watcher(s) not running: {', '.join(dead)}",
               f"{len(dead)} 个监视器未运行:{', '.join(dead)}")
            + (_t(f" (alive: {', '.join(alive)})", f"(存活:{', '.join(alive)})") if alive else ""),
            fix=_t("Restart the daemon to respawn them; check the log for the "
                   "exception that killed the watcher (UIA/hook errors are common).",
                   "重启守护进程以重建它们;查看日志中导致监视器崩溃的异常"
                   "(UIA/钩子错误较常见)。"),
        )
    if not alive:
        return CheckResult(
            name, WARN, _t("no a11y watchers found running", "未发现任何无障碍监视器在运行"),
            fix=_t("If a11y capture is enabled, the daemon may still be starting — "
                   "re-run shortly.",
                   "若已启用无障碍捕获,守护进程可能仍在启动 —— 稍后重试。"),
        )
    return CheckResult(name, OK, _t(f"{len(alive)} running ({', '.join(alive)})",
                                    f"{len(alive)} 个在运行({', '.join(alive)})"))


def _check_ocr(cfg: Any) -> CheckResult:
    """Does the *configured* OCR engine actually load?

    OCR text is what makes captured frames searchable / answerable. The engine
    is config-selected (winrt | rapidocr | tesseract | off) and each silently
    falls back when its deps are missing — so a user who picked 'rapidocr' but
    never installed it gets degraded OCR with no error. This verifies the chosen
    engine is genuinely available; if not, it names the fallback that will run.
    """
    name = _t("OCR engine", "OCR 引擎")
    engine = str(getattr(getattr(cfg, "ocr", None), "engine", "winrt") or "winrt").lower()
    if engine == "off":
        return CheckResult(name, OK, _t("disabled (engine = off)", "已禁用(engine = off)"))

    try:
        from ..screen import ocr as ocr_mod  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return CheckResult(name, WARN, _t(f"could not load OCR module ({type(exc).__name__})",
                                          f"无法加载 OCR 模块({type(exc).__name__})"))

    if engine == "rapidocr":
        # Build (or reuse the cached) engine — None means deps missing/load failed.
        try:
            ok = ocr_mod._rapidocr_engine() is not None
        except Exception:  # noqa: BLE001
            ok = False
        if ok:
            return CheckResult(name, OK, _t("rapidocr (PP-OCR) loaded", "rapidocr(PP-OCR)已加载"))
        # Mirror perform_ocr's fallback chain: rapidocr → winrt → tesseract.
        nxt = "winrt" if ocr_mod._winrt_available() else "tesseract"
        return CheckResult(
            name, WARN,
            _t(f"rapidocr selected but unavailable — falling back to {nxt}",
               f"已选 rapidocr 但不可用 —— 将退回到 {nxt}"),
            fix=_t("pip install rapidocr (PP-OCR on OpenVINO CPU) in the daemon's "
                   "environment, or switch [ocr] engine to winrt.",
                   "在守护进程环境中 pip install rapidocr(基于 OpenVINO CPU 的 PP-OCR),"
                   "或把 [ocr] engine 切换为 winrt。"),
        )

    if engine == "winrt":
        if ocr_mod._winrt_available():
            return CheckResult(name, OK, _t("winrt OCR available", "winrt OCR 可用"))
        return CheckResult(
            name, WARN,
            _t("winrt selected but unavailable — falling back to tesseract",
               "已选 winrt 但不可用 —— 将退回到 tesseract"),
            fix=_t("Install winrt in the daemon's environment, or switch [ocr] "
                   "engine to rapidocr/tesseract.",
                   "在守护进程环境中安装 winrt,或把 [ocr] engine 切换为 rapidocr/tesseract。"),
        )

    if engine == "tesseract":
        import shutil  # noqa: PLC0415

        cmd = getattr(getattr(cfg, "ocr", None), "tesseract_cmd", None)
        found = (cmd and os.path.isfile(cmd)) or shutil.which("tesseract")
        try:
            import pytesseract  # noqa: F401, PLC0415
            have_pkg = True
        except Exception:  # noqa: BLE001
            have_pkg = False
        if found and have_pkg:
            return CheckResult(name, OK, _t("tesseract available", "tesseract 可用"))
        missing = []
        if not have_pkg:
            missing.append(_t("pytesseract not installed", "未安装 pytesseract"))
        if not found:
            missing.append(_t("tesseract binary not on PATH", "tesseract 可执行文件不在 PATH 中"))
        joiner = " & " if _LANG.get() != "zh" else "、"
        return CheckResult(
            name, WARN,
            _t("tesseract selected but " + " & ".join(missing),
               "已选 tesseract,但" + joiner.join(missing)),
            fix=_t("Install Tesseract and pytesseract, set [ocr] tesseract_cmd, or "
                   "switch to winrt/rapidocr.",
                   "安装 Tesseract 与 pytesseract,设置 [ocr] tesseract_cmd,"
                   "或切换为 winrt/rapidocr。"),
        )

    return CheckResult(name, OK, f"engine = {engine}")


def _check_capture_freshness(cfg: Any, db: Any = None) -> CheckResult:
    """Is capture not just configured, but actually producing data *right now*?

    The Recording check confirms frames exist at all; this one confirms the most
    recent frame is recent. A stalled capture loop (crashed grabber, paused, a
    permissions revocation) leaves a healthy-looking frame count but a frozen
    last_frame_timestamp — this is what catches that."""
    name = _t("Capture freshness", "捕获时效")
    if not bool(getattr(getattr(cfg, "capture", None), "enabled", True)):
        return CheckResult(name, OK, _t("capture disabled in config", "配置中已禁用捕获"))
    if db is None:
        return CheckResult(name, WARN, _t("database handle unavailable", "数据库句柄不可用"))
    try:
        stats = db.health()
    except Exception as exc:  # noqa: BLE001
        return CheckResult(name, WARN, _t(f"could not read stats ({type(exc).__name__})",
                                          f"无法读取统计({type(exc).__name__})"))

    last = stats.get("last_frame_timestamp")
    if not last:
        return CheckResult(
            name, WARN, _t("no frames captured yet", "尚未捕获任何帧"),
            fix=_t("Keep DeskMate running; confirm capture isn't paused.",
                   "让 DeskMate 持续运行;确认捕获未被暂停。"),
        )
    age = _age_seconds(last)
    if age is None:
        return CheckResult(name, OK, _t(f"last frame at {last}", f"最近一帧:{last}"))

    # Stale threshold: a generous multiple of the heartbeat floor so adaptive
    # idle-throttling (which legitimately slows capture when the user is away)
    # doesn't trip a false alarm. Event-driven capture has no fixed cadence, so
    # we use a flat 15-minute ceiling there.
    hb = int(getattr(getattr(cfg, "capture", None), "heartbeat_seconds", 60) or 60)
    threshold = max(15 * 60, hb * 10)
    mins = int(age // 60)
    if age > threshold:
        return CheckResult(
            name, WARN,
            _t(f"last frame was {mins} min ago — capture may be stalled",
               f"最近一帧在 {mins} 分钟前 —— 捕获可能已停滞"),
            fix=_t("Check whether the user is idle (expected) or the capture loop "
                   "died (see 'Background workers' and the daemon log).",
                   "确认是用户空闲(正常)还是捕获循环已崩溃"
                   "(参见「后台工作线程」与守护进程日志)。"),
        )
    return CheckResult(name, OK, _t(f"last frame {mins} min ago", f"最近一帧在 {mins} 分钟前"))


def _check_disk(cfg: Any) -> CheckResult:
    """Is there enough free disk where DeskMate stores frames/videos/audio? These
    grow continuously; a full disk silently breaks capture and the DB."""
    import shutil  # noqa: PLC0415

    name = _t("Disk space", "磁盘空间")
    try:
        from .. import paths  # noqa: PLC0415

        target = paths.root()
    except Exception as exc:  # noqa: BLE001
        return CheckResult(name, WARN, _t(f"could not resolve data dir ({type(exc).__name__})",
                                          f"无法解析数据目录({type(exc).__name__})"))
    try:
        usage = shutil.disk_usage(str(target))
    except Exception as exc:  # noqa: BLE001
        return CheckResult(name, WARN, _t(f"could not read disk usage ({type(exc).__name__})",
                                          f"无法读取磁盘用量({type(exc).__name__})"))

    free_gb = usage.free / (1024 ** 3)
    pct_free = (usage.free / usage.total * 100) if usage.total else 0.0
    vol = target.anchor or target
    where = _t(f"{free_gb:.1f} GB free ({pct_free:.0f}%) on the {vol} volume",
               f"{vol} 卷剩余 {free_gb:.1f} GB({pct_free:.0f}%)")
    if free_gb < 1.0 or pct_free < 3.0:
        return CheckResult(
            name, FAIL, where,
            fix=_t("Free up space or lower retention (Settings → retention days). "
                   "A full disk stops capture and can corrupt the DB.",
                   "清理磁盘或降低保留天数(设置 → 保留天数)。"
                   "磁盘写满会停止捕获并可能损坏数据库。"),
        )
    if free_gb < 5.0 or pct_free < 10.0:
        return CheckResult(
            name, WARN, where,
            fix=_t("Consider lowering retention days; frames/videos/audio grow over time.",
                   "建议降低保留天数;帧/视频/音频会随时间持续增长。"),
        )
    return CheckResult(name, OK, where)


def _check_db_integrity(cfg: Any, db: Any = None) -> CheckResult:
    """Is the SQLite DB healthy and writable? A read-only mount, a stale lock, or
    on-disk corruption all surface as cryptic failures elsewhere — probe directly
    with a fast quick_check plus a no-op write inside a rolled-back transaction."""
    name = _t("Database", "数据库")
    if db is None:
        return CheckResult(name, WARN, _t("database handle unavailable", "数据库句柄不可用"))

    conn = getattr(db, "_conn", None)
    lock = getattr(db, "_lock", None)
    if conn is None:
        return CheckResult(name, WARN, _t("no live connection to probe", "无可探测的活动连接"))

    import contextlib  # noqa: PLC0415

    guard = lock if lock is not None else contextlib.nullcontext()
    try:
        with guard:
            row = conn.execute("PRAGMA quick_check").fetchone()
            # The connection uses a dict row_factory, so a tuple-index would
            # yield the column name. Pull the single value out either way.
            if isinstance(row, dict):
                verdict = next(iter(row.values()), "")
            elif row is not None:
                verdict = row[0]
            else:
                verdict = ""
            # Write probe: create+drop a temp table inside a savepoint we roll
            # back, so a read-only/locked DB fails here without mutating anything.
            conn.execute("SAVEPOINT _doctor_w")
            try:
                conn.execute("CREATE TEMP TABLE _doctor_probe(x)")
                conn.execute("DROP TABLE _doctor_probe")
            finally:
                conn.execute("ROLLBACK TO _doctor_w")
                conn.execute("RELEASE _doctor_w")
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).lower()
        if "readonly" in msg or "read-only" in msg or "attempt to write" in msg:
            return CheckResult(
                name, FAIL, _t("database is not writable (read-only/locked)",
                               "数据库不可写(只读/被锁定)"),
                fix=_t("Check file permissions and that no other process holds the DB; "
                       "ensure the data volume is mounted read-write.",
                       "检查文件权限以及是否有其他进程占用数据库;"
                       "确保数据卷以可读写方式挂载。"),
            )
        return CheckResult(name, FAIL, _t(f"probe failed: {type(exc).__name__}: {exc}",
                                          f"探测失败:{type(exc).__name__}: {exc}"),
                           fix=_t("The DB may be locked or corrupt; restart the daemon.",
                                  "数据库可能被锁定或已损坏;请重启守护进程。"))

    if str(verdict).lower() != "ok":
        return CheckResult(
            name, FAIL, _t(f"integrity check reported: {verdict}",
                           f"完整性检查报告:{verdict}"),
            fix=_t("Back up ~/.deskmate/data.db, then restore from a backup or "
                   "re-create it (corruption won't self-heal).",
                   "备份 ~/.deskmate/data.db,然后从备份恢复或重建"
                   "(损坏不会自行修复)。"),
        )
    return CheckResult(name, OK, _t("integrity ok, writable", "完整性正常,可写"))


def _check_managed_process(cfg: Any) -> CheckResult:
    """Health of the background *process* DeskMate launches (the Ollama model
    service), as opposed to the in-process worker threads checked above.

    DeskMate can spawn Ollama as a detached child and track it via a PID file.
    This surfaces the three states that matter:
      • managed & alive  — we launched it and the process is up (OK)
      • stale PID file    — we recorded a launch but the process is gone, so a
                            crashed/killed service looks 'configured' but is dead
                            (WARN: capture/Ask depending on it will fail)
      • external service  — something is answering the port that we didn't start
                            (OK, informational — we won't manage/stop it)
      • nothing           — no managed process and no service at all (OK; the
                            Ollama-service check already covers reachability)
    """
    from ..modelsvc import service as modelsvc  # noqa: PLC0415

    name = _t("Managed model process", "受管模型进程")
    try:
        st = modelsvc.status(cfg)
    except Exception as exc:  # noqa: BLE001
        return CheckResult(name, WARN,
                           _t(f"could not read service status ({type(exc).__name__})",
                              f"无法读取服务状态({type(exc).__name__})"))

    pid = st.get("pid")
    managed = bool(st.get("managed_by_deskmate"))
    external = bool(st.get("external"))
    backend = st.get("running_backend") or st.get("backend") or "?"

    if managed:
        return CheckResult(
            name, OK,
            _t(f"DeskMate-launched {backend} service running (pid {pid})",
               f"DeskMate 启动的 {backend} 服务正在运行(pid {pid})"),
        )
    if pid and not st.get("pid_alive"):
        # We have a PID file (we launched it) but that process is no longer
        # alive — the classic 'silently died' case.
        return CheckResult(
            name, WARN,
            _t(f"DeskMate launched a {backend} service (pid {pid}) but it is no "
               "longer running",
               f"DeskMate 曾启动 {backend} 服务(pid {pid}),但它已不再运行"),
            fix=_t("Restart it from the Model Service page (its Start button); check "
                   "the service log for why it exited.",
                   "在「模型服务」页面重新启动它(启动按钮);查看服务日志了解退出原因。"),
        )
    if external:
        return CheckResult(
            name, OK,
            _t(f"an external {backend} service is running (not launched by DeskMate)",
               f"有一个外部 {backend} 服务在运行(非 DeskMate 启动)"),
        )
    return CheckResult(
        name, OK,
        _t("no DeskMate-launched service (started manually or not in use)",
           "没有 DeskMate 启动的服务(手动启动或未使用)"),
    )


def _check_model_autostart(cfg: Any) -> CheckResult:
    """If [model_service] auto_start is on, is a service actually up? A failed
    boot-time auto-start is otherwise invisible until Ask fails."""
    name = _t("Model auto-start", "模型自启动")
    ms = getattr(cfg, "model_service", None)
    if not (ms is not None and getattr(ms, "auto_start", False)):
        return CheckResult(name, OK, _t("disabled (started manually)", "已禁用(手动启动)"))
    from ..modelsvc import service as modelsvc  # noqa: PLC0415

    try:
        running = modelsvc.probe_running(cfg.ollama.base) is not None
    except Exception:  # noqa: BLE001
        running = False
    if running:
        return CheckResult(name, OK, _t("enabled and a service is running",
                                        "已启用,且有服务在运行"))
    return CheckResult(
        name, WARN,
        _t("auto_start is on but no service is answering — boot launch may have failed",
           "auto_start 已开启,但没有服务应答 —— 开机启动可能失败了"),
        fix=_t("Open Model Service and start the backend manually; check the service log.",
               "打开「模型服务」并手动启动后端;查看服务日志。"),
    )


def _check_connectors(cfg: Any) -> CheckResult:
    """Are any mailbox connectors configured? (Informational — absence is fine.)
    A connector counts as configured when it has OAuth client credentials."""
    name = _t("Mail connectors", "邮箱连接器")
    try:
        gmail = bool(getattr(getattr(cfg, "gmail", None), "client_id", ""))
        outlook = bool(getattr(getattr(cfg, "outlook", None), "client_id", ""))
    except Exception:  # noqa: BLE001
        gmail = outlook = False
    connected = [n for n, on in (("Gmail", gmail), ("Outlook", outlook)) if on]
    if connected:
        return CheckResult(name, OK, _t(", ".join(connected) + " configured",
                                        "、".join(connected) + " 已配置"))
    return CheckResult(
        name, OK, _t("none configured (optional)", "未配置(可选)"),
        fix=_t("Connect Gmail/Outlook on the Email page for mail-aware answers.",
               "在「邮件」页面连接 Gmail/Outlook,以获得感知邮件的回答。"),
    )


# Order matters: most fundamental first. Checks dispatch by what they need:
# plain (cfg,), DB checks (cfg, db), or the daemon check (cfg, daemon).
_CHECKS: list[Callable[..., CheckResult]] = [
    _check_api,
    _check_ollama,
    _check_running_backend,
    _check_active_model,
    _check_genai_runtime,
    _check_model_autostart,
    _check_winrt,
    _check_proxy,
    _check_workers,
    _check_a11y_watchers,
    _check_managed_process,
    _check_ocr,
    _check_recording,
    _check_capture_freshness,
    _check_disk,
    _check_db_integrity,
    _check_schema,
    _check_connectors,
]

# Checks that also need a DB handle.
_DB_CHECKS = {_check_recording, _check_schema, _check_capture_freshness, _check_db_integrity}
# Checks that also need the live daemon (in-process deployment only).
_DAEMON_CHECKS = {_check_workers, _check_a11y_watchers}


# ── Orchestration ─────────────────────────────────────────────────────────────


def run_all(cfg: Any, db: Any = None, daemon: Any = None) -> list[CheckResult]:
    """Run every check against ``cfg`` (plus ``db``/``daemon`` for the ones that
    need them). A check that itself blows up becomes a ``fail`` line rather than
    propagating."""
    out: list[CheckResult] = []
    for check in _CHECKS:
        try:
            if check in _DB_CHECKS:
                out.append(check(cfg, db))
            elif check in _DAEMON_CHECKS:
                out.append(check(cfg, daemon))
            else:
                out.append(check(cfg))
        except Exception as exc:  # noqa: BLE001
            out.append(CheckResult(getattr(check, "__name__", "check"), FAIL,
                                   f"check raised {type(exc).__name__}: {exc}"))
    return out


def report(cfg: Any, db: Any = None, daemon: Any = None, lang: str = "en") -> dict[str, Any]:
    """Run all checks and return a JSON-serialisable report with a summary.

    ``lang`` ("en" | "zh") selects the language of every name/message/fix; it's
    set into a context variable for the duration of this call and reset after,
    so concurrent reports in different languages don't interfere.
    """
    token = _LANG.set("zh" if str(lang).lower().startswith("zh") else "en")
    try:
        results = run_all(cfg, db, daemon)
    finally:
        _LANG.reset(token)
    counts = {OK: 0, WARN: 0, FAIL: 0}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    overall = FAIL if counts[FAIL] else (WARN if counts[WARN] else OK)
    return {
        "overall": overall,
        "summary": {"ok": counts[OK], "warn": counts[WARN], "fail": counts[FAIL]},
        "checks": [asdict(r) for r in results],
    }
