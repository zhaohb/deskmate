# 21 — Diagnostics (Doctor)

## Purpose

A dependency-light self-check that surfaces the environment problems DeskMate
users actually hit — a wrong Ollama backend or GenAI runtime, an unreachable /
mis-named model, a dead background worker or capture watcher, a stalled capture,
a missing OCR engine, low disk, a locked DB, a proxy hijacking localhost — and
tells the user how to fix each one. Backs the **Diagnostics** UI page and the
`GET /health/doctor` endpoint.

Covers `deskmate/engine/doctor.py`.

## Key files

| Path | Role |
|------|------|
| `deskmate/engine/doctor.py` | All checks, the `CheckResult` type, `run_all` / `report`, and the bilingual `_t` helper |
| `deskmate/engine/api.py` | `GET /health/doctor?lang=` — calls `doctor.report(cfg, db, daemon, lang=...)` |
| `deskmate/ui/static/app.js` | `runDoctor()` — fetches the report, renders rows generically |
| `deskmate/ui/static/i18n.js` | `doctor.*` UI-chrome keys (title/help/pills); check text is server-side |

## Design & data flow

Every check is a small function `(_check_*)` that returns a uniform
`CheckResult(name, status, message, fix)`, where `status` is `"ok" | "warn" |
"fail"` and `fix` is an optional actionable hint shown only when not OK. The API
and any CLI render them identically.

```
UI (Diagnostics page)
  └─ GET /health/doctor?lang=zh|en
       └─ doctor.report(cfg, db, daemon, lang)
            ├─ set language contextvar
            ├─ run_all() → [CheckResult, ...]   (each check, dispatched by need)
            └─ tally → {overall, summary{ok,warn,fail}, checks:[{name,status,message,fix}]}
  └─ render rows generically (icon by status, name / message / fix)
```

**Never-raise contract.** A check must not propagate exceptions: it wraps risky
work in `try/except` and degrades to a `warn`/`fail` line. `run_all` is the
backstop — if a check itself blows up, it becomes a single `fail` row rather
than taking down the whole report. So one broken probe can never blank the page.

**Overall status** = `fail` if any check failed, else `warn` if any warned, else
`ok`.

### Dispatch by dependency

Checks declare what they need; `run_all` dispatches accordingly:

| Group | Signature | Members |
|-------|-----------|---------|
| plain | `(cfg)` | most checks |
| DB | `(cfg, db)` | `_check_recording`, `_check_schema`, `_check_capture_freshness`, `_check_db_integrity` (in `_DB_CHECKS`) |
| daemon | `(cfg, daemon)` | `_check_workers`, `_check_a11y_watchers` (in `_DAEMON_CHECKS`) |

The daemon-dependent checks are only meaningful in the **in-process**
deployment, where the API can see the live `Daemon` object (passed as
`app.state.daemon`). In a split API/daemon setup `daemon is None`, and these
degrade to an informational **OK** ("not introspectable") rather than a false
alarm.

## The checks

Ordered most-fundamental first (`_CHECKS`):

| Check | Looks at | Typical non-OK |
|-------|----------|----------------|
| **DeskMate API** | configured host:port answers `/health` | WARN if unreachable |
| **Ollama service** | a service answers `{base}/api/tags` | FAIL if nothing answers |
| **Running backend** | which backend is up + its GenAI runtime build | WARN if the loaded GenAI nightly is older than `_MIN_GENAI_BUILD` (garbles long-context output on the Intel GPU plugin) |
| **Active model** | `[ollama] model` is installed **and accepted by `/api/chat`** | FAIL on `"invalid model name"` (this build needs a fully-qualified `host/ns/model:tag`); WARN if not in the tag list |
| **GenAI runtime** | an OpenVINO GenAI runtime is installed | WARN if absent (only needed for the OpenVINO backend) |
| **Model auto-start** | `[model_service] auto_start` on ⇒ a service is up | WARN if on but nothing answers (boot launch may have failed) |
| **Windows toasts (winrt)** | `winrt` importable | WARN if not (reminders fall back to the in-app inbox) |
| **HTTP proxy** | a proxy is set that bypasses localhost | WARN if `NO_PROXY` lacks `127.0.0.1`/`localhost` (may hijack local calls) |
| **Background workers** | daemon loop threads + sub-workers are alive | FAIL listing any dead worker (capture/audio/retention/translate, AppScheduler, PipeScheduler, HabitWatcher, ContextFusionBus, RedactReconciler) |
| **A11y capture watchers** | the accessibility-capture threads are alive | FAIL listing any dead one (WinEventWatcher, InputHooks + worker, ClipboardWatcher, UiEventPipeline, FrameLinkerActor) |
| **Managed model process** | the Ollama **process** DeskMate launched is alive | WARN if we have a PID file but the process is gone (silently died); OK for managed-alive / external / none |
| **OCR engine** | the **configured** engine actually loads | WARN if the chosen engine (rapidocr/winrt/tesseract) is unavailable, naming the fallback that will run |
| **Recording** | any frames captured at all | WARN if none yet |
| **Capture freshness** | the most recent frame is recent | WARN if the last frame is older than `max(15 min, heartbeat×10)` — capture may be stalled |
| **Disk space** | free space on the `~/.deskmate` volume | WARN < 5 GB / 10%; FAIL < 1 GB / 3% (a full disk stops capture and can corrupt the DB) |
| **Database** | SQLite `quick_check` + a rolled-back write probe | FAIL on corruption or a read-only/locked DB |
| **DB schema** | on-disk schema version == code's | WARN if a migration is pending (restart to apply) |
| **Mail connectors** | Gmail/Outlook OAuth configured | always OK (informational — absence is fine) |

> **Threads vs. processes.** Three checks form a fleet-health trio that the
> older checks missed: **Background workers** and **A11y capture watchers**
> inspect *threads inside the daemon process* (a crashed worker dies silently),
> while **Managed model process** inspects the *detached child process* DeskMate
> launched (the Ollama service — see [20 — Model Service](20-model-service.md)).

## Localization (中英文)

All user-facing text — names, messages, and fix hints — is bilingual
(English / Chinese). The status codes (`ok`/`warn`/`fail`) are
language-independent.

- A `contextvars.ContextVar` (`_LANG`) holds the active language for the
  duration of one `report()` call; `report` sets it from its `lang` argument
  and resets it in a `finally`, so concurrent reports in different languages
  never interfere.
- Every string is built with `_t(en, zh)`, which keeps both languages adjacent
  in the source and picks one based on `_LANG`. `"zh"`/`"zh-CN"` → Chinese; any
  other value → English.
- The endpoint takes `?lang=`; the UI passes its current language
  (`I18N.lang`). Because the verdicts are computed identically, switching
  language only changes the text — `report(..., "en")` and `report(..., "zh")`
  always agree on `summary` and `overall`.

**Language follows the Settings page.** The Diagnostics language is bound to the
UI language selector on the Settings page: changing it fires `I18N.onChange`,
which re-runs `runDoctor()` so the report is re-fetched in the new language. No
manual refresh needed; the choice persists in `localStorage` like the rest of
the UI.

## Interfaces & dependencies

- **Exposes**: `run_all(cfg, db=None, daemon=None) -> list[CheckResult]` and
  `report(cfg, db=None, daemon=None, lang="en") -> dict` (JSON-serialisable:
  `{overall, summary, checks}`).
- **Consumes**: `cfg` (always), the DB manager (`db.health()`, `schema_version`,
  the raw connection for the integrity probe), and the live `Daemon` (thread
  handles) when in-process. It also calls into `modelsvc.service` (probe /
  status / runtime detection), `screen.ocr` (engine availability), and
  `paths.root()` (disk target).
- **Consumed by**: the `GET /health/doctor` route and the `#view-doctor` UI.

## Design trade-offs

- **Server-side localization, not frontend keys.** The messages are densely
  interpolated (pids, GB, counts, build dates, file names). Splitting each into
  an i18n key + params would be brittle and hard to read; bilingual `_t(en, zh)`
  pairs at the call site keep the two languages in lockstep and easy to audit.
- **Generic UI rendering.** `runDoctor()` maps over `rep.checks` and shows the
  status icon + name/message/fix verbatim — it hard-codes no check names, so
  adding a check in `doctor.py` needs **no UI change**.
- **OK-when-unknowable beats false alarms.** Daemon-only checks return OK with a
  "not introspectable" message in split deployments rather than failing; the
  managed-process check treats an *external* Ollama as OK (we don't own it).
- **The probe must be cheap and safe.** The DB integrity check uses a fast
  `quick_check` and a no-op write inside a savepoint it rolls back, so it never
  mutates data; the OCR check reuses the cached engine singleton rather than
  building a throwaway one.
