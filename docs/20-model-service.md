# 20 — Model Service

## Purpose

Let a user provision and run the local Ollama server from inside DeskMate —
download the executable, pull models, set a custom model source, and start/stop
the background service — without touching a terminal. Backs the **Model Service**
UI page.

Covers `deskmate/modelsvc/` and the `/models/*` API routes.

## Key files

| Path | Role |
|------|------|
| `deskmate/modelsvc/service.py` | Download/extract, exe discovery, launch env, start/stop, status probe, model pull |
| `deskmate/modelsvc/__init__.py` | Public re-exports |
| `deskmate/engine/api.py` | `/models/*` endpoints inside `create_app` |
| `deskmate/config.py` | `ModelServiceConfig` (`[model_service]`) |
| `deskmate/paths.py` | `bin_dir()`, `ollama_official_dir()`, `genai_runtime_dir()`, `modelsvc_pid_file()`, `modelsvc_log_file()` |
| `deskmate/ui/static/` | `view-models` section, `app.js` handlers, `i18n.js` keys |

## Two backends

DeskMate supports two Ollama builds, which are distributed very differently:

- **Official** (`backend = "official"`) — auto-downloaded from a GitHub release.
  Tag `v0.30.7`, asset `ollama-windows-amd64.zip`. The Download button streams
  the zip into `~/.deskmate/bin/ollama-official/` and extracts it.
- **OpenVINO** (`backend = "openvino"`, [zhaohb/ollama_openvino](https://github.com/zhaohb/ollama_openvino)) —
  has **no GitHub releases**; its prebuilt `ollama.exe` lives only on a Google
  Drive folder that can't be fetched programmatically. So the OV build is two
  **independent downloads** into a folder the user can choose
  (`[model_service] download_dir`, default `~/.deskmate/bin/ollama-openvino/`):
  1. **`ollama.exe`** — `obtain_openvino_exe` accepts either an **http(s) URL**
     (downloaded, with progress) or a **local path** (copied in); either way it
     lands at `<download_dir>/ollama.exe`.
  2. **GenAI runtime** — `download_genai` fetches a GenAI zip and extracts it to
     `<download_dir>/runtime/<version>/`. The URL defaults to
     `GENAI_RUNTIME_URL` but the user can supply a **custom URL** to pull a
     different version (persisted as `genai_url`). Because each version unpacks
     to its own subfolder, **multiple versions coexist**: `list_genai_versions`
     enumerates them (newest first) and the user can **pick the active one** —
     stored as `genai_runtime_dir`, which is what goes on `PATH` at launch (when
     unset, the newest is used).

  At launch the GenAI runtime dirs are prepended to `PATH` (`build_launch_env`)
  — this replicates what the upstream `setupvars.bat` does for *running*
  (the DLL search path), applied programmatically, so there is **no separate
  setup step**. We add **both** the OpenVINO/GenAI DLL dir
  (`runtime/bin/intel64/Release`) **and** the Intel **TBB** dir
  (`runtime/3rdparty/tbb/bin`, holding `tbb12.dll`) — TBB is a separate
  dependency in its own folder that OpenVINO can't load without
  (`_ov_runtime_path_dirs`). (`GODEBUG=cgocheck=0` is *not* set — it isn't
  needed for this build.)

> **Why no single fused `.exe`?** `ollama.exe` is a Go binary and the GenAI
> runtime is a set of C++ DLLs (+ tbb); they can't be cleanly merged into one
> executable, and a self-extracting wrapper would unpack hundreds of MB on every
> launch and break code-signing. Instead both files live together in one
> portable `download_dir`, which gives the same "drop-in and run" result.

## `[model_service]` vs `[ollama]`

These are intentionally separate:

- **`[ollama]`** = *connection*: `base`, `model`, `chat_timeout`. Used by Ask and
  the pipe apps to talk to whatever is listening. Unchanged by this feature.
- **`[model_service]`** = *provisioning & lifecycle*: which backend, the
  OpenVINO exe path, the custom `registry`, the GenAI runtime dir, the
  `download_dir`, and `auto_start`.

The service is launched on the host:port parsed from `[ollama] base`, so both
views agree on one endpoint; models are pulled into that running service, and
the Settings page's `[ollama] model` still selects which installed model Ask
uses.

## Lifecycle: detached & persistent

The service is launched **detached** so it outlives the DeskMate UI process:

- Windows: `Popen([exe, "serve"], creationflags=DETACHED_PROCESS |
  CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW)`.
- Other OSes (best-effort): `start_new_session=True`.

A JSON PID file (`~/.deskmate/modelsvc.pid`: `{pid, backend, exe, started_at}`)
records what we started; combined stdout/stderr go to
`~/.deskmate/logs/ollama-service.log`. **Status** = an HTTP probe of
`{base}/api/tags` (reusing `engine.llm.http_get`) plus PID liveness:

- `running` — the probe succeeded.
- `managed_by_deskmate` — we have a PID file and that pid is alive.
- `external: true` — running but **not** started by us. The Stop button is
  disabled in this case so DeskMate never kills an Ollama the user launched
  themselves (this also covers the "port already in use" situation: `start` is
  idempotent — it probes first and no-ops if something already answers).

Because persistence is the whole point, **`stop` is explicit only** (the Stop
button → Windows `taskkill /F /T /PID`, posix SIGTERM→SIGKILL). There is no
teardown on DeskMate shutdown. The daemon has a single **opt-in** hook: when
`[model_service] auto_start = true`, `Daemon.start()` calls
`modelsvc.start_service(cfg)` best-effort (and logs on failure) — with no
matching `stop()`.

## Launch environment

`build_launch_env(cfg, backend, exe)`:

| Var | When | Value |
|-----|------|-------|
| `OLLAMA_HOST` | always | host:port from `cfg.ollama.base` |
| `OLLAMA_REGISTRY` | `registry` set | the custom model source |
| `PATH` | openvino | GenAI runtime dir prepended (replaces `setupvars.bat`) |

The custom **repository address** asked for by the feature is exactly
`OLLAMA_REGISTRY` — the source Ollama pulls models from.

## API routes

All under `create_app` (no router), `cfg` mutated in-process after persisting
via `set_config_value`:

| Route | Method | Behavior |
|-------|--------|----------|
| `/models/status` | GET | `modelsvc.status(cfg)` — polled by the UI |
| `/models/config` | POST | Validate+persist any of `backend, ollama_exe_path, registry, genai_runtime_dir, download_dir, auto_start` |
| `/models/download-ollama` | POST | Stream NDJSON download/extract progress (official build) |
| `/models/download-openvino-exe` | POST | `{exe}` URL→download or local path→copy into `download_dir`; persists `backend=openvino` + exe path |
| `/models/download-genai` | POST | Stream NDJSON progress for the GenAI runtime into `download_dir/runtime` |
| `/models/pull` | POST | Proxy Ollama `/api/pull` NDJSON to the client |
| `/models/start` | POST | `start_service` (400 if exe missing) |
| `/models/stop` | POST | `stop_service` |

The long jobs stream **chunked NDJSON** (request-scoped) rather than the global
SSE event bus; blocking generators are pumped through `asyncio.to_thread` so the
server isn't stalled (same technique as `/events/stream`). The UI consumes them
with a small `streamNdjson()` helper (`fetch().body.getReader()`).

## UI: guided step flow

The page is a single scrollable view (`#view-models`, in the scrollable-overflow
group so nothing is clipped) laid out as a guided flow:

- a **sticky status strip** at the top — always-visible state pill + compact
  facts (version / model count / exe), so the current status never scrolls away;
- **Step 1 Backend** (segmented toggle) → **Step 2 Install** (the official
  Download button, or the OpenVINO `download_dir` + exe + GenAI sub-steps, each
  with a readiness chip) → **Step 3 Start the service** (the primary Start /
  Stop buttons as their own clearly-labelled step) → **Step 4 Pull a model**;
- **Advanced** (custom registry) and **Activity log** are collapsed
  `<details>` so they don't lengthen the page until needed.

Only the selected backend's install card is shown. Readiness chips turn green
when the exe / runtime are present, so it's obvious what's left before Start.

## Security notes

- **Zip-Slip**: `extract_zip` rejects any member that resolves outside the
  destination, even for our own server-controlled URLs.
- **taskkill**: the PID is an int we wrote to our own PID file and is passed as a
  fixed argv list (never a shell), so there is no injection surface; we only kill
  a process we manage and that is alive.
- **User exe path**: `validate_exe_path` resolves the path and requires an
  existing regular `.exe` file, so a typo or directory can't be launched.
- **Downloads** land under `~/.deskmate/bin/` — already covered by the
  `.deskmate/` gitignore rule, so binaries never reach git.
