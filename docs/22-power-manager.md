# 22 — Power Manager

## Purpose

On battery, push DeskMate's background AI tasks (semantic index, redaction scan,
screen capture/OCR, data cleanup) onto efficient cores (E-cores) via **thread-level
EcoQoS**, and let the user pick third-party apps to throttle as well — extending
laptop battery life while the Ask conversation path keeps performance cores
(P-cores).

**Design principles: purely additive, zero-invasive, zero new dependencies.** All
Win32 / NT API calls use the stdlib `ctypes` — no `psutil` or `pywin32`. Existing
worker code is untouched — a central `PowerManager` thread finds workers by
**thread name** and tags them with EcoQoS from the outside. Non-Windows / older
platforms degrade gracefully to no-ops and never raise.

---

## Key files

| File | Role |
|------|------|
| `platform/qos.py` | Thread-level + process-level EcoQoS ctypes wrapper (`SetThreadInformation` / `SetProcessInformation` `*PowerThrottling`) |
| `platform/battery.py` | `GetSystemPowerStatus` → AC/battery/percent/OS runtime estimate; fail-open to UNKNOWN |
| `platform/power_manager.py` | Central controller thread `daemon-power-manager`: polls power source → tags workers by thread name |
| `platform/processes.py` | Enumerates user-facing apps (visible windows via `EnumWindows`) + `AppPowerController` (user-selected process eco/restore) |
| `platform/cores.py` | Per-core load grouped by P/E efficiency class (`NtQuerySystemInformation` + `GetSystemCpuSetInformation`). Backend retained, not currently shown in UI |
| `engine/api.py` | `/power/*` routes (see below) |
| `ui/static/index.html` `app.js` `i18n.js` | Battery Saver SPA view + topbar battery capsule; all static and dynamic copy is localized through `i18n.js` |
| `config.py` `PowerConfig` | `[power] enabled / poll_seconds` |

---

## Design & data flow

```
Unplug/plug  ──poll(15s)──▶  PowerManager.tick()
                               │  battery?  → find workers by thread name → set_thread_eco(tid)
                               │  ac?       → clear_thread(tid)
                               ▼
         threading.enumerate() matches DEFAULT_ECO_THREAD_NAMES:
         ├─ daemon-semantic-index   (embedding index, CPU-bound)
         ├─ RedactReconciler        (ONNX redaction, CPU-bound)
         ├─ event-driven-capture    (screen capture + OCR; OCR pinned to CPU)
         ├─ daemon-heartbeat        (legacy capture loop)
         └─ daemon-retention        (periodic DB cleanup)

User picks app ──▶ POST /power/apps/eco {pid} ──▶ AppPowerController.eco(pid)
                                                    set_process_eco(pid)  # process-level
```

### EcoQoS semantics (`EXECUTION_SPEED` knob)

| ControlMask / StateMask | Meaning |
|---|---|
| SPEED / SPEED | "Throttle me": scheduler prefers E-cores + caps frequency → saves power |
| SPEED / 0 | "Do NOT throttle me": HighQoS, stays on P-cores even under power-save |
| 0 / 0 | Clear our override; follow system default |

### Why thread-level (not process-level) for our own workers

The daemon runs Ask, capture, OCR, redaction, and indexing in **one process**.
Process-level EcoQoS would throttle Ask too. Thread-level
`SetThreadInformation` lets us push only background workers onto E-cores while
leaving Ask alone. The controller finds workers by `threading.enumerate()` +
`native_id`, opens them with `OpenThread(THREAD_SET_INFORMATION)` from the
outside — so worker code needs zero changes.

### Why these workers are throttled and Ask is not

- **Throttled**: semantic index, redaction, OCR/capture, cleanup — all CPU-bound
  and **the user is not waiting** for their results.
- **Not throttled (Ask)**: request-driven, heavy lifting on GPU/Ollama, must stay
  responsive when the user asks a question.
- **Whisper defaults to NPU** (`openvino_genai` backend); EcoQoS only governs CPU
  cores, so it has no effect on NPU workloads — and no need to. Only when the user
  switches to the `onnx_cpu` backend does Whisper run on CPU.

### Local topology (Core Ultra X7 358H dev machine)

4 P-cores (EfficiencyClass 1) + 12 E/LPE-cores (EfficiencyClass 0), 16 total,
no hyper-threading. Ample E-cores to absorb background workers without starving
the 4 P-cores.

---

## HTTP API

| Route | Purpose |
|------|------|
| `GET /power/status` | Power source + eco active + eco thread count (feeds capsule & main card). Degrades to raw OS read when daemon is absent |
| `GET /power/cores` | Per-core load grouped by P/E (backend retained, frontend not currently using) |
| `GET /power/apps` | Enumerate visible-window apps + per-app throttle-ability + current eco state |
| `POST /power/apps/eco` `{pid}` | Push user-selected process onto E-cores; returns 409 if unable (likely needs admin) |
| `POST /power/apps/restore` `{pid}` or `{}` | Restore a specific pid, or omit pid to restore all |
| `GET /power/ui` | Legacy standalone page entry; now 307-redirects to `/ui` (Battery Saver is a SPA view) |

The UI is a SPA view (`#view-power`, nav button "续航管家" / "Battery Saver",
`data-view="power"`). Lifecycle mirrors Model Service: enter → start poll, leave
→ stop poll. The topbar also carries a battery capsule that auto-hides when no
battery / unsupported.

### Localization

Power Manager follows the global Settings language. Static labels in
`index.html` use `data-i18n`, and runtime-rendered state in `app.js` (battery
state, EcoQoS worker labels, projected runtime, app-control badges, alerts, and
the topbar battery capsule) uses `T("power.*")` keys from `i18n.js`. This avoids
the Battery Saver page showing Chinese-only status text while the rest of the UI
is set to English.

---

## "Extra X minutes" estimate (honestly labeled as projected)

No published "EcoQoS whole-system battery gain" measurement exists (confirmed via
web research). A two-step model with sourced factors is used, with an orange
warning box in the UI explicitly marking it as projected:

```
Whole-system saving ≈ P × S
  P = background task share of system power ≈ 8%  (conservative estimate; no public figure)
  S = EcoQoS saving on that share ≈ 45%
      (Microsoft official: same work at < 50% CPU energy
       devblogs.microsoft.com/sustainable-software/introducing-ecoqos)
  ⇒ ≈ 3.6%
```

Reference data points (none are whole-system measurements; anchors only):
- **Microsoft EcoQoS**: tagged task CPU power up to −90%, energy < 50% (**per-task
  CPU**, not system).
- **Intel E-cores**: same work at ~1/5 P-core energy (Architecture Day marketing,
  vs old Skylake).
- **Microsoft Energy Saver** (different mechanism, whole-system): up to +14%.

**True value requires an on-device A/B discharge test** to replace the coefficient.

