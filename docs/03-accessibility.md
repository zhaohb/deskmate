# 03 — Accessibility & Input

## Purpose

Observe what the user is doing via Windows UI Automation (UIA) and low-level
hooks: window focus changes, clicks, keystrokes, clipboard activity, and the
structured accessibility tree of the focused window.

Covers `deskmate/a11y/`.

## Key files

| File | Role |
|------|------|
| `win_events.py` | `WinEventWatcher` — owns a Win32 message pump + `SetWinEventHook` for focus/foreground/name-change events |
| `input_hooks.py` | Low-level `WH_KEYBOARD_LL` / `WH_MOUSE_LL` hooks with keystroke debouncing |
| `clipboard.py` | `ClipboardWatcher` polling `GetClipboardSequenceNumber` to detect copy events |
| `uia_thread.py` | Dedicated COM/STA thread (`UIAutomationInitializerInThread`) for all UIA calls |
| `uia_tree.py` | UIA tree walker with `CacheRequest` batching → structured text + tree JSON |
| `browser_url.py` | Extracts the active browser tab URL from the address-bar UIA element |
| `recorder.py` | `UiRecorder` — assembles watchers + threads into one start/stop unit |
| `activity_feed.py` | Rolls raw events into a higher-level activity stream |
| `ui_event_types.py` | Event dataclasses/enums shared across the a11y layer |

## Threading model

Windows hooks and UIA each require specific thread affinity, so the a11y layer
isolates them:

```mermaid
flowchart LR
    subgraph WinEvent thread
        MP["message pump"] --> HOOK["SetWinEventHook"]
    end
    subgraph Input thread
        KB["WH_KEYBOARD_LL"]
        MS["WH_MOUSE_LL"]
    end
    subgraph UIA STA thread
        INIT["COM init"] --> WALK["CacheRequest tree walk"]
    end
    HOOK --> REC["UiRecorder"]
    KB --> REC
    MS --> REC
    REC --> PIPE["capture/ui_event_pipeline"]
```

- **WinEventWatcher** runs its own message pump because `SetWinEventHook`
  callbacks are only delivered to a thread that pumps messages.
- **InputHooks** install global low-level keyboard/mouse hooks; keystrokes are
  **debounced (~300 ms)** so a burst of typing becomes a single "typed" event
  instead of thousands.
- **UIA thread** initializes COM in single-threaded apartment (STA) mode and is the
  *only* thread allowed to touch UIA objects, avoiding cross-apartment marshalling
  errors.

## UIA tree extraction

`uia_tree.py` walks the focused window's element tree to produce both flattened
text and a structured JSON tree. It uses a **`CacheRequest`** to batch the
properties it needs (name, control type, value, bounding rect) into a single
cross-process round-trip per element, which is dramatically faster (~10×) than
fetching properties one at a time over COM.

`browser_url.py` is a specialized walk that finds the address-bar element of known
browsers and reads its value — giving each frame a `browser_url`, which downstream
search, meeting detection, and incognito filtering rely on.

## Output

All watchers feed the `UiRecorder`, which normalizes them into UI event objects
and hands them to `capture/ui_event_pipeline.py` for batched persistence and
capture triggering (see [02 — Capture](02-capture.md)).

## Design trade-offs

1. **One thread per affinity constraint** — Message pump, hooks, and UIA each get a
   thread that satisfies their OS requirements, instead of fighting threading rules.
2. **CacheRequest batching** — Trades a slightly larger single request for far
   fewer COM round-trips; the main lever for acceptable capture latency.
3. **Keystroke debouncing** — Privacy- and noise-friendly: stores *that* typing
   happened, not a keylogger stream.
4. **Polling the clipboard sequence number** — Cheap and reliable; avoids fragile
   clipboard-viewer chain APIs.
5. **Graceful absence** — If `comtypes`/UIA is unavailable, the recorder still runs
   window-focus + input capture and simply omits tree data.
