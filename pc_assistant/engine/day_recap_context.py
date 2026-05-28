"""Helpers for rich day-recap agent context."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

_LOW_VALUE_PHRASES = (
    "activate windows",
    "go to settings to activate windows",
    "windows license",
    "settings to activate",
)

_SKIP_UI_EVENTS = frozenset({
    "move", "title_change", "value_change", "window_focus", "app_switch", "scroll",
})

_NOISE_LINE_RE = re.compile(
    r"^("
    r"Chrome Legacy Window|Minimize|Maximize|Restore|Close$|"
    r"Back$|Forward$|Reload$|Home$|New tab$|New Tab$|"
    r"File$|Edit$|Selection$|View$|Go$|Run$|Terminal$|Help$|"
    r"Toggle |Navigation actions|Title actions|Agents Window|"
    r"Open Cursor|Explorer Section|Toggle Panel|Toggle Agents|"
    r"Go Back|Go Forward|Files Explorer$|"
    r"Open tab|View site|Address and|Search this tab|Translate$|"
    r"Bookmark this|Extensions$|Hidden toolbar|Mute tab|"
    r"Check your|Open Gemini|Control your|Performance issue|"
    r"Energy Saver|Side Panel|Infobar|Split View|Create split|"
    r"Menu containing|All Bookmarks|Separator$|Close side|Close website|"
    r"Left title|Right title|Has access|Tab active|"
    r"JavaScript opt|To open this|Reading mode|Offer available|"
    r"This page is|Shopping|Discount|Google Password|Addresses and|"
    r"Browse with|Gemini in|Ask AI|Ask Google|Zoom:|Third-party|"
    r"Save card|View your virtual|Add virtual|Enable mandatory|"
    r"Clear input$|Finish update$|Bookmarks$|Managed bookmarks$|"
    r"Saved Tab Groups$|Tab groups$|Close website$|hongbo$|"
    r"Indigo$|Omni - Bookmark|SciSpace|Ghelper|有道灵动|"
    r"page wants to install|"
    r"Running applications|DesktopWindowXamlSource|"
    r"Tray Input|Right-click for|To switch input|"
    r"Safely Remove|Windows Security|OneDrive|"
    r"Privacy Microphone|Network Network|"
    r"Volume |Power Battery|Clock \d|"
    r"Show Desktop$|"
    r"Sunday|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|"
    r"Hours$|Minutes$|AM/PM|Month |Day |Year |"
    r"Show |Clear$|Today$|Su$|Mo$|Tu$|We$|Th$|Fr$|Sa$"
    r")", re.I,
)


def is_low_value_text(text: str) -> bool:
    """Omit repetitive OS chrome / license nag from recap narrative."""
    t = " ".join((text or "").lower().split())
    if len(t) < 12:
        return True
    if any(p in t for p in _LOW_VALUE_PHRASES) and len(t) < 200:
        return True
    return False


_global_line_freq: dict[str, int] = {}


def reset_line_freq() -> None:
    """Reset the global line frequency counter (call at start of each summary build)."""
    _global_line_freq.clear()


def count_line_frequencies(texts: list[str]) -> None:
    """Count how often each normalized line appears across all texts."""
    for text in texts:
        local_seen: set[str] = set()
        for line in (text or "").split("\n"):
            norm = line.strip().lower()
            if len(norm) < 6 or norm in local_seen:
                continue
            local_seen.add(norm)
            _global_line_freq[norm] = _global_line_freq.get(norm, 0) + 1


def extract_valuable_lines(text: str, *, freq_threshold: int = 0) -> str:
    """Strip UI-chrome noise lines from OCR/a11y text, keep only meaningful content.

    If freq_threshold > 0 and line frequencies have been counted, lines appearing
    in more than freq_threshold distinct frames are also filtered as repetitive noise.
    """
    if not text:
        return ""
    seen: set[str] = set()
    valuable: list[str] = []
    for line in text.split("\n"):
        s = line.strip()
        if len(s) < 6:
            continue
        if s.isdigit():
            continue
        if _NOISE_LINE_RE.match(s):
            continue
        norm = s.lower()
        if norm in seen:
            continue
        # Filter lines that appear in many different frames (repetitive chrome)
        if freq_threshold > 0 and _global_line_freq.get(norm, 0) > freq_threshold:
            continue
        seen.add(norm)
        valuable.append(s)
    return "\n".join(valuable)


def normalize_text_key(text: str) -> str:
    return " ".join((text or "").lower().split())[:120]


def format_ts_local(ts: str) -> str:
    """Format a timestamp as a clock time like '2:30 PM' for recap reports."""
    if not ts:
        return ""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.astimezone()
        hour12 = dt.hour % 12 or 12
        meridiem = "AM" if dt.hour < 12 else "PM"
        return f"{hour12}:{dt.minute:02d} {meridiem}"
    except ValueError:
        return ts[:16]


def format_search_items(
    items: list[dict[str, Any]],
    *,
    max_text: int = 450,
    include_ui_text: bool = True,
) -> list[str]:
    """Format /search results; filter noise and bare UI focus events."""
    lines: list[str] = []
    seen: set[str] = set()
    for item in sorted(items, key=lambda x: (x.get("content") or {}).get("timestamp", "")):
        item_type = item.get("type", "?")
        c = item.get("content", {})
        ts = format_ts_local(c.get("timestamp", ""))
        if item_type == "OCR":
            raw_text = (c.get("text") or "").strip()
            text = extract_valuable_lines(raw_text)
            if len(text) < 20 or is_low_value_text(text):
                continue
            key = normalize_text_key(text)
            if key in seen:
                continue
            seen.add(key)
            app = c.get("app_name", "")
            win = c.get("window_name", "")
            url_str = c.get("browser_url") or ""
            line = f"  [{ts}] SCREEN | {app} | {win}"
            if url_str:
                line += f" | {url_str}"
            if text:
                line += f"\n    {text[:max_text]}"
            lines.append(line)
        elif item_type == "Audio":
            text = (c.get("transcription") or "").strip()
            if len(text) < 8:
                continue
            key = normalize_text_key(text)
            if key in seen:
                continue
            seen.add(key)
            device = c.get("device", "")
            lines.append(f"  [{ts}] AUDIO | {device}\n    {text[:max_text]}")
        elif item_type == "UI" and include_ui_text:
            evt = c.get("event_type", "")
            text_content = (c.get("text_content") or "").strip()
            if evt in _SKIP_UI_EVENTS and evt not in ("text", "clipboard", "click", "key"):
                continue
            app = c.get("app_name", "")
            win = c.get("window_title", "")
            if text_content and not is_low_value_text(text_content):
                key = normalize_text_key(text_content)
                if key in seen:
                    continue
                seen.add(key)
                lines.append(f"  [{ts}] INPUT | {app} | {win}\n    {text_content[:max_text]}")
    return lines


def dedupe_timeline(entries: list[dict[str, Any]], *, max_items: int = 40) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for e in sorted(entries, key=lambda x: x.get("timestamp", "")):
        text = (e.get("text") or "").strip()
        if is_low_value_text(text):
            continue
        key = normalize_text_key(text)
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
        if len(out) >= max_items:
            break
    return out
