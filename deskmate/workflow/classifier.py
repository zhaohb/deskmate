"""Local heuristic workflow classifier with optional HTTP backend."""

from __future__ import annotations

import os

from ..logger import get

logger = get("workflow")

_DEFAULT_RULES: dict[str, tuple[tuple[str, ...], ...]] = {
    "coding": (
        ("Cursor", "Code", "Visual Studio", "WebStorm", "PyCharm", "IntelliJ",
         "RustRover", "Vim", "Emacs", "Sublime"),
        ("WindowsTerminal", "Terminal", "Powershell", "WSL", "cmd"),
    ),
    "browsing": (
        ("chrome", "Chrome", "Edge", "Firefox", "Safari", "Brave", "Vivaldi", "Opera"),
    ),
    "email": (
        ("Outlook", "Thunderbird", "Mailspring", "Mail"),
    ),
    "communication": (
        ("Teams", "Slack", "Discord", "Zoom", "Meet", "Lark", "WeChat", "QQ"),
    ),
    "writing": (
        ("Word", "Pages", "WPS", "Notion", "Obsidian", "Bear", "TextEdit"),
    ),
    "meeting": (
        ("Teams - Meeting", "Zoom Meeting", "Google Meet"),
    ),
}

# Browser process names. When the foreground *app* is a browser, the workflow is
# browsing regardless of what the page title says — a YouTube tutorial titled
# "... Code ..." or a Slack web tab whose title contains "meet" must not be
# reclassified as coding/meeting just because a keyword appears in the title.
_BROWSER_APPS = (
    "chrome", "msedge", "edge", "firefox", "safari", "brave", "vivaldi",
    "opera", "arc",
)


def _classify_local(app_name: str, window_title: str) -> str:
    """Heuristic classification: match the app name first, fall back to title.

    Matching the *app* before the title avoids title-substring false positives
    (e.g. a browser tab titled "...Code..." → coding). Browser apps get a
    negative override: their workflow is always ``browsing``, since the page
    title is not a reliable signal of what the user is actually doing.
    """
    app_l = (app_name or "").lower()
    title_l = (window_title or "").lower()

    # Pass 1: app-name match wins (most authoritative signal).
    for workflow, keyword_groups in _DEFAULT_RULES.items():
        for group in keyword_groups:
            if any(k.lower() in app_l for k in group):
                # Browser negative override: a browser app is browsing even when
                # its own rule also matched (the "Chrome" → browsing case) and
                # even if a later title check would have matched something else.
                if any(b in app_l for b in _BROWSER_APPS):
                    return "browsing"
                return workflow

    # Browser with no other app match (e.g. process name only in title rules).
    if any(b in app_l for b in _BROWSER_APPS):
        return "browsing"

    # Pass 2: title fallback — only when the app name gave no signal at all, and
    # never for browser apps (handled above).
    for workflow, keyword_groups in _DEFAULT_RULES.items():
        for group in keyword_groups:
            if any(k.lower() in title_l for k in group):
                return workflow
    return "other"


def classify_frame(app_name: str, window_title: str = "") -> str:
    endpoint = os.environ.get("WORKFLOW_CLASSIFIER")
    if endpoint:
        try:
            import httpx  # type: ignore[import-not-found]
            r = httpx.post(endpoint, json={"app": app_name, "title": window_title}, timeout=2.0)
            r.raise_for_status()
            wf = r.json().get("workflow")
            if wf:
                return str(wf)
        except Exception as exc:  # noqa: BLE001
            logger.debug("remote workflow classifier failed: %s", exc)

    return _classify_local(app_name, window_title)


class WorkflowClassifier:
    def __init__(self) -> None:
        self.cache: dict[tuple[str, str], str] = {}

    def classify(self, app_name: str, window_title: str = "") -> str:
        key = (app_name or "", window_title or "")
        if key in self.cache:
            return self.cache[key]
        v = classify_frame(*key)
        self.cache[key] = v
        return v
