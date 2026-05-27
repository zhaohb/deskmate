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

    app_l = (app_name or "").lower()
    title_l = (window_title or "").lower()
    for workflow, keyword_groups in _DEFAULT_RULES.items():
        for group in keyword_groups:
            if any(k.lower() in app_l or k.lower() in title_l for k in group):
                return workflow
    return "other"


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
