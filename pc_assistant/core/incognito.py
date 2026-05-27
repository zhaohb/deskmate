"""Best-effort private/incognito window detection from the window title.

Pure string matching is enough for mainstream browsers; full incognito
detection requires per-browser APIs we deliberately skip."""

from __future__ import annotations

_KEYWORDS = [
    "incognito",
    "private browsing",
    "inprivate",
    "private window",
    "无痕",
    "隐私浏览",
    "プライベートブラウズ",
    "비공개",
]


def is_title_private(title: str) -> bool:
    if not title:
        return False
    low = title.lower()
    return any(k in low for k in _KEYWORDS)
