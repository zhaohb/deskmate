"""Browser URL extraction via Windows UI Automation.

This mirrors screenpipe's Windows strategy: for a focused browser process,
inspect the UIA tree, read the address bar Edit control value, normalize it,
and validate the URL locally without making network requests.
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

from ..logger import get

logger = get("a11y.browser_url")

BROWSER_APP_TOKENS = (
    "chrome",
    "msedge",
    "edge",
    "firefox",
    "brave",
    "vivaldi",
    "opera",
    "arc",
)

ADDRESS_HINTS = (
    "address",
    "地址",
    "url",
    "search or enter",
    "search with",
    "web address",
    "adresse",
    "dirección",
)


def is_browser_app(app_name: str) -> bool:
    """Return True when the process name looks like a supported browser."""
    app = (app_name or "").lower()
    return any(token in app for token in BROWSER_APP_TOKENS)


def normalize_url_text(text: str) -> str | None:
    """Normalize address-bar text to an http(s) URL, or None for search text."""
    raw = (text or "").strip()
    if not raw or any(ch.isspace() for ch in raw):
        return None

    lower = raw.lower()
    if lower.startswith(("http://", "https://")):
        candidate = raw
    elif "://" in raw:
        return None
    else:
        candidate = f"https://{raw}"

    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return candidate


def resolve_browser_url(
    app_name: str,
    *,
    pid: int = 0,
    hwnd: int = 0,
) -> str | None:
    """Best-effort URL resolver for the active browser window on Windows."""
    if os.name != "nt" or not is_browser_app(app_name):
        return None
    try:
        import uiautomation as auto  # noqa: PLC0415
    except ImportError:
        logger.debug("uiautomation not installed; browser URL disabled")
        return None

    root = _root_control(auto, hwnd=hwnd, pid=pid)
    if root is None:
        return None

    edit_controls = list(_iter_edit_controls(root, max_nodes=1500))
    if not edit_controls:
        return None

    # Prefer controls whose accessible name looks like an address bar. If those
    # fail validation, fall back to screenpipe's "first Edit with URLish value".
    hinted = [ctrl for ctrl in edit_controls if _looks_like_address_control(ctrl)]
    hinted_ids = {id(ctrl) for ctrl in hinted}
    for ctrl in [*hinted, *[ctrl for ctrl in edit_controls if id(ctrl) not in hinted_ids]]:
        value = _read_value(ctrl)
        url = normalize_url_text(value)
        if url:
            return url
    return None


def _root_control(auto: Any, *, hwnd: int, pid: int) -> Any | None:
    if hwnd:
        try:
            return auto.ControlFromHandle(hwnd)
        except Exception as exc:  # noqa: BLE001
            logger.debug("ControlFromHandle failed for URL extraction: %s", exc)

    if pid:
        try:
            return auto.Control(searchDepth=1, ProcessId=pid)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Process root lookup failed for URL extraction: %s", exc)
    return None


def _iter_edit_controls(root: Any, *, max_nodes: int) -> list[Any]:
    out: list[Any] = []
    seen = 0

    def walk(elem: Any) -> None:
        nonlocal seen
        if elem is None or seen >= max_nodes:
            return
        seen += 1
        try:
            if (getattr(elem, "ControlTypeName", "") or "").lower() == "edit":
                out.append(elem)
        except Exception:  # noqa: BLE001
            pass

        try:
            child = elem.GetFirstChildControl()
        except Exception:  # noqa: BLE001
            child = None
        while child is not None and seen < max_nodes:
            walk(child)
            try:
                child = child.GetNextSiblingControl()
            except Exception:  # noqa: BLE001
                break

    walk(root)
    return out


def _looks_like_address_control(ctrl: Any) -> bool:
    parts: list[str] = []
    for attr in ("Name", "AutomationId", "ClassName", "LocalizedControlType"):
        try:
            value = getattr(ctrl, attr, "") or ""
        except Exception:  # noqa: BLE001
            value = ""
        if value:
            parts.append(str(value).lower())
    haystack = " ".join(parts)
    return any(hint in haystack for hint in ADDRESS_HINTS)


def _read_value(ctrl: Any) -> str:
    try:
        getter = getattr(ctrl, "GetValuePattern", None)
        if getter is not None:
            pattern = getter()
            if pattern is not None:
                value = getattr(pattern, "Value", "") or ""
                if value:
                    return str(value).strip()
    except Exception:  # noqa: BLE001
        pass
    try:
        return str(getattr(ctrl, "Value", "") or "").strip()
    except Exception:  # noqa: BLE001
        return ""
