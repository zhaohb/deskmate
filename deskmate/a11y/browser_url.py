"""Browser URL extraction via Windows UI Automation.

Uses the **Python** ``uiautomation`` package API (``EditControl``, ``ProcessId``,
``searchFromControl``) to read the address bar on supported browsers.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
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
    "omnibox",
)

_EDIT_TYPE_NAMES = frozenset({"edit", "editcontrol"})


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

    pid = _pid_for_uia_lookup(hwnd, pid)
    search_from = _search_root(auto, hwnd=hwnd)

    # Primary: Python uiautomation typed search (Omnibox is EditControl).
    url = _find_omnibox_url(auto, pid=pid, search_from=search_from)
    if url:
        return url

    root = search_from
    if root is None and pid:
        try:
            root = auto.Control(searchDepth=3, ProcessId=pid)
            if root is not None and not root.Exists(0, 0):
                root = None
        except Exception as exc:  # noqa: BLE001
            logger.debug("Process Control lookup failed: %s", exc)

    if root is None:
        return None

    edit_controls = list(_iter_edit_controls(root, max_nodes=1500))
    if not edit_controls:
        return None

    hinted = [ctrl for ctrl in edit_controls if _looks_like_address_control(ctrl)]
    hinted_ids = {id(ctrl) for ctrl in hinted}
    for ctrl in [*hinted, *[ctrl for ctrl in edit_controls if id(ctrl) not in hinted_ids]]:
        url = normalize_url_text(_read_value(ctrl))
        if url:
            return url
    return None


def _pid_for_uia_lookup(hwnd: int, pid: int) -> int:
    """Map renderer/content HWNDs to the top-level browser process id."""
    if os.name != "nt" or not hwnd:
        return pid
    try:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        root_hwnd = int(user32.GetAncestor(hwnd, 2) or 0)  # GA_ROOT
        if root_hwnd and root_hwnd != hwnd:
            root_pid = wt.DWORD(0)
            user32.GetWindowThreadProcessId(root_hwnd, ctypes.byref(root_pid))
            if root_pid.value:
                return int(root_pid.value)
    except Exception as exc:  # noqa: BLE001
        logger.debug("top-level pid lookup failed: %s", exc)
    return pid


def _search_root(auto: Any, *, hwnd: int) -> Any | None:
    if not hwnd:
        return None
    try:
        ctrl = auto.ControlFromHandle(hwnd)
        if ctrl is not None and ctrl.Exists(0, 0):
            return ctrl
    except Exception as exc:  # noqa: BLE001
        logger.debug("ControlFromHandle failed for URL extraction: %s", exc)
    return None


def _find_omnibox_url(auto: Any, *, pid: int, search_from: Any | None) -> str | None:
    """Locate the browser omnibox via ``EditControl`` search properties."""
    attempts: list[dict[str, Any]] = []
    if search_from is not None:
        attempts.append({"searchFromControl": search_from})
    if pid:
        attempts.append({"ProcessId": pid})
    if search_from is not None and pid:
        attempts.append({"searchFromControl": search_from, "ProcessId": pid})

    for props in attempts:
        try:
            bar = auto.EditControl(searchDepth=0xFFFFFFFF, **props)
            # Brief wait: omnibox may not be synchronously populated at capture time.
            if bar is None or not bar.Exists(0.3, 0):
                continue
            url = normalize_url_text(_read_value(bar))
            if url:
                return url
        except Exception as exc:  # noqa: BLE001
            logger.debug("EditControl search %s failed: %s", props, exc)
    return None


def _is_edit_control(elem: Any) -> bool:
    try:
        ct = (getattr(elem, "ControlTypeName", "") or "").lower()
    except Exception:  # noqa: BLE001
        return False
    return ct in _EDIT_TYPE_NAMES


def _iter_edit_controls(root: Any, *, max_nodes: int) -> list[Any]:
    out: list[Any] = []
    seen = 0

    def walk(elem: Any) -> None:
        nonlocal seen
        if elem is None or seen >= max_nodes:
            return
        seen += 1
        if _is_edit_control(elem):
            out.append(elem)

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
