"""UI Automation tree walker for Windows.

Two design choices:
1. **Field schema is identical** — `control_type`, `name`, `automation_id`,
   `class_name`, `value`, `bounds: {x,y,width,height}`, `is_enabled`,
   `is_focused`, `is_keyboard_focusable`, `help_text`, `is_password`,
   `is_selected`, `is_expanded`, `accelerator_key`, `access_key`,
   `localized_control_type`, `children`. Optional fields are omitted when
   empty (matches Rust's `skip_serializing_if = "Option::is_none"`).
2. **CacheRequest batching** — the full-window walk prefetches the entire
   subtree into a UIA cache in a single `BuildUpdatedCache` COM round trip
   (subtree scope, raw-view filter to match the walker), then reads every
   node's properties and children from that cache. This replaces 5000+
   per-property cross-process COM calls with one, ~5-9× faster on complex
   trees. The cache surface lives on the raw `comtypes` element
   (`uiautomation`'s wrapper comments out the cached accessors), so when that
   surface is unavailable the walk transparently falls back to live
   per-property reads — never worse than before, just slower.

Additionally we expose `on_screen` (whether the window's foreground rect
overlaps the visible monitor area), `depth`, and a `lines[]` projection of
the textual content for downstream consumers that want line-level OCR-like
data.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from ..logger import get
from .uia_thread import uia_com_session

logger = get("a11y.uia")

SKIP_TYPES = {"ScrollBar", "Image", "Separator", "Thumb", "ToolTip", "ProgressBar"}
INTERACTIVE_TYPES = {
    "Button", "Edit", "Document", "ComboBox", "CheckBox", "RadioButton",
    "Slider", "Tab", "TabItem", "MenuItem", "Hyperlink",
}


def _omit_none(d: dict[str, Any]) -> dict[str, Any]:
    """Drop keys whose value is None / "" / False-but-flagged-optional /
    empty containers (matches serde `skip_serializing_if = Option::is_none`)."""
    return {k: v for k, v in d.items() if v not in (None, "", [], {})}


@dataclass
class ElementBounds:
    """Element bounding rectangle."""
    x: float
    y: float
    width: float
    height: float

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass
class AccessibilityNode:
    """Accessibility tree node.

    Required fields are emitted unconditionally; optional fields are
    suppressed when None / empty.
    """

    # Required
    control_type: str = ""
    is_enabled: bool = True
    depth: int = 0

    # Optional
    name: str | None = None
    automation_id: str | None = None
    class_name: str | None = None
    value: str | None = None
    bounds: ElementBounds | None = None
    is_focused: bool | None = None
    is_keyboard_focusable: bool | None = None
    help_text: str | None = None
    is_password: bool | None = None
    is_selected: bool | None = None
    is_expanded: bool | None = None
    accelerator_key: str | None = None
    access_key: str | None = None
    localized_control_type: str | None = None
    on_screen: bool | None = None
    lines: list[str] | None = None

    children: list[AccessibilityNode] = field(default_factory=list)

    # ----- traversal helpers ------------------------------------------------
    def node_count(self) -> int:
        return 1 + sum(c.node_count() for c in self.children)

    def max_depth(self) -> int:
        if not self.children:
            return 0
        return 1 + max(c.max_depth() for c in self.children)

    def find_by_type(self, control_type: str) -> AccessibilityNode | None:
        if self.control_type.lower() == control_type.lower():
            return self
        for c in self.children:
            found = c.find_by_type(control_type)
            if found is not None:
                return found
        return None

    def find_by_name(self, name: str) -> AccessibilityNode | None:
        if self.name and name in self.name:
            return self
        for c in self.children:
            found = c.find_by_name(name)
            if found is not None:
                return found
        return None

    def named_node_count(self) -> int:
        self_count = 1 if (self.name and self.name.strip()) else 0
        return self_count + sum(c.named_node_count() for c in self.children)

    def interactive_count(self) -> int:
        self_count = 1 if any(
            self.control_type.lower() == t.lower() for t in INTERACTIVE_TYPES
        ) else 0
        return self_count + sum(c.interactive_count() for c in self.children)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "control_type": self.control_type,
            "is_enabled": self.is_enabled,
            "depth": self.depth,
        }
        if self.name is not None:           d["name"] = self.name
        if self.automation_id is not None:  d["automation_id"] = self.automation_id
        if self.class_name is not None:     d["class_name"] = self.class_name
        if self.value is not None:          d["value"] = self.value
        if self.bounds is not None:         d["bounds"] = self.bounds.to_dict()
        if self.is_focused is not None:     d["is_focused"] = self.is_focused
        if self.is_keyboard_focusable is not None:
            d["is_keyboard_focusable"] = self.is_keyboard_focusable
        if self.help_text is not None:      d["help_text"] = self.help_text
        if self.is_password is not None:    d["is_password"] = self.is_password
        if self.is_selected is not None:    d["is_selected"] = self.is_selected
        if self.is_expanded is not None:    d["is_expanded"] = self.is_expanded
        if self.accelerator_key is not None: d["accelerator_key"] = self.accelerator_key
        if self.access_key is not None:     d["access_key"] = self.access_key
        if self.localized_control_type is not None:
            d["localized_control_type"] = self.localized_control_type
        if self.on_screen is not None:      d["on_screen"] = self.on_screen
        if self.lines:                      d["lines"] = self.lines
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d


@dataclass
class WindowTreeSnapshot:
    app_name: str
    window_name: str
    pid: int
    hwnd: int
    captured_at: str
    focused_role: str | None
    focused_name: str | None
    focused_value: str | None
    on_screen: bool
    root: AccessibilityNode | None
    text: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "app_name": self.app_name,
                "window_name": self.window_name,
                "pid": self.pid,
                "hwnd": self.hwnd,
                "captured_at": self.captured_at,
                "focused_role": self.focused_role,
                "focused_name": self.focused_name,
                "focused_value": self.focused_value,
                "on_screen": self.on_screen,
                "root": self.root.to_dict() if self.root else None,
                "text": self.text,
            },
            ensure_ascii=False,
        )


# ─── Win32 / UIA glue ───────────────────────────────────────────────────────
def foreground_window() -> tuple[int, int, str]:
    """(hwnd, pid, title)."""
    if os.name != "nt":
        return (0, 0, "")
    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    hwnd = user32.GetForegroundWindow() or 0
    if not hwnd:
        return (0, 0, "")
    return window_info(int(hwnd))


def window_info(hwnd: int) -> tuple[int, int, str]:
    """Return (hwnd, pid, title) for a specific window handle."""
    if os.name != "nt" or not hwnd:
        return (0, 0, "")
    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    pid = wt.DWORD(0)
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    buf = ctypes.create_unicode_buffer(1024)
    user32.GetWindowTextW(hwnd, buf, 1024)
    return int(hwnd), int(pid.value), buf.value or ""


def _safe_str(value: Any, max_len: int = 1000) -> str:
    s = str(value or "").strip()
    return s[:max_len] + "…" if len(s) > max_len else s


def _safe_control_type(elem: Any) -> str | None:
    """Read ``ControlTypeName``; return None when the UIA element is stale.

    ``getattr(elem, "ControlTypeName", "")`` does not help: the property getter
    calls COM and can raise ``COMError`` (e.g. 0x80040201) while the window tree
    is changing.
    """
    try:
        ct = getattr(elem, "ControlTypeName", "") or ""
        return ct or "Unknown"
    except Exception:  # noqa: BLE001
        return None


def _safe_class_name(elem: Any) -> str:
    """Read the focused element's ``ClassName`` defensively.

    The ClassName disambiguates chat input boxes that share a generic
    ``EditControl`` role — e.g. Cursor's ``aislash-editor-input`` vs VS Code's
    ``native-edit-context`` Monaco editor. Returns ``""`` on any COM failure.
    """
    try:
        return (getattr(elem, "ClassName", "") or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def _safe_name(elem: Any) -> str:
    """Read the focused element's accessibility ``Name`` defensively.

    The Name disambiguates chat inputs that share both a generic ``EditControl``
    role and a shared ClassName — e.g. VS Code Copilot's chat composer (Name
    starts with ``"Chat Input"``) vs an ordinary Monaco editor, both of which
    use ClassName ``native-edit-context``. Returns ``""`` on any COM failure.
    """
    try:
        return (getattr(elem, "Name", "") or "").strip()
    except Exception:  # noqa: BLE001
        return ""


# UIA property IDs we prefetch into the cache and read back per node. The
# constants live on the UIAutomationCore typelib (NOT the top-level
# ``uiautomation`` module — an earlier version of this file read them off the
# module via getattr and silently got ``None`` for every one, so the cache was
# built empty and never used). We resolve them by name once and reuse the ints.
_CACHE_PROP_NAMES = (
    "UIA_NamePropertyId", "UIA_AutomationIdPropertyId", "UIA_ClassNamePropertyId",
    "UIA_ControlTypePropertyId", "UIA_IsEnabledPropertyId",
    "UIA_HasKeyboardFocusPropertyId", "UIA_IsKeyboardFocusablePropertyId",
    "UIA_HelpTextPropertyId", "UIA_IsPasswordPropertyId",
    "UIA_BoundingRectanglePropertyId", "UIA_AcceleratorKeyPropertyId",
    "UIA_AccessKeyPropertyId", "UIA_LocalizedControlTypePropertyId",
    "UIA_IsOffscreenPropertyId", "UIA_ValueValuePropertyId",
    "UIA_SelectionItemIsSelectedPropertyId",
    "UIA_ExpandCollapseExpandCollapseStatePropertyId",
    "UIA_IsValuePatternAvailablePropertyId", "UIA_IsTextPatternAvailablePropertyId",
    "UIA_IsSelectionItemPatternAvailablePropertyId",
    "UIA_IsExpandCollapsePatternAvailablePropertyId",
)


class _Uia:
    """Resolved UIA COM handles + property IDs for one cached walk.

    Holding the factory, the configured CacheRequest and the property-id map in
    one object keeps the walk readable and avoids re-resolving constants per
    node. Construction returns ``None`` (via :func:`_resolve_uia`) when the
    installed ``uiautomation`` build doesn't expose the raw COM surface, so the
    caller transparently falls back to the live per-property walk.
    """

    def __init__(self, factory: Any, core: Any, control_type_names: dict[int, str]) -> None:
        self.factory = factory
        self.core = core
        self.control_type_names = control_type_names
        self.pid = {name: getattr(core, name) for name in _CACHE_PROP_NAMES}

    def build_cache_request(self) -> Any:
        cache = self.factory.CreateCacheRequest()
        for pid in self.pid.values():
            cache.AddProperty(pid)
        # Subtree scope caches the whole tree in ONE COM round trip (vs one per
        # property per node). The Raw view filter matches the library's
        # ViewWalker (RawViewWalker) so the cached node set is identical to the
        # live walk's. AutomationElementMode is left at its default (Full) so
        # pattern-backed properties (Value, ExpandCollapse…) populate.
        cache.TreeScope = self.core.TreeScope_Subtree
        try:
            cache.TreeFilter = self.factory.RawViewCondition
        except Exception:  # noqa: BLE001
            pass
        return cache


def _resolve_uia() -> _Uia | None:
    """Resolve the raw IUIAutomation factory + core typelib, or None.

    Both live on ``uiautomation.uiautomation._AutomationClient`` in the
    supported package version (2.0.x). Any AttributeError/ImportError means the
    build is too old or too new to drive caching directly — return None so the
    caller uses the live walk.
    """
    try:
        import uiautomation as _auto  # noqa: PLC0415
        from uiautomation import uiautomation as _uia_impl  # noqa: PLC0415

        client = _uia_impl._AutomationClient.instance()
        return _Uia(client.IUIAutomation, client.UIAutomationCore, _auto.ControlTypeNames)
    except Exception as exc:  # noqa: BLE001
        logger.debug("raw UIA surface unavailable; using live property reads: %s", exc)
        return None


def walk_focused_window(
    *,
    max_depth: int = 60,
    hwnd: int = 0,
    max_nodes: int = 5000,
) -> WindowTreeSnapshot | None:
    """Walk the focused window's UIA subtree. Returns None when UIA isn't
    available (off-Windows, package missing, no foreground window)."""
    if os.name != "nt":
        return None
    try:
        import uiautomation as auto  # noqa: PLC0415
    except ImportError:
        logger.warning("uiautomation not installed; UIA tree disabled")
        return None

    with uia_com_session():
        return _walk_focused_window_impl(
            auto, max_depth=max_depth, hwnd=hwnd, max_nodes=max_nodes,
        )


def read_focused_value(
    hwnd: int = 0, *, max_len: int = 4000,
) -> tuple[str, str, str, str]:
    """Read the focused control's ``(role, value, class_name, name)`` — fast, no walk.

    Used for on-demand prompt capture (e.g. when the user pauses typing or
    sends a message) so a chat input box's *full* text — including pasted and
    IME-composed content the low-level keystroke buffer may miss — is recorded.
    ``class_name`` disambiguates chat inputs that share a generic role (e.g.
    Cursor's ``aislash-editor-input`` vs an ordinary editor); ``name``
    disambiguates those that share both role *and* class (e.g. VS Code
    Copilot's chat composer, whose Name starts with ``"Chat Input"``, vs an
    ordinary Monaco editor — both ``native-edit-context``). Returns
    ``("", "", "", "")`` off-Windows, when UIA is unavailable, for password
    fields, or on any failure. ``hwnd`` is currently advisory; the system-wide
    focused control is always the one the user is typing into.
    """
    if os.name != "nt":
        return ("", "", "", "")
    try:
        import uiautomation as auto  # noqa: PLC0415
    except ImportError:
        return ("", "", "", "")

    def _do() -> tuple[str, str, str, str]:
        try:
            f = auto.GetFocusedControl()
        except Exception:  # noqa: BLE001
            return ("", "", "", "")
        if f is None:
            return ("", "", "", "")
        role = _safe_control_type(f) or ""
        cls = _safe_class_name(f)
        name = _safe_name(f)
        if _safe_is_password(f):
            return (role, "", cls, name)
        value = _read_value(f) or ""
        if max_len and len(value) > max_len:
            value = value[:max_len]
        return (role, value, cls, name)

    try:
        with uia_com_session():
            return _do()
    except Exception:  # noqa: BLE001
        return ("", "", "", "")


def _walk_focused_window_impl(
    auto: Any,
    *,
    max_depth: int,
    hwnd: int,
    max_nodes: int,
) -> WindowTreeSnapshot | None:
    from datetime import datetime, timezone  # noqa: PLC0415

    if hwnd == 0:
        hwnd, pid, title = foreground_window()
    else:
        hwnd, pid, title = window_info(hwnd)
    if not hwnd:
        return None

    try:
        from .win_events import foreground_app_name  # noqa: PLC0415
        app_name = foreground_app_name(pid)
    except Exception:  # noqa: BLE001
        app_name = ""

    try:
        root = auto.ControlFromHandle(hwnd)
    except Exception as exc:  # noqa: BLE001
        logger.debug("ControlFromHandle failed: %s", exc)
        return None
    if root is None:
        return None

    flat_text: list[str] = []
    nodes_seen = 0

    focused_role: str | None = None
    focused_name: str | None = None
    focused_value: str | None = None
    try:
        f = auto.GetFocusedControl()
        if f is not None:
            focused_role = _safe_control_type(f)
            focused_name = getattr(f, "Name", None) or None
            focused_value = "[REDACTED]" if _safe_is_password(f) else _read_value(f)
    except Exception:  # noqa: BLE001
        pass

    # Fast path: prefetch the whole subtree into a UIA cache in one COM round
    # trip, then read every node's properties from the cache. On a 5000-node
    # tree this replaces 5000+ cross-process COM calls with a single one. Any
    # failure (old/new package, COM error mid-build) falls through to the live
    # per-property walk below, so behaviour is never worse than before.
    uia = _resolve_uia()
    if uia is not None:
        try:
            cached_root = root.Element.BuildUpdatedCache(uia.build_cache_request())
            root_node, cached_text = _walk_cached(
                cached_root, uia, max_depth=max_depth, max_nodes=max_nodes,
            )
            if root_node is not None:
                visible_text = "\n".join(cached_text)[:10_000]
                win_on_screen = root_node.on_screen
                return WindowTreeSnapshot(
                    app_name=app_name,
                    window_name=title,
                    pid=pid,
                    hwnd=hwnd,
                    captured_at=datetime.now(timezone.utc).astimezone().isoformat(),
                    focused_role=focused_role,
                    focused_name=focused_name,
                    focused_value=focused_value,
                    on_screen=bool(win_on_screen) if win_on_screen is not None else True,
                    root=root_node,
                    text=visible_text,
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("cached UIA walk failed (%s); falling back to live walk", exc)
            flat_text = []
            nodes_seen = 0

    def _walk(elem: Any, depth: int) -> AccessibilityNode | None:
        nonlocal nodes_seen
        if nodes_seen >= max_nodes:
            return None
        nodes_seen += 1
        ct = _safe_control_type(elem)
        if ct is None:
            return None
        if ct in SKIP_TYPES:
            return None
        try:
            name = (getattr(elem, "Name", "") or "").strip() or None
            aid = (getattr(elem, "AutomationId", "") or "").strip() or None
            cls = (getattr(elem, "ClassName", "") or "").strip() or None
            secure = _safe_is_password(elem)
            value_str = "[REDACTED]" if secure else _read_value(elem)
            bounds = _read_bounds(elem)
            help_text = _read_optional_attr(elem, "Tooltip")
            localized = _read_optional_attr(elem, "LocalizedControlType")
            acc_key = _read_optional_attr(elem, "AcceleratorKey")
            access_key = _read_optional_attr(elem, "AccessKey")

            is_focused = bool(getattr(elem, "HasKeyboardFocus", False))
            try:
                is_kb_focusable = bool(getattr(elem, "IsKeyboardFocusable", False))
            except Exception:  # noqa: BLE001
                is_kb_focusable = None
            try:
                is_enabled = bool(getattr(elem, "IsEnabled", True))
            except Exception:  # noqa: BLE001
                is_enabled = True
            try:
                is_off = bool(getattr(elem, "IsOffscreen", False))
                on_screen = (not is_off)
            except Exception:  # noqa: BLE001
                on_screen = None

            is_selected = _selection_state(elem)
            is_expanded = _expand_state(elem)

            node = AccessibilityNode(
                control_type=ct or "Unknown",
                is_enabled=is_enabled,
                depth=depth,
                name=name,
                automation_id=aid,
                class_name=cls,
                value=value_str or None,
                bounds=bounds,
                is_focused=is_focused if is_focused else None,
                is_keyboard_focusable=is_kb_focusable,
                help_text=help_text,
                is_password=True if secure else None,
                is_selected=is_selected,
                is_expanded=is_expanded,
                accelerator_key=acc_key,
                access_key=access_key,
                localized_control_type=localized,
                on_screen=on_screen,
            )

            collectible = name or value_str
            if collectible:
                flat_text.append(_safe_str(collectible))
                if value_str and "\n" in value_str:
                    node.lines = [ln for ln in value_str.splitlines() if ln.strip()]
        except Exception:  # noqa: BLE001
            return None

        if depth + 1 >= max_depth:
            return node
        try:
            child = elem.GetFirstChildControl()
        except Exception:  # noqa: BLE001
            child = None
        while child is not None:
            try:
                sub = _walk(child, depth + 1)
                if sub is not None:
                    node.children.append(sub)
                child = child.GetNextSiblingControl()
            except Exception:  # noqa: BLE001
                break
        return node

    try:
        root_node = _walk(root, 0)
    except Exception as exc:  # noqa: BLE001
        logger.debug("UIA walk aborted for hwnd=%s: %s", hwnd, exc)
        root_node = None
    visible_text = "\n".join(flat_text)[:10_000]
    win_on_screen = (root_node.on_screen if root_node else None)
    return WindowTreeSnapshot(
        app_name=app_name,
        window_name=title,
        pid=pid,
        hwnd=hwnd,
        captured_at=datetime.now(timezone.utc).astimezone().isoformat(),
        focused_role=focused_role,
        focused_name=focused_name,
        focused_value=focused_value,
        on_screen=bool(win_on_screen) if win_on_screen is not None else True,
        root=root_node,
        text=visible_text,
    )


# ─── cached walk (fast path) ─────────────────────────────────────────────────
def _cached(elem: Any, pid: int) -> Any:
    """Read one cached property; None on any COM hiccup."""
    try:
        return elem.GetCachedPropertyValue(pid)
    except Exception:  # noqa: BLE001
        return None


def _cached_str(elem: Any, pid: int) -> str | None:
    v = _cached(elem, pid)
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _cached_bool(elem: Any, pid: int, default: bool | None) -> bool | None:
    v = _cached(elem, pid)
    if v is None:
        return default
    return bool(v)


def _walk_cached(
    cached_root: Any,
    uia: _Uia,
    *,
    max_depth: int,
    max_nodes: int,
) -> tuple[AccessibilityNode | None, list[str]]:
    """Build the node tree by reading from the prefetched cache only.

    Mirrors the live ``_walk`` field-for-field (same SKIP_TYPES filtering, same
    optional-field semantics, same ``lines``/``flat_text`` projection) so the
    snapshot is byte-identical to the live path — only the data source differs
    (cached reads vs per-property COM calls). Children come from
    ``GetCachedChildren`` (also served from the single cached subtree).
    """
    pid = uia.pid
    flat_text: list[str] = []
    state = {"seen": 0}

    def _walk(elem: Any, depth: int) -> AccessibilityNode | None:
        if state["seen"] >= max_nodes:
            return None
        state["seen"] += 1

        ct_id = _cached(elem, pid["UIA_ControlTypePropertyId"])
        ct = uia.control_type_names.get(ct_id, "Unknown") if ct_id is not None else "Unknown"
        if ct in SKIP_TYPES:
            return None

        secure = bool(_cached_bool(elem, pid["UIA_IsPasswordPropertyId"], False))

        # Value: prefer ValuePattern's cached value; fall back to None. (The
        # live path also consults TextPattern, which isn't a plain cacheable
        # property — a node that only exposes text via TextPattern keeps its
        # name as collectible text, matching the common case.)
        value_str: str | None = None
        if secure:
            value_str = "[REDACTED]"
        elif _cached_bool(elem, pid["UIA_IsValuePatternAvailablePropertyId"], False):
            value_str = _cached_str(elem, pid["UIA_ValueValuePropertyId"])
            if value_str:
                value_str = _safe_str(value_str)

        name = _cached_str(elem, pid["UIA_NamePropertyId"])
        bounds = _cached_bounds(elem, pid["UIA_BoundingRectanglePropertyId"])

        is_focused = bool(_cached_bool(elem, pid["UIA_HasKeyboardFocusPropertyId"], False))
        is_kb_focusable = _cached_bool(elem, pid["UIA_IsKeyboardFocusablePropertyId"], None)
        is_enabled = bool(_cached_bool(elem, pid["UIA_IsEnabledPropertyId"], True))
        is_off = _cached_bool(elem, pid["UIA_IsOffscreenPropertyId"], None)
        on_screen = (not is_off) if is_off is not None else None

        is_selected: bool | None = None
        if _cached_bool(elem, pid["UIA_IsSelectionItemPatternAvailablePropertyId"], False):
            is_selected = _cached_bool(elem, pid["UIA_SelectionItemIsSelectedPropertyId"], None)
        is_expanded: bool | None = None
        if _cached_bool(elem, pid["UIA_IsExpandCollapsePatternAvailablePropertyId"], False):
            state_val = _cached(elem, pid["UIA_ExpandCollapseExpandCollapseStatePropertyId"])
            if state_val is not None:
                # 1 == Expanded (0 Collapsed, 2 Partial, 3 LeafNode).
                is_expanded = int(state_val) == 1

        node = AccessibilityNode(
            control_type=ct,
            is_enabled=is_enabled,
            depth=depth,
            name=name,
            automation_id=_cached_str(elem, pid["UIA_AutomationIdPropertyId"]),
            class_name=_cached_str(elem, pid["UIA_ClassNamePropertyId"]),
            value=value_str or None,
            bounds=bounds,
            is_focused=is_focused if is_focused else None,
            is_keyboard_focusable=is_kb_focusable,
            help_text=_cached_str(elem, pid["UIA_HelpTextPropertyId"]),
            is_password=True if secure else None,
            is_selected=is_selected,
            is_expanded=is_expanded,
            accelerator_key=_cached_str(elem, pid["UIA_AcceleratorKeyPropertyId"]),
            access_key=_cached_str(elem, pid["UIA_AccessKeyPropertyId"]),
            localized_control_type=_cached_str(elem, pid["UIA_LocalizedControlTypePropertyId"]),
            on_screen=on_screen,
        )

        collectible = name or value_str
        if collectible:
            flat_text.append(_safe_str(collectible))
            if value_str and "\n" in value_str:
                node.lines = [ln for ln in value_str.splitlines() if ln.strip()]

        if depth + 1 >= max_depth:
            return node
        children, n_children = _cached_children(elem)
        for i in range(n_children):
            if state["seen"] >= max_nodes:
                break
            try:
                child_elem = children.GetElement(i)
            except Exception:  # noqa: BLE001
                continue
            sub = _walk(child_elem, depth + 1)
            if sub is not None:
                node.children.append(sub)
        return node

    root_node = _walk(cached_root, 0)
    return root_node, flat_text


def _cached_children(elem: Any) -> tuple[Any, int]:
    """Return ``(children_array, count)``; ``(None, 0)`` when absent.

    ``GetCachedChildren`` can hand back a NULL COM pointer (leaf node) that is
    not ``None`` yet raises ``ValueError: NULL COM pointer access`` on ``.Length``
    — so the count read must be guarded, not just the call.
    """
    try:
        children = elem.GetCachedChildren()
        return children, int(children.Length)
    except Exception:  # noqa: BLE001
        return None, 0


def _cached_bounds(elem: Any, pid: int) -> ElementBounds | None:
    """Cached BoundingRectangle is ``(x, y, width, height)`` floats."""
    v = _cached(elem, pid)
    try:
        if v and len(v) == 4:
            x, y, w, h = (float(n) for n in v)
            if w > 0 and h > 0:
                return ElementBounds(x=x, y=y, width=w, height=h)
    except Exception:  # noqa: BLE001
        pass
    return None


# ─── tiny readers ───────────────────────────────────────────────────────────
def _safe_is_password(elem: Any) -> bool:
    try:
        return bool(getattr(elem, "CurrentIsPassword", False))
    except Exception:  # noqa: BLE001
        return False


def _read_value(elem: Any) -> str:
    try:
        vp = getattr(elem, "GetValuePattern", None)
        if vp is not None:
            pat = vp()
            if pat is not None:
                v = getattr(pat, "Value", "") or ""
                if v:
                    return _safe_str(v)
    except Exception:  # noqa: BLE001
        pass
    try:
        tp = getattr(elem, "GetTextPattern", None)
        if tp is not None:
            pat = tp()
            if pat is not None:
                rng = pat.DocumentRange
                if rng is not None:
                    return _safe_str(rng.GetText(2000))
    except Exception:  # noqa: BLE001
        pass
    return ""


def _read_bounds(elem: Any) -> ElementBounds | None:
    try:
        r = elem.BoundingRectangle
        if r and (r.right > r.left and r.bottom > r.top):
            return ElementBounds(
                x=float(r.left), y=float(r.top),
                width=float(r.right - r.left), height=float(r.bottom - r.top),
            )
    except Exception:  # noqa: BLE001
        pass
    return None


def _read_optional_attr(elem: Any, attr: str) -> str | None:
    try:
        v = getattr(elem, attr, "") or ""
        v = str(v).strip()
        return v or None
    except Exception:  # noqa: BLE001
        return None


def _selection_state(elem: Any) -> bool | None:
    try:
        getter = getattr(elem, "GetSelectionItemPattern", None)
        if getter is not None:
            pat = getter()
            if pat is not None:
                return bool(getattr(pat, "IsSelected", False))
    except Exception:  # noqa: BLE001
        pass
    return None


def _expand_state(elem: Any) -> bool | None:
    try:
        getter = getattr(elem, "GetExpandCollapsePattern", None)
        if getter is not None:
            pat = getter()
            if pat is not None:
                # ExpandCollapseState: 0=Collapsed, 1=Expanded, 2=PartiallyExpanded, 3=LeafNode
                state = int(getattr(pat, "ExpandCollapseState", 0))
                return state == 1
    except Exception:  # noqa: BLE001
        pass
    return None


# Compatibility shim — old code path expected dataclass-style asdict on snapshots
def _legacy_asdict(snap: WindowTreeSnapshot) -> dict[str, Any]:
    return asdict(snap)
