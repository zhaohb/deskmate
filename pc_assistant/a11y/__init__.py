"""Accessibility + input + clipboard capture."""

from .activity_feed import ActivityFeed, ActivityKind, CaptureParams
from .activity_feed import default as activity_default
from .browser_url import is_browser_app, normalize_url_text, resolve_browser_url
from .clipboard import ClipboardWatcher
from .input_hooks import InputHooks
from .recorder import UiRecorder
from .uia_tree import (
    AccessibilityNode,
    ElementBounds,
    WindowTreeSnapshot,
    foreground_window,
    window_info,
    walk_focused_window,
)
from .win_events import WinEventWatcher, foreground_app_name

__all__ = [
    "AccessibilityNode",
    "ActivityFeed",
    "ActivityKind",
    "CaptureParams",
    "ClipboardWatcher",
    "ElementBounds",
    "InputHooks",
    "UiRecorder",
    "WindowTreeSnapshot",
    "WinEventWatcher",
    "activity_default",
    "foreground_app_name",
    "foreground_window",
    "is_browser_app",
    "normalize_url_text",
    "resolve_browser_url",
    "window_info",
    "walk_focused_window",
]
