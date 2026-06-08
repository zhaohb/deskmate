"""Extract the active document/file from an editor window title.

`frames.document_path` answers "which file was the user working on" — a strong
signal for activity summaries that window/app names alone can't give. The most
portable, zero-cost source is the window title: every mainstream editor puts the
current file at the front (e.g. ``config.toml - project - Visual Studio Code``,
``*notes.md - Notepad``). We parse that conservatively and return ``None`` when
the title doesn't clearly name a document, so we never invent a path.

This is title-only by design: editor *contents* don't reach the UIA tree, but
the filename in the title bar is reliable and free (we already have the title).
"""

from __future__ import annotations

import re

# Process names whose window title is "<file> - <...> - <editor>" shaped. We
# only trust the title→document mapping for known editors; a generic app's title
# is not a filename.
_EDITOR_APP_TOKENS = (
    "code",          # VS Code / VSCodium (Code.exe)
    "cursor",        # Cursor
    "windsurf",      # Windsurf
    "devenv",        # Visual Studio
    "notepad",       # Notepad / Notepad++
    "sublime_text",  # Sublime Text
    "idea", "pycharm", "webstorm", "goland", "clion", "rider", "phpstorm",
    "rubymine", "datagrip",  # JetBrains family
    "gvim", "vim", "nvim",   # (Neo)vim GUIs
    "atom", "zed",
)

# Trailing " - <Editor name>" segments to strip before isolating the filename.
_EDITOR_SUFFIXES = (
    "visual studio code",
    "visual studio",
    "cursor",
    "windsurf",
    "notepad++",
    "notepad",
    "记事本",
    "sublime text",
    "atom",
    "zed",
)

# A filename looks like ``name.ext`` (1–8 char extension). Optional leading
# markers for unsaved state ("●", "*") are stripped by the caller.
_FILENAME_RE = re.compile(r"^[^\\/:*?\"<>|]+\.[A-Za-z0-9]{1,8}$")

# JetBrains/VS often show "file.py – project [/abs/path]" or "project – file.py".
# We keep this simple: split on common separators and look for a filename token.
_SEPARATORS = (" - ", " — ", " – ", " | ")


def is_editor_app(app_name: str) -> bool:
    """True when the process name looks like a known code/text editor."""
    app = (app_name or "").lower().removesuffix(".exe")
    return any(token == app or token in app for token in _EDITOR_APP_TOKENS)


def _strip_dirty_marker(token: str) -> str:
    """Drop leading unsaved-state markers an editor prefixes to the filename."""
    return token.lstrip("●*•◦ \t").strip()


def _looks_like_filename(token: str) -> bool:
    t = _strip_dirty_marker(token)
    if not t or len(t) > 200:
        return False
    return bool(_FILENAME_RE.match(t))


def extract_document_path(app_name: str, window_title: str) -> str | None:
    """Return the document/file named in an editor's window title, or None.

    Conservative: only for known editor apps, and only when a title segment
    clearly looks like ``filename.ext``. Returns the bare filename (titles rarely
    carry a full absolute path); callers store it as ``document_path``.
    """
    if not window_title or not is_editor_app(app_name):
        return None

    title = window_title.strip()

    # Remove a trailing editor-name suffix so it can't be mistaken for content.
    low = title.lower()
    for suffix in _EDITOR_SUFFIXES:
        marker = f" - {suffix}"
        idx = low.rfind(marker)
        if idx != -1:
            title = title[:idx].strip()
            break

    if not title:
        return None

    # Split on the usual title separators and pick the first filename-shaped part
    # (editors lead with the active file; JetBrains sometimes trails it, so we
    # scan all segments and prefer the earliest match).
    segments: list[str] = [title]
    for sep in _SEPARATORS:
        if sep in title:
            segments = [s.strip() for s in title.split(sep) if s.strip()]
            break

    for seg in segments:
        if _looks_like_filename(seg):
            return _strip_dirty_marker(seg)

    return None
