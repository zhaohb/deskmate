"""Paired capture pipeline — the heart of the recorder.

Each capture:
  1. Snapshot the focused app/window/title/url.
  2. Apply window filters (exclude apps, incognito titles, etc.).
  3. Take one screenshot per requested monitor.
  4. Walk the UIA tree of the focused window.
  5. Optionally OCR the screenshot (engine-driven fallback).
  6. Optionally PII-redact OCR & accessibility text.
  7. Insert a `frames` row + attached `ocr_text` + `accessibility` rows.

The order is intentionally stable so each frame is paired with the matching
OCR and accessibility data.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any

from .. import events as bus
from ..a11y.browser_url import resolve_browser_url
from ..a11y.document import extract_document_path
from ..a11y.uia_tree import INTERACTIVE_TYPES, foreground_window, walk_focused_window
from ..a11y.win_events import foreground_app_name
from ..config import Config
from ..core import is_app_excluded, is_title_private
from ..core.filter import WindowFilter
from ..db import DatabaseManager
from ..logger import get
from ..redact import maybe_redact
from ..screen.capture import Monitor, downscale, grab_monitor, list_monitors
from ..screen.ocr import OcrEngine, perform_ocr
from ..screen.snapshot import SnapshotWriter

logger = get("capture.paired")


def _flatten_elements(root: Any, cfg: Config, max_rows: int) -> list[dict[str, Any]]:
    """Flatten an AccessibilityNode tree into normalized element rows (P1).

    Preorder walk over the whole tree (so ``node_index`` is stable), but only
    *meaningful* nodes are emitted: those carrying a name/value, or that are
    interactive, or focused. ``parent_index`` points at the nearest *kept*
    ancestor so the emitted rows still form a navigable forest. Password fields
    never expose their value.
    """
    if root is None:
        return []
    rows: list[dict[str, Any]] = []
    counter = 0

    def visit(node: Any, kept_parent: int | None) -> None:
        nonlocal counter
        if len(rows) >= max_rows:
            return
        idx = counter
        counter += 1
        role = node.control_type or ""
        is_interactive = any(role.lower() == t.lower() for t in INTERACTIVE_TYPES)
        name = (node.name or "").strip() or None
        if node.is_password:
            value = None
        else:
            raw = (node.value or "").strip()
            value = maybe_redact(raw, cfg) if raw else None
        keep = bool(name or value or is_interactive or node.is_focused)
        next_parent = kept_parent
        if keep:
            bounds = None
            b = node.bounds
            if b is not None:
                bounds = f"{b.x:.0f},{b.y:.0f},{b.width:.0f},{b.height:.0f}"
            rows.append({
                "node_index": idx,
                "parent_index": kept_parent,
                "depth": node.depth,
                "role": role,
                "name": name,
                "value": value,
                "automation_id": (node.automation_id or "").strip() or None,
                "is_focused": bool(node.is_focused),
                "is_interactive": is_interactive,
                "bounds": bounds,
            })
            next_parent = idx
        for child in node.children:
            if len(rows) >= max_rows:
                break
            visit(child, next_parent)

    visit(root, None)
    return rows


class PairedCapture:
    """Stateful capture coordinator. Owns dedup/min-gap state."""

    def __init__(self, cfg: Config, db: DatabaseManager) -> None:
        self.cfg = cfg
        self.db = db
        self.snapshot = SnapshotWriter()
        self._monitors: list[Monitor] = list_monitors()
        self._filter = WindowFilter(
            ignored_apps=cfg.filters.ignored_apps,
            ignored_windows=cfg.filters.ignored_windows,
            included_windows=cfg.filters.included_windows,
        )
        self._lock = threading.Lock()
        self._last_at = 0.0
        self._last_signature = ""

    def _now(self) -> datetime:
        return datetime.now(timezone.utc).astimezone()

    def _select_monitors(self) -> list[Monitor]:
        if not self._monitors:
            self._monitors = list_monitors()
        if self.cfg.capture.all_monitors:
            return self._monitors
        return self._monitors[:1]

    def capture_once(self, *, trigger: str = "idle", trigger_data: dict[str, Any] | None = None) -> list[int]:
        """Run one paired capture. Returns the list of frame_ids written."""
        if not self.cfg.capture.enabled:
            return []
        with self._lock:
            gap = time.time() - self._last_at
            if gap < self.cfg.capture.min_capture_gap_seconds:
                return []
            self._last_at = time.time()

        hwnd, pid, title = foreground_window()
        app = foreground_app_name(pid)

        if is_app_excluded(app, self.cfg.filters.ignored_apps):
            logger.debug("skip: excluded app %r", app)
            return []
        if self.cfg.filters.ignore_incognito and is_title_private(title):
            logger.debug("skip: incognito window %r", title)
            return []
        if not self._filter.passes(app, title):
            logger.debug("skip: filter denied app=%r title=%r", app, title)
            return []

        signature = f"{app}\x1f{title}\x1f{trigger}"
        with self._lock:
            same_as_last = signature == self._last_signature
            self._last_signature = signature
        if same_as_last and trigger in ("heartbeat", "idle"):
            return []

        tree = walk_focused_window(
            max_depth=self.cfg.a11y.ax_depth,
            max_nodes=self.cfg.a11y.ax_max_nodes,
            hwnd=hwnd,
        ) if self.cfg.a11y.enabled else None
        browser_url = resolve_browser_url(app, pid=pid, hwnd=hwnd)
        captured_at = self._now()
        frame_ids: list[int] = []

        for mon in self._select_monitors():
            frame_id = self._write_one(
                monitor=mon,
                captured_at=captured_at,
                hwnd=hwnd,
                pid=pid,
                app=app,
                title=title,
                browser_url=browser_url,
                tree=tree,
                trigger=trigger,
            )
            if frame_id:
                frame_ids.append(frame_id)
        # ui_events.frame_id is linked via FrameLinker — no capture/* rows.
        return frame_ids

    def _write_one(
        self,
        *,
        monitor: Monitor,
        captured_at: datetime,
        hwnd: int,
        pid: int,
        app: str,
        title: str,
        browser_url: str | None,
        tree: Any,
        trigger: str,
    ) -> int | None:
        snapshot_path: str | None = None
        width = height = 0
        # Grab at full resolution; OCR runs on this so small/high-DPI text isn't
        # lost. The on-disk snapshot is downscaled separately (storage only).
        ocr_img = None
        if self.cfg.capture.include_screenshot:
            ocr_img = grab_monitor(monitor)
            if ocr_img is not None:
                img = downscale(ocr_img, self.cfg.capture.screenshot_max_width)
                width, height = img.size
                p = self.snapshot.write(
                    img, monitor_id=monitor.id, captured_at=captured_at,
                    quality=self.cfg.capture.screenshot_jpeg_quality,
                )
                snapshot_path = str(p)

        frame_id = self.db.insert_frame(
            monitor_id=monitor.id,
            device_name=monitor.name,
            app_name=app,
            window_name=title,
            browser_url=browser_url,
            document_path=extract_document_path(app, title),
            focused=True,
            snapshot_path=snapshot_path,
            width=width, height=height,
            capture_trigger=trigger,
            timestamp=captured_at.replace(microsecond=0).isoformat(),
        )

        # OCR (best-effort; never blocks frame insertion above). Runs on the
        # full-resolution grab; word boxes are normalized 0–1 so they map onto
        # the (possibly downscaled) stored snapshot unchanged.
        if ocr_img is not None and self.cfg.ocr.engine != "off":
            try:
                engine = OcrEngine(self.cfg.ocr.engine)
                text, words_json, conf = perform_ocr(
                    ocr_img,
                    engine,
                    self.cfg.ocr.languages,
                    tesseract_cmd=self.cfg.ocr.tesseract_cmd,
                )
                text = maybe_redact(text, self.cfg) if text else text
                if text:
                    self.db.attach_ocr(frame_id, text=text, text_json=words_json, engine=engine.value, confidence=conf)
            except Exception as exc:  # noqa: BLE001
                logger.warning("ocr failed: %s", exc)

        # Accessibility tree.
        if tree is not None and tree.text:
            ax_text = maybe_redact(tree.text, self.cfg) if tree.text else tree.text
            try:
                self.db.attach_accessibility(
                    frame_id,
                    text=ax_text,
                    focused_role=tree.focused_role,
                    focused_name=tree.focused_name,
                    focused_value=maybe_redact(tree.focused_value or "", self.cfg) if tree.focused_value else None,
                    tree_json=tree.to_json(),
                    on_screen=1 if tree.on_screen else 0,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("a11y attach failed: %s", exc)

            # P1: normalized element rows (gradual rollout; off by default).
            if self.cfg.a11y.persist_elements and tree.root is not None:
                try:
                    rows = _flatten_elements(
                        tree.root, self.cfg, self.cfg.a11y.elements_max_rows_per_frame
                    )
                    self.db.attach_elements(frame_id, rows)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("elements attach failed: %s", exc)

        bus.send(bus.EventType.FRAME_WRITTEN, frame_id=frame_id, monitor_id=monitor.id, app_name=app, window_title=title, trigger=trigger)
        return frame_id


def paired_capture(cfg: Config, db: DatabaseManager, *, trigger: str = "manual") -> list[int]:
    """One-shot helper. Equivalent of `paired_capture::paired_capture` for ad-hoc calls."""
    return PairedCapture(cfg, db).capture_once(trigger=trigger)
