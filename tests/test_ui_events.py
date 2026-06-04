"""UI event pipeline tests — frame linker pairing."""

from __future__ import annotations

from pathlib import Path

import json
import sqlite3

from deskmate.a11y.input_hooks import return_flush_reason, return_inserts_newline
from deskmate.a11y.ui_event_types import (
    CaptureTrigger,
    ScrollBurstTracker,
    TriggerGates,
    UiEventInsert,
    UiEventType,
    capture_trigger_kind,
    ui_event_text_sql,
)
from deskmate.capture.frame_linker import (
    EventPersisted,
    FrameCaptured,
    FrameLinker,
    FrameLinkerConfig,
)


def test_ui_event_text_sql_reads_content_and_text_alias() -> None:
    con = sqlite3.connect(":memory:")
    expr = ui_event_text_sql("data_json")
    con.execute("CREATE TABLE ui_events (data_json TEXT)")
    con.execute(
        "INSERT INTO ui_events VALUES (?)",
        (json.dumps({"content": "hello from hook"}, ensure_ascii=False),),
    )
    row = con.execute(f"SELECT {expr} AS t FROM ui_events").fetchone()
    assert row[0] == "hello from hook"
    con.execute("UPDATE ui_events SET data_json = ?", (json.dumps({"text": "legacy"}, ensure_ascii=False),))
    row = con.execute(f"SELECT {expr} AS t FROM ui_events").fetchone()
    assert row[0] == "legacy"


def test_to_db_row_writes_text_and_content() -> None:
    evt = UiEventInsert(
        event_type=UiEventType.TEXT,
        text_content="用户 prompt",
        extra={"reason": "enter"},
    )
    _et, _app, _title, _url, data_json = evt.to_db_row()
    data = json.loads(data_json)
    assert data["content"] == "用户 prompt"
    assert data["text"] == "用户 prompt"
    assert data["reason"] == "enter"


def test_return_flush_reason() -> None:
    assert return_flush_reason(ctrl_down=False, shift_down=False) == "enter"
    assert return_flush_reason(ctrl_down=True, shift_down=False) == "ctrl_enter"
    assert return_flush_reason(ctrl_down=False, shift_down=True) is None
    assert return_flush_reason(ctrl_down=True, shift_down=True) is None
    assert return_inserts_newline(shift_down=True) is True
    assert return_inserts_newline(shift_down=False) is False


def test_capture_trigger_kind_text_is_typing_pause() -> None:
    evt = UiEventInsert(event_type=UiEventType.TEXT, app_name="chrome.exe", window_title="x")
    assert capture_trigger_kind(evt, ignored_patterns=[], gates=TriggerGates()) == CaptureTrigger.TYPING_PAUSE


def test_capture_trigger_kind_send_snapshot_is_send() -> None:
    evt = UiEventInsert(
        event_type=UiEventType.TEXT,
        app_name="chrome.exe",
        window_title="x",
        extra={"source": "send"},
    )
    # SEND fires regardless of the keystroke gate (it's an explicit user send).
    assert capture_trigger_kind(evt, ignored_patterns=[], gates=TriggerGates()) == CaptureTrigger.SEND


def test_capture_trigger_kind_clipboard_gated() -> None:
    evt = UiEventInsert(event_type=UiEventType.CLIPBOARD, app_name="app", window_title="w")
    assert capture_trigger_kind(evt, ignored_patterns=[], gates=TriggerGates(capture_on_clipboard=False)) is None
    assert capture_trigger_kind(evt, ignored_patterns=[], gates=TriggerGates(capture_on_clipboard=True)) == CaptureTrigger.CLIPBOARD


def test_frame_linker_pairs_event_before_frame() -> None:
    linker = FrameLinker(FrameLinkerConfig(ttl_s=60, capacity=16))
    upd = linker.on_event_persisted(EventPersisted(correlation_id=1, row_id=10))
    assert upd is None
    updates = linker.on_frame_captured(FrameCaptured(frame_id=99, correlation_ids=[1]))
    assert len(updates) == 1
    assert updates[0].row_id == 10
    assert updates[0].frame_id == 99


def test_frame_linker_pairs_frame_before_event() -> None:
    linker = FrameLinker(FrameLinkerConfig(ttl_s=60, capacity=16))
    updates = linker.on_frame_captured(FrameCaptured(frame_id=99, correlation_ids=[2]))
    assert updates == []
    upd = linker.on_event_persisted(EventPersisted(correlation_id=2, row_id=11))
    assert upd is not None
    assert upd.frame_id == 99


def test_update_ui_event_frame_id_idempotent(tmp_path: Path) -> None:
    from deskmate.db import DatabaseManager

    db = DatabaseManager(tmp_path / "t.db")
    try:
        fid = db.insert_frame(
            monitor_id=1,
            device_name="m1",
            app_name="a",
            window_name="w",
            browser_url=None,
            focused=True,
            snapshot_path=None,
            width=0,
            height=0,
            capture_trigger="click",
        )
        eid = db.insert_ui_event(
            event_type="click", app_name="a", window_title="w", data={"x": 1},
        )
        db.update_ui_event_frame_id(eid, fid)
        db.update_ui_event_frame_id(eid, fid + 999)
        row = db._conn.execute("SELECT frame_id FROM ui_events WHERE id=?", (eid,)).fetchone()
        assert row["frame_id"] == fid
    finally:
        db.close()


def test_scroll_burst_tracker() -> None:
    import time

    tracker = ScrollBurstTracker(delay_s=0.05)
    tracker.record(7)
    assert tracker.poll_burst_end() is None
    time.sleep(0.06)
    assert tracker.poll_burst_end() == 7
    assert tracker.poll_burst_end() is None
