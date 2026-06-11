"""Tests for the enhanced meeting pipeline:
- audio-corroborated detection (fewer false positives from parked tabs),
- MEETING_STARTED / MEETING_ENDED event emission,
- participant capture persisted to meeting metadata,
- action-item extraction parsing for the summary → todos flow.
"""

from __future__ import annotations

import json
from pathlib import Path

from deskmate import events as bus
from deskmate.meeting import MeetingDetector, detect_meeting


# ── detection accuracy ───────────────────────────────────────────────────────


def test_detect_url_only_needs_audio_when_known_silent() -> None:
    # A parked Meet tab (URL but no call control). Audio known-silent → not a meeting.
    obs = detect_meeting(
        app_name="chrome.exe",
        window_title="Meet",
        browser_url="https://meet.google.com/abc-defg-hij",
        text="",  # no "leave call" control
        audio_active=False,
    )
    assert obs.in_meeting is False


def test_detect_url_only_opens_when_audio_active() -> None:
    obs = detect_meeting(
        app_name="chrome.exe",
        window_title="Meet",
        browser_url="https://meet.google.com/abc-defg-hij",
        text="",
        audio_active=True,
    )
    assert obs.in_meeting is True


def test_detect_url_only_unknown_audio_is_backward_compatible() -> None:
    # audio_active=None (default) keeps the original URL-only behavior.
    obs = detect_meeting(
        app_name="chrome.exe",
        window_title="Meet",
        browser_url="https://meet.google.com/abc-defg-hij",
        text="",
    )
    assert obs.in_meeting is True


def test_detect_call_control_always_wins() -> None:
    # An explicit hang-up control beats a silent-audio hint.
    obs = detect_meeting(
        app_name="zoom.exe",
        window_title="Zoom Meeting",
        browser_url=None,
        text="Leave Meeting",
        audio_active=False,
    )
    assert obs.in_meeting is True


# ── lifecycle events + participants ──────────────────────────────────────────


def test_meeting_events_and_participants(tmp_path: Path) -> None:
    from deskmate.db import DatabaseManager

    seen: list[tuple[str, dict]] = []
    unsub = bus.subscribe(lambda e: seen.append((e.type.value, dict(e.data))))
    try:
        db = DatabaseManager(tmp_path / "m.db")
        # Seed two named speakers so participants resolve to names.
        sid_a = db._conn.execute(  # noqa: SLF001
            "INSERT INTO speakers(name, centroid_json, sample_count) VALUES (?,?,?)",
            ("Alice", "[1.0]", 1),
        ).lastrowid
        sid_b = db._conn.execute(  # noqa: SLF001
            "INSERT INTO speakers(name, centroid_json, sample_count) VALUES (?,?,?)",
            ("Bob", "[0.0]", 1),
        ).lastrowid

        # Generous grace so link_transcript() (which calls expire_if_idle first)
        # doesn't end the meeting mid-accumulation; we end it explicitly below.
        detector = MeetingDetector(db, end_grace_seconds=10_000.0)
        detector.observe(
            app_name="zoom.exe", window_title="Zoom Meeting",
            browser_url=None, text="Leave Meeting",
        )
        mid = detector.active_meeting_id
        assert mid is not None
        assert ("meeting_started", {"meeting_id": mid, "profile_name": "Zoom"}) in [
            (t, {k: d[k] for k in ("meeting_id", "profile_name")}) for t, d in seen
        ]

        for sid in (sid_a, sid_b, sid_a):  # Alice speaks twice, Bob once
            tid = db.insert_transcript(
                device="mic", text="hi", language="en",
                speaker_id=int(sid), start_time=1.0, end_time=2.0,
            )
            detector.link_transcript(
                transcription_id=tid, speaker_id=int(sid),
                text="hi", start_time=1.0, end_time=2.0,
            )

        # Force end: drop the grace to 0 so the next idle check expires it.
        detector.end_grace_seconds = 0.0
        detector.expire_if_idle()
        assert detector.active_meeting_id is None

        # MEETING_ENDED emitted.
        assert any(t == "meeting_ended" and d.get("meeting_id") == mid for t, d in seen)

        # Participants persisted to metadata, deduped, by name.
        meeting = db.list_meetings()[0]
        meta = meeting.get("metadata")
        meta = json.loads(meta) if isinstance(meta, str) else (meta or {})
        participants = meta.get("participants") or []
        assert set(participants) == {"Alice", "Bob"}
        db.close()
    finally:
        unsub()


def test_note_participant_from_ui(tmp_path: Path) -> None:
    from deskmate.db import DatabaseManager

    db = DatabaseManager(tmp_path / "m2.db")
    detector = MeetingDetector(db, end_grace_seconds=10_000.0)
    detector.observe(
        app_name="teams.exe", window_title="Microsoft Teams",
        browser_url=None, text="Hang up",
    )
    detector.note_participant("Carol Chen")
    detector.note_participant("Carol Chen")  # dup
    detector.note_participant("David Li")
    detector.end_grace_seconds = 0.0
    detector.expire_if_idle()

    meeting = db.list_meetings()[0]
    meta = meeting.get("metadata")
    meta = json.loads(meta) if isinstance(meta, str) else (meta or {})
    assert set(meta.get("participants") or []) == {"Carol Chen", "David Li"}
    db.close()


# ── action-item extraction ───────────────────────────────────────────────────


def _agent():
    import sys
    import deskmate.apps.agent as agent
    sys.modules.setdefault("agent", agent)
    return agent


def test_parse_action_items_structured() -> None:
    agent = _agent()
    body = (
        "## Summary\n- discussed the roadmap\n\n"
        "## Action Items\n"
        "- [ ] Ship the OCR fix | owner: Alice | due: 2026-06-10\n"
        "- [ ] Draft the release notes | owner: unknown | due: none\n"
        "- [x] Review PR | owner: Bob\n"
    )
    items = agent._parse_action_items(body)
    assert len(items) == 3
    assert items[0]["task"] == "Ship the OCR fix"
    assert items[0]["owner"] == "Alice" and items[0]["due"] == "2026-06-10"
    assert items[1]["due"] == ""  # 'none' normalized to empty
    assert items[2]["task"] == "Review PR" and items[2]["owner"] == "Bob"


def test_parse_action_items_none_section() -> None:
    agent = _agent()
    body = "## Summary\n- nothing actionable\n\n## Action Items\nNONE\n"
    assert agent._parse_action_items(body) == []


def test_parse_action_items_stops_at_next_section() -> None:
    agent = _agent()
    body = (
        "## Action Items\n- [ ] Do thing | owner: X | due: none\n\n"
        "## Notes\n- [ ] this is not an action item\n"
    )
    items = agent._parse_action_items(body)
    assert len(items) == 1
    assert items[0]["task"] == "Do thing"
