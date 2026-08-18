"""Detection-accuracy tests for the tightened meeting detector:

* call signals matched as a CONTROL (title / line-isolated button), not as a
  bare substring of the OCR dump → kills chat / sidebar false positives.
* NEGATIVE_SIGNALS veto a waiting-room / join / ended-call screen.
* audio_active corroboration: a meeting URL with no recent speech is a parked
  tab; listen-only (others talking via loopback) still counts as a meeting.
* db.has_recent_speech() reads recent non-empty transcriptions.
"""

from __future__ import annotations

from pathlib import Path

from deskmate.db import DatabaseManager
from deskmate.meeting import detect_meeting


# ── false positives that the OLD substring match got wrong, now fixed ─────────

def test_chat_message_containing_leave_is_not_a_meeting() -> None:
    obs = detect_meeting(
        app_name="slack.exe", window_title="general - Slack", browser_url=None,
        text="Sarah: are you going to leave early today?",
    )
    assert obs.in_meeting is False


def test_teams_chat_hang_up_phrase_is_not_a_meeting() -> None:
    obs = detect_meeting(
        app_name="teams.exe", window_title="Chat | Microsoft Teams", browser_url=None,
        text="please hang up your coat by the door",
    )
    assert obs.in_meeting is False


def test_discord_sidebar_label_is_not_a_meeting() -> None:
    # "voice connected" as a sidebar label embedded among other text — not a control line.
    obs = detect_meeting(
        app_name="discord.exe", window_title="Discord", browser_url=None,
        text="general\nrandom\nvoice connected to lounge channel earlier",
    )
    assert obs.in_meeting is False


# ── veto: waiting room / join / ended ─────────────────────────────────────────

def test_zoom_waiting_room_vetoed() -> None:
    obs = detect_meeting(
        app_name="zoom.exe", window_title="Zoom", browser_url=None,
        text="Please wait, the meeting host will let you in soon.\nLeave Meeting",
    )
    assert obs.in_meeting is False


def test_meeting_ended_screen_vetoed() -> None:
    obs = detect_meeting(
        app_name="zoom.exe", window_title="Zoom Meeting", browser_url=None,
        text="This meeting has ended\nLeave Meeting",
    )
    assert obs.in_meeting is False


# ── true positives still detected ─────────────────────────────────────────────

def test_control_in_title_detected() -> None:
    obs = detect_meeting(
        app_name="zoom.exe", window_title="Zoom Meeting - Leave", browser_url=None,
        text="",
    )
    assert obs.in_meeting is True and obs.profile_name == "Zoom"


def test_control_as_isolated_button_line_detected() -> None:
    # a11y/OCR renders the button on its own line — accepted as a control.
    obs = detect_meeting(
        app_name="zoom.exe", window_title="Zoom Meeting", browser_url=None,
        text="Mute\nStart Video\nParticipants\nChat\nLeave Meeting",
    )
    assert obs.in_meeting is True


def test_listen_only_with_others_talking_is_a_meeting() -> None:
    # Mic muted, but loopback captured others → audio_active True; URL-only opens.
    obs = detect_meeting(
        app_name="chrome.exe", window_title="Meet", browser_url="https://meet.google.com/abc-defg-hij",
        text="", audio_active=True,
    )
    assert obs.in_meeting is True


def test_listen_only_with_call_control_opens_even_when_silent() -> None:
    # Everyone momentarily silent (audio_active False) but the in-call Leave
    # control is visible → still a meeting (control beats the audio corroborator).
    obs = detect_meeting(
        app_name="zoom.exe", window_title="Zoom Meeting", browser_url=None,
        text="Mute\nLeave Meeting", audio_active=False,
    )
    assert obs.in_meeting is True


def test_parked_tab_no_speech_not_a_meeting() -> None:
    obs = detect_meeting(
        app_name="chrome.exe", window_title="Meet", browser_url="https://meet.google.com/abc-defg-hij",
        text="", audio_active=False,
    )
    assert obs.in_meeting is False


# ── end-of-meeting tick (event-driven path) ──────────────────────────────────

def test_expire_if_idle_closes_after_grace_without_new_frames(tmp_path: Path) -> None:
    """The timer-driven expire tick must end a meeting on its grace even when no
    further observe()/link_transcript() arrives (user walked away at call end)."""
    from deskmate.meeting import MeetingDetector

    db = DatabaseManager(tmp_path / "e.db")
    detector = MeetingDetector(db, end_grace_seconds=120.0)
    detector.observe(
        app_name="zoom.exe", window_title="Zoom Meeting", browser_url=None,
        text="Mute\nLeave Meeting",
    )
    mid = detector.active_meeting_id
    assert mid is not None

    # Within grace → tick is a no-op.
    detector.expire_if_idle()
    assert detector.active_meeting_id == mid

    # Simulate the grace having elapsed (no new frames since), then tick.
    detector._last_seen -= 200.0  # noqa: SLF001
    detector.expire_if_idle()
    assert detector.active_meeting_id is None
    assert db.meeting_by_id(mid).get("ended_at")  # MEETING_ENDED path ran
    db.close()


def test_capture_loop_fires_meeting_expire_on_first_iteration() -> None:
    """The event-driven loop invokes meeting_expire on its timer — deterministic:
    last starts at -inf so iteration 1 fires it; the callback stops the loop."""
    import queue as _q
    import threading
    from deskmate.capture import event_driven_capture as ed
    from deskmate.config import load

    fired = threading.Event()
    stop = threading.Event()

    def _expire() -> None:
        fired.set()
        stop.set()

    class _NullState:
        def can_capture(self): return False
        def poll_activity(self): return None
        def mark_captured(self): ...
        # The loop also probes for on-screen change; a double that stops short
        # of the real interface fails here for the wrong reason.
        def visual_ready(self, _now): return False
        def visual_changed(self): return False

    orig_state = ed.EventDrivenCapture
    ed.EventDrivenCapture = lambda cfg: _NullState()  # type: ignore[assignment]
    try:
        ed.run_event_driven_capture_loop(
            cfg=load(), db=None, paired=None, trigger_rx=_q.Queue(),
            linker=None, stop=stop, meeting_observe=None,
            meeting_expire=_expire, meeting_expire_interval_s=0.0,
        )
    finally:
        ed.EventDrivenCapture = orig_state  # type: ignore[assignment]
    assert fired.is_set()

# ── db.has_recent_speech ──────────────────────────────────────────────────────

def test_has_recent_speech(tmp_path: Path) -> None:
    db = DatabaseManager(tmp_path / "s.db")
    assert db.has_recent_speech() is False  # nothing yet
    # An empty (VAD-silence) transcription must NOT count.
    db.insert_transcript(device="loopback", text="", language="en")
    assert db.has_recent_speech() is False
    # Real speech (e.g. from loopback = others talking) counts.
    db.insert_transcript(device="loopback", text="so the plan is to ship friday", language="en")
    assert db.has_recent_speech() is True
    db.close()
