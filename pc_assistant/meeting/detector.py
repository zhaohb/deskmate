"""Meeting detection and transcript linking.

The design uses meeting app profiles to identify calls from window titles,
URLs, and UI signals; active-call detection prefers leave/hang-up controls
over weak hints like a mute button.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

from ..db import DatabaseManager
from ..logger import get

logger = get("meeting.detector")


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class MeetingProfile:
    name: str
    app_tokens: tuple[str, ...]
    url_tokens: tuple[str, ...]
    title_tokens: tuple[str, ...]
    call_signals: tuple[str, ...]


@dataclass(frozen=True)
class MeetingObservation:
    in_meeting: bool
    app_name: str
    window_title: str
    browser_url: str | None
    profile_name: str | None
    matched_signals: tuple[str, ...]


PROFILES: tuple[MeetingProfile, ...] = (
    MeetingProfile(
        name="Microsoft Teams",
        app_tokens=("teams.exe", "ms-teams.exe", "msteams", "teams"),
        url_tokens=("teams.microsoft.com", "teams.live.com"),
        title_tokens=("microsoft teams", "teams"),
        call_signals=("hang up", "leave", "leave call", "离开", "挂断"),
    ),
    MeetingProfile(
        name="Zoom",
        app_tokens=("zoom.exe", "zoom"),
        url_tokens=("zoom.us/j", "zoom.us/wc", "zoom.us/my"),
        title_tokens=("zoom meeting", "zoom workplace"),
        call_signals=("leave", "leave meeting", "end meeting", "return to meeting", "zoom video container", "离开", "结束会议"),
    ),
    MeetingProfile(
        name="Google Meet",
        app_tokens=(),
        url_tokens=("meet.google.com",),
        title_tokens=("meet", "google meet"),
        call_signals=("leave call", "end call", "离开通话", "结束通话"),
    ),
    MeetingProfile(
        name="Slack Huddle",
        app_tokens=("slack.exe", "slack"),
        url_tokens=("app.slack.com/huddle",),
        title_tokens=("huddle",),
        call_signals=("leave huddle", "leave", "离开"),
    ),
    MeetingProfile(
        name="Discord",
        app_tokens=("discord.exe", "discord"),
        url_tokens=("discord.com", "discordapp.com"),
        title_tokens=("discord",),
        call_signals=("disconnect", "leave call", "voice connected", "断开连接"),
    ),
    MeetingProfile(
        name="Webex",
        app_tokens=("webex.exe", "webex"),
        url_tokens=("webex.com",),
        title_tokens=("webex",),
        call_signals=("leave meeting", "end meeting", "离开会议"),
    ),
)


class MeetingDetector:
    """Small state machine that opens, keeps alive and closes meeting rows."""

    def __init__(self, db: DatabaseManager, *, end_grace_seconds: float = 120.0) -> None:
        self.db = db
        self.end_grace_seconds = end_grace_seconds
        self._meeting_id: int | None = None
        self._profile_name: str | None = None
        self._last_seen = 0.0

    @property
    def active_meeting_id(self) -> int | None:
        return self._meeting_id

    def is_in_meeting(self) -> bool:
        self.expire_if_idle()
        return self._meeting_id is not None

    def observe(
        self,
        *,
        app_name: str,
        window_title: str,
        browser_url: str | None = None,
        text: str = "",
    ) -> MeetingObservation:
        observation = detect_meeting(
            app_name=app_name,
            window_title=window_title,
            browser_url=browser_url,
            text=text,
        )
        now = time.time()
        if observation.in_meeting:
            self._last_seen = now
            if self._meeting_id is None:
                self._meeting_id = self.db.insert_meeting(
                    name=observation.profile_name or observation.app_name or "Meeting",
                    started_at=_now_iso(),
                    metadata={
                        "detection_source": "ui_scan",
                        "app_name": observation.app_name,
                        "window_title": observation.window_title,
                        "browser_url": observation.browser_url,
                        "matched_signals": list(observation.matched_signals),
                    },
                )
                self._profile_name = observation.profile_name
                logger.info("meeting started id=%s profile=%s", self._meeting_id, self._profile_name)
            elif observation.profile_name and observation.profile_name != self._profile_name:
                self._profile_name = observation.profile_name
                self.db.update_meeting_metadata(
                    self._meeting_id,
                    {
                        "detection_source": "ui_scan",
                        "app_name": observation.app_name,
                        "window_title": observation.window_title,
                        "browser_url": observation.browser_url,
                        "profile_name": observation.profile_name,
                    },
                )
        else:
            self.expire_if_idle(now=now)
        return observation

    def expire_if_idle(self, *, now: float | None = None) -> None:
        if self._meeting_id is None:
            return
        current = time.time() if now is None else now
        if current - self._last_seen < self.end_grace_seconds:
            return
        meeting_id = self._meeting_id
        self.db.end_meeting(meeting_id, ended_at=_now_iso())
        logger.info("meeting ended id=%s", meeting_id)
        self._meeting_id = None
        self._profile_name = None
        self._last_seen = 0.0

    def link_transcript(
        self,
        *,
        transcription_id: int,
        speaker_id: int | None,
        text: str,
        start_time: float | None,
        end_time: float | None,
    ) -> int | None:
        self.expire_if_idle()
        if self._meeting_id is None:
            return None
        return self.db.insert_meeting_segment(
            meeting_id=self._meeting_id,
            transcription_id=transcription_id,
            speaker_id=speaker_id,
            text=text,
            start_time=start_time or 0.0,
            end_time=end_time or start_time or 0.0,
        )


def detect_meeting(
    *,
    app_name: str,
    window_title: str,
    browser_url: str | None = None,
    text: str = "",
) -> MeetingObservation:
    app_l = (app_name or "").lower()
    title_l = (window_title or "").lower()
    url_l = (browser_url or "").lower()
    text_l = (text or "").lower()
    combined = "\n".join((title_l, url_l, text_l))

    for profile in PROFILES:
        app_match = bool(profile.app_tokens) and any(token in app_l for token in profile.app_tokens)
        url_match = bool(profile.url_tokens) and any(token in url_l for token in profile.url_tokens)
        title_match = bool(profile.title_tokens) and any(token in title_l for token in profile.title_tokens)
        if not (app_match or url_match or title_match):
            continue

        matched = tuple(signal for signal in profile.call_signals if signal.lower() in combined)
        # Browser meeting URLs are stronger than app/title identity, but still
        # require either an active-call control or a meeting-specific URL.
        in_meeting = bool(matched) or url_match
        if in_meeting:
            return MeetingObservation(
                in_meeting=True,
                app_name=app_name or "",
                window_title=window_title or "",
                browser_url=browser_url,
                profile_name=profile.name,
                matched_signals=matched,
            )

    return MeetingObservation(
        in_meeting=False,
        app_name=app_name or "",
        window_title=window_title or "",
        browser_url=browser_url,
        profile_name=None,
        matched_signals=(),
    )

