"""Unit tests for the email-digest prefetch.

Covers `_do_email_digest_prefetch`:
- dedicated mail-client processes contribute hits by process name alone;
- webmail tools only count when the host literally appears in URL/title;
- the returned `verified` list matches the labels with substantive hits.

Network calls are stubbed by monkeypatching `_do_content_search`.
"""

from __future__ import annotations

import sys
from types import ModuleType

import pytest


def _load_agent_module() -> ModuleType:
    import deskmate.apps.agent as agent  # noqa: PLC0415

    sys.modules.setdefault("agent", agent)
    return agent


@pytest.fixture()
def agent() -> ModuleType:
    return _load_agent_module()


def _ocr_item(
    *,
    text: str,
    app_name: str = "chrome.exe",
    window_name: str = "",
    browser_url: str = "",
    timestamp: str = "2026-05-29T09:00:00+08:00",
) -> dict:
    return {
        "type": "OCR",
        "content": {
            "text": text,
            "app_name": app_name,
            "window_name": window_name,
            "browser_url": browser_url,
            "timestamp": timestamp,
            "frame_id": 1,
        },
    }


def test_email_targets_cover_major_mail_clients(agent: ModuleType) -> None:
    labels = set(agent.EMAIL_TOOL_TARGETS)
    # Must cover native clients and major webmail (Gmail + Outlook web).
    assert {"Outlook", "Thunderbird", "Gmail (web)", "Outlook (web)"} <= labels


def test_email_prefetch_verifies_native_client_by_process(
    agent: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict] = []

    def fake_search(start, end, *, limit=10, app_name=None, q=None, verbose=False):
        calls.append({"app_name": app_name, "q": q})
        if app_name and app_name.lower() == "outlook.exe":
            return [
                _ocr_item(
                    text="Re: Q3 planning meeting agenda",
                    app_name="OUTLOOK.EXE",
                    window_name="Inbox - Outlook",
                ),
            ]
        return []

    monkeypatch.setattr(agent, "_do_content_search", fake_search)

    data, verified = agent._do_email_digest_prefetch(
        "2026-05-29T00:00:00+08:00", "2026-05-29T23:59:59+08:00", verbose=False
    )

    assert "Outlook" in verified
    assert "Q3 planning meeting agenda" in data
    # We must have probed Outlook by process name at least once.
    assert any(c["app_name"] and c["app_name"].lower() == "outlook.exe" for c in calls)


def test_email_prefetch_rejects_browser_noise_for_webmail(
    agent: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A chrome.exe hit that mentions 'gmail' only in OCR (not in URL / title)
    must NOT verify Gmail (web). This is the key anti-fabrication property."""

    def fake_search(start, end, *, limit=10, app_name=None, q=None, verbose=False):
        # Native mail clients: nothing.
        if app_name:
            return []
        # Keyword fallback for Gmail: return a chrome hit whose URL is unrelated.
        if q == "mail.google.com":
            return [
                _ocr_item(
                    text="article mentioning gmail keyboard shortcuts",
                    app_name="chrome.exe",
                    window_name="Best Gmail tips - HackerNews",
                    browser_url="https://news.ycombinator.com/item?id=1",
                ),
            ]
        return []

    monkeypatch.setattr(agent, "_do_content_search", fake_search)

    _, verified = agent._do_email_digest_prefetch(
        "2026-05-29T00:00:00+08:00", "2026-05-29T23:59:59+08:00", verbose=False
    )

    assert "Gmail (web)" not in verified, (
        "Gmail (web) must require the host to appear in URL/title, not raw OCR"
    )


def test_email_prefetch_verifies_webmail_only_when_url_matches(
    agent: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_search(start, end, *, limit=10, app_name=None, q=None, verbose=False):
        if app_name:
            return []
        if q == "mail.google.com":
            return [
                _ocr_item(
                    text="Inbox (3) - alice@example.com - Gmail",
                    app_name="chrome.exe",
                    window_name="Inbox (3) - alice@example.com - Gmail",
                    browser_url="https://mail.google.com/mail/u/0/#inbox",
                ),
            ]
        return []

    monkeypatch.setattr(agent, "_do_content_search", fake_search)

    data, verified = agent._do_email_digest_prefetch(
        "2026-05-29T00:00:00+08:00", "2026-05-29T23:59:59+08:00", verbose=False
    )

    assert "Gmail (web)" in verified
    assert "alice@example.com" in data


def test_email_prefetch_empty_when_no_hits(
    agent: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        agent,
        "_do_content_search",
        lambda *a, **k: [],
    )
    # Isolate from any locally-connected OAuth account so the assertion reflects
    # the screen/UI evidence only (a live Gmail/Outlook login must not leak in).
    monkeypatch.setattr(agent, "_fetch_gmail_oauth_messages", lambda *a, **k: ("", False))
    monkeypatch.setattr(agent, "_fetch_outlook_oauth_messages", lambda *a, **k: ("", False))

    data, verified = agent._do_email_digest_prefetch(
        "2026-05-29T00:00:00+08:00", "2026-05-29T23:59:59+08:00", verbose=False
    )

    assert verified == []
    # Every email target must still appear with substantive_hits=0.
    for label in agent.EMAIL_TOOL_TARGETS:
        assert f"### {label} | substantive_hits=0" in data
        assert f"NO USAGE RECORDED for {label}" in data


def test_email_prefetch_uses_outlook_oauth_messages(
    agent: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(agent, "_do_content_search", lambda *a, **k: [])

    def fake_get(url: str, timeout: int = 15):
        if url.endswith("/connections/outlook/instances"):
            return {"data": [{"instance": "me@example.com", "email": "me@example.com"}]}
        if "/connections/outlook/messages" in url:
            return {
                "data": {
                    "messages": [
                        {
                            "date": "2026-05-29T09:00:00Z",
                            "from": "Ada <ada@example.com>",
                            "subject": "Quarterly planning",
                            "snippet": "Please review the planning notes.",
                        }
                    ]
                }
            }
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(agent, "_http_get", fake_get)

    data, verified = agent._do_email_digest_prefetch(
        "2026-05-29T00:00:00+08:00", "2026-05-29T23:59:59+08:00", verbose=False
    )

    assert "Outlook (OAuth)" in verified
    assert "Microsoft Graph" not in data  # data stays factual, prompt explains source
    assert "Quarterly planning" in data
    assert "Ada <ada@example.com>" in data


def test_email_prefetch_uses_gmail_oauth_messages(
    agent: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(agent, "_do_content_search", lambda *a, **k: [])

    def fake_get(url: str, timeout: int = 15):
        if url.endswith("/connections/gmail/instances"):
            return {"data": [{"instance": "me@gmail.com", "email": "me@gmail.com"}]}
        if "/connections/gmail/messages/msg-1" in url:
            return {
                "data": {
                    "date": "Fri, 29 May 2026 09:00:00 +0800",
                    "from": "Ada <ada@example.com>",
                    "subject": "Gmail planning",
                    "snippet": "Please review the Gmail planning notes.",
                }
            }
        if "/connections/gmail/messages" in url:
            return {"data": {"messages": [{"id": "msg-1", "threadId": "thread-1"}]}}
        if url.endswith("/connections/outlook/instances"):
            return {"data": []}
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(agent, "_http_get", fake_get)

    data, verified = agent._do_email_digest_prefetch(
        "2026-05-29T00:00:00+08:00", "2026-05-29T23:59:59+08:00", verbose=False
    )

    assert "Gmail (OAuth)" in verified
    assert "Gmail planning" in data
    assert "Ada <ada@example.com>" in data


def test_gmail_oauth_constrains_to_window_and_drops_stale(
    agent: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gmail OAuth must pass a `newer_than` query and drop pre-window mail."""
    seen_urls: list[str] = []

    def fake_get(url: str, timeout: int = 15):
        seen_urls.append(url)
        if url.endswith("/connections/gmail/instances"):
            return {"data": [{"instance": "me@gmail.com", "email": "me@gmail.com"}]}
        if "/connections/gmail/messages/fresh" in url:
            return {
                "data": {
                    "date": "Fri, 29 May 2026 09:00:00 +0800",
                    "from": "Ada <ada@example.com>",
                    "subject": "Fresh thread",
                    "snippet": "in window",
                }
            }
        if "/connections/gmail/messages/stale" in url:
            return {
                "data": {
                    "date": "Mon, 01 Jan 2026 09:00:00 +0800",
                    "from": "Old <old@example.com>",
                    "subject": "Stale thread",
                    "snippet": "way before the window",
                }
            }
        if "/connections/gmail/messages" in url:
            return {"data": {"messages": [{"id": "fresh"}, {"id": "stale"}]}}
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(agent, "_http_get", fake_get)
    text, ok = agent._fetch_gmail_oauth_messages(
        since_iso="2026-05-29T00:00:00+08:00"
    )

    assert any("newer_than" in u for u in seen_urls), "list call must constrain by newer_than"
    assert ok is True
    assert "Fresh thread" in text
    assert "Stale thread" not in text, "pre-window mail must be dropped"


def test_gmail_oauth_unparseable_date_is_kept(
    agent: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A message whose date cannot be parsed must NOT be silently dropped."""
    assert agent._msg_date_within("not a date", "2026-05-29T00:00:00+08:00") is True
    assert agent._msg_date_within("(no date)", "2026-05-29T00:00:00+08:00") is True
    assert agent._msg_date_within("", "2026-05-29T00:00:00+08:00") is True
    # No floor → always within.
    assert agent._msg_date_within("Mon, 01 Jan 2020 00:00:00 +0000", None) is True
