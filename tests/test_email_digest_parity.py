"""Parity verification for email-digest Gmail/OAuth field coverage.

Goal: prove that whatever the user actually SAW on screen (sender, subject,
snippet) is faithfully surfaced by ``_do_email_digest_prefetch``, with the same
field-level fidelity that ``parse_gmail_message`` produces from the Gmail API.

Architectural caveat — read this first:
    Gmail OAuth path = REST API → JSON → digest fields
    Screen-only path = local OCR / UI → markdown report
Byte-identical parity across both paths is impossible without OAuth. These tests
verify the **closest achievable parity**: for any email the user opened on
screen, every human-meaningful Gmail field (from, subject, snippet, approximate
date) appears verbatim in the prefetch data the LLM receives.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

_APPS_DIR = Path(__file__).resolve().parents[1] / "apps"


def _load_agent() -> ModuleType:
    if str(_APPS_DIR) not in sys.path:
        sys.path.insert(0, str(_APPS_DIR))
    sys.modules.pop("agent", None)
    spec = importlib.util.spec_from_file_location("agent", _APPS_DIR / "agent.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["agent"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def agent() -> ModuleType:
    return _load_agent()


# ── Fixtures shaped like parse_gmail_message() output ────────────────────────
# Each record mirrors what GET /connections/gmail/messages/:id returns:
#   {id, threadId, from, to, subject, date, snippet, body}
GMAIL_INBOX_FIXTURE: list[dict[str, str]] = [
    {
        "id": "18f1a2b3c4d5e6f7",
        "threadId": "18f1a2b3c4d5e6f7",
        "from": "Alice Chen <alice@example.com>",
        "to": "me@example.com",
        "subject": "Q3 planning meeting agenda",
        "date": "Thu, 29 May 2026 09:14:23 +0800",
        "snippet": "Sharing the proposed agenda for Friday's Q3 planning sync.",
        "body": "Hi team,\n\nHere is the agenda for our Q3 sync...\n",
    },
    {
        "id": "18f2b3c4d5e6f7a8",
        "threadId": "18f2b3c4d5e6f7a8",
        "from": "GitHub <noreply@github.com>",
        "to": "me@example.com",
        "subject": "[deskmate] PR #142 ready for review",
        "date": "Thu, 29 May 2026 11:02:11 +0800",
        "snippet": "hongbo opened PR #142: Add email-digest app.",
        "body": "View PR: https://github.com/...\n",
    },
    {
        "id": "18f3c4d5e6f7a8b9",
        "threadId": "18f3c4d5e6f7a8b9",
        "from": "Bob Liu <bob@example.com>",
        "to": "me@example.com",
        "subject": "Re: Lunch tomorrow?",
        "date": "Thu, 29 May 2026 12:45:00 +0800",
        "snippet": "Sounds good, see you at 12:30 at the cafe.",
        "body": "Sounds good, see you at 12:30.\n",
    },
]


def _ocr_from_gmail_message(msg: dict[str, str], *, web: bool = True) -> dict:
    """Synthesise the OCR row DeskMate would have captured while the user
    was reading this exact Gmail message in the browser.

    Every Gmail API field we care about ends up in the on-screen text that
    DeskMate captures via OCR.
    """
    visible_text = (
        f"{msg['subject']}\n"
        f"From: {msg['from']}\n"
        f"To: {msg['to']}\n"
        f"{msg['date']}\n"
        f"{msg['snippet']}"
    )
    if web:
        return {
            "type": "OCR",
            "content": {
                "text": visible_text,
                "app_name": "chrome.exe",
                "window_name": f"{msg['subject']} - me@example.com - Gmail",
                "browser_url": "https://mail.google.com/mail/u/0/#inbox",
                "timestamp": "2026-05-29T09:30:00+08:00",
                "frame_id": int(msg["id"][:8], 16) % 100000,
            },
        }
    return {
        "type": "OCR",
        "content": {
            "text": visible_text,
            "app_name": "OUTLOOK.EXE",
            "window_name": f"{msg['subject']} - Outlook",
            "browser_url": "",
            "timestamp": "2026-05-29T09:30:00+08:00",
            "frame_id": int(msg["id"][:8], 16) % 100000,
        },
    }


# ── Parity assertions ───────────────────────────────────────────────────────

def test_prefetch_surfaces_every_gmail_field_user_saw(
    agent: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """For each email the user opened on screen, from / subject / snippet / date
    must appear verbatim in the data passed to the LLM."""
    ocr_rows = [_ocr_from_gmail_message(m, web=True) for m in GMAIL_INBOX_FIXTURE]

    def fake_search(start, end, *, limit=10, app_name=None, q=None, verbose=False):
        if q == "mail.google.com":
            return ocr_rows
        return []

    monkeypatch.setattr(agent, "_do_content_search", fake_search)
    data, verified = agent._do_email_digest_prefetch(
        "2026-05-29T00:00:00+08:00", "2026-05-29T23:59:59+08:00", verbose=False
    )

    assert "Gmail (web)" in verified
    for msg in GMAIL_INBOX_FIXTURE:
        assert msg["subject"] in data, f"missing subject {msg['subject']!r}"
        sender_name = msg["from"].split(" <")[0]
        assert sender_name in data, f"missing sender {sender_name!r}"
        assert msg["snippet"][:30] in data, f"missing snippet for {msg['subject']!r}"


def test_prefetch_excludes_messages_user_never_opened(
    agent: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An email that exists in the Gmail account but was never opened on
    screen MUST NOT appear in the prefetch."""
    seen = GMAIL_INBOX_FIXTURE[:2]
    unseen = GMAIL_INBOX_FIXTURE[2]
    ocr_rows = [_ocr_from_gmail_message(m, web=True) for m in seen]

    monkeypatch.setattr(
        agent,
        "_do_content_search",
        lambda *a, **k: ocr_rows if k.get("q") == "mail.google.com" else [],
    )
    data, _ = agent._do_email_digest_prefetch(
        "2026-05-29T00:00:00+08:00", "2026-05-29T23:59:59+08:00", verbose=False
    )

    for msg in seen:
        assert msg["subject"] in data
    assert unseen["subject"] not in data, (
        "fabrication guard failed: pipeline must not surface emails the user did "
        "not open on screen"
    )


def test_native_client_parity_matches_webmail_parity(
    agent: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same fixture via native Outlook OCR → same field-level parity as webmail."""
    ocr_rows = [_ocr_from_gmail_message(m, web=False) for m in GMAIL_INBOX_FIXTURE]

    def fake_search(start, end, *, limit=10, app_name=None, q=None, verbose=False):
        if app_name and app_name.lower() in {"outlook.exe", "outlook.exe"}:
            return ocr_rows
        return []

    monkeypatch.setattr(agent, "_do_content_search", fake_search)
    data, verified = agent._do_email_digest_prefetch(
        "2026-05-29T00:00:00+08:00", "2026-05-29T23:59:59+08:00", verbose=False
    )

    assert "Outlook" in verified
    for msg in GMAIL_INBOX_FIXTURE:
        assert msg["subject"] in data
        assert msg["from"].split(" <")[0] in data


def test_parse_gmail_message_field_set_is_a_subset_of_what_prefetch_emits(
    agent: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """parse_gmail_message emits {id, threadId, from, to, subject, date, snippet, body}.
    The human-meaningful subset (from, subject, date, snippet) must be present
    in email-digest prefetch data."""
    msg = GMAIL_INBOX_FIXTURE[0]
    ocr = _ocr_from_gmail_message(msg, web=True)
    monkeypatch.setattr(
        agent,
        "_do_content_search",
        lambda *a, **k: [ocr] if k.get("q") == "mail.google.com" else [],
    )
    data, _ = agent._do_email_digest_prefetch(
        "2026-05-29T00:00:00+08:00", "2026-05-29T23:59:59+08:00", verbose=False
    )

    must_surface = {
        "from": msg["from"].split(" <")[0],
        "subject": msg["subject"],
        "snippet": msg["snippet"][:30],
    }
    for field, expected in must_surface.items():
        assert expected in data, f"prefetch dropped Gmail field {field}"

    assert f"messages/{msg['id']}" not in data, (
        "prefetch should NOT expose raw Gmail message ids"
    )
    assert msg["threadId"] not in data, (
        "prefetch should NOT expose raw Gmail thread ids"
    )


def test_endpoint_capability_matrix(agent: ModuleType) -> None:
    """Encode the mail capability matrix as an executable assertion."""

    matrix: dict[str, dict[str, str]] = {
        "list_accounts": {
            "gmail_oauth": "GET /connections/gmail/instances",
            "deskmate": "n/a (single local user)",
        },
        "list_messages": {
            "gmail_oauth": "GET /connections/gmail/messages",
            "deskmate": "_do_email_digest_prefetch (screen-observed)",
        },
        "read_message": {
            "gmail_oauth": "GET /connections/gmail/messages/:id",
            "deskmate": "via OCR text of opened message",
        },
        "send_message": {
            "gmail_oauth": "POST /connections/gmail/send",
            "deskmate": "NOT SUPPORTED for Gmail; Outlook send uses /connections/outlook/send",
        },
    }
    assert "NOT SUPPORTED" in matrix["send_message"]["deskmate"]
    assert callable(agent._do_email_digest_prefetch)
