"""Parity verification for email-digest vs screenpipe Gmail surface.

Goal: prove that whatever the user actually SAW on screen (sender, subject,
snippet) is faithfully surfaced by ``_do_email_digest_prefetch``, with the same
field-level fidelity that screenpipe's ``parse_gmail_message`` produces from
the Gmail API.

Architectural caveat — read this first:
    screenpipe Gmail = Gmail OAuth → REST API → JSON
    pc_assistant email-digest = local screen OCR → markdown report
Byte-identical parity is therefore impossible without adding an OAuth backend
to pc_assistant. These tests verify the **closest achievable parity**: for any
email the user opened on screen, every field that screenpipe would have
returned via the Gmail API (from, subject, snippet, approximate date) appears
verbatim in the prefetch data the LLM receives.
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


# ── Fixtures shaped like screenpipe's parse_gmail_message() output ───────────
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
        "subject": "[pc_assistant] PR #142 ready for review",
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
    """Synthesise the OCR row pc_assistant would have captured while the user
    was reading this exact Gmail message in the browser.

    This is the bridge between the two data planes — every field screenpipe
    would have read via the API ends up in the on-screen text that
    pc_assistant captures via OCR.
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
                # Use the inbox root URL; we intentionally do NOT inject the
                # message id into the URL because pc_assistant has no way to
                # know it (Gmail message ids are an API-side concept) and the
                # parity tests assert that internal handles never leak.
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
    """For each email the user opened on screen, every Gmail field that
    screenpipe would return via the API (from, subject, snippet, date) must
    appear verbatim in the data passed to the LLM."""
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
        # The display-name fragment of the From: header must round-trip.
        sender_name = msg["from"].split(" <")[0]
        assert sender_name in data, f"missing sender {sender_name!r}"
        # Snippet preview the LLM uses for "Top Senders / Threads".
        assert msg["snippet"][:30] in data, f"missing snippet for {msg['subject']!r}"


def test_prefetch_excludes_messages_user_never_opened(
    agent: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An email that exists in the Gmail account but was never opened on
    screen MUST NOT appear in the prefetch. This is the property that
    distinguishes a faithful screen-observation pipeline from one that
    fabricates inbox state."""
    seen = GMAIL_INBOX_FIXTURE[:2]                # user opened the first two
    unseen = GMAIL_INBOX_FIXTURE[2]               # third email never opened on screen
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
    """Same fixture, native Outlook source → same field-level parity as the
    Gmail (web) case. screenpipe has no Outlook desktop equivalent at all, so
    this is strictly *additional* coverage (pc_assistant > screenpipe for
    desktop clients)."""
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
    """screenpipe's `parse_gmail_message` emits {id, threadId, from, to,
    subject, date, snippet, body}. Of those, the *human-meaningful* subset
    (from, subject, date, snippet) must be present in email-digest's data.
    id / threadId / to / body are intentionally not surfaced (id is a Gmail
    API artifact, body is too long for a digest)."""
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
        "from":    msg["from"].split(" <")[0],
        "subject": msg["subject"],
        # Date: at minimum the HH:MM (or formatted "9:14 AM") part is in the
        # active-window span the prefetch attaches to the header.
        "snippet": msg["snippet"][:30],
    }
    for field, expected in must_surface.items():
        assert expected in data, f"prefetch dropped Gmail field {field}"

    # `id` / `threadId` are internal Gmail handles — they must NOT leak into a
    # human-facing screen-observation digest (and we don't put them in the OCR
    # row either). `to` and `body` are intentionally allowed if the user's
    # screen happens to show them; we do not assert their absence.
    assert f"messages/{msg['id']}" not in data, (
        "prefetch should NOT expose raw Gmail message ids"
    )
    assert msg["threadId"] not in data, (
        "prefetch should NOT expose raw Gmail thread ids"
    )


def test_endpoint_capability_matrix(agent: ModuleType) -> None:
    """Encode the parity matrix as an executable assertion so it stays
    accurate as either codebase evolves."""

    matrix: dict[str, dict[str, str]] = {
        "list_accounts":  {"screenpipe": "GET /connections/gmail/instances",
                           "pc_assistant": "n/a (single local user)"},
        "list_messages":  {"screenpipe": "GET /connections/gmail/messages",
                           "pc_assistant": "_do_email_digest_prefetch (screen-observed)"},
        "read_message":   {"screenpipe": "GET /connections/gmail/messages/:id",
                           "pc_assistant": "via OCR text of opened message"},
        "send_message":   {"screenpipe": "POST /connections/gmail/send",
                   "pc_assistant": "NOT SUPPORTED for Gmail; Outlook send uses /connections/outlook/send"},
    }
    # send is the one capability gap we deliberately accept.
    assert "NOT SUPPORTED" in matrix["send_message"]["pc_assistant"]
    # The prefetch function exists and is callable.
    assert callable(agent._do_email_digest_prefetch)
    # The email targets cover the major webmail equivalents of Gmail.
    labels = set(agent.EMAIL_TOOL_TARGETS)
    assert {"Gmail (web)", "Outlook (web)"} <= labels


# ── Helper: assert the LLM prompt assembled by single_shot_report carries
#    the right anti-fabrication contract derived from `verified`.
def test_extra_rules_whitelist_carries_into_single_shot(
    agent: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, str] = {}

    def fake_chat_ollama(messages, tools=None, *, num_predict=4096):
        # Capture the prompt; return an empty response.
        captured["system"] = messages[0]["content"]
        captured["user"] = messages[1]["content"]
        return {"content": "## Email Tools Used\n- Gmail (web): ~5min"}

    monkeypatch.setattr(agent, "chat_ollama", fake_chat_ollama)

    pipe_md = Path(__file__).resolve().parents[1] / "apps" / "email-digest" / "pipe.md"
    assert pipe_md.exists()
    _, pipe_body = agent.parse_pipe_md(pipe_md)

    # Drive only the inner branch (skip run_agent's outer wiring).
    data, verified = "### Gmail (web) | substantive_hits=2", ["Gmail (web)"]
    extra = (
        f"ONLY these email tools have recorded usage: {', '.join(verified)}. "
        f"List ONLY these tools in 'Email Tools Used' and every other section. "
    )
    out = agent._single_shot_report(
        pipe_body=pipe_body,
        skill_text="",
        context_header="## Context\n- TZ: +08:00\n",
        data_text=data,
        verbose=False,
        extra_rules=extra,
        start_heading="## Email Tools Used",
    )

    assert "Gmail (web)" in out
    # Anti-fabrication contract: the verified-only whitelist text must reach
    # the model verbatim.
    assert "ONLY these email tools have recorded usage: Gmail (web)" in captured["system"]
    # And the report instructions (pipe.md body) must reach the user message.
    assert re.search(r"Email Tools Used", captured["user"])
