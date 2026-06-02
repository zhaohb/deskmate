from __future__ import annotations

import base64
from urllib.parse import parse_qs, urlparse

from fastapi.testclient import TestClient

from pc_assistant.config import Config, GmailConfig
from pc_assistant.connections.gmail import GmailConnection, build_raw_message, parse_gmail_message
from pc_assistant.db import DatabaseManager
from pc_assistant.engine.api import create_app


def test_gmail_connect_requires_client_id(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PC_ASSISTANT_HOME", str(tmp_path))
    cfg = Config(gmail=GmailConfig(client_id=""))
    app = create_app(cfg=cfg, db=DatabaseManager(tmp_path / "test.db"))
    response = TestClient(app).get("/connections/gmail/auth-url")
    assert response.status_code == 400
    assert "PCA_GMAIL__CLIENT_ID" in response.json()["error"]


def test_config_endpoint_redacts_gmail_client_secret(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PC_ASSISTANT_HOME", str(tmp_path))
    cfg = Config(gmail=GmailConfig(client_id="google-client", client_secret="secret-value"))
    app = create_app(cfg=cfg, db=DatabaseManager(tmp_path / "test.db"))
    response = TestClient(app).get("/config")
    assert response.status_code == 200
    assert response.json()["gmail"]["client_secret"] == "********"


def test_gmail_auth_url_uses_pkce_and_gmail_scopes(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PC_ASSISTANT_HOME", str(tmp_path))
    conn = GmailConnection(GmailConfig(client_id="google-client"))
    data = conn.auth_url("me@example.com")
    parsed = urlparse(data["authorization_url"])
    params = parse_qs(parsed.query)

    assert parsed.netloc == "accounts.google.com"
    assert params["client_id"] == ["google-client"]
    assert params["code_challenge_method"] == ["S256"]
    assert params["access_type"] == ["offline"]
    assert params["login_hint"] == ["me@example.com"]
    assert "https://www.googleapis.com/auth/gmail.readonly" in params["scope"][0]
    assert "https://www.googleapis.com/auth/gmail.send" in params["scope"][0]
    assert data["state"] == params["state"][0]


def test_parse_gmail_message_parses_standard_fields() -> None:
    encoded_body = base64.urlsafe_b64encode("Full Gmail body".encode()).decode().rstrip("=")
    parsed = parse_gmail_message(
        {
            "id": "msg-1",
            "threadId": "thread-1",
            "snippet": "Please review",
            "payload": {
                "headers": [
                    {"name": "From", "value": "Ada <ada@example.com>"},
                    {"name": "To", "value": "Me <me@example.com>"},
                    {"name": "Subject", "value": "Planning"},
                    {"name": "Date", "value": "Fri, 29 May 2026 09:00:00 +0800"},
                ],
                "body": {"data": encoded_body},
            },
        }
    )

    assert parsed == {
        "id": "msg-1",
        "threadId": "thread-1",
        "from": "Ada <ada@example.com>",
        "to": "Me <me@example.com>",
        "subject": "Planning",
        "date": "Fri, 29 May 2026 09:00:00 +0800",
        "snippet": "Please review",
        "body": "Full Gmail body",
    }


def test_build_raw_message_for_gmail_send() -> None:
    raw = build_raw_message(
        from_addr="me@example.com",
        to="ada@example.com",
        subject="Hello",
        body="Body",
    )
    decoded = base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)).decode()
    assert "From: me@example.com" in decoded
    assert "To: ada@example.com" in decoded
    assert "Subject: Hello" in decoded
    assert "Body" in decoded


def test_gmail_instances_are_loaded_from_token_store(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PC_ASSISTANT_HOME", str(tmp_path))
    conn = GmailConnection(GmailConfig(client_id="google-client"))
    conn._write_token(
        "me@example.com",
        {
            "access_token": "token",
            "refresh_token": "refresh",
            "expires_at": 9999999999,
            "instance": "me@example.com",
            "email": "me@example.com",
            "displayName": "Me",
        },
    )

    assert conn.list_instances() == [
        {
            "instance": "me@example.com",
            "email": "me@example.com",
            "display_name": "Me",
            "expires_at": 9999999999,
            "connected": True,
        }
    ]
