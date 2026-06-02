from __future__ import annotations

from urllib.parse import parse_qs, urlparse

from fastapi.testclient import TestClient

from deskmate.config import Config, OutlookConfig
from deskmate.connections.outlook import OutlookConnection, build_send_payload, parse_graph_message
from deskmate.db import DatabaseManager
from deskmate.engine.api import create_app


def test_outlook_connect_requires_client_id(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    cfg = Config(outlook=OutlookConfig(client_id=""))
    app = create_app(cfg=cfg, db=DatabaseManager(tmp_path / "test.db"))
    response = TestClient(app).get("/connections/outlook/auth-url")
    assert response.status_code == 400
    assert "DESKMATE_OUTLOOK__CLIENT_ID" in response.json()["error"]


def test_outlook_auth_url_uses_pkce_and_graph_scopes(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    conn = OutlookConnection(OutlookConfig(client_id="client-123"))
    data = conn.auth_url("me@example.com")
    parsed = urlparse(data["authorization_url"])
    params = parse_qs(parsed.query)

    assert parsed.netloc == "login.microsoftonline.com"
    assert params["client_id"] == ["client-123"]
    assert params["code_challenge_method"] == ["S256"]
    assert params["login_hint"] == ["me@example.com"]
    assert "Mail.Read" in params["scope"][0]
    assert "Mail.Send" in params["scope"][0]
    assert "offline_access" in params["scope"][0]
    assert data["state"] == params["state"][0]


def test_parse_graph_message_matches_email_shape() -> None:
    parsed = parse_graph_message(
        {
            "id": "AAMk-id",
            "conversationId": "thread-1",
            "subject": "Quarterly planning",
            "receivedDateTime": "2026-05-29T09:00:00Z",
            "bodyPreview": "Please review the plan",
            "body": {"content": "Full body"},
            "isRead": True,
            "webLink": "https://outlook.office.com/mail/id/AAMk-id",
            "from": {"emailAddress": {"name": "Ada", "address": "ada@example.com"}},
            "toRecipients": [
                {"emailAddress": {"name": "Me", "address": "me@example.com"}},
            ],
        }
    )

    assert parsed["id"] == "AAMk-id"
    assert parsed["threadId"] == "thread-1"
    assert parsed["from"] == "Ada <ada@example.com>"
    assert parsed["to"] == "Me <me@example.com>"
    assert parsed["subject"] == "Quarterly planning"
    assert parsed["snippet"] == "Please review the plan"
    assert parsed["body"] == "Full body"


def test_build_send_payload_for_graph_send_mail() -> None:
    payload = build_send_payload(
        to="Ada <ada@example.com>, bob@example.com",
        cc="cc@example.com",
        subject="Hello",
        body="Body",
    )

    assert payload["message"]["subject"] == "Hello"
    assert payload["message"]["body"] == {"contentType": "Text", "content": "Body"}
    assert payload["message"]["toRecipients"][0]["emailAddress"]["address"] == "ada@example.com"
    assert payload["message"]["toRecipients"][1]["emailAddress"]["address"] == "bob@example.com"
    assert payload["message"]["ccRecipients"][0]["emailAddress"]["address"] == "cc@example.com"
    assert payload["saveToSentItems"] is True


def test_outlook_instances_are_loaded_from_token_store(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    conn = OutlookConnection(OutlookConfig(client_id="client-123"))
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
