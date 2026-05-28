"""Gmail OAuth and Gmail API helpers."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import secrets
import time
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx

from .. import paths
from ..config import GmailConfig


AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
GMAIL_BASE = "https://gmail.googleapis.com/gmail/v1"
USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"
PENDING_TTL_SECONDS = 10 * 60


class GmailError(RuntimeError):
    def __init__(self, message: str, *, status_code: int = 500, upstream_status: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.upstream_status = upstream_status


@dataclass(frozen=True)
class PendingOAuth:
    state: str
    code_verifier: str
    instance_hint: str | None
    created_at: float


_PENDING: dict[str, PendingOAuth] = {}


def _now() -> float:
    return time.time()


def _urlsafe_random(num_bytes: int = 32) -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(num_bytes)).decode("ascii").rstrip("=")


def _pkce_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _sanitize_instance(instance: str | None) -> str:
    value = (instance or "default").strip().lower() or "default"
    return re.sub(r"[^a-z0-9_.@-]+", "_", value)


def _decode_base64url(data: str | None) -> str:
    if not data:
        return ""
    padded = data + "=" * (-len(data) % 4)
    try:
        return base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8", errors="replace")
    except Exception:
        return ""


def _extract_text_body(payload: dict[str, Any]) -> str:
    body = payload.get("body") if isinstance(payload.get("body"), dict) else {}
    direct = _decode_base64url(body.get("data"))
    if direct:
        return direct
    parts = payload.get("parts") if isinstance(payload.get("parts"), list) else []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if part.get("mimeType") == "text/plain":
            text = _decode_base64url((part.get("body") or {}).get("data"))
            if text:
                return text
        nested = _extract_text_body(part)
        if nested:
            return nested
    return ""


def parse_gmail_message(msg: dict[str, Any]) -> dict[str, Any]:
    payload = msg.get("payload") if isinstance(msg.get("payload"), dict) else {}
    headers = payload.get("headers") if isinstance(payload.get("headers"), list) else []

    def header(name: str) -> str:
        for item in headers:
            if not isinstance(item, dict):
                continue
            if str(item.get("name") or "").lower() == name.lower():
                return str(item.get("value") or "")
        return ""

    return {
        "id": msg.get("id") or "",
        "threadId": msg.get("threadId") or "",
        "from": header("From"),
        "to": header("To"),
        "subject": header("Subject"),
        "date": header("Date"),
        "snippet": msg.get("snippet") or "",
        "body": _extract_text_body(payload),
    }


def build_raw_message(*, to: str, subject: str, body: str, from_addr: str | None = None) -> str:
    message = EmailMessage()
    if from_addr:
        message["From"] = from_addr
    message["To"] = to
    message["Subject"] = subject
    message.set_content(body)
    return base64.urlsafe_b64encode(message.as_bytes()).decode("ascii").rstrip("=")


class GmailConnection:
    def __init__(self, cfg: GmailConfig) -> None:
        self.cfg = cfg

    @property
    def redirect_uri(self) -> str:
        return self.cfg.redirect_uri or "http://127.0.0.1:3030/connections/gmail/oauth/callback"

    def auth_url(self, instance: str | None = None) -> dict[str, str]:
        self._require_client_id()
        self._cleanup_pending()
        state = _urlsafe_random(24)
        verifier = _urlsafe_random(48)
        _PENDING[state] = PendingOAuth(state, verifier, instance, _now())
        params = {
            "client_id": self.cfg.client_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "scope": " ".join(self.cfg.scopes),
            "state": state,
            "code_challenge": _pkce_challenge(verifier),
            "code_challenge_method": "S256",
            "access_type": "offline",
            "prompt": "consent select_account",
        }
        if instance:
            params["login_hint"] = instance
        return {"authorization_url": f"{AUTH_URL}?{urlencode(params)}", "state": state, "redirect_uri": self.redirect_uri}

    async def complete_oauth(self, code: str, state: str) -> dict[str, Any]:
        self._require_client_id()
        pending = _PENDING.pop(state, None)
        if pending is None or _now() - pending.created_at > PENDING_TTL_SECONDS:
            raise GmailError("Gmail OAuth state is missing or expired", status_code=400)
        async with httpx.AsyncClient(timeout=30) as client:
            token = await self._exchange_code(client, code, pending.code_verifier)
            profile = await self._google_json(client, "GET", USERINFO_URL, token["access_token"])
        instance = str(profile.get("email") or pending.instance_hint or "").strip().lower()
        if not instance:
            raise GmailError("Gmail profile did not include an email", status_code=502)
        token.update({"instance": instance, "email": instance, "displayName": profile.get("name")})
        self._write_token(instance, token)
        return {"connected": True, "instance": instance, "email": instance, "display_name": profile.get("name")}

    def list_instances(self) -> list[dict[str, Any]]:
        items = []
        for path in sorted(self._store_dir().glob("gmail-*.json")):
            data = self._read_path(path)
            if data is None:
                continue
            items.append({
                "instance": data.get("instance") or path.stem.removeprefix("gmail-"),
                "email": data.get("email"),
                "display_name": data.get("displayName"),
                "expires_at": data.get("expires_at"),
                "connected": bool(data.get("access_token") or data.get("refresh_token")),
            })
        return items

    async def status(self, instance: str | None = None) -> dict[str, Any]:
        try:
            selected, token = self._load_token(instance)
        except GmailError as exc:
            if exc.status_code == 401:
                return {"connected": False, "error": str(exc), "instances": self.list_instances()}
            raise
        return {"connected": True, "instance": selected, "email": token.get("email"), "display_name": token.get("displayName"), "expires_at": token.get("expires_at")}

    async def list_messages(self, *, query: str | None = None, max_results: int = 20, page_token: str | None = None, instance: str | None = None) -> dict[str, Any]:
        token = await self._valid_access_token(instance)
        params: dict[str, str] = {"maxResults": str(min(max(max_results, 1), 500))}
        if query:
            params["q"] = query
        if page_token:
            params["pageToken"] = page_token
        async with httpx.AsyncClient(timeout=30) as client:
            data = await self._google_json(client, "GET", f"{GMAIL_BASE}/users/me/messages", token, params=params)
        return data

    async def get_message(self, message_id: str, *, instance: str | None = None) -> dict[str, Any]:
        token = await self._valid_access_token(instance)
        async with httpx.AsyncClient(timeout=30) as client:
            data = await self._google_json(client, "GET", f"{GMAIL_BASE}/users/me/messages/{message_id}", token, params={"format": "full"})
        return parse_gmail_message(data)

    async def send_message(self, body: dict[str, Any]) -> dict[str, Any]:
        token = await self._valid_access_token(body.get("instance"))
        to_addr = str(body.get("to") or "").strip()
        if not to_addr:
            raise GmailError("Gmail send requires 'to'", status_code=400)
        raw = build_raw_message(
            to=to_addr,
            subject=str(body.get("subject") or ""),
            body=str(body.get("body") or ""),
            from_addr=body.get("from"),
        )
        async with httpx.AsyncClient(timeout=30) as client:
            return await self._google_json(client, "POST", f"{GMAIL_BASE}/users/me/messages/send", token, json_body={"raw": raw})

    def disconnect(self, instance: str | None = None) -> bool:
        selected, _ = self._load_token(instance)
        path = self._token_path(selected)
        if path.exists():
            path.unlink()
            return True
        return False

    async def _valid_access_token(self, instance: str | None) -> str:
        selected, token = self._load_token(instance)
        if int(token.get("expires_at") or 0) > int(_now()) + 60 and token.get("access_token"):
            return str(token["access_token"])
        refresh_token = token.get("refresh_token")
        if not refresh_token:
            raise GmailError(f"Gmail account {selected} needs reconnect", status_code=401)
        async with httpx.AsyncClient(timeout=30) as client:
            refreshed = await self._refresh_token(client, str(refresh_token))
        refreshed["refresh_token"] = refreshed.get("refresh_token") or refresh_token
        refreshed["instance"] = selected
        refreshed["email"] = token.get("email")
        refreshed["displayName"] = token.get("displayName")
        self._write_token(selected, refreshed)
        return str(refreshed["access_token"])

    async def _exchange_code(self, client: httpx.AsyncClient, code: str, verifier: str) -> dict[str, Any]:
        data = {
            "client_id": self.cfg.client_id,
            "code": code,
            "code_verifier": verifier,
            "redirect_uri": self.redirect_uri,
            "grant_type": "authorization_code",
        }
        if self.cfg.client_secret:
            data["client_secret"] = self.cfg.client_secret
        resp = await client.post(TOKEN_URL, data=data)
        return await self._token_json(resp)

    async def _refresh_token(self, client: httpx.AsyncClient, refresh_token: str) -> dict[str, Any]:
        data = {"client_id": self.cfg.client_id, "refresh_token": refresh_token, "grant_type": "refresh_token"}
        if self.cfg.client_secret:
            data["client_secret"] = self.cfg.client_secret
        resp = await client.post(TOKEN_URL, data=data)
        return await self._token_json(resp)

    async def _token_json(self, resp: httpx.Response) -> dict[str, Any]:
        if not resp.is_success:
            raise GmailError(resp.text, status_code=401, upstream_status=resp.status_code)
        data = resp.json()
        if "access_token" not in data:
            raise GmailError("Gmail token response did not include access_token", status_code=502)
        data["expires_at"] = int(_now()) + int(data.get("expires_in") or 3600) - 60
        return data

    async def _google_json(self, client: httpx.AsyncClient, method: str, url: str, token: str, *, params: dict[str, str] | None = None, json_body: dict[str, Any] | None = None) -> dict[str, Any]:
        resp = await client.request(method, url, params=params, headers={"Authorization": f"Bearer {token}", "Accept": "application/json"}, json=json_body)
        if not resp.is_success:
            raise GmailError(resp.text, status_code=resp.status_code, upstream_status=resp.status_code)
        if not resp.content:
            return {}
        return resp.json()

    def _store_dir(self) -> Path:
        path = paths.root() / "oauth"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _token_path(self, instance: str | None) -> Path:
        return self._store_dir() / f"gmail-{_sanitize_instance(instance)}.json"

    def _write_token(self, instance: str, token: dict[str, Any]) -> None:
        self._token_path(instance).write_text(json.dumps(token, indent=2, ensure_ascii=False), encoding="utf-8")

    def _read_path(self, path: Path) -> dict[str, Any] | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _load_token(self, instance: str | None) -> tuple[str, dict[str, Any]]:
        if instance:
            data = self._read_path(self._token_path(instance))
            if data is None:
                raise GmailError(f"Gmail account {instance} is not connected", status_code=401)
            return instance, data
        instances = self.list_instances()
        if not instances:
            raise GmailError("Gmail is not connected", status_code=401)
        if len(instances) > 1:
            names = ", ".join(str(item.get("instance")) for item in instances)
            raise GmailError(f"Multiple Gmail accounts connected; pass instance= one of: {names}", status_code=400)
        selected = str(instances[0]["instance"])
        data = self._read_path(self._token_path(selected))
        if data is None:
            raise GmailError(f"Gmail account {selected} token is unreadable", status_code=401)
        return selected, data

    def _require_client_id(self) -> None:
        if not self.cfg.client_id.strip():
            raise GmailError(
                "Gmail OAuth is not configured. Set PCA_GMAIL__CLIENT_ID or [gmail] client_id.",
                status_code=400,
            )

    def _cleanup_pending(self) -> None:
        cutoff = _now() - PENDING_TTL_SECONDS
        for state, pending in list(_PENDING.items()):
            if pending.created_at < cutoff:
                _PENDING.pop(state, None)
