"""Microsoft Outlook OAuth and Graph Mail helpers."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import secrets
import time
from dataclasses import dataclass
from email.utils import parseaddr
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx

from .. import paths
from ..config import OutlookConfig


AUTH_BASE = "https://login.microsoftonline.com"
GRAPH_BASE = "https://graph.microsoft.com/v1.0"
PENDING_TTL_SECONDS = 10 * 60


class OutlookError(RuntimeError):
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


def _json_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def _content_text(body: Any) -> str:
    if not isinstance(body, dict):
        return ""
    return str(body.get("content") or "")


def parse_graph_message(msg: dict[str, Any]) -> dict[str, Any]:
    sender = msg.get("from") or {}
    sender_addr = sender.get("emailAddress") if isinstance(sender, dict) else {}
    to_recipients = msg.get("toRecipients") if isinstance(msg.get("toRecipients"), list) else []
    to_values = []
    for recipient in to_recipients:
        email = (recipient or {}).get("emailAddress") if isinstance(recipient, dict) else None
        if isinstance(email, dict):
            name = str(email.get("name") or "").strip()
            address = str(email.get("address") or "").strip()
            to_values.append(f"{name} <{address}>" if name and address else address or name)
    sender_name = str((sender_addr or {}).get("name") or "").strip()
    sender_email = str((sender_addr or {}).get("address") or "").strip()
    return {
        "id": msg.get("id") or "",
        "threadId": msg.get("conversationId") or "",
        "conversationId": msg.get("conversationId") or "",
        "from": f"{sender_name} <{sender_email}>" if sender_name and sender_email else sender_email or sender_name,
        "to": ", ".join(v for v in to_values if v),
        "subject": msg.get("subject") or "",
        "date": msg.get("receivedDateTime") or msg.get("sentDateTime") or "",
        "snippet": msg.get("bodyPreview") or "",
        "body": _content_text(msg.get("body")),
        "isRead": bool(msg.get("isRead", False)),
        "webLink": msg.get("webLink") or "",
    }


def build_send_payload(
    *,
    to: str | list[str],
    subject: str,
    body: str,
    cc: str | list[str] | None = None,
    bcc: str | list[str] | None = None,
    save_to_sent: bool = True,
) -> dict[str, Any]:
    def recipients(value: str | list[str] | None) -> list[dict[str, dict[str, str]]]:
        if value is None:
            return []
        raw_values = value if isinstance(value, list) else [v.strip() for v in value.split(",")]
        out = []
        for raw in raw_values:
            name, address = parseaddr(raw)
            address = (address or raw).strip()
            if not address:
                continue
            email_address = {"address": address}
            if name:
                email_address["name"] = name
            out.append({"emailAddress": email_address})
        return out

    payload = {
        "message": {
            "subject": subject,
            "body": {"contentType": "Text", "content": body},
            "toRecipients": recipients(to),
        },
        "saveToSentItems": save_to_sent,
    }
    cc_values = recipients(cc)
    bcc_values = recipients(bcc)
    if cc_values:
        payload["message"]["ccRecipients"] = cc_values
    if bcc_values:
        payload["message"]["bccRecipients"] = bcc_values
    return payload


class OutlookConnection:
    def __init__(self, cfg: OutlookConfig) -> None:
        self.cfg = cfg

    @property
    def redirect_uri(self) -> str:
        return self.cfg.redirect_uri or "http://127.0.0.1:3030/connections/outlook/oauth/callback"

    @property
    def token_url(self) -> str:
        return f"{AUTH_BASE}/{self.cfg.tenant}/oauth2/v2.0/token"

    def auth_url(self, instance: str | None = None) -> dict[str, str]:
        self._require_client_id()
        self._cleanup_pending()
        state = _urlsafe_random(24)
        verifier = _urlsafe_random(48)
        _PENDING[state] = PendingOAuth(
            state=state,
            code_verifier=verifier,
            instance_hint=instance,
            created_at=_now(),
        )
        params = {
            "client_id": self.cfg.client_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "response_mode": "query",
            "scope": " ".join(self.cfg.scopes),
            "state": state,
            "code_challenge": _pkce_challenge(verifier),
            "code_challenge_method": "S256",
            "prompt": "select_account",
        }
        if instance:
            params["login_hint"] = instance
        return {
            "authorization_url": f"{AUTH_BASE}/{self.cfg.tenant}/oauth2/v2.0/authorize?{urlencode(params)}",
            "state": state,
            "redirect_uri": self.redirect_uri,
        }

    async def complete_oauth(self, code: str, state: str) -> dict[str, Any]:
        self._require_client_id()
        pending = _PENDING.pop(state, None)
        if pending is None or _now() - pending.created_at > PENDING_TTL_SECONDS:
            raise OutlookError("Outlook OAuth state is missing or expired", status_code=400)
        async with httpx.AsyncClient(timeout=30) as client:
            token = await self._exchange_code(client, code, pending.code_verifier)
            profile = await self._graph_json(client, "GET", "/me", token["access_token"])
        instance = self._instance_from_profile(profile, pending.instance_hint)
        token.update({
            "instance": instance,
            "email": profile.get("mail") or profile.get("userPrincipalName") or instance,
            "displayName": profile.get("displayName"),
        })
        self._write_token(instance, token)
        return {"connected": True, "instance": instance, "email": token["email"], "display_name": token.get("displayName")}

    def list_instances(self) -> list[dict[str, Any]]:
        items = []
        for path in sorted(self._store_dir().glob("outlook-*.json")):
            data = self._read_path(path)
            if data is None:
                continue
            items.append({
                "instance": data.get("instance") or path.stem.removeprefix("outlook-"),
                "email": data.get("email"),
                "display_name": data.get("displayName"),
                "expires_at": data.get("expires_at"),
                "connected": bool(data.get("access_token") or data.get("refresh_token")),
            })
        return items

    async def status(self, instance: str | None = None) -> dict[str, Any]:
        try:
            selected, token = self._load_token(instance)
        except OutlookError as exc:
            if exc.status_code == 401:
                return {"connected": False, "error": str(exc), "instances": self.list_instances()}
            raise
        return {
            "connected": True,
            "instance": selected,
            "email": token.get("email"),
            "display_name": token.get("displayName"),
            "expires_at": token.get("expires_at"),
        }

    async def list_messages(
        self,
        *,
        query: str | None = None,
        top: int = 20,
        skip: int | None = None,
        instance: str | None = None,
    ) -> dict[str, Any]:
        token = await self._valid_access_token(instance)
        params: dict[str, str] = {
            "$top": str(min(max(top, 1), 100)),
            "$select": "id,conversationId,subject,from,toRecipients,receivedDateTime,bodyPreview,isRead,webLink",
            "$orderby": "receivedDateTime desc",
        }
        headers: dict[str, str] | None = None
        if skip is not None:
            params["$skip"] = str(max(skip, 0))
        if query:
            params.pop("$orderby", None)
            params["$search"] = f'"{query}"'
            headers = {"ConsistencyLevel": "eventual"}
        async with httpx.AsyncClient(timeout=30) as client:
            data = await self._graph_json(client, "GET", "/me/messages", token, params=params, headers=headers)
        messages = [parse_graph_message(m) for m in data.get("value", []) if isinstance(m, dict)]
        return {"messages": messages, "nextLink": data.get("@odata.nextLink")}

    async def get_message(self, message_id: str, *, instance: str | None = None) -> dict[str, Any]:
        token = await self._valid_access_token(instance)
        params = {"$select": "id,conversationId,subject,from,toRecipients,receivedDateTime,bodyPreview,body,isRead,webLink"}
        async with httpx.AsyncClient(timeout=30) as client:
            data = await self._graph_json(client, "GET", f"/me/messages/{message_id}", token, params=params)
        return parse_graph_message(data)

    async def send_message(self, body: dict[str, Any]) -> dict[str, Any]:
        token = await self._valid_access_token(body.get("instance"))
        payload = build_send_payload(
            to=body.get("to") or [],
            subject=str(body.get("subject") or ""),
            body=str(body.get("body") or ""),
            cc=body.get("cc"),
            bcc=body.get("bcc"),
            save_to_sent=bool(body.get("save_to_sent", body.get("saveToSentItems", True))),
        )
        if not payload["message"]["toRecipients"]:
            raise OutlookError("Outlook send requires at least one recipient", status_code=400)
        async with httpx.AsyncClient(timeout=30) as client:
            await self._graph_json(client, "POST", "/me/sendMail", token, json_body=payload, expect_json=False)
        return {"sent": True, "saveToSentItems": payload["saveToSentItems"]}

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
            raise OutlookError(f"Outlook account {selected} needs reconnect", status_code=401)
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
            "scope": " ".join(self.cfg.scopes),
            "code": code,
            "redirect_uri": self.redirect_uri,
            "grant_type": "authorization_code",
            "code_verifier": verifier,
        }
        resp = await client.post(self.token_url, data=data)
        return await self._token_json(resp)

    async def _refresh_token(self, client: httpx.AsyncClient, refresh_token: str) -> dict[str, Any]:
        data = {
            "client_id": self.cfg.client_id,
            "scope": " ".join(self.cfg.scopes),
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }
        resp = await client.post(self.token_url, data=data)
        return await self._token_json(resp)

    async def _token_json(self, resp: httpx.Response) -> dict[str, Any]:
        if not resp.is_success:
            raise OutlookError(resp.text, status_code=401, upstream_status=resp.status_code)
        data = resp.json()
        if "access_token" not in data:
            raise OutlookError("Outlook token response did not include access_token", status_code=502)
        data["expires_at"] = int(_now()) + int(data.get("expires_in") or 3600) - 60
        return data

    async def _graph_json(
        self,
        client: httpx.AsyncClient,
        method: str,
        path: str,
        token: str,
        *,
        params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
        expect_json: bool = True,
    ) -> dict[str, Any]:
        merged_headers = _json_headers(token)
        if headers:
            merged_headers.update(headers)
        resp = await client.request(method, f"{GRAPH_BASE}{path}", params=params, headers=merged_headers, json=json_body)
        if not resp.is_success:
            raise OutlookError(resp.text, status_code=resp.status_code, upstream_status=resp.status_code)
        if not expect_json or resp.status_code == 202 or not resp.content:
            return {}
        return resp.json()

    def _store_dir(self) -> Path:
        path = paths.root() / "oauth"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _token_path(self, instance: str | None) -> Path:
        return self._store_dir() / f"outlook-{_sanitize_instance(instance)}.json"

    def _write_token(self, instance: str, token: dict[str, Any]) -> None:
        path = self._token_path(instance)
        path.write_text(json.dumps(token, indent=2, ensure_ascii=False), encoding="utf-8")

    def _read_path(self, path: Path) -> dict[str, Any] | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _load_token(self, instance: str | None) -> tuple[str, dict[str, Any]]:
        if instance:
            data = self._read_path(self._token_path(instance))
            if data is None:
                raise OutlookError(f"Outlook account {instance} is not connected", status_code=401)
            return instance, data
        instances = self.list_instances()
        if not instances:
            raise OutlookError("Outlook is not connected", status_code=401)
        if len(instances) > 1:
            names = ", ".join(str(i.get("instance")) for i in instances)
            raise OutlookError(f"Multiple Outlook accounts connected; pass instance= one of: {names}", status_code=400)
        selected = str(instances[0]["instance"])
        data = self._read_path(self._token_path(selected))
        if data is None:
            raise OutlookError(f"Outlook account {selected} token is unreadable", status_code=401)
        return selected, data

    def _instance_from_profile(self, profile: dict[str, Any], hint: str | None) -> str:
        email = profile.get("mail") or profile.get("userPrincipalName") or hint
        if not email:
            raise OutlookError("Outlook profile did not include an account email", status_code=502)
        return str(email).strip().lower()

    def _require_client_id(self) -> None:
        if not self.cfg.client_id.strip():
            raise OutlookError(
                "Outlook OAuth is not configured. Set DESKMATE_OUTLOOK__CLIENT_ID or [outlook] client_id.",
                status_code=400,
            )

    def _cleanup_pending(self) -> None:
        cutoff = _now() - PENDING_TTL_SECONDS
        for state, pending in list(_PENDING.items()):
            if pending.created_at < cutoff:
                _PENDING.pop(state, None)
