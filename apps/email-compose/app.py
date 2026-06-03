"""Email Compose — LLM drafts an email for a connected Gmail / Outlook account.

Fetches the connected accounts via /connections/{provider}/instances. When a
``--reply-to`` message id is supplied, also pulls the source message via
``/connections/{provider}/messages/{id}`` to ground the LLM. The pipe.md then
asks the LLM to draft Subject + Body + up to 2 alternatives.

When ``--send`` is passed, after the draft is generated the script POSTs the
primary subject + body to ``/connections/{provider}/send`` using the chosen
account; otherwise the draft is only written to disk for the user to review.

This app is the LLM-backed replacement for the old ``Send email`` UI form.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

sys.path.append(str(Path(__file__).resolve().parents[1]))
from agent import run_agent  # noqa: E402
from common import add_agent_time_args, agent_time_kwargs_from_args, output_dir, run_cli, write_markdown  # noqa: E402

APP_NAME = "email-compose"
PIPE_MD = Path(__file__).with_name("pipe.md")
API_BASE = os.environ.get("DESKMATE_API", "http://127.0.0.1:3030")

_GMAIL_URL_ID_RE = re.compile(
    r"mail\.google\.com/mail[^\s#]*#(?:inbox|all|sent|drafts|spam|trash|label|search)/"
    r"([A-Za-z0-9_-]+)",
    re.IGNORECASE,
)


def _raw_request(method: str, url: str, body: bytes | None = None,
                 headers: dict[str, str] | None = None, timeout: int = 60) -> Any:
    parsed = urlparse(url)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=timeout)
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    conn.request(method, path, body=body, headers=headers or {})
    resp = conn.getresponse()
    data = resp.read().decode("utf-8")
    conn.close()
    if resp.status >= 400:
        raise RuntimeError(f"HTTP {resp.status}: {data[:500]}")
    return json.loads(data)


def normalize_message_id(provider: str, raw: str) -> str:
    """Accept a raw Gmail/Outlook message id or a pasted webmail URL."""
    value = (raw or "").strip()
    if not value:
        return ""
    if provider == "gmail":
        match = _GMAIL_URL_ID_RE.search(value)
        if match:
            return match.group(1)
    if "@" in value and not value.startswith("http"):
        raise SystemExit(
            "Reply-to must be a message id, not an email address. "
            "Pick a recent message in the UI or leave blank for a new email."
        )
    return value


def _resolve_account(provider: str, account: str | None) -> str:
    payload = _raw_request("GET", f"{API_BASE}/connections/{provider}/instances")
    instances = payload.get("data") or [] if isinstance(payload, dict) else []
    candidates = [i.get("instance") or i.get("email") for i in instances if isinstance(i, dict)]
    candidates = [c for c in candidates if c]
    if not candidates:
        raise SystemExit(f"No {provider} account connected. Open the Email tab and click Connect.")
    if account:
        if account not in candidates:
            raise SystemExit(f"Account {account!r} not connected for {provider}. Have: {candidates}")
        return account
    return candidates[0]


def _fetch_source_message(provider: str, instance: str, message_id: str) -> dict[str, Any]:
    message_id = normalize_message_id(provider, message_id)
    url = (
        f"{API_BASE}/connections/{provider}/messages/{quote(message_id)}"
        f"?instance={quote(instance)}"
    )
    try:
        payload = _raw_request("GET", url)
    except RuntimeError as exc:
        msg = str(exc)
        if "Invalid id" in msg or "invalidArgument" in msg or "ItemNotFound" in msg:
            raise SystemExit(
                f"Could not load message {message_id!r} from {provider}. "
                "Use a recent message from the dropdown (not a browser URL or thread id), "
                "or leave reply blank to draft a new email."
            ) from exc
        raise SystemExit(f"Failed to fetch source message: {msg[:400]}") from exc
    return payload.get("data", {}) if isinstance(payload, dict) else {}


def _clean_subject(raw: str) -> str:
    """Normalize a subject line the model may wrap in markdown / quotes."""
    text = raw.strip()
    # Drop a leading "Subject:" the model sometimes repeats inline.
    text = re.sub(r"^subject\s*:\s*", "", text, flags=re.IGNORECASE)
    # Unwrap **bold**, `code`, and surrounding bullets / quotes.
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    return text.strip(" -*`\"'“”‘’").strip()


def _parse_draft(report: str) -> tuple[str, str]:
    """Extract Subject and Body from the LLM markdown report.

    Tolerates the heading variants a small model emits: ``## Subject`` on its
    own line followed by the value, or an inline ``## Subject: value``. The body
    capture stops at the next ``##`` heading so Alternatives / Send Preview are
    never folded into the sent body.
    """
    subject = ""
    subject_match = re.search(
        r"##\s*Subject\b[ \t]*:?[ \t]*\n*[ \t]*-?[ \t]*(.+)", report
    )
    if subject_match:
        # Keep only the first non-empty line of the captured text.
        for line in subject_match.group(1).splitlines():
            cleaned = _clean_subject(line)
            if cleaned:
                subject = cleaned
                break

    body_match = re.search(
        r"##\s*Body\b[ \t]*:?[ \t]*\n+(.+?)(?=\n##\s|\Z)", report, flags=re.DOTALL,
    )
    body_raw = body_match.group(1).strip() if body_match else ""
    # strip leading "- " bullets that the LLM may emit
    body_lines = [re.sub(r"^\s*[-*]\s*", "", line) for line in body_raw.splitlines()]
    body = "\n".join(body_lines).strip()
    return subject, body


def _send(provider: str, instance: str, to: str, subject: str, body: str) -> dict[str, Any]:
    payload = {"instance": instance, "to": to, "subject": subject, "body": body}
    return _raw_request(
        "POST", f"{API_BASE}/connections/{provider}/send",
        body=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )


def _build_context(provider: str, instance: str, intent: str, to: str,
                   source: dict[str, Any] | None) -> str:
    parts = [
        "## Compose request",
        f"- provider: {provider}",
        f"- account: {instance}",
        f"- to: {to or '(unknown)'}",
        "",
        "### Intent (user-supplied)",
        intent or "(no intent provided)",
    ]
    if source:
        parts += [
            "",
            "### Source message (for reply)",
            f"- id: {source.get('id', '')}",
            f"- from: {source.get('from', '')}",
            f"- subject: {source.get('subject', '')}",
            f"- date: {source.get('date', '')}",
            "",
            "Snippet:",
            (source.get("snippet") or "").strip(),
            "",
            "Body:",
            ((source.get("body") or "")[:2000]).strip(),
        ]
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LLM-drafted email for a connected Gmail / Outlook account."
    )
    parser.add_argument("--provider", choices=["gmail", "outlook"], required=True)
    parser.add_argument("--to", required=True, help="Recipient email address.")
    parser.add_argument("--intent", required=True, help="What you want to say (one sentence).")
    parser.add_argument("--account", default=None, help="Connected account email (defaults to first).")
    parser.add_argument("--reply-to", default=None, help="Source message id to ground a reply.")
    parser.add_argument("--send", action="store_true", help="Actually send via the provider after drafting.")
    add_agent_time_args(parser, default_hours=1.0)
    parser.add_argument("--model", type=str, default=None, help="Ollama model override.")
    parser.add_argument("--verbose", action="store_true", help="Print agent rounds to stderr.")
    args = parser.parse_args()

    if args.model:
        import agent
        agent.OLLAMA_MODEL = args.model

    instance = _resolve_account(args.provider, args.account)
    source = None
    if args.reply_to:
        source = _fetch_source_message(args.provider, instance, args.reply_to)

    compose_context = _build_context(args.provider, instance, args.intent, args.to, source)

    # Stash the compose context where run_agent's data_text picks it up. We
    # reuse the generic single-shot path by writing it to an env var the pipe
    # body interpolates — but to keep the change minimal we inline it as a
    # one-shot temp pipe.md addendum.
    base_pipe = PIPE_MD.read_text(encoding="utf-8")
    tmp_pipe = PIPE_MD.with_name("_compose_runtime.md")
    tmp_pipe.write_text(
        f"{base_pipe}\n\n---\n\n## Per-compose context (verified, do NOT fetch more)\n\n"
        f"Provider for this draft: **{args.provider}**\n\n{compose_context}\n",
        encoding="utf-8",
    )
    time_kwargs = agent_time_kwargs_from_args(args)
    try:
        report = run_agent(tmp_pipe, verbose=args.verbose, **time_kwargs)
    finally:
        try:
            tmp_pipe.unlink()
        except OSError:
            pass

    subject, body = _parse_draft(report)

    out = output_dir(APP_NAME)
    write_markdown(out / "email-compose.md", report)
    print(out / "email-compose.md")

    if args.send:
        if not subject or not body:
            print("Could not parse Subject/Body from draft; refusing to send.", file=sys.stderr)
            return 2
        result = _send(args.provider, instance, args.to, subject, body)
        print(f"Sent via {args.provider}: {json.dumps(result)[:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
