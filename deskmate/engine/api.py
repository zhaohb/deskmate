"""FastAPI HTTP server for the local activity recorder.

The API exposes health checks, search, frames, events, audio transcripts,
speakers, monitors, configuration and the browser UI. Routes for optional
subsystems that are not implemented return a 501 `Not Implemented`.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import time
from collections.abc import AsyncIterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .. import events as bus
from .. import paths
from ..a11y.activity_feed import default as activity_default
from ..capture import paired_capture as run_paired_capture
from ..connections.gmail import GmailConnection, GmailError
from ..connections.outlook import OutlookConnection, OutlookError
from ..config import Config, load as load_config
from ..db import DatabaseManager
from ..habits import HabitMiner, HabitStore
from ..habits import rules as habit_rules
from ..fusion import CaptureControl, ContextStore
from ..fusion.control import TOGGLEABLE
from ..logger import get
from ..pipes import PipeRuntime, load_pipes
from ..screen.capture import list_monitors
from ..screen.redact_image import redact_image_bytes, regions_from_ocr
from ..screen.video_chunks import video_chunk_path
from ..ui import index_file, static_dir
from ..workflow import WorkflowClassifier
from . import app_schedules

logger = get("engine.api")


# ── editable-settings schema ──────────────────────────────────────────────
# Single source of truth for the user-friendly Settings UI. Each field names a
# config (section, key), a type that drives the input widget + validation, a
# human label/help string, and whether changing it needs a restart. Kept to the
# high-frequency settings a normal user actually tunes — not every config key.
#   type: "bool" | "int" | "float" | "str" | "choice" | "csv"
#   restart: True  → takes effect only after a restart (most fields)
#            False → hot-applied live by _hot_apply_settings
SETTINGS_SCHEMA: list[dict[str, Any]] = [
    {
        "id": "recording",
        "title": "录制 Recording",
        "desc": "控制 DeskMate 是否以及如何捕获你的屏幕与输入。",
        "fields": [
            {"section": "capture", "key": "enabled", "type": "bool", "restart": True,
             "label": "启用屏幕录制",
             "help": "关闭后将完全停止截图与画面捕获。"},
            {"section": "capture", "key": "include_screenshot", "type": "bool", "restart": True,
             "label": "保存截图",
             "help": "关闭则只记录文字/事件元数据，不存 JPEG 画面，省磁盘。"},
            {"section": "capture", "key": "screenshot_max_width", "type": "int", "restart": True,
             "label": "截图最大宽度 (px)", "min": 640, "max": 3840,
             "help": "更小=更省空间、OCR 略糊；1920 通常足够。"},
            {"section": "a11y", "key": "capture_keystrokes", "type": "bool", "restart": True,
             "label": "记录键盘输入",
             "help": "用于「你输入了什么/在做什么」。关闭更注重隐私。"},
            {"section": "a11y", "key": "capture_clipboard", "type": "bool", "restart": True,
             "label": "记录剪贴板",
             "help": "捕获复制的文本片段，增强活动理解。"},
        ],
    },
    {
        "id": "ocr",
        "title": "文字识别 OCR",
        "desc": "把屏幕上的文字转成可搜索内容。",
        "fields": [
            {"section": "ocr", "key": "engine", "type": "choice", "restart": True,
             "choices": ["rapidocr", "winrt", "tesseract", "off"],
             "label": "OCR 引擎",
             "help": "rapidocr 对中文/小字最好；winrt 无需额外依赖；off 关闭。"},
        ],
    },
    {
        "id": "audio",
        "title": "音频与转录 Audio",
        "desc": "本地语音转写与实时翻译。语言可即时生效，其余需重启。",
        "fields": [
            {"section": "audio", "key": "enabled", "type": "bool", "restart": True,
             "label": "启用音频转录",
             "help": "开启后用 Whisper 把麦克风/系统声音转成文字。"},
            {"section": "audio", "key": "languages", "type": "csv", "restart": False,
             "label": "转录语言", "placeholder": "zh, en",
             "help": "逗号分隔的 ISO 代码；留空=自动检测。下一段音频即生效，无需重启。"},
            {"section": "audio", "key": "translate_enabled", "type": "bool", "restart": False,
             "label": "实时翻译",
             "help": "把每句转录翻译成目标语言。即时生效。注意会持续占用 Ollama。"},
            {"section": "audio", "key": "translate_target_lang", "type": "str", "restart": False,
             "label": "翻译目标语言", "placeholder": "zh",
             "help": "ISO 639-1 代码，如 zh / en / ja。即时生效。"},
        ],
    },
    {
        "id": "ollama",
        "title": "本地大模型 Ollama",
        "desc": "Ask 与各类 App 使用的本地 LLM。改完需重启生效。",
        "fields": [
            {"section": "ollama", "key": "model", "type": "str", "restart": True,
             "label": "模型名称", "placeholder": "qwen3.5_4b_ov:v1",
             "help": "Ollama 中已安装的模型 id。用 `ollama list` 查看可用模型。"},
            {"section": "ollama", "key": "chat_timeout", "type": "int", "restart": True,
             "label": "请求超时 (秒)", "min": 30, "max": 1800,
             "help": "等待模型响应的上限。模型慢/首次加载久时可调大。"},
            {"section": "ollama", "key": "think", "type": "bool", "restart": False,
             "label": "思考模式 (thinking)",
             "help": "让模型先推理再回答/调用工具,质量更好;会增加延迟与耗时,慢硬件可关闭。"},
        ],
    },
    {
        "id": "retention",
        "title": "数据保留 Retention",
        "desc": "自动清理多久以前的数据，控制磁盘占用。",
        "fields": [
            {"section": "retention", "key": "frame_days", "type": "int", "restart": True,
             "label": "画面保留天数", "min": 1, "max": 3650,
             "help": "超过这个天数的截图/画面会被自动删除。"},
            {"section": "retention", "key": "audio_days", "type": "int", "restart": True,
             "label": "音频保留天数", "min": 1, "max": 3650,
             "help": "超过这个天数的音频转录会被自动删除。"},
            {"section": "retention", "key": "db_max_mb", "type": "int", "restart": True,
             "label": "数据库上限 (MB)", "min": 100, "max": 100000,
             "help": "数据库体积软上限，超出时触发更激进的清理。"},
        ],
    },
]


def _coerce_setting(field: dict[str, Any], raw: Any) -> Any:
    """Validate + coerce one incoming setting value per its schema type."""
    t = field["type"]
    if t == "bool":
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in ("1", "true", "yes", "on")
    if t == "int":
        try:
            v = int(raw)
        except (TypeError, ValueError):
            raise ValueError("must be an integer")
        if "min" in field and v < field["min"]:
            raise ValueError(f"must be ≥ {field['min']}")
        if "max" in field and v > field["max"]:
            raise ValueError(f"must be ≤ {field['max']}")
        return v
    if t == "float":
        try:
            v = float(raw)
        except (TypeError, ValueError):
            raise ValueError("must be a number")
        if "min" in field and v < field["min"]:
            raise ValueError(f"must be ≥ {field['min']}")
        if "max" in field and v > field["max"]:
            raise ValueError(f"must be ≤ {field['max']}")
        return v
    if t == "choice":
        s = str(raw).strip()
        if s not in field.get("choices", []):
            raise ValueError(f"must be one of {field.get('choices')}")
        return s
    if t == "csv":
        if isinstance(raw, list):
            items = raw
        else:
            items = str(raw).split(",")
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            code = str(item).strip().lower()
            if code and code not in seen:
                seen.add(code)
                out.append(code)
        return out
    # default: trimmed string
    return str(raw).strip()


def _hot_apply_settings(daemon: Any, cfg: Config, saved: list[str]) -> list[str]:
    """Apply the few settings the running daemon can adopt without a restart.

    Returns the dotted keys that were hot-applied. Everything else still
    persisted to disk and will take effect on the next start."""
    hot: list[str] = []
    if daemon is None:
        return hot
    if "audio.languages" in saved:
        transcriber = getattr(daemon, "transcriber", None)
        if transcriber is not None and hasattr(transcriber, "set_languages"):
            transcriber.set_languages(cfg.audio.languages)
            hot.append("audio.languages")
    translate_keys = {"audio.translate_enabled", "audio.translate_target_lang"}
    if translate_keys & set(saved) and hasattr(daemon, "set_translation"):
        daemon.set_translation(
            enabled=cfg.audio.translate_enabled,
            target_lang=cfg.audio.translate_target_lang,
        )
        hot.extend(sorted(translate_keys & set(saved)))
    return hot


def _all_hot(saved: list[str], hot: list[str]) -> bool:
    """True when every saved key was hot-applied (so no restart is needed)."""
    return bool(saved) and set(saved) == set(hot)


def _is_deskmate_app(app_name: str | None) -> bool:
    """Exclude DeskMate's own UI from search results."""
    if not app_name:
        return False
    low = app_name.lower()
    return "deskmate" in low


def _content_item_for_frame(row: dict[str, Any]) -> dict[str, Any]:
    """Shape an OCR content item."""
    return {
        "type": "OCR",
        "content": {
            "frame_id": row.get("frame_id") or row.get("id"),
            "text": row.get("ocr_text") or "",
            "timestamp": row["timestamp"],
            "file_path": row.get("snapshot_path") or "",
            "offset_index": row.get("offset_index", 0),
            "app_name": row.get("app_name") or "",
            "window_name": row.get("window_name") or "",
            "browser_url": row.get("browser_url"),
            "tags": [],
            "frame": None,
        },
    }


def _content_item_for_transcript(row: dict[str, Any]) -> dict[str, Any]:
    """Shape an audio content item."""
    return {
        "type": "Audio",
        "content": {
            "id": row.get("id") or row.get("transcription_id"),
            "audio_chunk_id": row.get("audio_chunk_id"),
            "transcription": row.get("transcription") or row.get("snippet") or "",
            "timestamp": row.get("timestamp"),
            "file_path": "",
            "offset_index": row.get("offset_index", 0),
            "device": row.get("device") or row.get("device_name") or "",
            "device_name": row.get("device") or "",
            "language": row.get("language"),
            "speaker_id": row.get("speaker_id"),
            "start_time": row.get("start_time"),
            "end_time": row.get("end_time"),
            "text_length": row.get("text_length"),
            "redacted_transcription": row.get("redacted_transcription"),
            "translation": row.get("translation"),
            "translation_lang": row.get("translation_lang"),
            "tags": [],
        },
    }


def _content_item_for_ui(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "UI",
        "content": {
            "id": row.get("id") or row.get("frame_id") or row.get("event_id"),
            "frame_id": row.get("frame_id") or row.get("id"),
            "text": row.get("text") or "",
            "timestamp": row.get("timestamp"),
            "app_name": row.get("app_name") or "",
            "window_name": row.get("window_name") or row.get("window_title") or "",
            "window_title": row.get("window_title") or row.get("window_name") or "",
            "browser_url": row.get("browser_url"),
            "file_path": row.get("file_path") or "",
            "data": _parse_json(row.get("data_json")),
            "element": _parse_json(row.get("element_json")),
        },
    }


def _content_item_for_input(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "Input",
        "content": {
            "id": row.get("id"),
            "timestamp": row.get("timestamp"),
            "event_type": row.get("event_type"),
            "app_name": row.get("app_name") or "",
            "window_title": row.get("window_title") or "",
            "browser_url": row.get("browser_url"),
            "text_content": row.get("text_content"),
            "frame_id": row.get("frame_id"),
            "data": _parse_json(row.get("data")),
            "element": _parse_json(row.get("element")),
        },
    }


def _content_item_for_element(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "Element",
        "content": {
            "id": row.get("element_id") or row.get("id"),
            "element_id": row.get("element_id"),
            "frame_id": row.get("frame_id"),
            "role": row.get("role") or "",
            "name": row.get("name") or "",
            "value": row.get("value") or "",
            "text": row.get("text") or "",
            "automation_id": row.get("automation_id"),
            "is_focused": row.get("is_focused"),
            "bounds": row.get("bounds"),
            "timestamp": row.get("timestamp"),
            "app_name": row.get("app_name") or "",
            "window_name": row.get("window_name") or "",
            "browser_url": row.get("browser_url"),
            "file_path": row.get("file_path") or "",
        },
    }


def _content_item_for_memory(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "Memory",
        "content": {
            "id": row.get("id"),
            "content": row.get("content") or "",
            "created_at": row.get("created_at"),
            "frame_id": row.get("frame_id"),
        },
    }


def _search_result_to_content_item(result: Any) -> dict[str, Any] | None:
    from ..db.search_engine import SearchResultKind

    payload = result.payload
    app_name = payload.get("app_name") or payload.get("window_title")
    if _is_deskmate_app(app_name if isinstance(app_name, str) else None):
        return None

    if result.kind == SearchResultKind.OCR:
        return {
            "type": "OCR",
            "content": {
                "frame_id": payload.get("frame_id"),
                "text": payload.get("text") or "",
                "timestamp": payload["timestamp"],
                "file_path": payload.get("file_path") or "",
                "offset_index": 0,
                "app_name": payload.get("app_name") or "",
                "window_name": payload.get("window_name") or "",
                "browser_url": payload.get("browser_url"),
                "tags": [],
                "frame": None,
            },
        }
    if result.kind == SearchResultKind.AUDIO:
        return _content_item_for_transcript(payload)
    if result.kind == SearchResultKind.UI:
        return _content_item_for_ui(payload)
    if result.kind == SearchResultKind.INPUT:
        return _content_item_for_input(payload)
    if result.kind == SearchResultKind.ELEMENT:
        return _content_item_for_element(payload)
    if result.kind == SearchResultKind.MEMORY:
        return _content_item_for_memory(payload)
    return None


def _dedupe_ocr_ui_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge duplicate OCR/UI hits at the same moment."""
    ocr_by_moment: dict[tuple[int, str], int] = {}
    for i, item in enumerate(items):
        if item.get("type") != "OCR":
            continue
        content = item["content"]
        ts = content.get("timestamp")
        ts_key = 0
        if ts:
            try:
                from datetime import datetime

                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                ts_key = int(dt.timestamp())
            except ValueError:
                pass
        app_key = content.get("app_name") or ""
        ocr_by_moment.setdefault((ts_key, app_key), i)

    ui_remove: list[int] = []
    text_upgrades: list[tuple[int, str]] = []
    for i, item in enumerate(items):
        if item.get("type") != "UI":
            continue
        content = item["content"]
        ts_key = 0
        if content.get("timestamp"):
            try:
                from datetime import datetime

                dt = datetime.fromisoformat(str(content["timestamp"]).replace("Z", "+00:00"))
                ts_key = int(dt.timestamp())
            except ValueError:
                pass
        app_key = content.get("app_name") or ""
        ocr_idx = ocr_by_moment.get((ts_key, app_key))
        if ocr_idx is None:
            continue
        ui_remove.append(i)
        ocr_text = items[ocr_idx]["content"].get("text") or ""
        ui_text = content.get("text") or ""
        if len(ui_text) > len(ocr_text):
            text_upgrades.append((ocr_idx, ui_text))

    for idx, text in text_upgrades:
        items[idx]["content"]["text"] = text
    for idx in sorted(ui_remove, reverse=True):
        items.pop(idx)
    return items


def _parse_json(value: Any) -> Any:
    if not value:
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def _is_readonly_sql(query: str) -> bool:
    stripped = query.lstrip().lower()
    if not stripped:
        return False
    return stripped.startswith(("select", "with", "explain"))


def _write_ffconcat(path: Path, frames: list[dict[str, Any]], fps: float) -> None:
    duration = 1.0 / max(fps, 0.1)
    lines: list[str] = []
    for frame in frames:
        image_path = str(Path(str(frame["snapshot_path"])).resolve()).replace("'", "'\\''")
        lines.append(f"file '{image_path}'")
        lines.append(f"duration {duration:.6f}")
    image_path = str(Path(str(frames[-1]["snapshot_path"])).resolve()).replace("'", "'\\''")
    lines.append(f"file '{image_path}'")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _oauth_success_html(data: dict[str, Any]) -> str:
    email = data.get("email") or data.get("instance") or "account"
    return f"""<!doctype html>
<html><head><meta charset=\"utf-8\"><title>Email connected</title></head>
<body><h1>Email connected</h1><p>{email} is ready for DeskMate.</p></body></html>"""


def _audio_pipeline_payload(cfg: Config, db: DatabaseManager, daemon: Any) -> dict[str, Any]:
    from ..audio.pipeline_status import build_audio_status  # noqa: PLC0415

    transcriber = getattr(daemon, "transcriber", None) if daemon else None
    capture_active = getattr(getattr(daemon, "audio", None), "capture_active", None) if daemon else None
    stats = db.health()
    return build_audio_status(
        cfg,
        transcriber=transcriber,
        capture_active=capture_active,
        transcript_count=int(stats.get("transcripts") or 0),
    )


def create_app(
    cfg: Config | None = None,
    db: DatabaseManager | None = None,
    daemon: Any = None,
) -> FastAPI:
    cfg = cfg or load_config()
    db = db or DatabaseManager()
    workflow = WorkflowClassifier()
    gmail = GmailConnection(cfg.gmail)
    outlook = OutlookConnection(cfg.outlook)

    app = FastAPI(title="deskmate", version="0.2.0")
    app.state.cfg = cfg
    app.state.db = db
    app.state.daemon = daemon
    app.mount("/ui/assets", StaticFiles(directory=static_dir()), name="ui-assets")
    started_at = time.time()

    # ─── health ───────────────────────────────────────────────────────────
    @app.get("/health")
    def health() -> dict[str, Any]:
        stats = db.health()
        feed = activity_default()
        params = feed.get_capture_params()
        audio_info = _audio_pipeline_payload(cfg, db, app.state.daemon)
        if not audio_info.get("enabled"):
            audio_status = "disabled"
        elif audio_info.get("transcription_ready"):
            audio_status = "ok" if stats.get("transcripts") else "ready, waiting for speech"
        else:
            audio_status = audio_info.get("error_code") or "transcription unavailable"
        return {
            "status": "ok",
            "status_code": 200,
            "frames": stats.get("frames", 0),
            "events": stats.get("events", 0),
            "transcripts": stats.get("transcripts", 0),
            "last_frame_timestamp": stats.get("last_frame_timestamp"),
            "last_audio_timestamp": stats.get("last_audio_timestamp"),
            "frame_status": "ok" if stats.get("frames") else "no frames yet",
            "audio_status": audio_status,
            "audio_hint": audio_info.get("hint"),
            "audio_error_code": audio_info.get("error_code"),
            "transcription_ready": audio_info.get("transcription_ready"),
            "meeting_status": "active" if db.active_meeting() else "idle",
            "message": "deskmate running",
            "verbose_instructions": audio_info.get("hint"),
            "device_status_details": audio_info,
            "monitors": [m.name for m in list_monitors()],
            "schema_version": db.schema_version(),
            "uptime_seconds": int(time.time() - started_at),
            "activity": {
                "idle_ms": feed.idle_ms(),
                "keyboard_idle_ms": feed.keyboard_idle_ms(),
                "is_typing": feed.is_typing(),
                "recommended_interval_ms": params.interval_ms,
                "skip_threshold": params.skip_threshold,
            },
        }

    @app.get("/health/doctor")
    def health_doctor() -> dict[str, Any]:
        """Run DeskMate self-diagnostics and return a structured report.

        Surfaces the environment issues users actually hit (Ollama backend +
        GenAI runtime version, active model, winrt, proxy hijacking localhost,
        DB/recording state). Each check is {name, status, message, fix}.
        """
        from . import doctor  # noqa: PLC0415

        return doctor.report(cfg, db)

    def _outlook_error(exc: OutlookError) -> JSONResponse:
        body: dict[str, Any] = {"error": str(exc)}
        if exc.upstream_status is not None:
            body["upstream_status"] = exc.upstream_status
        return JSONResponse(status_code=exc.status_code, content=body)

    def _gmail_error(exc: GmailError) -> JSONResponse:
        body: dict[str, Any] = {"error": str(exc)}
        if exc.upstream_status is not None:
            body["upstream_status"] = exc.upstream_status
        return JSONResponse(status_code=exc.status_code, content=body)

    # ─── Gmail OAuth / Gmail API mail ────────────────────────────────────
    @app.get("/connections/gmail/auth-url")
    def gmail_auth_url(instance: str | None = None) -> JSONResponse:
        try:
            return JSONResponse(gmail.auth_url(instance))
        except GmailError as exc:
            return _gmail_error(exc)

    @app.get("/connections/gmail/connect")
    def gmail_connect(instance: str | None = None) -> Response:
        try:
            return RedirectResponse(gmail.auth_url(instance)["authorization_url"], status_code=307)
        except GmailError as exc:
            return _gmail_error(exc)

    @app.get("/connections/gmail/oauth/callback")
    async def gmail_oauth_callback(code: str | None = None, state: str | None = None, error: str | None = None) -> Response:
        if error:
            return JSONResponse(status_code=400, content={"error": error})
        if not code or not state:
            return JSONResponse(status_code=400, content={"error": "missing OAuth code or state"})
        try:
            data = await gmail.complete_oauth(code, state)
            return HTMLResponse(_oauth_success_html(data), status_code=200)
        except GmailError as exc:
            return _gmail_error(exc)

    @app.get("/connections/gmail/status")
    async def gmail_status(instance: str | None = None) -> JSONResponse:
        try:
            return JSONResponse(await gmail.status(instance))
        except GmailError as exc:
            return _gmail_error(exc)

    @app.get("/connections/gmail/instances")
    def gmail_instances() -> dict[str, Any]:
        return {"data": gmail.list_instances()}

    @app.delete("/connections/gmail/instances/{instance}")
    def gmail_disconnect(instance: str) -> JSONResponse:
        try:
            return JSONResponse({"success": gmail.disconnect(instance)})
        except GmailError as exc:
            return _gmail_error(exc)

    @app.get("/connections/gmail/messages")
    async def gmail_messages(
        q: str | None = None,
        maxResults: int = 20,
        pageToken: str | None = None,
        instance: str | None = None,
    ) -> JSONResponse:
        try:
            data = await gmail.list_messages(query=q, max_results=maxResults, page_token=pageToken, instance=instance)
            return JSONResponse({"data": data})
        except GmailError as exc:
            return _gmail_error(exc)

    @app.get("/connections/gmail/messages/{message_id:path}")
    async def gmail_message(message_id: str, instance: str | None = None) -> JSONResponse:
        try:
            return JSONResponse({"data": await gmail.get_message(message_id, instance=instance)})
        except GmailError as exc:
            return _gmail_error(exc)

    @app.post("/connections/gmail/send")
    async def gmail_send(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            body = {}
        try:
            return JSONResponse({"data": await gmail.send_message(body)})
        except GmailError as exc:
            return _gmail_error(exc)

    # ─── Outlook OAuth / Microsoft Graph mail ────────────────────────────
    @app.get("/connections/outlook/auth-url")
    def outlook_auth_url(instance: str | None = None) -> JSONResponse:
        try:
            return JSONResponse(outlook.auth_url(instance))
        except OutlookError as exc:
            return _outlook_error(exc)

    @app.get("/connections/outlook/connect")
    def outlook_connect(instance: str | None = None) -> Response:
        try:
            return RedirectResponse(outlook.auth_url(instance)["authorization_url"], status_code=307)
        except OutlookError as exc:
            return _outlook_error(exc)

    @app.get("/connections/outlook/oauth/callback")
    async def outlook_oauth_callback(code: str | None = None, state: str | None = None, error: str | None = None) -> Response:
        if error:
            return JSONResponse(status_code=400, content={"error": error})
        if not code or not state:
            return JSONResponse(status_code=400, content={"error": "missing OAuth code or state"})
        try:
            data = await outlook.complete_oauth(code, state)
            return HTMLResponse(_oauth_success_html(data), status_code=200)
        except OutlookError as exc:
            return _outlook_error(exc)

    @app.get("/connections/outlook/status")
    async def outlook_status(instance: str | None = None) -> JSONResponse:
        try:
            return JSONResponse(await outlook.status(instance))
        except OutlookError as exc:
            return _outlook_error(exc)

    @app.get("/connections/outlook/instances")
    def outlook_instances() -> dict[str, Any]:
        return {"data": outlook.list_instances()}

    @app.delete("/connections/outlook/instances/{instance}")
    def outlook_disconnect(instance: str) -> JSONResponse:
        try:
            return JSONResponse({"success": outlook.disconnect(instance)})
        except OutlookError as exc:
            return _outlook_error(exc)

    @app.get("/connections/outlook/messages")
    async def outlook_messages(
        q: str | None = None,
        top: int = 20,
        skip: int | None = None,
        instance: str | None = None,
    ) -> JSONResponse:
        try:
            data = await outlook.list_messages(query=q, top=top, skip=skip, instance=instance)
            return JSONResponse({"data": data})
        except OutlookError as exc:
            return _outlook_error(exc)

    @app.get("/connections/outlook/messages/{message_id:path}")
    async def outlook_message(message_id: str, instance: str | None = None) -> JSONResponse:
        try:
            return JSONResponse({"data": await outlook.get_message(message_id, instance=instance)})
        except OutlookError as exc:
            return _outlook_error(exc)

    @app.post("/connections/outlook/send")
    async def outlook_send(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            body = {}
        try:
            return JSONResponse({"data": await outlook.send_message(body)})
        except OutlookError as exc:
            return _outlook_error(exc)

    @app.post("/connections/gmail/disconnect")
    def gmail_disconnect_post(instance: str) -> JSONResponse:
        try:
            return JSONResponse({"success": gmail.disconnect(instance)})
        except GmailError as exc:
            return _gmail_error(exc)

    @app.post("/connections/outlook/disconnect")
    def outlook_disconnect_post(instance: str) -> JSONResponse:
        try:
            return JSONResponse({"success": outlook.disconnect(instance)})
        except OutlookError as exc:
            return _outlook_error(exc)

    # ─── search (the big one) ─────────────────────────────────────────────
    @app.get("/search")
    def search(
        q: str | None = Query(default=None, description="FTS5 MATCH query"),
        content_type: str = Query(default="all"),
        app_name: str | None = None,
        window_name: str | None = None,
        frame_name: str | None = None,
        browser_url: str | None = None,
        focused: bool | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        speaker_ids: str | None = None,  # comma-separated
        role: str | None = Query(default=None, description="Filter elements by UIA role"),
        include_frames: bool = False,
        semantic: bool = Query(
            default=False,
            description="Blend semantic (embedding) results with keyword search",
        ),
        offset: int = 0,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Search content (FTS sanitize + merge)."""
        speaker_filter = [int(s) for s in (speaker_ids or "").split(",") if s.strip().isdigit()] or None
        use_semantic = semantic and cfg.search.semantic_enabled
        if use_semantic:
            results = db.hybrid_search(
                q,
                content_type,
                model_name=cfg.search.embedding_model,
                limit=limit,
                offset=offset,
                start_time=start_time,
                end_time=end_time,
                app_name=app_name,
                window_name=window_name,
                frame_name=frame_name,
                browser_url=browser_url,
                focused=focused,
                min_length=min_length,
                max_length=max_length,
                speaker_ids=speaker_filter,
                rrf_k=cfg.search.rrf_k,
                candidate_pool=cfg.search.candidate_pool,
            )
        else:
            results = db.search(
                q,
                content_type,
                limit=limit,
                offset=offset,
                start_time=start_time,
                end_time=end_time,
                app_name=app_name,
                window_name=window_name,
                frame_name=frame_name,
                browser_url=browser_url,
                focused=focused,
                min_length=min_length,
                max_length=max_length,
                speaker_ids=speaker_filter,
                role=role,
            )
        items: list[dict[str, Any]] = []
        for result in results:
            item = _search_result_to_content_item(result)
            if item is not None:
                items.append(item)
        items = _dedupe_ocr_ui_items(items)
        return {
            "data": items,
            "pagination": {"limit": limit, "offset": offset, "total": len(items)},
        }

    # ─── frames ───────────────────────────────────────────────────────────
    @app.get("/frames")
    def frames(limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        return db.recent_frames(limit=limit, offset=offset)

    @app.post("/frames/export")
    async def export_frames(request: Request) -> dict[str, Any]:
        body = await request.json()
        start_time = str(body.get("start_time") or body.get("startTime") or "")
        end_time = str(body.get("end_time") or body.get("endTime") or "")
        fps = float(body.get("fps") or 1.0)
        limit = int(body.get("limit") or 1000)
        if not start_time or not end_time:
            raise HTTPException(status_code=400, detail="start_time and end_time are required")

        with db._lock:  # noqa: SLF001
            frame_rows = db._conn.execute(  # noqa: SLF001
                """SELECT id, timestamp, app_name, window_name, browser_url, snapshot_path
                     FROM frames
                    WHERE timestamp >= ?
                      AND timestamp <= ?
                      AND snapshot_path IS NOT NULL
                    ORDER BY timestamp ASC
                    LIMIT ?""",
                (start_time, end_time, limit),
            ).fetchall()
        frames_for_export = [dict(row) for row in frame_rows if row["snapshot_path"] and Path(row["snapshot_path"]).exists()]
        if not frames_for_export:
            raise HTTPException(status_code=404, detail="no snapshot frames found in time range")

        output_dir = paths.root() / "exports" / datetime.now().strftime("%Y%m%dT%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "manifest.json"
        concat_path = output_dir / "frames.ffconcat"
        video_path = output_dir / "screen-activity.mp4"
        manifest_path.write_text(json.dumps({
            "start_time": start_time,
            "end_time": end_time,
            "fps": fps,
            "frames": frames_for_export,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_ffconcat(concat_path, frames_for_export, fps)

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return {
                "success": False,
                "reason": "ffmpeg_not_found",
                "frame_count": len(frames_for_export),
                "manifest_path": str(manifest_path),
                "concat_path": str(concat_path),
            }

        result = subprocess.run(  # noqa: S603
            [
                ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_path),
                "-r", str(fps), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-pix_fmt", "yuv420p", str(video_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            log_path = video_path.with_suffix(".ffmpeg.log")
            log_path.write_text(result.stderr, encoding="utf-8")
            raise HTTPException(status_code=500, detail={"error": "ffmpeg failed", "log_path": str(log_path)})
        return {
            "success": True,
            "file_path": str(video_path),
            "frame_count": len(frames_for_export),
            "manifest_path": str(manifest_path),
        }

    @app.get("/frames/next-valid")
    def next_valid_frame(start_frame_id: int | None = None) -> dict[str, Any]:
        rows = db.recent_frames(limit=200)
        candidates = [r for r in rows if start_frame_id is None or int(r["id"]) >= start_frame_id]
        for row in candidates or rows:
            if row.get("snapshot_path") or row.get("video_chunk_id"):
                return row
        raise HTTPException(status_code=404, detail="no valid frame found")

    @app.get("/frames/{frame_id}")
    def frame_metadata(frame_id: int) -> dict[str, Any]:
        row = db.frame_by_id(frame_id)
        if not row:
            raise HTTPException(status_code=404, detail="frame not found")
        return row

    @app.get("/frames/{frame_id}/metadata")
    def frame_metadata_alias(frame_id: int) -> dict[str, Any]:
        return frame_metadata(frame_id)

    @app.get("/frames/{frame_id}/text")
    @app.get("/frames/{frame_id}/ocr")
    def frame_text(frame_id: int) -> dict[str, Any]:
        row = db.frame_by_id(frame_id)
        if not row:
            raise HTTPException(status_code=404, detail="frame not found")
        return {
            "frame_id": frame_id,
            "ocr_text": row.get("ocr_text") or "",
            "ocr_text_json": _parse_json(row.get("ocr_text_json")) or [],
            "accessibility_text": row.get("accessibility_text") or "",
        }

    @app.get("/frames/{frame_id}/context")
    def frame_context(frame_id: int) -> dict[str, Any]:
        row = db.frame_by_id(frame_id)
        if not row:
            raise HTTPException(status_code=404, detail="frame not found")
        return {
            "frame": row,
            "tags": db.frame_tags(frame_id),
            "nearby_events": [
                e for e in db.recent_events(limit=50)
                if e.get("app_name") == row.get("app_name")
            ][:10],
        }

    @app.get("/frames/{frame_id}/image")
    def frame_image(frame_id: int, redact_pii: bool = False):  # noqa: ANN201
        row = db.frame_by_id(frame_id)
        if not row or not row.get("snapshot_path"):
            raise HTTPException(status_code=404, detail="frame has no snapshot")
        p = Path(row["snapshot_path"])
        if not p.exists():
            raise HTTPException(status_code=410, detail="snapshot file gone")
        if redact_pii:
            image_width = int(row.get("width") or 0)
            image_height = int(row.get("height") or 0)
            if image_width <= 0 or image_height <= 0:
                with Image.open(p) as img:
                    image_width, image_height = img.size
            regions = regions_from_ocr(
                ocr_text=row.get("ocr_text") or "",
                ocr_text_json=row.get("ocr_text_json"),
                image_width=image_width,
                image_height=image_height,
                cfg=cfg,
            )
            data = redact_image_bytes(p, regions, quality=cfg.capture.screenshot_jpeg_quality)
            if regions:
                db.mark_frame_image_redacted(frame_id)
            return Response(
                content=data,
                media_type="image/jpeg",
                headers={"x-redact": "pixel", "x-redact-regions": str(len(regions))},
            )
        return FileResponse(p, media_type="image/jpeg")

    # ─── video chunks ─────────────────────────────────────────────────────
    @app.get("/video-chunks")
    @app.get("/video_chunks")
    def video_chunks(limit: int = 50) -> list[dict[str, Any]]:
        return db.list_video_chunks(limit=limit)

    @app.get("/video-chunks/path")
    @app.get("/video_chunks/path")
    def next_video_chunk_path(device_name: str = "screen", extension: str = "mp4") -> dict[str, str]:
        return {"file_path": str(video_chunk_path(device_name=device_name, extension=extension))}

    @app.post("/video-chunks/register")
    @app.post("/video_chunks/register")
    async def register_video_chunk(request: Request) -> dict[str, Any]:
        body = await request.json()
        file_path = str(body.get("file_path") or "").strip()
        if not file_path:
            file_path = str(video_chunk_path(
                device_name=str(body.get("device_name") or "screen"),
                extension=str(body.get("extension") or "mp4"),
            ))
        chunk_id = db.insert_video_chunk(
            file_path=file_path,
            device_name=str(body.get("device_name") or ""),
            fps=float(body.get("fps") or 1.0),
        )
        return {"id": chunk_id, "file_path": file_path}

    @app.get("/video-chunks/{chunk_id}")
    @app.get("/video_chunks/{chunk_id}")
    def video_chunk_detail(chunk_id: int) -> dict[str, Any]:
        row = db.video_chunk_by_id(chunk_id)
        if not row:
            raise HTTPException(status_code=404, detail="video chunk not found")
        return row

    # ─── audio ────────────────────────────────────────────────────────────
    @app.get("/audio/list")
    def audio_list(limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        return db.recent_transcripts(limit=limit, offset=offset)

    @app.get("/audio/device/status")
    def audio_device_status() -> dict[str, Any]:
        payload = _audio_pipeline_payload(cfg, db, app.state.daemon)
        payload.update({
            "microphone": cfg.audio.microphone,
            "loopback": cfg.audio.loopback,
            "recent_transcripts": len(db.recent_transcripts(limit=10)),
        })
        return payload

    @app.post("/audio/start")
    @app.post("/audio/stop")
    def audio_runtime_control() -> JSONResponse:
        return JSONResponse(status_code=501, content={"error": "audio runtime control requires daemon restart"})

    # ─── ui events ────────────────────────────────────────────────────────
    @app.get("/events/recent")
    def recent_events(limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        return db.recent_events(limit=limit, offset=offset)

    @app.get("/events/stream")
    async def stream_events(request: Request):  # noqa: ANN201
        def _next_or_none(stream: Any) -> Any:
            """Blocking queue wait; returns ``None`` on timeout (no event)."""
            try:
                return next(stream)
            except StopIteration:
                return None

        async def _gen() -> AsyncIterator[str]:
            stream = bus.stream(timeout=0.5)
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    # Run the blocking Queue.get off the event loop so the
                    # SSE wait never stalls other HTTP requests / the UI.
                    evt = await asyncio.to_thread(_next_or_none, stream)
                    if evt is None:
                        continue
                    yield "data: " + json.dumps({
                        "type": evt.type.value,
                        "timestamp": evt.timestamp,
                        "data": evt.data,
                    }, ensure_ascii=False) + "\n\n"
            finally:
                stream.close()

        return StreamingResponse(_gen(), media_type="text/event-stream")

    # ─── speakers ─────────────────────────────────────────────────────────
    @app.get("/speakers/search")
    def speakers_search(q: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        sql = """SELECT s.id, s.name, s.sample_count, s.metadata, s.created_at, s.updated_at,
                        COUNT(e.id) AS embedding_count
                   FROM speakers s
                   LEFT JOIN speaker_embeddings e ON e.speaker_id = s.id"""
        args: tuple[Any, ...] = ()
        if q:
            sql += " WHERE s.name LIKE ?"
            args = (f"%{q}%",)
        sql += " GROUP BY s.id ORDER BY s.id LIMIT ?"
        with db._lock:  # noqa: SLF001
            return db._conn.execute(sql, (*args, limit)).fetchall()  # noqa: SLF001

    @app.post("/speakers/{speaker_id}/name")
    async def rename_speaker(speaker_id: int, request: Request) -> dict[str, Any]:
        body = await request.json()
        name = body.get("name", "")
        with db._lock:  # noqa: SLF001
            db._conn.execute(  # noqa: SLF001
                "UPDATE speakers SET name = ?, updated_at = datetime('now') WHERE id = ?",
                (name, speaker_id),
            )
        return {"ok": True}

    @app.post("/speakers/update")
    async def update_speaker(request: Request) -> dict[str, Any]:
        body = await request.json()
        speaker_id = int(body.get("id") or body.get("speaker_id") or 0)
        if speaker_id <= 0:
            raise HTTPException(status_code=400, detail="speaker id required")
        name = str(body.get("name") or "")
        metadata = body.get("metadata")
        with db._lock:  # noqa: SLF001
            if metadata is not None:
                db._conn.execute(  # noqa: SLF001
                    "UPDATE speakers SET name = ?, metadata = ?, updated_at = datetime('now') WHERE id = ?",
                    (name, json.dumps(metadata, ensure_ascii=False), speaker_id),
                )
            else:
                db._conn.execute(  # noqa: SLF001
                    "UPDATE speakers SET name = ?, updated_at = datetime('now') WHERE id = ?",
                    (name, speaker_id),
                )
        return {"ok": True}

    @app.post("/speakers/delete")
    async def delete_speaker(request: Request) -> dict[str, Any]:
        body = await request.json()
        speaker_id = int(body.get("id") or body.get("speaker_id") or 0)
        with db._lock:  # noqa: SLF001
            db._conn.execute("DELETE FROM speakers WHERE id = ?", (speaker_id,))  # noqa: SLF001
        return {"ok": True}

    # ─── meetings ─────────────────────────────────────────────────────────
    @app.get("/meetings")
    def meetings(limit: int = 50) -> list[dict[str, Any]]:
        return db.list_meetings(limit=limit)

    @app.get("/meetings/status")
    def meeting_status() -> dict[str, Any]:
        active = db.active_meeting()
        return {"in_meeting": active is not None, "meeting": active}

    @app.get("/meetings/{meeting_id}")
    def meeting_detail(meeting_id: int) -> dict[str, Any]:
        meeting = db.meeting_by_id(meeting_id)
        if not meeting:
            raise HTTPException(status_code=404, detail="meeting not found")
        return {
            "meeting": meeting,
            "segments": db.list_meeting_segments(meeting_id),
        }

    @app.get("/meetings/{meeting_id}/transcript")
    def meeting_transcript(meeting_id: int) -> dict[str, Any]:
        meeting = db.meeting_by_id(meeting_id)
        if not meeting:
            raise HTTPException(status_code=404, detail="meeting not found")
        segments = db.list_meeting_segments(meeting_id)
        return {
            "meeting_id": meeting_id,
            "text": "\n".join(s.get("text") or "" for s in segments),
            "segments": segments,
        }

    @app.post("/meetings/start")
    async def start_meeting(request: Request) -> dict[str, Any]:
        body = await request.json()
        meeting_id = db.insert_meeting(
            name=str(body.get("name") or "Manual meeting"),
            note=str(body.get("note") or ""),
            metadata={"detection_source": "manual", **(body.get("metadata") or {})},
        )
        return {"id": meeting_id}

    @app.post("/meetings/stop")
    async def stop_meeting(request: Request) -> dict[str, Any]:
        body = await request.json()
        meeting_id = body.get("id") or body.get("meeting_id")
        if meeting_id is None:
            active = db.active_meeting()
            meeting_id = active.get("id") if active else None
        if meeting_id is None:
            raise HTTPException(status_code=404, detail="no active meeting")
        db.end_meeting(int(meeting_id))
        return {"ok": True, "id": int(meeting_id)}

    @app.patch("/meetings/{meeting_id}")
    async def patch_meeting(meeting_id: int, request: Request) -> dict[str, Any]:
        meeting = db.meeting_by_id(meeting_id)
        if not meeting:
            raise HTTPException(status_code=404, detail="meeting not found")
        body = await request.json()
        name = body.get("name")
        if name is None:
            name = body.get("title")
        note = body.get("note")
        updated = db.update_meeting(
            meeting_id,
            name=str(name) if name is not None else None,
            note=str(note) if note is not None else None,
        )
        return {"ok": updated, "meeting": db.meeting_by_id(meeting_id)}

    # ─── todos ────────────────────────────────────────────────────────────
    @app.get("/todos")
    def list_todos(
        status: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 200,
    ) -> dict[str, Any]:
        rows = db.list_todos(status=status, since=since, until=until, limit=limit)
        open_count = sum(1 for r in rows if r.get("status") != "done")
        return {"data": rows, "total": len(rows), "open": open_count}

    @app.post("/todos")
    async def create_todos(request: Request) -> dict[str, Any]:
        body = await request.json()
        if isinstance(body, dict) and isinstance(body.get("todos"), list):
            items = body["todos"]
        elif isinstance(body, list):
            items = body
        elif isinstance(body, dict) and body.get("text"):
            items = [body]
        else:
            raise HTTPException(status_code=400, detail="provide 'text' or a 'todos' list")

        ids: list[int] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            text = str(it.get("text") or "").strip()
            if not text:
                continue
            raw_mid = it.get("meeting_id")
            try:
                meeting_id = int(raw_mid) if raw_mid not in (None, "") else None
            except (TypeError, ValueError):
                meeting_id = None
            todo_id = db.upsert_todo(
                text=text,
                source=str(it.get("source") or ""),
                source_ref=str(it.get("source_ref") or ""),
                source_detail=str(it.get("source_detail") or ""),
                meeting_id=meeting_id,
                priority=str(it.get("priority") or ""),
                due=str(it.get("due") or ""),
                origin_app=str(it.get("origin_app") or ""),
                evidence_start=str(it.get("evidence_start") or ""),
                evidence_end=str(it.get("evidence_end") or ""),
                dedup_key=str(it.get("dedup_key") or ""),
            )
            ids.append(todo_id)
        return {"ok": True, "ids": ids, "count": len(ids)}

    @app.patch("/todos/{todo_id}")
    async def update_todo(todo_id: int, request: Request) -> dict[str, Any]:
        if not db.todo_by_id(todo_id):
            raise HTTPException(status_code=404, detail="todo not found")
        body = await request.json()
        status = str(body.get("status") or "").strip().lower()
        if status not in ("open", "done"):
            raise HTTPException(status_code=400, detail="status must be 'open' or 'done'")
        db.set_todo_status(todo_id, status)
        return {"ok": True, "todo": db.todo_by_id(todo_id)}

    @app.delete("/todos/{todo_id}")
    def remove_todo(todo_id: int) -> dict[str, Any]:
        if not db.delete_todo(todo_id):
            raise HTTPException(status_code=404, detail="todo not found")
        return {"ok": True}

    # ─── monitors ─────────────────────────────────────────────────────────
    @app.get("/monitors")
    def monitors() -> list[dict[str, Any]]:
        return [
            {
                "id": m.id,
                "stable_id": str(m.id),
                "name": m.name,
                "width": m.width,
                "height": m.height,
                "is_default": (m.id == 1),
            }
            for m in list_monitors()
        ]

    # ─── capture trigger ──────────────────────────────────────────────────
    @app.post("/capture")
    def capture_now() -> dict[str, Any]:
        ids = run_paired_capture(cfg, db, trigger="manual")
        return {"frame_ids": ids}

    # ─── workflow classifier ──────────────────────────────────────────────
    @app.post("/workflow/classify")
    async def workflow_classify(request: Request) -> dict[str, Any]:
        body = await request.json()
        wf = workflow.classify(body.get("app_name", ""), body.get("window_title", ""))
        return {"workflow": wf}

    # ─── activity feed ────────────────────────────────────────────────────
    @app.get("/activity/params")
    def activity_params() -> dict[str, Any]:
        feed = activity_default()
        params = feed.get_capture_params()
        return {
            "idle_ms": feed.idle_ms(),
            "keyboard_idle_ms": feed.keyboard_idle_ms(),
            "is_typing": feed.is_typing(),
            "is_keyboard_burst": feed.is_keyboard_burst(),
            "interval_ms": params.interval_ms,
            "skip_threshold": params.skip_threshold,
        }

    # ─── config ───────────────────────────────────────────────────────────
    @app.get("/config")
    def get_config() -> dict[str, Any]:
        data = cfg.model_dump()
        gmail_cfg = data.get("gmail")
        if isinstance(gmail_cfg, dict) and gmail_cfg.get("client_secret"):
            gmail_cfg["client_secret"] = "********"
        return data

    @app.post("/config/audio/languages")
    async def set_audio_languages(request: Request) -> dict[str, Any]:
        """Hot-update the Whisper transcription language list.

        `languages` is read per audio clip, not at model load, so this takes
        effect on the next chunk without reloading the (possibly NPU-compiled)
        model. The value is also persisted to config.toml so it survives a
        restart. Hot-update only reaches a transcriber when the daemon runs in
        this process; split API/daemon deployments still get persistence.
        """
        from ..config import set_audio_languages as persist_languages  # noqa: PLC0415

        body = await request.json()
        raw = body.get("languages", [])
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            raise HTTPException(status_code=400, detail="languages must be a list of codes")
        # Normalize: trim, drop blanks, lowercase ISO-639-1-ish codes, dedupe.
        seen: set[str] = set()
        languages: list[str] = []
        for item in raw:
            code = str(item).strip().lower()
            if code and code not in seen:
                seen.add(code)
                languages.append(code)

        # Persist first (works regardless of daemon presence), then hot-apply.
        persist_languages(languages)
        cfg.audio.languages = languages
        applied = False
        transcriber = getattr(app.state.daemon, "transcriber", None)
        if transcriber is not None and hasattr(transcriber, "set_languages"):
            transcriber.set_languages(languages)
            applied = True
        return {"languages": languages, "hot_applied": applied}

    @app.get("/config/audio/translate")
    def get_audio_translate() -> dict[str, Any]:
        """Return the current live-translation settings for the UI controls."""
        a = cfg.audio
        return {
            "translate_enabled": a.translate_enabled,
            "translate_target_lang": a.translate_target_lang,
            "translate_latency_mode": a.translate_latency_mode,
        }

    @app.post("/config/audio/translate")
    async def set_audio_translate(request: Request) -> dict[str, Any]:
        """Hot-toggle / reconfigure live translation from the UI.

        Persists each provided key to config.toml and, when the daemon runs in
        this process, applies it live (building/dropping the translator and
        starting the worker) so no restart is needed. ``enabled`` (re)takes
        effect on the next utterance; ``target_lang`` also drives the next
        translation; ``latency_mode`` changes the endpoint pause threshold for
        new audio chunks. Split API/daemon deployments still get persistence.
        """
        from ..config import set_audio_value  # noqa: PLC0415

        body = await request.json()
        enabled = body.get("enabled")
        target_lang = body.get("target_lang")
        latency_mode = body.get("latency_mode")

        if enabled is not None:
            enabled = bool(enabled)
            set_audio_value("translate_enabled", enabled)
            cfg.audio.translate_enabled = enabled
        if target_lang is not None:
            target_lang = str(target_lang).strip().lower()
            if not target_lang:
                raise HTTPException(status_code=400, detail="target_lang must be non-empty")
            set_audio_value("translate_target_lang", target_lang)
            cfg.audio.translate_target_lang = target_lang
        if latency_mode is not None:
            latency_mode = str(latency_mode).strip().lower()
            if latency_mode not in {"fast", "balanced", "quality"}:
                raise HTTPException(status_code=400, detail="latency_mode must be fast|balanced|quality")
            set_audio_value("translate_latency_mode", latency_mode)
            cfg.audio.translate_latency_mode = latency_mode

        applied = False
        daemon = app.state.daemon
        if daemon is not None and hasattr(daemon, "set_translation"):
            daemon.set_translation(
                enabled=enabled,
                target_lang=target_lang,
                latency_mode=latency_mode,
            )
            applied = True
        return {
            "translate_enabled": cfg.audio.translate_enabled,
            "translate_target_lang": cfg.audio.translate_target_lang,
            "translate_latency_mode": cfg.audio.translate_latency_mode,
            "hot_applied": applied,
        }

    # ─── user-friendly settings (schema-driven, UI-editable) ──────────────
    @app.get("/config/settings")
    def get_settings() -> dict[str, Any]:
        """Return the editable-settings schema plus each field's current value.

        The schema is the single source of truth shared with the Settings UI:
        it groups fields, carries human labels/help/choices, and flags whether a
        change hot-applies or needs a restart. The UI renders straight from this
        so backend and frontend can never drift.
        """
        groups = []
        for group in SETTINGS_SCHEMA:
            fields = []
            for f in group["fields"]:
                section = cfg.__getattribute__(f["section"])
                fields.append({**f, "value": getattr(section, f["key"], None)})
            groups.append({**group, "fields": fields})
        return {"groups": groups}

    @app.post("/config/settings")
    async def update_settings(request: Request) -> dict[str, Any]:
        """Persist a batch of settings, hot-applying what we can.

        Body: ``{"values": {"section.key": value, ...}}``. Each value is
        validated against the schema, written to config.toml (comments
        preserved), and applied to the live config object. We report which keys
        took effect immediately vs. which need a restart so the UI can prompt.
        """
        from ..config import set_config_value  # noqa: PLC0415

        body = await request.json()
        values = body.get("values") or {}
        if not isinstance(values, dict):
            raise HTTPException(status_code=400, detail="values must be an object")

        index = {f"{f['section']}.{f['key']}": f
                 for g in SETTINGS_SCHEMA for f in g["fields"]}
        saved: list[str] = []
        needs_restart = False
        errors: dict[str, str] = {}

        for dotted, raw in values.items():
            field = index.get(dotted)
            if field is None:
                errors[dotted] = "unknown setting"
                continue
            try:
                coerced = _coerce_setting(field, raw)
            except ValueError as exc:
                errors[dotted] = str(exc)
                continue
            set_config_value(field["section"], field["key"], coerced)
            setattr(getattr(cfg, field["section"]), field["key"], coerced)
            saved.append(dotted)
            if field.get("restart", True):
                needs_restart = True

        # Hot-apply the handful of fields the running daemon can adopt live.
        hot = _hot_apply_settings(app.state.daemon, cfg, saved)
        return {
            "saved": saved,
            "errors": errors,
            "needs_restart": needs_restart and not _all_hot(saved, hot),
            "hot_applied": hot,
        }

    @app.post("/restart")
    def restart_server() -> JSONResponse:
        """Ask the host process to restart so config changes take effect.

        We can't reliably re-exec uvicorn in-process on Windows (NPU/COM state,
        port reuse), so this signals the launcher: it writes a restart-request
        marker and schedules a clean exit. A supervising script (or the user's
        ``deskmate ui`` shell) is expected to relaunch. If no supervisor exists,
        the UI tells the user to start DeskMate again manually.
        """
        import os  # noqa: PLC0415
        import threading  # noqa: PLC0415

        try:
            paths.restart_marker_path().write_text(
                datetime.now().astimezone().isoformat(), encoding="utf-8"
            )
        except Exception:  # noqa: BLE001
            pass

        def _exit() -> None:
            time.sleep(0.6)  # let the HTTP response flush first
            # Best-effort graceful daemon shutdown (flush DB, join threads) before
            # the hard exit; WAL-mode SQLite also recovers if this is skipped.
            daemon = app.state.daemon
            if daemon is not None and hasattr(daemon, "stop"):
                try:
                    daemon.stop()
                except Exception:  # noqa: BLE001
                    pass
            os._exit(0)

        threading.Thread(target=_exit, name="restart-exit", daemon=True).start()
        return JSONResponse({"restarting": True})

    # ─── tags / memories ──────────────────────────────────────────────────
    @app.post("/tags/vision/batch")
    async def tags_batch(request: Request) -> dict[str, Any]:
        body = await request.json()
        ids = body.get("frame_ids") or body.get("ids") or []
        frame_ids = [int(x) for x in ids if str(x).isdigit()]
        return {"data": db.tag_batch(frame_ids)}

    @app.post("/tags/{content_type}/{item_id}")
    async def add_tags(content_type: str, item_id: int, request: Request) -> dict[str, Any]:
        if content_type not in {"vision", "frame", "frames"}:
            raise HTTPException(status_code=400, detail="only frame tags are supported")
        body = await request.json()
        tags = body.get("tags") or body.get("tag") or []
        if isinstance(tags, str):
            tags = [tags]
        return {"tags": db.set_frame_tags(item_id, [str(t) for t in tags])}

    @app.delete("/tags/{content_type}/{item_id}")
    async def remove_tags(content_type: str, item_id: int, request: Request) -> dict[str, Any]:
        if content_type not in {"vision", "frame", "frames"}:
            raise HTTPException(status_code=400, detail="only frame tags are supported")
        body = await request.json()
        tags = body.get("tags") or body.get("tag") or []
        if isinstance(tags, str):
            tags = [tags]
        return {"tags": db.remove_frame_tags(item_id, [str(t) for t in tags])}

    @app.get("/memories")
    def list_memories(limit: int = 50) -> list[dict[str, Any]]:
        return db.list_memories(limit=limit)

    @app.get("/memories/tags")
    def memory_tags() -> dict[str, Any]:
        return {"data": []}

    @app.post("/memories")
    async def create_memory(request: Request) -> dict[str, Any]:
        body = await request.json()
        memory_id = db.create_memory(
            str(body.get("content") or ""),
            frame_id=body.get("frame_id"),
            sync_id=body.get("sync_id"),
        )
        return {"id": memory_id}

    @app.get("/memories/{memory_id}")
    def get_memory(memory_id: int) -> dict[str, Any]:
        row = db.memory_by_id(memory_id)
        if not row:
            raise HTTPException(status_code=404, detail="memory not found")
        return row

    @app.put("/memories/{memory_id}")
    async def update_memory(memory_id: int, request: Request) -> dict[str, Any]:
        body = await request.json()
        db.update_memory(memory_id, str(body.get("content") or ""))
        return {"ok": True}

    @app.delete("/memories/{memory_id}")
    def delete_memory(memory_id: int) -> dict[str, Any]:
        db.delete_memory(memory_id)
        return {"ok": True}

    # ─── pipes ─────────────────────────────────────────────────────────────
    @app.get("/pipes")
    def list_pipes(include_executions: bool = False) -> dict[str, Any]:
        pipes = load_pipes(paths.pipes_dir())
        rows: list[dict[str, Any]] = []
        for pipe in pipes:
            item: dict[str, Any] = {
                "name": pipe.frontmatter.name,
                "description": pipe.frontmatter.description,
                "runtime": pipe.frontmatter.runtime,
                "interval_seconds": pipe.frontmatter.interval_seconds,
                "schedule": pipe.frontmatter.schedule,
                "permissions": pipe.frontmatter.permissions.__dict__,
                "path": str(pipe.path),
            }
            if include_executions:
                item["recent_executions"] = db.list_pipe_executions(pipe.frontmatter.name, limit=5)
            rows.append(item)
        return {"data": rows, "total": len(rows)}

    @app.get("/pipes/{pipe_name}/executions")
    def pipe_executions(pipe_name: str, limit: int = 20) -> dict[str, Any]:
        return {"data": db.list_pipe_executions(pipe_name, limit=limit)}

    @app.post("/pipes/{pipe_name}/run")
    def run_pipe(pipe_name: str) -> dict[str, Any]:
        for pipe in load_pipes(paths.pipes_dir()):
            if pipe.frontmatter.name == pipe_name:
                execution_id = PipeRuntime(db, cfg).run(pipe, trigger="manual")
                return {"success": True, "execution_id": execution_id}
        raise HTTPException(status_code=404, detail="pipe not found")

    @app.post("/add")
    async def add_to_database(request: Request) -> dict[str, Any]:
        body = await request.json()
        content_type = str(body.get("content_type") or body.get("type") or "").lower()
        if content_type in {"memory", "memories"}:
            memory_id = db.create_memory(str(body.get("content") or body.get("text") or ""))
            return {"success": True, "memory_id": memory_id}
        if content_type in {"video_chunk", "video_chunks"}:
            chunk_id = db.insert_video_chunk(
                file_path=str(body.get("file_path") or video_chunk_path(device_name=str(body.get("device_name") or "screen"))),
                device_name=str(body.get("device_name") or ""),
                fps=float(body.get("fps") or 1.0),
            )
            return {"success": True, "video_chunk_id": chunk_id}
        raise HTTPException(status_code=400, detail="unsupported content_type")

    @app.api_route("/raw_sql", methods=["POST"])
    async def raw_sql(request: Request) -> dict[str, Any]:
        body = await request.json()
        query = str(body.get("query") or "").strip()
        if not _is_readonly_sql(query):
            raise HTTPException(status_code=403, detail="raw_sql only allows SELECT, WITH and EXPLAIN")
        raw_params = body.get("params") or []
        if not isinstance(raw_params, list):
            raise HTTPException(status_code=400, detail="params must be a list when provided")
        params: list[Any] = [p for p in raw_params]
        with db._lock:  # noqa: SLF001
            rows = db._conn.execute(query, params).fetchall()  # noqa: SLF001
        return {"data": rows, "total": len(rows)}

    @app.api_route("/transcribe", methods=["POST"])
    def transcribe_stub() -> JSONResponse:
        return JSONResponse(
            status_code=501,
            content={"error": "upload-and-transcribe not implemented"},
        )

    # ─── activity summary (aggregated overview for LLM agents) ──────────
    @app.get("/activity-summary")
    def activity_summary(
        start_time: str | None = None,
        end_time: str | None = None,
        app_name: str | None = None,
        q: str | None = None,
        include_recording: bool = True,
        include_memories: bool = True,
        include_snippets: bool = True,
        include_guidance: bool = True,
        max_snippets: int = 8,
        max_snippet_chars: int = 500,
        max_memories: int = 5,
        limit: int = 200,  # noqa: ARG001 — kept for API compat
    ) -> dict[str, Any]:
        """Rich activity bundle for LLM agents (`/activity-summary`)."""
        from .activity_summary import build_activity_summary

        return build_activity_summary(
            db,
            start_time=start_time,
            end_time=end_time,
            app_name=app_name,
            q=q,
            include_recording=include_recording,
            include_memories=include_memories,
            include_snippets=include_snippets,
            include_guidance=include_guidance,
            max_snippets=max_snippets,
            max_snippet_chars=max_snippet_chars,
            max_memories=max_memories,
        )

    # ─── apps (LLM-agent pipe apps) ─────────────────────────────────────
    # Apps are discovered across BOTH the built-in app dir and the user plugin
    # dir (~/.deskmate/apps/plugins); see deskmate.paths. User apps shadow
    # built-ins of the same name. Resolve an app's source dir with
    # paths.find_app_dir(name) — never a hardcoded source-relative path.

    def _scan_apps() -> list[dict[str, Any]]:
        apps_list: list[dict[str, Any]] = []
        for app_dir in paths.discover_app_dirs():
            pipe_md = app_dir / "pipe.md"
            text = pipe_md.read_text(encoding="utf-8")
            fm: dict[str, Any] = {}
            fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
            if fm_match:
                for line in fm_match.group(1).splitlines():
                    if ":" in line:
                        k, _, v = line.partition(":")
                        v = v.strip().strip('"').strip("'")
                        fm[k.strip()] = v
            name = pipe_md.parent.name
            pipe_schedule = fm.get("schedule", "manual")
            sched = app_schedules.entry_for_api(name, pipe_schedule)
            apps_list.append({
                "name": name,
                "title": fm.get("title", name),
                "description": fm.get("description", ""),
                "icon": fm.get("icon", ""),
                "schedule": sched["display"],
                "schedule_source": sched["source"],
                "schedule_config": sched,
                "pipe_schedule": pipe_schedule,
                "has_app_py": (pipe_md.parent / "app.py").exists(),
            })
        return apps_list

    def _scan_outputs(app_name: str) -> list[dict[str, Any]]:
        out_root = paths.root() / "apps" / app_name / "output"
        if not out_root.is_dir():
            return []
        runs: list[dict[str, Any]] = []
        for run_dir in sorted(out_root.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            md_files = list(run_dir.glob("*.md"))
            runs.append({
                "run_id": run_dir.name,
                "timestamp": run_dir.name,
                "files": [f.name for f in sorted(run_dir.iterdir())],
                "report_file": md_files[0].name if md_files else None,
            })
        return runs[:20]

    # ─── Ask (LLM agent) ────────────────────────────────────────────────

    @app.post("/ask")
    async def ask_question(request: Request) -> dict[str, Any]:
        body = await request.json()
        question = (body.get("question") or body.get("q") or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="question is required")

        from ..engine.ask import run_ask

        api_base = f"http://{cfg.server.host}:{cfg.server.port}"
        result = await asyncio.to_thread(run_ask, question, api_base=api_base)
        # Drop the internal evidence pool (raw tool results kept for grounding
        # checks) — the UI only needs the tool/args/length summary.
        if isinstance(result, dict):
            for entry in result.get("tool_calls") or []:
                if isinstance(entry, dict):
                    entry.pop("result", None)
        # Additive: log answered queries as future LoRA training pairs.
        try:
            answer = (result or {}).get("answer") or ""
            if answer and not (result or {}).get("error"):
                ask_id = _ask_store().record(
                    question=question,
                    answer=answer,
                    tool_count=len((result or {}).get("tool_calls") or []),
                )
                if ask_id is not None and isinstance(result, dict):
                    result["ask_id"] = ask_id
        except Exception:  # noqa: BLE001
            logger.debug("ask_history logging skipped", exc_info=True)
        return result

    @app.post("/ask/stream")
    async def ask_question_stream(request: Request) -> StreamingResponse:
        """Like ``/ask`` but streams the final answer live as NDJSON.

        Emits one JSON object per line:
          ``{"type":"token","text":...}``  — a chunk of the final answer
          ``{"type":"done","answer":...,"tool_calls":[...],"ask_id":...}`` — the
          authoritative grounded result (use this, not the concatenated tokens)
          ``{"type":"error","error":...}`` — failure
        Tool-call rounds run first (no token output); tokens flow once the model
        writes its final answer, so the user sees text instead of a long wait.
        """
        body = await request.json()
        question = (body.get("question") or body.get("q") or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="question is required")

        from ..engine.ask import run_ask  # noqa: PLC0415

        api_base = f"http://{cfg.server.host}:{cfg.server.port}"
        q: queue.Queue = queue.Queue(maxsize=512)
        _DONE = object()

        def _on_token(text: str) -> None:
            try:
                q.put_nowait({"type": "token", "text": text})
            except queue.Full:
                pass

        def _on_reset() -> None:
            # A streamed partial was abandoned (model rambled then used a tool /
            # retried) — tell the UI to clear the live preview.
            try:
                q.put_nowait({"type": "reset"})
            except queue.Full:
                pass

        def _on_thinking(text: str) -> None:
            # The model's reasoning pass (when thinking is enabled). Streamed
            # separately so the UI can show it in its own collapsible area.
            try:
                q.put_nowait({"type": "thinking", "text": text})
            except queue.Full:
                pass

        def _worker() -> None:
            try:
                result = run_ask(
                    question, api_base=api_base, on_token=_on_token,
                    on_reset=_on_reset, on_thinking=_on_thinking,
                )
                if isinstance(result, dict):
                    for entry in result.get("tool_calls") or []:
                        if isinstance(entry, dict):
                            entry.pop("result", None)
                    try:
                        answer = result.get("answer") or ""
                        if answer and not result.get("error"):
                            ask_id = _ask_store().record(
                                question=question, answer=answer,
                                tool_count=len(result.get("tool_calls") or []),
                            )
                            if ask_id is not None:
                                result["ask_id"] = ask_id
                    except Exception:  # noqa: BLE001
                        logger.debug("ask_history logging skipped", exc_info=True)
                q.put({"type": "done", **(result or {})})
            except Exception as exc:  # noqa: BLE001
                q.put({"type": "error", "error": str(exc)})
            finally:
                q.put(_DONE)

        async def _gen() -> AsyncIterator[str]:
            worker = asyncio.create_task(asyncio.to_thread(_worker))
            try:
                while True:
                    item = await asyncio.to_thread(q.get)
                    if item is _DONE:
                        break
                    yield json.dumps(item, ensure_ascii=False) + "\n"
            finally:
                await worker

        return StreamingResponse(_gen(), media_type="application/x-ndjson")

    @app.post("/ask/{ask_id}/feedback")
    async def ask_feedback(ask_id: int, request: Request) -> dict[str, Any]:
        body = await _safe_json(request)
        try:
            score = int(body.get("feedback"))
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="feedback must be 1 or -1")
        if score not in (1, -1):
            raise HTTPException(status_code=400, detail="feedback must be 1 or -1")
        ok = _ask_store().set_feedback(ask_id, score)
        if not ok:
            raise HTTPException(status_code=404, detail="ask answer not found")
        return {"status": "ok", "ask_id": ask_id, "feedback": score}

    @app.get("/apps")
    def list_apps() -> dict[str, Any]:
        apps_list = _scan_apps()
        for a in apps_list:
            a["recent_outputs"] = _scan_outputs(a["name"])[:3]
        return {"data": apps_list, "total": len(apps_list)}

    def _pipe_schedule_for(app_dir: Path) -> str:
        pipe_md = app_dir / "pipe.md"
        if pipe_md.is_file():
            fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", pipe_md.read_text(encoding="utf-8"), re.DOTALL)
            if fm_match:
                for line in fm_match.group(1).splitlines():
                    if line.strip().startswith("schedule:"):
                        return line.split(":", 1)[1].strip().strip('"').strip("'")
        return "manual"

    @app.get("/apps/{app_name}/schedule")
    def get_app_schedule(app_name: str) -> dict[str, Any]:
        app_dir = paths.find_app_dir(app_name)
        if app_dir is None or not (app_dir / "app.py").is_file():
            raise HTTPException(status_code=404, detail=f"app '{app_name}' not found")
        return app_schedules.entry_for_api(app_name, _pipe_schedule_for(app_dir))

    @app.put("/apps/{app_name}/schedule")
    async def put_app_schedule(app_name: str, request: Request) -> dict[str, Any]:
        app_dir = paths.find_app_dir(app_name)
        if app_dir is None or not (app_dir / "app.py").is_file():
            raise HTTPException(status_code=404, detail=f"app '{app_name}' not found")
        body = await _safe_json(request)
        try:
            entry = app_schedules.validate_schedule_payload(body)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if entry.get("enabled"):
            app_schedules.save_entry(app_name, entry)
        else:
            app_schedules.save_entry(app_name, {"enabled": False, "mode": "manual"})
        daemon = getattr(app.state, "daemon", None)
        if daemon is not None and getattr(daemon, "app_scheduler", None) is not None:
            daemon.app_scheduler.reload()
        return {
            "status": "ok",
            "schedule": app_schedules.entry_for_api(app_name, _pipe_schedule_for(app_dir)),
        }

    @app.get("/apps/{app_name}/outputs")
    def app_outputs(app_name: str) -> dict[str, Any]:
        return {"data": _scan_outputs(app_name)}

    @app.get("/apps/{app_name}/outputs/{run_id}/{filename}")
    def app_output_file(app_name: str, run_id: str, filename: str) -> Response:
        file_path = paths.root() / "apps" / app_name / "output" / run_id / filename
        if not file_path.is_file():
            raise HTTPException(status_code=404, detail="file not found")
        content = file_path.read_text(encoding="utf-8")
        media = "text/markdown" if filename.endswith(".md") else "application/json"
        return Response(content=content, media_type=media)

    @app.post("/apps/{app_name}/run")
    async def run_app(app_name: str, request: Request) -> dict[str, Any]:
        app_dir = paths.find_app_dir(app_name)
        app_py = app_dir / "app.py" if app_dir else None
        if app_py is None or not app_py.is_file():
            raise HTTPException(status_code=404, detail=f"app '{app_name}' not found")
        body = {}
        try:
            body = await request.json()
        except Exception:
            pass
        hours_provided = "hours" in body and str(body.get("hours")).strip() != ""
        hours = str(body.get("hours", "16"))
        minutes = str(body.get("minutes", "5"))
        start_time = body.get("start_time") or body.get("startTime")
        end_time = body.get("end_time") or body.get("endTime")
        cmd_args = ["--verbose"]
        if app_name == "video-export":
            if start_time and end_time:
                cmd_args += ["--start", str(start_time), "--end", str(end_time)]
            elif start_time or end_time:
                raise HTTPException(
                    status_code=400,
                    detail="start_time and end_time must both be provided",
                )
            else:
                cmd_args += ["--minutes", minutes]
        elif app_name == "email-compose":
            provider = str(body.get("provider") or "").strip().lower()
            to_addr = str(body.get("to") or "").strip()
            intent = str(body.get("intent") or body.get("message") or "").strip()
            if provider not in ("gmail", "outlook"):
                raise HTTPException(status_code=400, detail="provider must be gmail or outlook")
            if not to_addr:
                raise HTTPException(status_code=400, detail="to (recipient email) is required")
            if not intent:
                raise HTTPException(status_code=400, detail="intent (what to write) is required")
            cmd_args += ["--provider", provider, "--to", to_addr, "--intent", intent]
            account = body.get("account")
            if account:
                cmd_args += ["--account", str(account)]
            reply_to = body.get("reply_to") or body.get("replyTo")
            if reply_to:
                cmd_args += ["--reply-to", str(reply_to)]
            if body.get("send"):
                cmd_args.append("--send")
            compose_hours = str(body.get("hours", "1"))
            if start_time and end_time:
                cmd_args += ["--start", str(start_time), "--end", str(end_time)]
            elif start_time or end_time:
                raise HTTPException(
                    status_code=400,
                    detail="start_time and end_time must both be provided",
                )
            else:
                cmd_args += ["--hours", compose_hours]
        elif app_name == "meeting-summary":
            # Scopes to a meeting record, not a look-back window. The Meetings
            # page may pass a specific meeting_id; otherwise the latest is used.
            meeting_id = body.get("meeting_id") or body.get("meetingId")
            if meeting_id is not None:
                cmd_args += ["--meeting-id", str(meeting_id)]
        elif start_time and end_time:
            cmd_args += ["--start", str(start_time), "--end", str(end_time)]
        elif start_time or end_time:
            raise HTTPException(
                status_code=400,
                detail="start_time and end_time must both be provided",
            )
        elif hours_provided:
            cmd_args += ["--hours", hours]
        # else: omit --hours so the app's own default_hours applies (e.g.
        # user-profile / habit-report default to 7 days). The UI always sends an
        # explicit hours for look-back apps, so this only affects direct callers.
        import asyncio as _aio
        proc = await _aio.create_subprocess_exec(
            sys.executable, str(app_py), *cmd_args,
            stdout=_aio.subprocess.PIPE, stderr=_aio.subprocess.PIPE,
            env={
                **dict(os.environ),
                "DESKMATE_API": f"http://{cfg.server.host}:{cfg.server.port}",
                "DESKMATE_HOME": str(paths.root()),
                "DESKMATE_DB": str(db.path),
            },
        )
        stdout_b, stderr_b = await proc.communicate()
        stdout_text = stdout_b.decode("utf-8", errors="replace").strip()
        stderr_text = stderr_b.decode("utf-8", errors="replace").strip()
        return {
            "success": proc.returncode == 0,
            "exit_code": proc.returncode,
            "output_path": stdout_text,
            "stderr": stderr_text,
            "outputs": _scan_outputs(app_name)[:1],
        }

    # ─── version / platform ───────────────────────────────────────────────
    @app.get("/")
    def root() -> RedirectResponse:
        return RedirectResponse(url="/ui", status_code=307)

    @app.get("/api")
    def api_root() -> dict[str, Any]:
        return {
            "app": "deskmate",
            "version": app.version,
            "platform": platform.system(),
            "api_routes": [
                "/health", "/search", "/frames", "/frames/{id}", "/frames/{id}/image",
                "/frames/{id}/text", "/frames/{id}/context", "/frames/export", "/frames/next-valid",
                "/video-chunks", "/video-chunks/path", "/video-chunks/register",
                "/audio/list", "/audio/device/status", "/events/recent", "/events/stream",
                "/speakers/search", "/speakers/{id}/name", "/meetings",
                "/meetings/status", "/meetings/{id}", "/meetings/{id}/transcript",
                "/todos", "/todos/{id}",
                "/tags/vision/batch", "/tags/{content_type}/{id}", "/memories",
                "/memories/{id}", "/monitors",
                "/pipes", "/pipes/{name}/run", "/pipes/{name}/executions",
                "/raw_sql", "/add", "/capture", "/workflow/classify", "/activity/params", "/config",
                "/config/audio/languages",
                "/connections/gmail/connect", "/connections/gmail/status",
                "/connections/gmail/instances", "/connections/gmail/messages",
                "/connections/gmail/messages/{id}", "/connections/gmail/send",
                "/connections/gmail/disconnect",
                "/connections/outlook/connect", "/connections/outlook/status",
                "/connections/outlook/instances", "/connections/outlook/messages",
                "/connections/outlook/messages/{id}", "/connections/outlook/send",
                "/connections/outlook/disconnect",
            ],
        }

    @app.get("/ui")
    def ui_index():  # noqa: ANN201
        return FileResponse(index_file(), media_type="text/html")

    # ─── habits (additive: routines, suggestions, proactive nudges) ───────
    def _habit_store() -> HabitStore:
        store = getattr(app.state, "habit_store", None)
        if store is None:
            store = HabitStore()
            try:
                store.ensure_rules(habit_rules.DEFAULT_RULES)
            except Exception:  # noqa: BLE001
                pass
            app.state.habit_store = store
        return store

    @app.get("/habits/profile")
    def habits_profile() -> dict[str, Any]:
        """Learned routines as a (day_type, slot) grid for the UI heatmap."""
        rows = _habit_store().all_profiles()
        grid: dict[str, dict[int, dict[str, Any]]] = {"weekday": {}, "weekend": {}}
        for r in rows:
            dtype = r.get("day_type") or "weekday"
            slot = int(r.get("slot") or 0)
            cell = grid.setdefault(dtype, {}).get(slot)
            # Keep the strongest (highest-frequency) category per slot for the grid.
            if cell is None or float(r.get("frequency") or 0) > float(cell.get("frequency") or 0):
                grid.setdefault(dtype, {})[slot] = {
                    "category": r.get("category"),
                    "top_app": r.get("top_app"),
                    "avg_minutes": r.get("avg_minutes"),
                    "frequency": r.get("frequency"),
                    "sample_days": r.get("sample_days"),
                }
        return {
            "slots_per_day": 48,
            "grid": {k: {str(s): v for s, v in slots.items()} for k, slots in grid.items()},
            "rows": rows,
            "total": len(rows),
        }

    @app.post("/habits/mine")
    def habits_mine() -> dict[str, Any]:
        """Manually trigger a re-mine of habit profiles from frames."""
        hcfg = cfg.habits
        miner = HabitMiner(
            _habit_store(),
            lookback_days=hcfg.mine_lookback_days,
            min_frequency=hcfg.min_frequency,
            min_sample_days=hcfg.min_sample_days,
        )
        return miner.mine()

    @app.get("/habits/suggestions")
    def habits_suggestions(status: str | None = None, limit: int = 50) -> dict[str, Any]:
        rows = _habit_store().list_suggestions(status=status, limit=limit)
        return {"data": rows, "total": len(rows)}

    @app.post("/habits/suggestions/{suggestion_id}/feedback")
    async def habits_suggestion_feedback(suggestion_id: int, request: Request) -> dict[str, Any]:
        body = await request.json()
        try:
            score = int(body.get("feedback"))
        except (TypeError, ValueError, AttributeError):
            raise HTTPException(status_code=400, detail="feedback must be 1 or -1")
        if score not in (1, -1):
            raise HTTPException(status_code=400, detail="feedback must be 1 or -1")
        ok = _habit_store().set_suggestion_feedback(suggestion_id, score)
        if not ok:
            raise HTTPException(status_code=404, detail="suggestion not found")
        return {"ok": True}

    @app.post("/habits/suggestions/{suggestion_id}/dismiss")
    def habits_suggestion_dismiss(suggestion_id: int) -> dict[str, Any]:
        ok = _habit_store().set_suggestion_status(suggestion_id, "dismissed")
        if not ok:
            raise HTTPException(status_code=404, detail="suggestion not found")
        return {"ok": True}

    @app.post("/habits/suggestions/{suggestion_id}/snooze")
    async def habits_suggestion_snooze(suggestion_id: int, request: Request) -> dict[str, Any]:
        """Snooze the RULE behind a suggestion. ``minutes`` mutes it for that long
        ("再等一会"); ``rest_of_day`` mutes it until local midnight ("今天别再提").
        Also marks the suggestion handled so it leaves the active inbox."""
        from datetime import datetime, timedelta  # noqa: PLC0415

        body = await _safe_json(request)
        store = _habit_store()
        rows = store.list_suggestions(limit=500)
        row = next((r for r in rows if int(r.get("id", -1)) == suggestion_id), None)
        if row is None:
            raise HTTPException(status_code=404, detail="suggestion not found")
        rule_name = row.get("rule_name") or ""
        now = datetime.now().astimezone()
        if body.get("rest_of_day"):
            until = (now + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        else:
            try:
                minutes = int(body.get("minutes", 15))
            except (TypeError, ValueError):
                minutes = 15
            minutes = max(1, min(minutes, 24 * 60))
            until = now + timedelta(minutes=minutes)
        store.snooze_rule(rule_name, until.replace(microsecond=0).isoformat())
        store.set_suggestion_status(suggestion_id, "dismissed")
        return {"ok": True, "rule": rule_name, "snoozed_until": until.isoformat()}

    @app.get("/habits/settings")
    def habits_settings() -> dict[str, Any]:
        """Module-wide reminder settings the UI reads/flips (global on/off, lang)."""
        store = _habit_store()
        return {
            "notifications_enabled": store.notifications_enabled(),
            "reminder_lang": store.get_setting("reminder_lang", cfg.habits.reminder_lang),
        }

    @app.post("/habits/settings")
    async def habits_settings_update(request: Request) -> dict[str, Any]:
        """Update reminder settings. Accepts ``notifications_enabled`` (bool) — the
        global "turn off reminders" switch — and/or ``reminder_lang`` ("zh"/"en")."""
        body = await _safe_json(request)
        store = _habit_store()
        if "notifications_enabled" in body:
            store.set_notifications_enabled(bool(body.get("notifications_enabled")))
        lang = body.get("reminder_lang")
        if lang in ("zh", "en"):
            store.set_setting("reminder_lang", lang)
            cfg.habits.reminder_lang = lang
        return {
            "ok": True,
            "notifications_enabled": store.notifications_enabled(),
            "reminder_lang": store.get_setting("reminder_lang", cfg.habits.reminder_lang),
        }

    @app.get("/habits/ui")
    def habits_ui():  # noqa: ANN201
        return FileResponse(static_dir() / "habits.html", media_type="text/html")

    # ─── capture control + unified timeline (additive) ───────────────────────
    async def _safe_json(request: Request) -> dict[str, Any]:
        try:
            body = await request.json()
            return body if isinstance(body, dict) else {}
        except Exception:  # noqa: BLE001
            return {}

    def _capture_control() -> CaptureControl:
        ctrl = getattr(app.state, "capture_control", None)
        if ctrl is None:
            ctrl = CaptureControl()
            app.state.capture_control = ctrl
        return ctrl

    def _context_store() -> ContextStore:
        store = getattr(app.state, "context_store", None)
        if store is None:
            store = ContextStore()
            app.state.context_store = store
        return store

    def _ask_store():  # noqa: ANN202
        store = getattr(app.state, "ask_store", None)
        if store is None:
            from ..learning.ask_store import AskHistoryStore  # noqa: PLC0415

            store = AskHistoryStore()
            app.state.ask_store = store
        return store

    @app.get("/capture/control")
    def capture_control_state() -> dict[str, Any]:
        return {"control": _capture_control().state(), "sources": list(TOGGLEABLE)}

    @app.post("/capture/pause")
    async def capture_pause(request: Request) -> dict[str, Any]:
        body = await _safe_json(request)
        minutes = body.get("minutes")
        try:
            minutes = int(minutes) if minutes is not None else None
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="minutes must be an integer")
        return {"control": _capture_control().set_paused(True, minutes=minutes)}

    @app.post("/capture/resume")
    def capture_resume() -> dict[str, Any]:
        return {"control": _capture_control().resume()}

    @app.post("/capture/source")
    async def capture_source(request: Request) -> dict[str, Any]:
        body = await _safe_json(request)
        source = str(body.get("source") or "")
        if source not in TOGGLEABLE:
            raise HTTPException(status_code=400, detail=f"source must be one of {list(TOGGLEABLE)}")
        if "enabled" not in body:
            raise HTTPException(status_code=400, detail="enabled (bool) is required")
        enabled = bool(body.get("enabled"))
        return {"control": _capture_control().set_source(source, enabled)}

    @app.post("/capture/forget")
    async def capture_forget(request: Request) -> dict[str, Any]:
        body = await _safe_json(request)
        try:
            minutes = int(body.get("minutes"))
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="minutes (int) is required")
        if minutes <= 0:
            raise HTTPException(status_code=400, detail="minutes must be > 0")
        cutoff = (datetime.now().astimezone() - timedelta(minutes=minutes)).replace(microsecond=0).isoformat()
        removed = db.forget_since(iso_cutoff=cutoff)
        removed["context_events"] = _context_store().forget_since(cutoff)
        return {"ok": True, "cutoff": cutoff, "removed": removed}

    @app.get("/timeline/unified")
    def timeline_unified(
        since: str | None = None,
        until: str | None = None,
        sources: str | None = None,
        limit: int = 200,
    ) -> dict[str, Any]:
        source_list = [s.strip() for s in sources.split(",") if s.strip()] if sources else None
        rows = _context_store().list_events(
            since=since, until=until, sources=source_list, limit=max(1, min(limit, 1000)),
        )
        return {"data": rows, "total": len(rows)}

    @app.get("/timeline/unified/breakdown")
    def timeline_breakdown(since: str | None = None) -> dict[str, Any]:
        return {"data": _context_store().source_breakdown(since=since)}

    @app.get("/capture/ui")
    def capture_ui():  # noqa: ANN201
        return FileResponse(static_dir() / "capture.html", media_type="text/html")

    # ─── LoRA training (additive, opt-in) ────────────────────────────────────
    def _training_sources(raw: str | None) -> list[str]:
        from ..learning.training.data import SOURCES  # noqa: PLC0415

        if not raw:
            return list(cfg.training.sources)
        return [s.strip() for s in raw.split(",") if s.strip() in SOURCES]

    def _training_apps(raw: str | None) -> list[str] | None:
        """Parse the optional per-app allow-list. ``None`` (param omitted) means
        all apps; an explicit empty string means *no* apps."""
        if raw is None:
            return None
        return [a.strip() for a in raw.split(",") if a.strip()]

    @app.get("/training/data")
    def training_data(
        sources: str | None = None, sample: int = 5, full: bool = False,
        apps: str | None = None,
    ) -> dict[str, Any]:
        """Preview the mined SFT dataset.

        ``sample`` controls how many example pairs are returned (capped at 200);
        ``full=true`` returns every mined pair so the UI can show the exact
        dataset that training will use. ``apps`` is an optional comma-separated
        allow-list restricting the ``apps`` source to those app folders.
        """
        from ..learning.training import DeskMateTrainingDataMiner  # noqa: PLC0415

        src = _training_sources(sources)
        app_filter = _training_apps(apps)
        tc = cfg.training
        miner = DeskMateTrainingDataMiner(min_feedback=tc.min_feedback, min_chars=tc.min_chars)
        try:
            breakdown = miner.source_breakdown(sources=src, limit_per_source=tc.limit_per_source)
            available_apps = miner.list_apps()
            pairs = miner.extract_sft_pairs(
                sources=src, limit_per_source=tc.limit_per_source, max_pairs=tc.max_pairs,
                apps=app_filter,
            )
        finally:
            miner.close()
        shown = pairs if full else pairs[: max(0, min(sample, 200))]
        return {
            "sources": src,
            "breakdown": breakdown,
            "apps": available_apps,
            "total": len(pairs),
            "returned": len(shown),
            "sample": shown,
        }

    @app.get("/training/data/export")
    def training_data_export(sources: str | None = None, apps: str | None = None) -> Response:
        """Download the full mined dataset as JSONL (one pair per line).

        This is exactly what a training run would consume, so the user can
        inspect or archive the dataset before fine-tuning."""
        from ..learning.training import DeskMateTrainingDataMiner  # noqa: PLC0415

        src = _training_sources(sources)
        app_filter = _training_apps(apps)
        tc = cfg.training
        miner = DeskMateTrainingDataMiner(min_feedback=tc.min_feedback, min_chars=tc.min_chars)
        try:
            pairs = miner.extract_sft_pairs(
                sources=src, limit_per_source=tc.limit_per_source, max_pairs=tc.max_pairs,
                apps=app_filter,
            )
        finally:
            miner.close()
        body = "\n".join(json.dumps(p, ensure_ascii=False) for p in pairs) + ("\n" if pairs else "")
        return Response(
            content=body,
            media_type="application/x-ndjson",
            headers={"Content-Disposition": 'attachment; filename="deskmate_sft_dataset.jsonl"'},
        )

    @app.post("/training/lora")
    async def training_lora(request: Request) -> dict[str, Any]:
        # Note: the trainer itself runs in a child process (see below); here we
        # only need the miner + the dep check, which don't import torch.
        from ..learning.training import (  # noqa: PLC0415
            DeskMateTrainingDataMiner,
            missing_training_deps,
        )

        body = await _safe_json(request)
        tc = cfg.training
        src = _training_sources(body.get("sources"))

        # Optional per-app allow-list: accept a list or a comma-string; omitted
        # → all apps. An explicit empty list means "no apps".
        raw_apps = body.get("apps")
        if isinstance(raw_apps, list):
            app_filter: list[str] | None = [str(a).strip() for a in raw_apps if str(a).strip()]
        else:
            app_filter = _training_apps(raw_apps)

        miner = DeskMateTrainingDataMiner(min_feedback=tc.min_feedback, min_chars=tc.min_chars)
        try:
            pairs = miner.extract_sft_pairs(
                sources=src,
                limit_per_source=tc.limit_per_source,
                max_pairs=int(body.get("max_pairs") or tc.max_pairs),
                apps=app_filter,
            )
        finally:
            miner.close()

        if not pairs:
            return {"status": "skipped", "reason": "no training data", "sources": src}
        missing = missing_training_deps()
        if missing:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"训练依赖未安装（缺少 {', '.join(missing)}）。"
                    "请运行： pip install 'deskmate[training]' 后重启 DeskMate。"
                ),
            )

        from .. import paths as _paths  # noqa: PLC0415

        out_dir = str(
            body.get("output_dir") or tc.output_dir or (_paths.root() / "checkpoints" / "lora")
        )

        config_kwargs = {
            "lora_rank": tc.lora_rank,
            "lora_alpha": tc.lora_alpha,
            "lora_dropout": tc.lora_dropout,
            "target_modules": list(tc.target_modules),
            "num_epochs": int(body.get("epochs") or tc.num_epochs),
            "batch_size": tc.batch_size,
            "learning_rate": tc.learning_rate,
            "max_seq_length": tc.max_seq_length,
            "use_4bit": tc.use_4bit,
            "output_dir": out_dir,
        }
        model_name = str(body.get("model") or tc.model_name)

        # Run training in a SEPARATE PROCESS, not a threadpool. Training loads the
        # base model + LoRA + optimizer onto the GPU/iGPU; doing it in the
        # long-lived UI process leaves that memory occupied until the server
        # exits. A child process that trains and then exits lets the OS reclaim
        # ALL of its accelerator memory immediately — cleaner than empty_cache.
        jobs_dir = _paths.root() / "checkpoints" / ".jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S_%f")
        job_file = jobs_dir / f"job_{stamp}.json"
        result_file = jobs_dir / f"result_{stamp}.json"
        job_file.write_text(
            json.dumps({"model_name": model_name, "config": config_kwargs, "pairs": pairs},
                       ensure_ascii=False),
            encoding="utf-8",
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "deskmate.learning.training._worker",
                str(job_file), str(result_file),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                env={**dict(os.environ),
                     "DESKMATE_HOME": str(_paths.root()),
                     "DESKMATE_DB": str(db.path)},
            )
            _out, err = await proc.communicate()
            if proc.returncode != 0 or not result_file.exists():
                detail = (err.decode("utf-8", "replace")[-800:] if err else "").strip()
                raise HTTPException(
                    status_code=500,
                    detail=f"训练子进程失败（退出码 {proc.returncode}）。{detail}",
                )
            summary = json.loads(result_file.read_text(encoding="utf-8"))
            if summary.get("status") == "error":
                raise HTTPException(status_code=500,
                                    detail=f"训练失败：{summary.get('error', 'unknown')}")
        finally:
            for f in (job_file, result_file):
                try:
                    f.unlink()
                except OSError:
                    pass

        summary["sources"] = src
        return summary

    # ── Model Service (local Ollama: download / pull / run) ──────────────────
    def _ndjson_install(install_fn, on_done=None) -> StreamingResponse:  # noqa: ANN001
        """Stream a blocking download/extract job as NDJSON progress lines.

        ``install_fn(progress)`` runs in a worker thread; its ``progress(done,
        total)`` callback feeds a queue that the async generator drains off the
        event loop (same off-loop technique as ``/events/stream``). Emits
        ``{phase: download|extract|done|error, ...}`` objects, one per line.
        ``on_done(result)`` runs in the worker after success (e.g. to persist
        config) and may return a dict merged into the ``done`` line.
        """
        q: queue.Queue = queue.Queue(maxsize=256)
        _DONE = object()

        def _progress(done: int, total: int) -> None:
            # Drop intermediate ticks if the consumer falls behind; never block
            # the download thread on a full queue.
            try:
                q.put_nowait({"phase": "download", "downloaded": done, "total": total})
            except queue.Full:
                pass

        def _worker() -> None:
            try:
                result = install_fn(_progress)
                q.put({"phase": "extract"})
                extra = on_done(result) if on_done else None
                done_line = {"phase": "done", "result": str(result)}
                if isinstance(extra, dict):
                    done_line.update(extra)
                q.put(done_line)
            except Exception as exc:  # noqa: BLE001
                q.put({"phase": "error", "error": str(exc)})
            finally:
                q.put(_DONE)

        async def _gen() -> AsyncIterator[str]:
            worker = asyncio.create_task(asyncio.to_thread(_worker))
            try:
                while True:
                    item = await asyncio.to_thread(q.get)
                    if item is _DONE:
                        break
                    yield json.dumps(item, ensure_ascii=False) + "\n"
            finally:
                await worker

        return StreamingResponse(_gen(), media_type="application/x-ndjson")

    @app.get("/models/status")
    def models_status() -> dict[str, Any]:
        from .. import modelsvc  # noqa: PLC0415

        return modelsvc.status(cfg)

    @app.post("/models/config")
    async def models_config(request: Request) -> dict[str, Any]:
        from .. import modelsvc  # noqa: PLC0415
        from ..config import set_config_value  # noqa: PLC0415

        body = await _safe_json(request)
        allowed = {"backend", "ollama_exe_path", "ollama_exe_url",
                   "registry", "genai_runtime_dir", "genai_url",
                   "download_dir", "auto_start", "pull_insecure", "stop_on_exit"}
        saved: list[str] = []
        errors: dict[str, str] = {}
        for key, raw in body.items():
            if key not in allowed:
                errors[key] = "unknown setting"
                continue
            try:
                if key == "backend":
                    value: Any = str(raw)
                    if value not in (modelsvc.BACKEND_OFFICIAL, modelsvc.BACKEND_OPENVINO):
                        raise ValueError("backend must be 'official' or 'openvino'")
                elif key in ("auto_start", "pull_insecure", "stop_on_exit"):
                    value = bool(raw)
                elif key == "ollama_exe_path" and str(raw).strip():
                    # Validate a user-supplied exe; store the resolved path.
                    value = str(modelsvc.validate_exe_path(str(raw)))
                else:
                    value = str(raw)
            except ValueError as exc:
                errors[key] = str(exc)
                continue
            set_config_value("model_service", key, value)
            setattr(cfg.model_service, key, value)
            saved.append(key)
        return {"saved": saved, "errors": errors, "status": modelsvc.status(cfg)}

    @app.post("/models/download-ollama")
    def models_download_ollama() -> StreamingResponse:
        from .. import modelsvc  # noqa: PLC0415

        return _ndjson_install(modelsvc.install_official)

    def _persist_ms(updates: dict) -> dict:
        """Write each ``[model_service]`` key to disk + live cfg; return status."""
        from .. import modelsvc  # noqa: PLC0415
        from ..config import set_config_value  # noqa: PLC0415

        for k, v in updates.items():
            set_config_value("model_service", k, v)
            setattr(cfg.model_service, k, v)
        return {"status": modelsvc.status(cfg)}

    @app.post("/models/download-openvino-exe")
    async def models_download_openvino_exe(request: Request) -> StreamingResponse:
        """Obtain the OpenVINO ``ollama.exe`` (download a URL or copy a local path).

        Body ``{exe: <url-or-path>}``. Streams progress; on success persists
        ``backend=openvino`` + the resolved exe path. The GenAI runtime is a
        separate download (``/models/download-genai``).
        """
        from .. import modelsvc  # noqa: PLC0415

        body = await _safe_json(request)
        exe_src = str(body.get("exe") or cfg.model_service.ollama_exe_path or "").strip()
        if not exe_src:
            raise HTTPException(status_code=400, detail="ollama.exe path or URL required")
        # A local path can be validated up front (a URL is checked on download).
        if not exe_src.lower().startswith(("http://", "https://")):
            try:
                modelsvc.validate_exe_path(exe_src)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        dl_dir = modelsvc.resolve_download_dir(cfg)
        return _ndjson_install(
            lambda progress: modelsvc.obtain_openvino_exe(exe_src, dl_dir, progress=progress),
            on_done=lambda result: _persist_ms({
                "backend": modelsvc.BACKEND_OPENVINO,
                "ollama_exe_path": str(result),
            }),
        )

    @app.post("/models/download-genai")
    async def models_download_genai(request: Request) -> StreamingResponse:
        """Download + extract the OpenVINO GenAI runtime into the download dir.

        Body ``{url?}`` overrides the default GenAI URL so the user can fetch a
        specific version. On success the persisted ``genai_runtime_dir`` points
        at the just-installed version (and the URL is remembered).
        """
        from .. import modelsvc  # noqa: PLC0415

        body = await _safe_json(request)
        url = str(body.get("url") or "").strip() or None
        if url and not url.lower().startswith(("http://", "https://")):
            raise HTTPException(status_code=400, detail="url must be http(s)")
        dl_dir = modelsvc.resolve_download_dir(cfg)

        def _on_done(result) -> dict:  # noqa: ANN001
            updates = {"genai_runtime_dir": str(result)}
            if url:
                updates["genai_url"] = url
            return _persist_ms(updates)

        return _ndjson_install(
            lambda progress: modelsvc.download_genai(dl_dir, progress=progress, url=url),
            on_done=_on_done,
        )

    @app.get("/models/genai-versions")
    def models_genai_versions() -> dict[str, Any]:
        """List installed GenAI runtime versions + which one is selected for PATH."""
        from .. import modelsvc  # noqa: PLC0415

        return {
            "versions": modelsvc.list_genai_versions(cfg),
            "selected": cfg.model_service.genai_runtime_dir or "",
        }

    @app.post("/models/pull")
    async def models_pull(request: Request) -> StreamingResponse:
        from .. import modelsvc  # noqa: PLC0415

        body = await _safe_json(request)
        model = str(body.get("model") or cfg.ollama.model or "").strip()
        if not model:
            raise HTTPException(status_code=400, detail="model name required")

        async def _gen() -> AsyncIterator[str]:
            gen = modelsvc.pull_model_stream(cfg, model)
            while True:
                item = await asyncio.to_thread(next, gen, None)
                if item is None:
                    break
                yield json.dumps(item, ensure_ascii=False) + "\n"

        return StreamingResponse(_gen(), media_type="application/x-ndjson")

    @app.post("/models/active")
    async def models_active(request: Request) -> dict[str, Any]:
        """Set the active model — the ``[ollama] model`` Ask / apps use.

        Body ``{model}``. Persisted to ``[ollama] model`` (not ``[model_service]``)
        so the selection the user makes here is the one every other surface
        reads. Returns the refreshed model-service status.
        """
        from .. import modelsvc  # noqa: PLC0415
        from ..config import set_config_value  # noqa: PLC0415

        body = await _safe_json(request)
        model = str(body.get("model") or "").strip()
        if not model:
            raise HTTPException(status_code=400, detail="model name required")
        set_config_value("ollama", "model", model)
        cfg.ollama.model = model
        return {"active_model": model, "status": modelsvc.status(cfg)}

    @app.post("/models/start")
    async def models_start(request: Request) -> dict[str, Any]:
        from .. import modelsvc  # noqa: PLC0415
        from ..config import set_config_value  # noqa: PLC0415

        # Optional {backend}: a panel's Start button names the backend it wants.
        # We make it the active backend before launching, so "start the OpenVINO
        # panel" launches OpenVINO regardless of the previously selected backend.
        body = await _safe_json(request)
        backend = str(body.get("backend") or "").strip()
        if backend in (modelsvc.BACKEND_OFFICIAL, modelsvc.BACKEND_OPENVINO):
            if backend != cfg.model_service.backend:
                set_config_value("model_service", "backend", backend)
                cfg.model_service.backend = backend
        elif backend:
            raise HTTPException(status_code=400, detail="backend must be 'official' or 'openvino'")

        try:
            return await run_in_threadpool(modelsvc.start_service, cfg)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/models/stop")
    async def models_stop() -> dict[str, Any]:
        from .. import modelsvc  # noqa: PLC0415

        return await run_in_threadpool(modelsvc.stop_service, cfg)

    @app.get("/models/log")
    def models_log(lines: int = 400, backend: str = "") -> dict[str, Any]:
        """Return the tail of a backend's Ollama service stdout/stderr log.

        ``backend`` ("openvino"/"official") selects that backend's log file;
        omitted returns the legacy combined log.
        """
        from .. import modelsvc  # noqa: PLC0415

        n = max(1, min(int(lines), 2000))
        be = backend if backend in (modelsvc.BACKEND_OFFICIAL, modelsvc.BACKEND_OPENVINO) else None
        return {"log": modelsvc.read_service_log(max_lines=n, backend=be), "backend": be or ""}

    @app.exception_handler(Exception)
    async def _eh(_request, exc):  # noqa: ANN001
        logger.exception("unhandled api error")
        return JSONResponse(status_code=500, content={"error": str(exc)})

    return app
