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
import re
import shutil
import subprocess
import sys
import time
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
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
from ..logger import get
from ..pipes import PipeRuntime, load_pipes
from ..screen.capture import list_monitors
from ..screen.redact_image import redact_image_bytes, regions_from_ocr
from ..screen.video_chunks import video_chunk_path
from ..ui import index_file, static_dir
from ..workflow import WorkflowClassifier

logger = get("engine.api")


def _is_pc_assistant_app(app_name: str | None) -> bool:
    """Exclude pc_assistant's own UI from search results."""
    if not app_name:
        return False
    return "pc_assistant" in app_name.lower() or "pc-assistant" in app_name.lower()


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
    if _is_pc_assistant_app(app_name if isinstance(app_name, str) else None):
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
<body><h1>Email connected</h1><p>{email} is ready for pc_assistant.</p></body></html>"""


def create_app(cfg: Config | None = None, db: DatabaseManager | None = None) -> FastAPI:
    cfg = cfg or load_config()
    db = db or DatabaseManager()
    workflow = WorkflowClassifier()
    gmail = GmailConnection(cfg.gmail)
    outlook = OutlookConnection(cfg.outlook)

    app = FastAPI(title="pc_assistant", version="0.2.0")
    app.state.cfg = cfg
    app.state.db = db
    app.mount("/ui/assets", StaticFiles(directory=static_dir()), name="ui-assets")

    started_at = time.time()

    # ─── health ───────────────────────────────────────────────────────────
    @app.get("/health")
    def health() -> dict[str, Any]:
        stats = db.health()
        feed = activity_default()
        params = feed.get_capture_params()
        return {
            "status": "ok",
            "status_code": 200,
            "frames": stats.get("frames", 0),
            "events": stats.get("events", 0),
            "transcripts": stats.get("transcripts", 0),
            "last_frame_timestamp": stats.get("last_frame_timestamp"),
            "last_audio_timestamp": stats.get("last_audio_timestamp"),
            "frame_status": "ok" if stats.get("frames") else "no frames yet",
            "audio_status": "ok" if stats.get("transcripts") else "off or no transcripts",
            "meeting_status": "active" if db.active_meeting() else "idle",
            "message": "pc_assistant running",
            "verbose_instructions": None,
            "device_status_details": None,
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
        include_frames: bool = False,
        offset: int = 0,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Search content (FTS sanitize + merge)."""
        speaker_filter = [int(s) for s in (speaker_ids or "").split(",") if s.strip().isdigit()] or None
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
        return {
            "enabled": cfg.audio.enabled,
            "microphone": cfg.audio.microphone,
            "loopback": cfg.audio.loopback,
            "recent_transcripts": len(db.recent_transcripts(limit=10)),
        }

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
        async def _gen() -> AsyncIterator[str]:
            stream = bus.stream(timeout=0.5)
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        evt = next(stream)
                    except StopIteration:
                        await asyncio.sleep(0.1)
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
    _APPS_SRC = Path(__file__).resolve().parents[2] / "apps"

    def _scan_apps() -> list[dict[str, Any]]:
        apps_list: list[dict[str, Any]] = []
        if not _APPS_SRC.is_dir():
            return apps_list
        for pipe_md in sorted(_APPS_SRC.glob("*/pipe.md")):
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
            apps_list.append({
                "name": name,
                "title": fm.get("title", name),
                "description": fm.get("description", ""),
                "icon": fm.get("icon", ""),
                "schedule": fm.get("schedule", "manual"),
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
        return result

    @app.get("/apps")
    def list_apps() -> dict[str, Any]:
        apps_list = _scan_apps()
        for a in apps_list:
            a["recent_outputs"] = _scan_outputs(a["name"])[:3]
        return {"data": apps_list, "total": len(apps_list)}

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
        app_py = _APPS_SRC / app_name / "app.py"
        if not app_py.is_file():
            raise HTTPException(status_code=404, detail=f"app '{app_name}' not found")
        body = {}
        try:
            body = await request.json()
        except Exception:
            pass
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
            # Scopes to the latest meeting record, not a look-back window.
            pass
        elif start_time and end_time:
            cmd_args += ["--start", str(start_time), "--end", str(end_time)]
        elif start_time or end_time:
            raise HTTPException(
                status_code=400,
                detail="start_time and end_time must both be provided",
            )
        else:
            cmd_args += ["--hours", hours]
        import asyncio as _aio
        proc = await _aio.create_subprocess_exec(
            sys.executable, str(app_py), *cmd_args,
            stdout=_aio.subprocess.PIPE, stderr=_aio.subprocess.PIPE,
            env={**dict(os.environ), "PC_ASSISTANT_API": f"http://{cfg.server.host}:{cfg.server.port}"},
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
            "app": "pc_assistant",
            "version": app.version,
            "platform": platform.system(),
            "api_routes": [
                "/health", "/search", "/frames", "/frames/{id}", "/frames/{id}/image",
                "/frames/{id}/text", "/frames/{id}/context", "/frames/export", "/frames/next-valid",
                "/video-chunks", "/video-chunks/path", "/video-chunks/register",
                "/audio/list", "/audio/device/status", "/events/recent", "/events/stream",
                "/speakers/search", "/speakers/{id}/name", "/meetings",
                "/meetings/status", "/meetings/{id}", "/meetings/{id}/transcript",
                "/tags/vision/batch", "/tags/{content_type}/{id}", "/memories",
                "/memories/{id}", "/monitors",
                "/pipes", "/pipes/{name}/run", "/pipes/{name}/executions",
                "/raw_sql", "/add", "/capture", "/workflow/classify", "/activity/params", "/config",
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

    @app.exception_handler(Exception)
    async def _eh(_request, exc):  # noqa: ANN001
        logger.exception("unhandled api error")
        return JSONResponse(status_code=500, content={"error": str(exc)})

    return app
