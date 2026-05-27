"""FastAPI HTTP server for the local activity recorder.

The API exposes health checks, search, frames, events, audio transcripts,
speakers, monitors, configuration and the browser UI. Routes for optional
subsystems that are not implemented return a 501 `Not Implemented`.
"""

from __future__ import annotations

import asyncio
import json
import platform
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .. import events as bus
from .. import paths
from ..a11y.activity_feed import default as activity_default
from ..capture import paired_capture as run_paired_capture
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


# Map content type values onto our search dispatch.
_CONTENT_TYPE_FRAMES = {"all", "frames", "ocr"}
_CONTENT_TYPE_AUDIO = {"all", "audio"}
_CONTENT_TYPE_UI = {"all", "ui"}


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
            "transcription": row.get("transcription") or row.get("snippet") or "",
            "timestamp": row.get("timestamp"),
            "file_path": "",
            "offset_index": row.get("offset_index", 0),
            "device_name": row.get("device") or "",
            "speaker_id": row.get("speaker_id"),
            "start_time": row.get("start_time"),
            "end_time": row.get("end_time"),
            "tags": [],
        },
    }


def _content_item_for_ui(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "UI",
        "content": {
            "id": row.get("id") or row.get("event_id"),
            "timestamp": row.get("timestamp"),
            "event_type": row.get("event_type"),
            "app_name": row.get("app_name") or "",
            "window_title": row.get("window_title") or "",
            "browser_url": row.get("browser_url"),
            "data": _parse_json(row.get("data_json")),
            "element": _parse_json(row.get("element_json")),
        },
    }


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


def create_app(cfg: Config | None = None, db: DatabaseManager | None = None) -> FastAPI:
    cfg = cfg or load_config()
    db = db or DatabaseManager()
    workflow = WorkflowClassifier()

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

    # ─── search (the big one) ─────────────────────────────────────────────
    @app.get("/search")
    def search(
        q: str | None = Query(default=None, description="FTS5 MATCH query"),
        content_type: str = Query(default="all"),
        app_name: str | None = None,
        window_name: str | None = None,
        frame_name: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        speaker_ids: str | None = None,  # comma-separated
        include_frames: bool = False,
        offset: int = 0,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Search content and return `{data: ContentItem[], pagination}`.
        `ContentItem.type` is one of "OCR" | "Audio" | "UI".
        """
        items: list[dict[str, Any]] = []

        if content_type in _CONTENT_TYPE_FRAMES:
            for row in db.search_frames(
                q, app_name=app_name, window_name=window_name,
                start=start_time, end=end_time,
                limit=limit, offset=offset,
            ):
                items.append(_content_item_for_frame(row))

        if content_type in _CONTENT_TYPE_AUDIO and q:
            speaker_filter = {int(s) for s in (speaker_ids or "").split(",") if s.strip().isdigit()}
            for row in db.search_transcripts(q, limit=limit):
                if speaker_filter and row.get("speaker_id") not in speaker_filter:
                    continue
                items.append(_content_item_for_transcript(row))

        if content_type in _CONTENT_TYPE_UI and q:
            for row in db.search_ui_events(q, limit=limit):
                items.append(_content_item_for_ui(row))

        # crude length filters applied post-hoc (FTS doesn't know "length")
        if min_length is not None or max_length is not None:
            def _len(item: dict[str, Any]) -> int:
                c = item["content"]
                return len(c.get("text") or c.get("transcription") or "")
            items = [
                i for i in items
                if (min_length is None or _len(i) >= min_length)
                and (max_length is None or _len(i) <= max_length)
            ]

        return {
            "data": items,
            "pagination": {"limit": limit, "offset": offset, "total": len(items)},
        }

    # ─── frames ───────────────────────────────────────────────────────────
    @app.get("/frames")
    def frames(limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        return db.recent_frames(limit=limit)

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
    def audio_list(limit: int = 50) -> list[dict[str, Any]]:
        return db.recent_transcripts(limit=limit)

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
    def recent_events(limit: int = 100) -> list[dict[str, Any]]:
        return db.recent_events(limit=limit)

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
        return cfg.model_dump()

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
        with db._lock:  # noqa: SLF001
            rows = db._conn.execute(query).fetchall()  # noqa: SLF001
        return {"data": rows, "total": len(rows)}

    @app.api_route("/transcribe", methods=["POST"])
    def transcribe_stub() -> JSONResponse:
        return JSONResponse(
            status_code=501,
            content={"error": "upload-and-transcribe not implemented"},
        )

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
                "/frames/{id}/text", "/frames/{id}/context", "/frames/next-valid",
                "/video-chunks", "/video-chunks/path", "/video-chunks/register",
                "/audio/list", "/audio/device/status", "/events/recent", "/events/stream",
                "/speakers/search", "/speakers/{id}/name", "/meetings",
                "/meetings/status", "/meetings/{id}", "/meetings/{id}/transcript",
                "/tags/vision/batch", "/tags/{content_type}/{id}", "/memories",
                "/memories/{id}", "/monitors",
                "/pipes", "/pipes/{name}/run", "/pipes/{name}/executions",
                "/raw_sql", "/add", "/capture", "/workflow/classify", "/activity/params", "/config",
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
