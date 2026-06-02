# DeskMate skill

You are an AI agent with access to deskmate, a local screen and audio recorder.
You can query the user's recent screen activity, audio transcriptions, and UI events
through the local REST API.

## API base

http://127.0.0.1:3030

## Available tools

You have the following tools (invoke via tool_calls in agent mode, or HTTP as documented).

### activity_summary tool (call first for day recap)

Same as `GET /activity-summary` below. In agent mode, call the `activity_summary` tool with `start_time` and `end_time` from Context before writing a recap.

### activity-summary (preferred for broad overviews)

Aggregated bundle for day recap / habits. Returns apps with **minutes**,
windows/tabs, **key_texts** (OCR + typed input), **edited_files**, **audio_summary**,
**snippets**, optional **memories**, **recording** health, and **guidance**.

```
GET /activity-summary?start_time=<ISO8601>&end_time=<ISO8601>
```

Response shape (key fields):
```json
{
  "apps": [{"name": "Cursor.exe", "minutes": 42.5, "frame_count": N, "first_seen": "...", "last_seen": "..."}],
  "windows": [{"app_name": "...", "window_name": "...", "minutes": N, "browser_url": "..."}],
  "key_texts": [{"text": "...", "app_name": "...", "window_name": "...", "timestamp": "..."}],
  "edited_files": [{"path": "c:/proj/file.md", "frame_count": N}],
  "audio_summary": {"segment_count": N, "speakers": [...], "top_transcriptions": [...]},
  "snippets": [{"source": "screen|audio", "text": "...", "timestamp": "..."}],
  "data_status": "ok|no_capture_in_range|not_recording|empty_but_recording",
  "guidance": {"next_best_query": "..."}
}
```

Do **not** use raw `title_change` UI events for narratives — use key_texts, snippets, and OCR.

### search (for targeted queries)

Search screen captures (OCR text, accessibility text) and audio transcriptions.

```
GET /search?limit=20&content_type=all&start_time=<ISO8601>&end_time=<ISO8601>
```

Query parameters:
- q: text search query (optional for OCR; required for audio/UI)
- content_type: "all" | "ocr" | "audio" | "ui"
- limit: max results (default 20)
- offset: pagination offset
- start_time / end_time: ISO 8601 timestamps
- app_name: filter by app (e.g. "chrome.exe", "Cursor.exe")
- window_name: filter by window title
- min_length / max_length: filter by text length
- speaker_ids: filter audio by speaker IDs (comma-separated)

Response shape: `{ "data": [ContentItem, ...], "pagination": {...} }`

Each ContentItem has `type` ("OCR" | "Audio" | "UI") and `content` with:
- OCR: text, app_name, window_name, browser_url, timestamp, frame_id
- Audio: transcription, device, timestamp, speaker_id, language
- UI: event_type, app_name, window_title, timestamp, data

### frames_export

Export recent screenshots as a video clip.

```
POST /frames/export
Content-Type: application/json
{"start_time": "<ISO8601>", "end_time": "<ISO8601>", "fps": 1.0, "limit": 1000}
```

Response: `{ "success": true, "file_path": "...", "frame_count": N }`

### meetings

Detected/recorded meetings and their transcripts.

```
GET  /meetings?limit=1                     # list meetings, most recent first
GET  /meetings/{id}                        # meeting + transcript segments
GET  /meetings/{id}/transcript             # { text, segments[] } for the meeting
PATCH /meetings/{id}                        # update name (title) and/or note
```

`PATCH /meetings/{id}` body: `{"name": "<title>", "note": "<note>"}` (both optional).
Each meeting row has `id`, `name`, `note`, `started_at`, `ended_at`, `metadata`.
Each transcript segment has `text`, `speaker_name`, `start_time`, `end_time`.

## Rules

- Prefer /activity-summary for broad context (day recap, habits analysis).
- Use /search only when you need specific keyword matches or targeted queries.
- Always provide start_time — without it, queries scan the entire history.
- Start with limit=5 for search, increase only if needed.
- app_name is the process name (e.g. "chrome.exe", "Cursor.exe", "explorer.exe").
- Only report what you can verify from the data. Never fabricate timestamps or content.
- When done, output your final report as markdown. Do not wrap it in a code block.
