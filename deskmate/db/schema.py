"""Consolidated SQLite schema for the local activity recorder.

The application uses one current schema snapshot instead of replaying every
historical migration. `_pca_migrations` stores the active schema version.
"""

from __future__ import annotations

# Bumped whenever the consolidated schema changes.
SCHEMA_VERSION = "20260906180000"

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;
PRAGMA temp_store = MEMORY;
PRAGMA busy_timeout = 5000;
PRAGMA cache_size = -64000;
PRAGMA mmap_size = 268435456;

-- ─── housekeeping ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS _pca_migrations (
    version TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL
);

-- ─── video chunks (recorded streams) ───────────────────────────────────────
CREATE TABLE IF NOT EXISTS video_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    device_name TEXT NOT NULL DEFAULT '',
    fps REAL NOT NULL DEFAULT 1.0
);
CREATE INDEX IF NOT EXISTS idx_video_chunks_device ON video_chunks(device_name);

-- ─── frames ────────────────────────────────────────────────────────────────
-- Frames can be stored as offsets into video chunks or as event-driven JPEG
-- snapshots (when video_chunk_id is NULL).
CREATE TABLE IF NOT EXISTS frames (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_chunk_id INTEGER,
    offset_index INTEGER NOT NULL DEFAULT 0,
    timestamp TIMESTAMP NOT NULL,
    name TEXT,
    app_name TEXT NOT NULL DEFAULT '',
    window_name TEXT NOT NULL DEFAULT '',
    focused INTEGER NOT NULL DEFAULT 1,
    browser_url TEXT,
    document_path TEXT,
    snapshot_path TEXT,           -- our extension; NULL when frame is in a video_chunk
    monitor_id INTEGER NOT NULL DEFAULT 0,
    device_name TEXT NOT NULL DEFAULT '',
    width INTEGER NOT NULL DEFAULT 0,
    height INTEGER NOT NULL DEFAULT 0,
    capture_trigger TEXT NOT NULL DEFAULT 'heartbeat',
    image_redacted INTEGER NOT NULL DEFAULT 0,
    text_length INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (video_chunk_id) REFERENCES video_chunks(id)
);
CREATE INDEX IF NOT EXISTS idx_frames_timestamp     ON frames(timestamp);
CREATE INDEX IF NOT EXISTS idx_frames_video_chunk   ON frames(video_chunk_id);
CREATE INDEX IF NOT EXISTS idx_frames_app           ON frames(app_name);
CREATE INDEX IF NOT EXISTS idx_frames_window        ON frames(window_name);
CREATE INDEX IF NOT EXISTS idx_frames_browser_url   ON frames(browser_url);

-- ─── ocr_text ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ocr_text (
    frame_id INTEGER NOT NULL,
    text TEXT NOT NULL DEFAULT '',
    text_json TEXT NOT NULL DEFAULT '[]',       -- [{text,left,top,width,height,conf} as strings]
    ocr_engine TEXT NOT NULL DEFAULT '',
    text_length INTEGER NOT NULL DEFAULT 0,
    redacted_text TEXT,
    redacted_text_json TEXT,
    PRIMARY KEY (frame_id),
    FOREIGN KEY (frame_id) REFERENCES frames(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_ocr_text_frame ON ocr_text(frame_id);

-- ─── accessibility (paired with frames) ────────────────────────────────────
CREATE TABLE IF NOT EXISTS frame_accessibility (
    frame_id INTEGER PRIMARY KEY,
    text TEXT NOT NULL DEFAULT '',
    tree_json TEXT NOT NULL DEFAULT '{}',
    focused_role TEXT,
    focused_name TEXT,
    focused_value TEXT,
    on_screen INTEGER,
    redacted_text TEXT,
    elements_ref_frame_id INTEGER,             -- dedup pointer when content unchanged
    FOREIGN KEY (frame_id) REFERENCES frames(id) ON DELETE CASCADE,
    FOREIGN KEY (elements_ref_frame_id) REFERENCES frames(id)
);
CREATE INDEX IF NOT EXISTS idx_acc_ref_frame    ON frame_accessibility(elements_ref_frame_id);

-- ─── elements (normalized accessibility nodes, P1) ─────────────────────────
-- One row per meaningful UIA node, flattened from frame_accessibility.tree_json.
-- Only written for "new content" frames (frame_accessibility.elements_ref_frame_id
-- IS NULL); dedup pointers reuse the referenced frame's rows. Rows cascade-delete
-- with their frame.
CREATE TABLE IF NOT EXISTS elements (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id       INTEGER NOT NULL,
    node_index     INTEGER NOT NULL,          -- preorder index within the frame's tree
    parent_index   INTEGER,                   -- node_index of parent (NULL = root)
    depth          INTEGER NOT NULL DEFAULT 0,
    role           TEXT NOT NULL DEFAULT '',  -- control_type
    name           TEXT,
    value          TEXT,                      -- redacted; NULL for password fields
    automation_id  TEXT,
    is_focused     INTEGER NOT NULL DEFAULT 0,
    is_interactive INTEGER NOT NULL DEFAULT 0,
    bounds         TEXT,                      -- "l,t,w,h" or NULL
    FOREIGN KEY (frame_id) REFERENCES frames(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_elements_frame ON elements(frame_id);
CREATE INDEX IF NOT EXISTS idx_elements_role  ON elements(role);
CREATE INDEX IF NOT EXISTS idx_elements_focus ON elements(frame_id, is_focused);

-- ─── ui events ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ui_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP NOT NULL,
    relative_ms INTEGER NOT NULL DEFAULT 0,
    event_type TEXT NOT NULL,                  -- click | move | scroll | key | text | app_switch | window_focus | clipboard
    app_name TEXT,
    window_title TEXT,
    browser_url TEXT,
    frame_id INTEGER,
    data_json TEXT NOT NULL DEFAULT '{}',
    element_json TEXT,
    FOREIGN KEY (frame_id) REFERENCES frames(id)
);
CREATE INDEX IF NOT EXISTS idx_ui_events_ts     ON ui_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_ui_events_type   ON ui_events(event_type);
CREATE INDEX IF NOT EXISTS idx_ui_events_app    ON ui_events(app_name);

-- ─── audio ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS audio_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    device_name TEXT NOT NULL DEFAULT '',
    timestamp TIMESTAMP NOT NULL,
    processing_status TEXT NOT NULL DEFAULT 'pending',  -- pending | processed | failed | evicted
    duration_ms INTEGER NOT NULL DEFAULT 0,
    evicted_at TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_audio_chunks_ts    ON audio_chunks(timestamp);
CREATE INDEX IF NOT EXISTS idx_audio_chunks_dev   ON audio_chunks(device_name);
CREATE INDEX IF NOT EXISTS idx_audio_chunks_stat  ON audio_chunks(processing_status);

CREATE TABLE IF NOT EXISTS audio_transcriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    audio_chunk_id INTEGER,
    offset_index INTEGER NOT NULL DEFAULT 0,
    timestamp TIMESTAMP NOT NULL,
    transcription TEXT NOT NULL DEFAULT '',
    device TEXT NOT NULL DEFAULT '',
    language TEXT,
    speaker_id INTEGER,
    start_time REAL,
    end_time REAL,
    text_length INTEGER NOT NULL DEFAULT 0,
    redacted_transcription TEXT,
    translation TEXT,                          -- live-translation target text (nullable)
    translation_lang TEXT,                     -- target language ISO code for `translation`
    FOREIGN KEY (audio_chunk_id) REFERENCES audio_chunks(id) ON DELETE SET NULL,
    FOREIGN KEY (speaker_id) REFERENCES speakers(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_transcripts_ts      ON audio_transcriptions(timestamp);
CREATE INDEX IF NOT EXISTS idx_transcripts_chunk   ON audio_transcriptions(audio_chunk_id);
CREATE INDEX IF NOT EXISTS idx_transcripts_speaker ON audio_transcriptions(speaker_id);

-- ─── speakers (diarization) ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS speakers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL DEFAULT '',
    metadata TEXT NOT NULL DEFAULT '{}',
    centroid_json TEXT,                        -- mean embedding (JSON-encoded floats)
    sample_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS speaker_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    speaker_id INTEGER NOT NULL,
    embedding_json TEXT NOT NULL,
    audio_chunk_id INTEGER,
    transcription_id INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (speaker_id) REFERENCES speakers(id) ON DELETE CASCADE,
    FOREIGN KEY (audio_chunk_id) REFERENCES audio_chunks(id) ON DELETE SET NULL,
    FOREIGN KEY (transcription_id) REFERENCES audio_transcriptions(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_speaker_emb_speaker ON speaker_embeddings(speaker_id);

-- ─── meetings + segments ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS meetings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL DEFAULT '',
    note TEXT NOT NULL DEFAULT '',
    started_at TIMESTAMP NOT NULL,
    ended_at TIMESTAMP,
    metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_meetings_started ON meetings(started_at);

CREATE TABLE IF NOT EXISTS meeting_transcript_segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id INTEGER NOT NULL,
    transcription_id INTEGER,
    speaker_id INTEGER,
    text TEXT NOT NULL DEFAULT '',
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE,
    FOREIGN KEY (transcription_id) REFERENCES audio_transcriptions(id) ON DELETE SET NULL,
    FOREIGN KEY (speaker_id) REFERENCES speakers(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_meeting_segments_meeting ON meeting_transcript_segments(meeting_id);

-- ─── todos (structured action items extracted from email + meetings) ────────
CREATE TABLE IF NOT EXISTS todos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',        -- open | done
    source TEXT NOT NULL DEFAULT '',            -- email | meeting | manual
    source_ref TEXT NOT NULL DEFAULT '',        -- sender / meeting display name
    source_detail TEXT NOT NULL DEFAULT '',     -- email:<tool> | meeting:<name>
    meeting_id INTEGER,
    priority TEXT NOT NULL DEFAULT '',          -- H | M | L
    due TEXT NOT NULL DEFAULT '',
    origin_app TEXT NOT NULL DEFAULT '',        -- app that created the todo
    evidence_start TEXT NOT NULL DEFAULT '',    -- activity window used at extraction
    evidence_end TEXT NOT NULL DEFAULT '',
    dedup_key TEXT NOT NULL DEFAULT '',         -- stable key for upsert dedup
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT,
    FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_todos_status  ON todos(status);
CREATE INDEX IF NOT EXISTS idx_todos_created ON todos(created_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_todos_dedup ON todos(dedup_key) WHERE dedup_key <> '';

-- ─── tags ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL
);
CREATE TABLE IF NOT EXISTS frame_tags (
    frame_id INTEGER NOT NULL,
    tag_id INTEGER NOT NULL,
    PRIMARY KEY (frame_id, tag_id),
    FOREIGN KEY (frame_id) REFERENCES frames(id) ON DELETE CASCADE,
    FOREIGN KEY (tag_id)   REFERENCES tags(id)   ON DELETE CASCADE
);

-- ─── memories (lightweight records used by local automations) ──────────────
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    frame_id INTEGER,
    sync_id TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ─── pipes (markdown files with frontmatter, see deskmate.pipes) ───────
CREATE TABLE IF NOT EXISTS pipe_executions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pipe_name TEXT NOT NULL,
    session_path TEXT,
    status TEXT NOT NULL DEFAULT 'pending',    -- pending | running | success | failed
    output TEXT,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    ended_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_pipe_exec_pipe ON pipe_executions(pipe_name);

-- ─── FTS5 indexes ──────────────────────────────────────────────────────────
-- Consolidated full-text search table.
CREATE VIRTUAL TABLE IF NOT EXISTS frames_full_text USING fts5(
    frame_id UNINDEXED,
    timestamp UNINDEXED,
    app_name, window_name, browser_url, document_path,
    ocr_text, accessibility_text,
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE VIRTUAL TABLE IF NOT EXISTS audio_transcriptions_fts USING fts5(
    transcription_id UNINDEXED,
    timestamp UNINDEXED,
    speaker_id UNINDEXED,
    transcription,
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE VIRTUAL TABLE IF NOT EXISTS ui_events_fts USING fts5(
    event_id UNINDEXED,
    timestamp UNINDEXED,
    event_type UNINDEXED,
    app_name, window_title, text_content,
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE VIRTUAL TABLE IF NOT EXISTS elements_fts USING fts5(
    element_id UNINDEXED,
    frame_id UNINDEXED,
    role UNINDEXED,
    name, value,
    tokenize = 'unicode61 remove_diacritics 2'
);

-- ─── Semantic search (vector embeddings) ───────────────────────────────────
-- One row per indexed piece of text content. The embedding is stored as a
-- little-endian float32 BLOB. Cosine similarity is computed in Python over a
-- bounded candidate set; this keeps the schema portable (no vector extension).
CREATE TABLE IF NOT EXISTS content_embeddings (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    content_type TEXT    NOT NULL,   -- 'ocr' | 'audio' | 'ui'
    content_id   INTEGER NOT NULL,   -- frame_id | transcription_id | event_id
    model        TEXT    NOT NULL,
    dim          INTEGER NOT NULL,
    embedding    BLOB    NOT NULL,
    timestamp    TEXT    NOT NULL,
    created_at   TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE(content_type, content_id, model)
);
CREATE INDEX IF NOT EXISTS idx_content_emb_type_ts
    ON content_embeddings(content_type, timestamp);
CREATE INDEX IF NOT EXISTS idx_content_emb_model
    ON content_embeddings(model);

-- ─── habits: learned routines (data → patterns) ────────────────────────────
-- One row per (day_type, half-hour slot, category). Aggregated from frames by
-- the habits miner. ``frequency`` (0..1) is the share of observed days the
-- behavior occurred and is the core "habit strength" signal.
CREATE TABLE IF NOT EXISTS habit_profiles (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    day_type     TEXT    NOT NULL,             -- 'weekday' | 'weekend'
    slot         INTEGER NOT NULL,             -- 0..47 (hour*2 + (minute>=30))
    category     TEXT    NOT NULL,             -- coding|browsing|email|meeting|writing|communication|other
    top_app      TEXT    NOT NULL DEFAULT '',  -- highest-frequency app in this slot
    avg_minutes  REAL    NOT NULL DEFAULT 0,   -- mean minutes spent (per active day)
    frequency    REAL    NOT NULL DEFAULT 0,   -- 0..1, share of days this behavior occurs
    sample_days  INTEGER NOT NULL DEFAULT 0,   -- distinct days backing this row (confidence)
    updated_at   TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE(day_type, slot, category)
);
CREATE INDEX IF NOT EXISTS idx_habit_slot ON habit_profiles(day_type, slot);

-- ─── habits: trigger rules (pattern + present → timing) ────────────────────
CREATE TABLE IF NOT EXISTS habit_rules (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    UNIQUE NOT NULL,        -- 'distraction_peak' | 'overwork' | ...
    rule_type     TEXT    NOT NULL,               -- deviation|routine_missing|threshold|streak
    enabled       INTEGER NOT NULL DEFAULT 1,
    params_json   TEXT    NOT NULL DEFAULT '{}',  -- thresholds / category / window
    cooldown_min  INTEGER NOT NULL DEFAULT 120,   -- min minutes between firings (anti-nag)
    quiet_hours   TEXT    NOT NULL DEFAULT '22-8', -- "start-end" local hours of silence
    priority      TEXT    NOT NULL DEFAULT 'M',   -- H|M|L
    created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- ─── habits: suggestion history (suggestion → user) ────────────────────────
CREATE TABLE IF NOT EXISTS habit_suggestions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_name     TEXT    NOT NULL,
    message       TEXT    NOT NULL,
    context_json  TEXT    NOT NULL DEFAULT '{}',  -- evidence at trigger time
    channel       TEXT    NOT NULL DEFAULT 'toast',-- toast|ui|telegram
    status        TEXT    NOT NULL DEFAULT 'sent', -- sent|suppressed|clicked|dismissed
    feedback      INTEGER,                          -- 1=useful -1=not useful NULL=none
    created_at    TEXT    NOT NULL DEFAULT (datetime('now')),
    acknowledged_at TEXT                            -- when the user clicked/dismissed/rated it
);
CREATE INDEX IF NOT EXISTS idx_suggestion_rule ON habit_suggestions(rule_name, created_at);
CREATE INDEX IF NOT EXISTS idx_suggestion_status ON habit_suggestions(status, created_at);

-- ─── fusion: unified multi-source context timeline ─────────────────────────
-- One normalized row per signal from any source (screen / audio / input /
-- clipboard / window). This is an additive, read-mostly "context bus" sink:
-- the fusion bus subscribes to the existing in-process event bus and projects
-- every event here, so downstream consumers (timeline, habits, recap) get a
-- single chronological stream with provenance + confidence. Producers are
-- never modified.
CREATE TABLE IF NOT EXISTS context_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT    NOT NULL,              -- local ISO-with-tz of the signal
    source       TEXT    NOT NULL,              -- screen|audio|input|clipboard|window
    kind         TEXT    NOT NULL,              -- frame|transcript|click|text|clipboard|focus|title|value
    app_name     TEXT    NOT NULL DEFAULT '',
    window_title TEXT    NOT NULL DEFAULT '',
    summary      TEXT    NOT NULL DEFAULT '',   -- short human-readable text
    payload_json TEXT    NOT NULL DEFAULT '{}', -- source-specific detail
    confidence   REAL    NOT NULL DEFAULT 1.0,  -- 0..1 (e.g. ASR < 1, UIA = 1)
    frame_id     INTEGER,                       -- link to frames when applicable
    created_at   TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_context_events_ts     ON context_events(ts);
CREATE INDEX IF NOT EXISTS idx_context_events_source ON context_events(source, ts);

-- ─── capture control: runtime pause / forget / per-source switches ─────────
-- Single-row control surface (id is pinned to 1). The daemon's capture
-- chokepoints read this (cached, fail-open) to honor pause + per-source
-- toggles; the API writes it. DB-mediated so it works whether the API and
-- daemon share a process or not, mirroring the habits module's pattern.
CREATE TABLE IF NOT EXISTS capture_control (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    paused      INTEGER NOT NULL DEFAULT 0,
    pause_until TEXT,                            -- auto-resume time (NULL = until manually resumed)
    screen      INTEGER NOT NULL DEFAULT 1,
    audio       INTEGER NOT NULL DEFAULT 1,
    input       INTEGER NOT NULL DEFAULT 1,
    clipboard   INTEGER NOT NULL DEFAULT 1,
    updated_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
INSERT OR IGNORE INTO capture_control(id) VALUES (1);

-- ─── ask history: (question → grounded answer) pairs ───────────────────────
-- Every answered Ask query is logged here. Additive and write-only from the
-- API; powers the optional LoRA training miner (the user's own questions and
-- the assistant's evidence-based answers are high-quality SFT pairs). Empty /
-- error answers are not persisted.
CREATE TABLE IF NOT EXISTS ask_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    question    TEXT    NOT NULL,
    answer      TEXT    NOT NULL,
    tool_count  INTEGER NOT NULL DEFAULT 0,     -- number of tool calls used
    feedback    INTEGER,                        -- user rating: 1=useful, -1=not, NULL=unrated
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_ask_history_created ON ask_history(created_at);
"""
