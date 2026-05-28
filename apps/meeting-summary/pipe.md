---
schedule: manual
enabled: true
template: true
trigger:
  events:
    - meeting_ended
title: Meeting Summary
description: Auto-summarizes the meeting that just ended and patches the summary back onto the meeting record (title + note).
icon: "🤝"
featured: false
---

A meeting just ended. Find it, summarize it, and patch the summary back onto its record so the user sees it next time they open the meeting.

Read pc_assistant skill first so you know the meetings + search endpoints.

step 1 — find the meeting that just ended:

  GET http://127.0.0.1:3030/meetings?limit=1

the most recent row is the one that just ended. capture its `id`, `name`, `note`, `started_at`, and `ended_at`.

step 2 — read what happened during this meeting and summarize it: key topics, decisions, action items. the transcript comes from:

  GET http://127.0.0.1:3030/meetings/<MEETING_ID>/transcript

each segment has `text`, `speaker_name`, and timing. if the transcript is empty, fall back to /search?content_type=audio scoped to the meeting's `started_at`/`ended_at` window.

step 3 — if your summary is worth saving, append it to the meeting note (and refresh the title in the same call) via:

  PATCH http://127.0.0.1:3030/meetings/<MEETING_ID>
  Content-Type: application/json
  {"name": "<NEW_TITLE_OR_OMIT>", "note": "<EXISTING_NOTE>\n\n## Summary\n<YOUR_SUMMARY>"}

replace `<EXISTING_NOTE>` with the meeting's current `note` field (empty string if none) so you don't overwrite the user's work; just append your summary under a `## Summary` heading. for the title: if the current title is missing, generic ("untitled", "meeting", just the app name) or doesn't capture what actually happened, replace it with a 5-8 word plain-english title (no quotes, no "meeting about…" prefix) — otherwise omit the field so a user-set title is left alone. if there's nothing useful to summarize (empty transcript, irrelevant audio), say so out loud and skip the PATCH — don't write a placeholder.
