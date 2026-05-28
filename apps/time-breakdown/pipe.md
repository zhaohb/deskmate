---
schedule: manual
enabled: true
template: true
title: Time Breakdown
description: "Where your time went — by app, project, and category"
icon: "⏱"
featured: false
---

Analyze my pc_assistant app usage from the supplied time range (defaults to the last 12 hours).

Read pc_assistant skill first. Call `activity_summary` once with the supplied `start_time` / `end_time` — it returns each app's `minutes`, the windows/tabs, `edited_files` and `audio_summary`, which is all you need. You may then call `search` at most 4 times (limit=5 each), only to verify the topic / project / category of a specific app when the window titles are ambiguous.

Use this exact format with durations and percentages (compute percentages from the `minutes` field of `apps`):

## By Application
- List each app with duration and percentage, sorted by time descending (e.g. "Cursor.exe: 2h 15min (28%)").
- Use the human-readable app name when obvious (e.g. "VS Code" for "Code.exe"); otherwise keep the process name.

## By Category
- Group into: coding, meetings, browsing, writing, communication, other.
- Show hours and percentage per category.
- Categorise based on app + window/tab titles you can see in the data — do not guess.

## By Project
- Group related activities by project / repo / topic, using `edited_files` paths and window titles. Name the specific repos or tasks.

## Productivity Score
- Calculate: focused_work_hours / total_hours as a percentage.
- Focused = coding + writing. Unfocused = browsing + idle switching.

End with: "**Suggestion:** [one specific change to improve tomorrow's productivity, grounded in what the data shows]".

Only report what you can verify from the data. Do not invent apps or projects that do not appear in `activity_summary`.
