---
schedule: manual
enabled: true
template: true
title: Time Breakdown
description: "Where your time went — by app, project, and category"
icon: "⏱"
featured: false
---

Analyze my DeskMate app usage from the supplied time range (defaults to the last 12 hours).

Read DeskMate skill first. The agent runner has **already pre-fetched** activity summary and **pre-computed** application minutes, category totals, and productivity score — you do **not** call tools.

Use this exact format with durations and percentages (use the pre-computed tables; do not invent minutes):

## By Application
- List each app with duration and percentage, sorted by time descending (e.g. "Cursor (`Cursor.exe`): 45 min (32%)").
- Use the human-readable name from pre-computed totals; include the process name in parentheses.

## By Category
- Group into: coding, meetings, browsing, writing, communication, other.
- Show minutes and percentage per category — must match pre-computed category minutes.

## By Project
- Group related activities by project / repo / topic, using `edited_files` paths and window/tab titles.
- Name specific folders or tasks (e.g. `UX/deskmate`, `ollama_openvino`).

## Productivity Score
- Use the pre-computed focused/unfocused minutes and score percentage.
- One line: `**Productivity Score:** N% (focused X min / total Y min)`.

End with: "**Suggestion:** [one specific change to improve tomorrow's productivity, grounded in what the data shows]".

Only report what you can verify from the pre-fetched data. Do not invent apps or projects that do not appear in the data.
