---
schedule: manual
enabled: true
template: true
title: Todo List Assistant
description: "One todolist from your email + meetings — task, source, due date, priority"
icon: "✅"
featured: true
---

Build ONE unified todolist from my activity in the supplied time range (defaults to the last 24 hours). Evidence comes from TWO sources, both pre-fetched into the data section below: my **email** (Gmail / Outlook OAuth + on-screen mail tools) and my **meetings** (detected video calls + their transcripts).

Read DeskMate skill first. The agent runner has ALREADY fetched all evidence — do NOT call any further tools. Use ONLY the verified data provided.

Use this EXACT format for every todo bullet. Fields are separated by a pipe `|`
and each carries an explicit `key:` label, so the order is fixed and parsing is
unambiguous. Put the task FIRST (no key), then the labelled fields:

`- [ ] <task> | from: <sender / meeting name> | due: <YYYY-MM-DD or "no date"> | source: <email:<tool> | meeting:<name> | screen:<app>> | priority: <high|medium|low>`

Rules for the `task` text: keep it on one line; do NOT put a `|` inside the task
(replace any literal pipe with a dash). Em-dashes and other punctuation inside
the task are fine — only the `|` separates fields.

## Todolist
- Extract EMAIL tasks from: explicit asks ("please review", "can you send", "需要你确认"), deadlines in subjects/snippets/bodies, calendar invites needing RSVP, unanswered questions, "reply later" flags.
- Extract MEETING tasks from the transcripts: spoken action items, commitments ("I'll send…", "我来跟进…"), assigned owners, and decisions that imply a follow-up. Only use what the transcript actually says — never invent dialogue.
- Extract SCREEN tasks from the on-screen evidence ONLY when the text states an explicit task assigned to or owned by the user: a `TODO:` / `FIXME:` note, a chat message directed at the user asking them to do something, or a checklist item. Tag these `source: screen:<app>`. Do NOT turn page titles, article text, menu labels, code you are merely reading, or generic UI text into todos. If unsure whether screen text is a real personal task, DROP it.
- For Gmail (OAuth) and Outlook (OAuth) entries, use the literal `From`, `Subject` and `Snippet` fields. For OCR / UI-only tools, extract literal subject lines and sender names from the data — never invent them.
- Infer priority from language cues: "urgent / ASAP / 紧急 / today" → high; explicit future date or "this week" → medium; everything else → low.
- Deduplicate across sources: if the same task appears in more than one place, keep one bullet and append `(also in: <other source>)` to the task text.

## By Source
- Group todo counts by source. One bullet each (omit a line whose count is 0):
  `- email: <count> tasks`, `- meetings: <count> tasks`, `- screen: <count> tasks`
- Then name the top 2 contributors (senders / meetings / apps) that produced the most todos: `- <name> — <count> tasks`.
- Skip this section if there are fewer than 2 todos total.

## Suggested Next Action
- One short paragraph (1–2 sentences) naming the single highest-priority todo and why it should be done first, grounded in the data.

If NO email, NO meetings and NO actionable screen evidence were found, say so clearly in one sentence and skip the empty sections. If only one source has data, still produce the Todolist from that source and note the others were empty. End with: "**Tip:** [one concrete suggestion for triage based on the patterns in the data]".

Only report what you can verify from the data. Do not invent senders, subjects, recipients, meeting names, due dates, or tasks.
