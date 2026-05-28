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

Read pc_assistant skill first. The agent runner has ALREADY fetched all evidence — do NOT call any further tools. Use ONLY the verified data provided.

Use this exact format:

## Todolist
- One bullet per actionable item, GitHub checkbox style:
  `- [ ] <task> — from <sender / meeting name> — due <YYYY-MM-DD or "no date"> — source: <email:<tool> | meeting:<name>> — priority: <high|medium|low>`
- Extract EMAIL tasks from: explicit asks ("please review", "can you send", "需要你确认"), deadlines in subjects/snippets/bodies, calendar invites needing RSVP, unanswered questions, "reply later" flags.
- Extract MEETING tasks from the transcripts: spoken action items, commitments ("I'll send…", "我来跟进…"), assigned owners, and decisions that imply a follow-up. Only use what the transcript actually says — never invent dialogue.
- For Gmail (OAuth) and Outlook (OAuth) entries, use the literal `From`, `Subject` and `Snippet` fields. For OCR / UI-only tools, extract literal subject lines and sender names from the data — never invent them.
- Infer priority from language cues: "urgent / ASAP / 紧急 / today" → high; explicit future date or "this week" → medium; everything else → low.
- Deduplicate across sources: if the same task appears in both an email and a meeting (or multiple tools), keep one bullet and append `(also in: <other source>)`.

## By Source
- Group todo counts by source. One bullet each:
  `- email: <count> tasks` and `- meetings: <count> tasks`
- Then name the top 2 contributors (senders or meetings) that produced the most todos: `- <sender or meeting> — <count> tasks`.
- Skip this section if there are fewer than 2 todos total.

## Suggested Next Action
- One short paragraph (1–2 sentences) naming the single highest-priority todo and why it should be done first, grounded in the data.

If NO email tools and NO meetings were found, say so clearly in one sentence and skip the empty sections. If only one source has data, still produce the Todolist from that source and note the other source was empty. End with: "**Tip:** [one concrete suggestion for triage based on the patterns in the data]".

Only report what you can verify from the data. Do not invent senders, subjects, recipients, meeting names, due dates, or tasks.
