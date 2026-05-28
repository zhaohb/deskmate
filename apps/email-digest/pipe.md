---
schedule: manual
enabled: true
template: true
title: Email Digest
description: "What you did in your inbox — apps, top senders, drafts, action items"
icon: "📧"
featured: true
---

Analyze my email-client and webmail activity from the supplied time range (defaults to the last 24 hours).

Read pc_assistant skill first. The agent runner has already executed one targeted `search` per email tool (native + webmail) and pre-formatted the substantive hits into the Per-email-tool data section below. Do NOT call any further tools. Use ONLY the verified tools listed in the extra rules.

Use this exact format:

## Email Tools Used
- List each verified tool with approximate time spent (e.g. "Outlook: ~35min", "Gmail (web): ~12min").
- Estimate time from the "active window" span and hit count shown in the data; if a tool has only 1 hit say "~few min".

## Top Senders / Threads
- Up to 5 bullets. Each bullet: sender or thread subject (literal text from OCR / window title), tool, approximate time of last interaction (HH:MM).
- Skip if the data does not show a clear sender or subject.

## Drafts in Progress
- Emails that look like they were being composed (subject typed, partial body) but not clearly sent.
- Bullet format: "<tool> — to <recipient or unknown> — <subject> — <one-line gist>". Skip if none.

## Action Items
- Things the data suggests need a follow-up: unanswered messages, "reply later" flags, calendar requests visible on screen.
- Skip if none.

## Todolist
- Structured tasks extracted from email subjects, snippets, OCR or UI text. Only include items that are clearly actionable.
- Bullet format (one per line, GitHub-style checkbox): `- [ ] <task> — from <sender or thread> — due <date or "no date"> — <tool>`
- Use literal phrasing from the data when possible; never invent task content, senders, recipients, or due dates.
- Skip this section entirely if no actionable items can be extracted.

## Patterns
- One short paragraph: do I batch email or check constantly? Native client vs. webmail mix? Heaviest hour?

If no email usage is found, say so clearly in one sentence and skip the empty sections. End with: "**Tip:** [one concrete suggestion to reduce inbox time or improve triage, grounded in what the data shows]".

Only report what you can verify from the data. Do not invent senders, subjects, recipients, or drafts.
