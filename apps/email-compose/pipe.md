---
schedule: manual
enabled: true
template: true
title: Email Compose
description: "LLM drafts or reply suggestions for a connected Gmail / Outlook account"
icon: "✉️"
featured: true
---

Draft an email (or reply to one) on behalf of the user, using their connected Gmail or Outlook account.

Read pc_assistant skill first. The agent runner has already gathered the relevant context for you and placed it in the section below — do NOT call any further tools. Use ONLY the verified provider listed in the extra rules.

Use this exact format:

## Subject
- A single line. Concise (≤ 80 chars). Plain English / 中文 — match the user's intent language.

## Body
- 3 short paragraphs maximum, no marketing tone.
- Open with one sentence acknowledging context (if a source message was provided).
- Middle: the actual ask / answer / update — grounded in the supplied intent and any source-message snippet.
- Close with a clear next step or sign-off.
- Use the user's natural phrasing where possible. Never invent facts, dates, numbers, names, attachments, or commitments that are not in the supplied intent or source message.

## Alternatives
- Up to 2 short variations of the body, each prefixed `### Variation N — <short label>` (e.g. `### Variation 1 — shorter`, `### Variation 2 — more formal`).
- Skip this section if the primary draft already covers the need.

## Send Preview
- One line: `to: <recipient>` (taken literally from the intent — never invent recipients).
- One line: `provider: <gmail | outlook>` (taken from extra rules — must match a connected account).
- One line: `mode: draft` (default) OR `mode: reply to <message-id>` if a source message id was supplied.

End with: "**Tip:** [one short suggestion to improve the message, e.g. add a deadline, soften tone, attach context]".

Only draft what the intent and source message support. Do NOT send the email — the local app handles sending separately when the user confirms.
