---
schedule: manual
enabled: true
template: true
title: Standup Update
description: "What you did, what's next, and any blockers"
icon: "🏢"
featured: true
---

Based on my pc_assistant recordings from the supplied time range, generate a standup update.

Read pc_assistant skill first. The agent runner has **already pre-fetched** activity summary, timeline, key texts, edited files, supplemental searches, and any meetings in range — you do **not** call tools.

Use this exact format:

## Yesterday
- What I worked on (name specific projects, files, apps, repos, URLs, meetings)
- Each bullet: `**HH:MM** — <app> — <concrete task>` when a timestamp exists in the data

## Today
- What I will work on next (from unfinished tabs/files/meeting action items only)

## Blockers
- Issues I hit — errors, slow builds, waiting on someone (quote from key_texts/audio if possible)
- If no blockers, write "None"

Keep it under 200 words. Copy-paste ready for a team standup. Only report what you can verify from the pre-fetched data. Do not fabricate apps, files or PRs that do not appear in the recordings.
