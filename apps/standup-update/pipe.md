---
schedule: manual
enabled: true
template: true
title: Standup Update
description: "What you did, what's next, and any blockers"
icon: "🏢"
featured: true
---

Based on my pc_assistant recordings from the last 24 hours, generate a standup update.

Read pc_assistant skill first. Call `activity_summary` once with the supplied time range, and use at most 2 follow-up `search` calls (limit=5 each) only if you need to confirm a specific blocker or unfinished task.

Use this exact format:

## Yesterday
- What I worked on (name specific projects, files, apps, repos, PRs)

## Today
- What I will work on next (based on unfinished tasks and recent activity)

## Blockers
- Issues I hit — errors, slow builds, waiting on someone
- If no blockers, write "None"

Keep it under 150 words. Copy-paste ready for a team standup. Only report what you can verify from the data. Do not fabricate apps, files or PRs that do not appear in the recordings.
