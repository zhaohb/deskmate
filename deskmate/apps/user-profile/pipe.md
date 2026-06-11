---
schedule: manual
enabled: true
template: true
title: 用户画像
description: "从近期活动总结你的角色、兴趣、工作习惯与协作风格（本地生成）"
icon: "🪪"
featured: true
---

Synthesize a USER PROFILE from the pre-fetched activity, habits, meetings, and
(if connected) email data in the Context. This is a multi-day portrait of WHO
this user is and HOW they work — not a day log.

Read the DeskMate skill first.

Ground every claim in the data. If a dimension has too little evidence, say so
briefly instead of inventing it. Use this exact format:

## 一句话画像
One sentence capturing who this user is and what they mainly do.

## 角色与职业
- The user's likely role / profession, inferred from the apps, files, and sites
  they use most. Name the concrete evidence (app, file type, repo, domain).

## 兴趣与主题
- The recurring technologies, domains, and topics they engage with. Cite the
  windows / OCR / search terms that show each interest.

## 工作习惯与节奏
- When they are most active, focus vs. context-switching, and their core tool
  chain. Use the habit profile (作息规律) and the per-app minutes — never guess
  durations.

## 沟通与协作
- Who/what they collaborate with: meeting cadence, frequent contacts, and
  communication style. If no email is connected and no meetings were detected,
  say collaboration data is limited and skip rather than fabricate.

## 数据说明
- One line on coverage: the time window analyzed and any gap (e.g. "未连接邮箱，
  协作维度仅基于会议与屏幕记录").

Only report what the data supports. Write in the user's language (Chinese).
