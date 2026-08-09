---
schedule: manual
enabled: true
template: true
title: 学习复盘
description: "检测学习阶段，从课件 OCR 与课堂录音归纳讲解重点/理解要点，并给出复习与下一步计划"
icon: "📚"
featured: true
---

Synthesize a LEARNING REPORT from the pre-fetched **learning slice** in the
Context. DeskMate already decided which screen/audio evidence belongs to study
sessions — you must NOT turn random browsing or chat into a study narrative.

Read the DeskMate skill first.

Prioritize **courseware content** and **audio transcriptions** (课堂/讲解录音).
Quote or paraphrase concrete terms that appear in OCR or transcripts. If the
context says `NO_LEARNING_SESSION`, say so clearly and do not invent coursework.
If audio/OCR is thin, say which sections lack evidence — never invent lecture
content.

Use this exact format:

## 是否在学习
One short paragraph: yes/no, total learning minutes, dominant kinds
(courseware_view / material_query / code_edit / problem). Cite session ids.
Mention whether audio transcripts were available.

## 学习摘要
- What the student studied (topics, courseware titles, materials). Name apps/URLs/files.
- What they practiced in code (files / problems), if any.
- Problems / errors they hit (quote short error text when present).

## 讲解重点
- 4–8 bullets grounded on **Pre-computed learning structure** when present.
- Prefer: LLM `主题:` / subtopics (with confidence) → `结构:定义` → `结构:步骤`
  → `结构:关系`, then remaining Audio / Courseware OCR.
- Each bullet: topic/concept + one taught sentence + cite `主题:` / `结构:*`
  and/or `录音` / `课件OCR` / `[session]`.
- Do not invent topics/definitions/steps/relations absent from the pre-computed
  block or direct OCR/audio quotes.
- If structure + transcripts/OCR are thin: "材料不足以还原完整讲解" and list
  only what is supported.

## 理解要点
- 3–6 bullets: **how to understand / remember** the same subjects as 讲解重点
  (intuition, analogy, common pitfalls, link to practice/code errors).
- Prefer explaining extracted concepts / definition subjects — not new chapters.
- Cite `结构:*` / `录音` / `课件OCR` / `[session]` / error text when grounding.

## 复习重点
- 3–5 points ranked by **复习队列** urgency: OVERDUE first, then WEAK /
  `exposure`/`recognition` mastery_tier, then hit count.
- Include open **问题队列** items when present.
- Each bullet: concept/problem + why now + cite `复习队列` / `问题队列` / `图谱:`.

## 下一步学习计划
- 3 ordered next actions for the next study block (30–90 minutes each).
- Start with OVERDUE / WEAK / low-tier concepts and open 问题队列 items;
  use `图谱:先决` to order prerequisites before dependents.
- Each step executable and trackable (review topic T, re-derive step S,
  contrast relation R, fix problem P).
- If queues empty: derive gentle steps from LLM topics / concepts only.
- No generic "keep learning" advice.

## 数据说明
- Time window, session count, whether audio was used for 讲解重点, and any gap
  (no audio, OCR-only slides, coding without courseware, detector found nothing).

Only report what the data supports. Write in the user's language (Chinese).
Maximize concrete courseware/lecture content when evidence exists.
