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
- 4–8 bullets: the key points the teacher / courseware / video **explained**.
- Prefer content from **Audio transcripts (lecture)** and courseware OCR slides.
- Each bullet: the concept name + one sentence of what was taught + cite source
  as `录音` / `课件OCR` / session id like `[1]`.
- Extract formulas, definitions, steps, contrasts (e.g. A vs B) when present.
- If transcripts/OCR are insufficient, write: "材料不足以还原完整讲解" and list
  only what is supported.

## 理解要点
- 3–6 bullets: **how to understand / remember** those points (intuition, analogy,
  common pitfalls, relationship to practice/code errors in the slice).
- Must connect to 讲解重点 — do not invent a textbook chapter that was not observed.
- Cite `录音` / `课件OCR` / `[session]` / error text when grounding a pitfall.

## 复习重点
- 3–5 concrete points to review, ranked by how often they appeared or blocked progress.
- Each bullet must cite a session id, 录音, or file/URL from the data.

## 下一步学习计划
- 3 ordered next actions for the next study block (30–90 minutes each idea).
- Make them actionable (re-watch X segment, re-derive Y, finish Z exercise, fix error W).
- Tie each step to evidence from the slice — no generic "keep learning" advice.

## 数据说明
- Time window, session count, whether audio was used for 讲解重点, and any gap
  (no audio, OCR-only slides, coding without courseware, detector found nothing).

Only report what the data supports. Write in the user's language (Chinese).
Maximize concrete courseware/lecture content when evidence exists.
