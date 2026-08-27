---
schedule: manual
enabled: true
template: true
title: 学习复盘
description: "从课件 OCR 与课堂录音总结课程内容、讲解重点、掌握状态与后续复习计划"
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

This is a **learning recap, not a transcript digest**. Before writing, silently
cluster repeated evidence by topic, remove greetings/transitions/repetition,
and reconstruct the course into a topic map, concrete explanations, and key
takeaways. Then separately assess the learner's demonstrated understanding or
unresolved questions and derive review actions. 不得按时间顺序复述转录，
不得把讲者的每句话换一种说法后排列输出。直接引用最多 2 处，每处不超过一句；其余内容必须跨多条证据归纳。
Hearing or viewing a concept is only exposure, not proof of mastery. Only mark
something 已掌握 when practice, a correct explanation, or successful problem-solving
supports it; otherwise mark it 待确认. Use `已掌握 / 待确认` explicitly where evidence
allows, and never manufacture a learner state from lecture content alone.

The course-content sections (`课程总结`, `主要内容`, `讲了什么`, `课程重点`,
`知识图谱`) are the core of the report and should contain most of its useful detail. Do not
replace them with activity statistics, mastery labels, or generic study advice.

Use this exact format:

## 是否在学习
One short paragraph: yes/no, total learning minutes, dominant kinds
(courseware_view / material_query / code_edit / problem). Cite session ids.
Mention whether audio transcripts were available.

## 课程总结
- Write one coherent 3–6 sentence overview answering: 这门课/这段课程在解决什么问题，
  核心思路是什么，最终得到什么结论或能力。
- Name the course/material and its central theme when evidence supports them.
- Merge repeated evidence and summarize across the whole session; do not list timestamps,
  apps, isolated transcript sentences, or study-management metadata here.

## 主要内容
- Give a 3–7 item topic map of the course. Each bullet has a clear topic title and
  one-sentence scope: what question that part addresses.
- Prefer: LLM `主题:` / subtopics (with confidence) → `结构:定义` → `结构:步骤`
  → `结构:关系`, then remaining Audio / Courseware OCR.
- Merge duplicate mentions of the same topic. Organize by logical dependency or
  importance, never by transcript timestamp.

## 讲了什么
- Explain the actual course content under 3–7 topic bullets. For each topic, state
  the definition/mechanism, how it works or is used, and an example/contrast when present.
- Be technically specific: name the concrete algorithm/API/data structure/hardware and
  keep any number, precision, or metric from the evidence (如 INT4、FP16、68%、tokens/s、
  CPU/GPU/NPU). 不要停留在「介绍了新特性」这类目录式描述。
- This must be a synthesized explanation a learner can reread without the transcript,
  not merely topic names, activity records, or phrases such as “老师提到了……”.
- Each bullet must be grounded with `主题:` / `结构:*` and/or `录音` / `课件OCR` /
  `[session]`; combine multiple evidence lines into one explanation.
- Every name in the context block `MUST COVER in 讲了什么 / 课程重点 / 知识图谱`
  must appear here, or this section must say `材料不足` for that name. Do not
  drop a short topic because a longer one was discussed more.
- Prefer `课件标题:` spellings over ASR near-misses of the same name.
- Do not invent topics/definitions/steps/relations absent from the pre-computed
  block or direct OCR/audio quotes.
- If structure + transcripts/OCR are thin: "材料不足以还原完整讲解" and list
  only what is supported.

## 课程重点
- Extract 3–6 high-value takeaways the learner should retain: key definitions,
  mechanisms, ordered steps, important distinctions, constraints, or common pitfalls.
- Each point states both `重点是什么` and `为什么重要`; avoid repeating the topic map.
- Keep the technical specifics that make the point verifiable — concrete technique names
  and any number/spec/precision from the evidence, not a generic paraphrase.
- Ground each point in `结构:*` / `录音` / `课件OCR` / `[session]`.
- Include every MUST COVER name that is a takeaway, or say why it is not.

## 知识图谱
- 只使用当前 session 的 `主题:*`、`结构:关系` 和 `Current-session concept graph
  edges`。不得混入其他 session 或全局历史图谱中的概念和关系。
- Copy that extracted graph into this section. Do **not** replace it with a
  shorter invented graph of "major" topics.
- First list 3–10 core nodes as `节点：A、B、C`; include every MUST COVER name
  that has an extracted node, plus any other concepts supported
  by this session's audio/OCR or pre-computed structure.
- Then list directed edges as `A --先决/相关/对比/导致--> B`, followed by one short
  explanation and a `图谱:*` or `结构:关系` citation.
- Map relations as: `prerequisite=先决`, `related=相关`, `contrasts=对比`,
  `leads_to=导致`. Preserve edge direction exactly as supplied.
- If there are concepts but no trustworthy edges, list the nodes and say
  `未抽取到可靠的概念关系`; do not invent edges to make the graph look complete.
- If course evidence is insufficient, say `材料不足，无法生成该 session 的知识图谱`.

## 掌握状态
- Group 3–6 bullets as `已掌握 / 待确认`. Treat lecture exposure alone as
  `待确认`; use `已掌握` only with behavioral evidence such as practice, a correct
  explanation, a resolved error, or a completed problem.
- For `待确认`, state the exact concept/question and what evidence is missing.
- Include code practice, files/problems, and resolved or unresolved errors when present.
- Cite `[session]`, practice/error text, `复习队列`, or `问题队列` when grounding.

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
- **Transcript coverage is mandatory here.** Read `Audio transcript lines
  available:` in the context and state it. If it says `sampled evenly` or a
  `⚠️ PARTIAL TRANSCRIPT` / `⚠️ PARTIAL SLIDES` block is present, you MUST say
  coverage was partial (give shown/total) and MUST NOT claim the lecture outline
  is complete — the un-shown lines are skipped content, not silence.
- If `⚠️ ORDER NOT GUARANTEED` is present, say the teaching order could not be
  established and do not present 讲解重点 as a sequence.

Only report what the data supports. Write in the user's language (Chinese).
Maximize concrete courseware/lecture content when evidence exists.
