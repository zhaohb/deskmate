const state = {
  health: null,
  frames: [],
  events: [],
  config: null,
  monitors: [],
};

const titles = {
  overview: ["总览", "健康状态、活动参数和最近记录"],
  search: ["搜索", "跨 OCR、UIA、UI 事件和音频转写检索"],
  timeline: ["时间线", "浏览最近帧并查看截图与元数据"],
  events: ["事件", "键鼠、剪贴板、窗口焦点和捕获事件"],
  settings: ["配置", "当前配置与监视器信息"],
};

const $ = (selector) => document.querySelector(selector);

function api(path, options = {}) {
  return fetch(path, {
    headers: { "content-type": "application/json" },
    ...options,
  }).then(async (response) => {
    const contentType = response.headers.get("content-type") || "";
    const body = contentType.includes("application/json")
      ? await response.json()
      : await response.text();
    if (!response.ok) {
      throw new Error(typeof body === "string" ? body : body.error || body.detail || response.statusText);
    }
    return body;
  });
}

function formatTime(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString();
}

function textPreview(value, max = 140) {
  const text = String(value || "").replace(/\s+/g, " ").trim();
  return text.length > max ? `${text.slice(0, max)}…` : text || "-";
}

function renderList(target, items, render, emptyText = "暂无数据") {
  const el = $(target);
  el.innerHTML = "";
  el.classList.toggle("empty", items.length === 0);
  if (!items.length) {
    el.textContent = emptyText;
    return;
  }
  for (const item of items) {
    el.appendChild(render(item));
  }
}

function listItem({ title, subtitle, actionLabel = "查看", onAction }) {
  const tpl = $("#itemTemplate").content.cloneNode(true);
  tpl.querySelector(".item-title").textContent = title;
  tpl.querySelector(".item-subtitle").textContent = subtitle;
  const button = tpl.querySelector(".item-action");
  button.textContent = actionLabel;
  if (onAction) {
    button.addEventListener("click", onAction);
  } else {
    button.remove();
  }
  return tpl;
}

function setView(name) {
  for (const el of document.querySelectorAll(".view")) {
    el.classList.toggle("active", el.id === `view-${name}`);
  }
  for (const el of document.querySelectorAll(".nav-item")) {
    el.classList.toggle("active", el.dataset.view === name);
  }
  $("#viewTitle").textContent = titles[name][0];
  $("#viewSubtitle").textContent = titles[name][1];
}

async function refreshAll() {
  const [health, frames, events, config, monitors] = await Promise.all([
    api("/health"),
    api("/frames?limit=30"),
    api("/events/recent?limit=50"),
    api("/config"),
    api("/monitors"),
  ]);
  state.health = health;
  state.frames = frames;
  state.events = events;
  state.config = config;
  state.monitors = monitors;
  renderHealth();
  renderFrames();
  renderEvents();
  renderSettings();
}

function renderHealth() {
  const h = state.health || {};
  $("#healthStatus").textContent = h.status || "unknown";
  $("#frameCount").textContent = h.frames ?? "-";
  $("#transcriptCount").textContent = h.transcripts ?? "-";
  $("#eventCount").textContent = h.events ?? "-";
  $("#healthMessage").textContent = h.message || h.status || "-";
  $("#healthDetails").textContent = `帧状态: ${h.frame_status || "-"} / 音频状态: ${h.audio_status || "-"}`;
  $("#activityInterval").textContent = h.activity?.recommended_interval_ms ?? "-";
  $("#schemaVersion").textContent = h.schema_version || "-";
}

function renderFrames() {
  const frames = state.frames || [];
  const render = (frame) => listItem({
    title: `${formatTime(frame.timestamp)} · ${frame.app_name || "unknown"}`,
    subtitle: `${frame.window_name || ""} ${textPreview(frame.ocr_text || frame.accessibility_text, 90)}`,
    actionLabel: "详情",
    onAction: () => selectFrame(frame.id || frame.frame_id),
  });
  renderList("#recentFrames", frames.slice(0, 8), render);
  renderList("#timelineFrames", frames, render);
}

function renderEvents() {
  const events = state.events || [];
  const render = (event) => listItem({
    title: `${formatTime(event.timestamp)} · ${event.event_type || "event"}`,
    subtitle: `${event.app_name || ""} ${event.window_title || ""} ${textPreview(rawEventData(event), 220)}`,
    actionLabel: "详情",
    onAction: () => selectEvent(event),
  });
  renderList("#recentEventsOverview", events.slice(0, 8), render);
  renderList("#eventsList", events, render);
}

function rawEventData(event) {
  if (event.data_json == null || event.data_json === "") return "-";
  if (typeof event.data_json === "string") return event.data_json;
  try {
    return JSON.stringify(event.data_json);
  } catch {
    return String(event.data_json);
  }
}

function parsedEventData(event) {
  if (event.data_json == null || event.data_json === "") return null;
  if (typeof event.data_json !== "string") return event.data_json;
  try {
    return JSON.parse(event.data_json);
  } catch {
    return event.data_json;
  }
}

function selectEvent(event) {
  setView("events");
  $("#eventDetails").textContent = JSON.stringify({
    timestamp: event.timestamp,
    event_type: event.event_type,
    app_name: event.app_name,
    window_title: event.window_title,
    browser_url: event.browser_url,
    frame_id: event.frame_id,
    data_json: parsedEventData(event),
  }, null, 2);
}

function renderSettings() {
  $("#configDump").textContent = JSON.stringify(state.config || {}, null, 2);
  renderList("#monitorList", state.monitors || [], (monitor) => listItem({
    title: `${monitor.name} (${monitor.width}×${monitor.height})`,
    subtitle: `id=${monitor.id} stable_id=${monitor.stable_id} default=${monitor.is_default}`,
    actionLabel: "",
  }));
}

async function selectFrame(frameId) {
  if (!frameId) return;
  setView("timeline");
  const detail = await api(`/frames/${frameId}`);
  $("#frameDetails").textContent = JSON.stringify(detail, null, 2);
  const img = $("#framePreview");
  if (detail.snapshot_path) {
    img.src = `/frames/${frameId}/image?ts=${Date.now()}`;
    img.classList.add("visible");
  } else {
    img.removeAttribute("src");
    img.classList.remove("visible");
  }
}

async function runSearch(event) {
  event.preventDefault();
  const q = $("#searchInput").value.trim();
  const contentType = $("#contentType").value;
  if (!q) return;
  const params = new URLSearchParams({ q, content_type: contentType, limit: "50" });
  const result = await api(`/search?${params.toString()}`);
  renderSearchResults(result.data || []);
}

function renderSearchResults(items) {
  renderList("#searchResults", items, (item) => {
    const content = item.content || {};
    const title = `${item.type} · ${formatTime(content.timestamp)}`;
    const subtitle = textPreview(
      content.text || content.transcription || content.window_title || JSON.stringify(content),
      180,
    );
    return listItem({
      title,
      subtitle,
      actionLabel: content.frame_id ? "帧" : "",
      onAction: content.frame_id ? () => selectFrame(content.frame_id) : null,
    });
  }, "没有结果");
}

async function captureNow() {
  const result = await api("/capture", { method: "POST" });
  await refreshAll();
  if (result.frame_ids?.[0]) {
    await selectFrame(result.frame_ids[0]);
  }
}

function wireEvents() {
  for (const button of document.querySelectorAll(".nav-item")) {
    button.addEventListener("click", () => setView(button.dataset.view));
  }
  $("#refreshButton").addEventListener("click", () => refreshAll().catch(showError));
  $("#captureButton").addEventListener("click", () => captureNow().catch(showError));
  $("#searchForm").addEventListener("submit", (event) => runSearch(event).catch(showError));
}

function showError(error) {
  console.error(error);
  $("#healthStatus").textContent = "错误";
  $("#healthMessage").textContent = "API 调用失败";
  $("#healthDetails").textContent = error.message || String(error);
}

wireEvents();
refreshAll().catch(showError);
setInterval(() => refreshAll().catch(showError), 10_000);
