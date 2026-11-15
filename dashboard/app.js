const MAX_POINTS = 120;
const TARGET_MS = 500;
const $ = (id) => document.getElementById(id);

const state = {
  gestures: {},
  voiceOn: false,
  ttsMode: "auto",
  viVoice: null,
  latest: null,
  dirty: false,
  e2e: [],
  transcript: [],
  lastSeen: 0,
  device: null,
  mode: null,
  modes: {},
  models: {},
  markModel: null,
  guide: [],
  speakLetters: false,
  activeTab: "live",
  activeSubtab: {
    live: "live-studio",
    telemetry: "tel-sensors",
    ai: "ai-perf",
    crm: "crm-translations",
    guide: "help-quickstart",
  },
  historyRows: [],
  rawLogs: [],
  dataset: null,
};

// ------------------------------------------------------------------ navigation (tabs & subtabs)
const allCharts = [];

function resizeVisibleCharts() {
  requestAnimationFrame(() => {
    allCharts.forEach((c) => {
      if (c && c.canvas && c.canvas.offsetParent !== null) {
        c.resize();
        c.update("none");
      }
    });
  });
}

function switchTab(tabId) {
  if (!tabId) return;
  state.activeTab = tabId;
  document.querySelectorAll(".nav-tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === tabId));
  document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${tabId}`));
  resizeVisibleCharts();
  if (tabId === "crm") {
    refreshCrmData();
  }
}

function switchSubtab(parentTab, subId) {
  if (!subId) return;
  state.activeSubtab[parentTab] = subId;
  const panel = $(`tab-${parentTab}`);
  if (!panel) return;
  panel.querySelectorAll(".subtab").forEach((b) => b.classList.toggle("active", b.dataset.subtab === subId));
  panel.querySelectorAll(".subpanel").forEach((sp) => sp.classList.toggle("active", sp.id === `sub-${subId}`));
  resizeVisibleCharts();
  if (subId === "crm-sensor-logs") fetchRawLogs();
  if (subId === "crm-dataset") fetchDatasetSummary();
}

document.querySelectorAll(".nav-tab").forEach((btn) => {
  btn.onclick = () => switchTab(btn.dataset.tab);
});

document.querySelectorAll(".subtabs").forEach((bar) => {
  const parent = bar.dataset.parent;
  bar.querySelectorAll(".subtab").forEach((btn) => {
    btn.onclick = () => switchSubtab(parent, btn.dataset.subtab);
  });
});

document.querySelectorAll("[data-goto-tab]").forEach((btn) => {
  btn.onclick = () => switchTab(btn.dataset.gotoTab);
});

document.querySelectorAll("[data-goto-sub]").forEach((btn) => {
  btn.onclick = () => {
    switchTab("live");
    switchSubtab("live", btn.dataset.gotoSub);
  };
});

// ------------------------------------------------------------------ charts
Chart.defaults.color = "#1e3a52";
Chart.defaults.borderColor = "rgba(7, 34, 54, 0.12)";
Chart.defaults.font.size = 11;

function lineChart(id, datasets, opts = {}) {
  const c = new Chart($(id), {
    type: "line",
    data: {
      labels: [],
      datasets: datasets.map((d) => ({ borderWidth: 2, pointRadius: 0, tension: 0.25, data: [], ...d })),
    },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      interaction: { intersect: false, mode: "index" },
      plugins: { legend: { labels: { boxWidth: 10 } } },
      scales: { x: { display: false }, y: opts.y || {} },
    },
  });
  allCharts.push(c);
  return c;
}

const flexChart = lineChart("chart-flex", [
  { label: "Ngón cái", borderColor: "#f472b6" },
  { label: "Ngón trỏ", borderColor: "#38bdf8" },
  { label: "Ngón giữa", borderColor: "#a78bfa" },
  { label: "Áp út", borderColor: "#34d399" },
  { label: "Ngón út", borderColor: "#fbbf24" },
], { y: { min: 0, max: 180, title: { display: true, text: "Góc uốn (độ)" } } });

const imuChart = lineChart("chart-imu", [
  { label: "Roll", borderColor: "#38bdf8", yAxisID: "y" },
  { label: "Pitch", borderColor: "#a78bfa", yAxisID: "y" },
  { label: "Tốc độ", borderColor: "#f472b6", yAxisID: "y1" },
  { label: "|a|", borderColor: "#fbbf24", yAxisID: "y1" },
]);
imuChart.options.scales = {
  x: { display: false },
  y: { position: "left", min: -180, max: 180, title: { display: true, text: "Góc (độ)" } },
  y1: { position: "right", min: 0, grid: { drawOnChartArea: false }, title: { display: true, text: "palm/s" } },
};

const latencyChart = lineChart("chart-latency", [
  { label: "End-to-End", borderColor: "#38bdf8", fill: true, backgroundColor: "rgba(56,189,248,.12)" },
  { label: "Edge → Cloud", borderColor: "#34d399" },
  { label: "Ngưỡng 500ms", borderColor: "#ef4444", borderDash: [6, 4], borderWidth: 1 },
], { y: { min: 0, suggestedMax: 550, title: { display: true, text: "ms" } } });

const rateChart = lineChart("chart-rate", [
  { label: "msg/s", borderColor: "#a78bfa", fill: true, backgroundColor: "rgba(167,139,250,.14)" },
], { y: { min: 0 } });
rateChart.options.plugins.legend.display = false;

const freqChart = new Chart($("chart-freq"), {
  type: "bar",
  data: { labels: [], datasets: [{ label: "Số lần dịch", data: [], backgroundColor: "#38bdf8", borderRadius: 5 }] },
  options: {
    animation: false,
    responsive: true,
    maintainAspectRatio: false,
    indexAxis: "y",
    plugins: { legend: { display: false } },
    scales: { x: { beginAtZero: true, ticks: { precision: 0 } } },
  },
});
allCharts.push(freqChart);

function push(chart, values) {
  chart.data.labels.push("");
  values.forEach((v, i) => chart.data.datasets[i].data.push(v));
  if (chart.data.labels.length > MAX_POINTS) {
    chart.data.labels.shift();
    chart.data.datasets.forEach((d) => d.data.shift());
  }
}

// ------------------------------------------------------------------ speech
function pickVoice() {
  const voices = window.speechSynthesis ? speechSynthesis.getVoices() : [];
  state.viVoice = voices.find((v) => v.lang && v.lang.toLowerCase().startsWith("vi")) || null;
}
if (window.speechSynthesis) {
  pickVoice();
  speechSynthesis.onvoiceschanged = pickVoice;
}

const audioCache = {};
function speak(text, force = false) {
  if ((!state.voiceOn && !force) || !text) return;
  const useBrowser = state.ttsMode === "browser" || (state.ttsMode === "auto" && state.viVoice);
  if (useBrowser && window.speechSynthesis) {
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = "vi-VN";
    if (state.viVoice) u.voice = state.viVoice;
    u.rate = 1.05;
    speechSynthesis.speak(u);
    return;
  }
  const key = text.toLowerCase();
  const audio = audioCache[key] || (audioCache[key] = new Audio(`/api/tts_text?text=${encodeURIComponent(text)}`));
  audio.currentTime = 0;
  audio.play().catch(() => {});
}

$("btn-voice").onclick = () => {
  state.voiceOn = !state.voiceOn;
  $("btn-voice").classList.toggle("on", state.voiceOn);
  $("btn-voice").textContent = state.voiceOn ? "Giọng nói: BẬT" : "Bật giọng nói";
  if (state.voiceOn) speak("Đã bật giọng nói tiếng Việt", true);
};
$("tts-mode").onchange = (e) => (state.ttsMode = e.target.value);
$("speak-letters").onchange = (e) => (state.speakLetters = e.target.checked);

$("btn-clear").onclick = () => {
  state.transcript = [];
  renderTranscript();
};

$("btn-copy-transcript").onclick = async () => {
  if (!state.transcript.length) return;
  await navigator.clipboard.writeText(state.transcript.join(" "));
  const btn = $("btn-copy-transcript");
  const old = btn.textContent;
  btn.textContent = "Đã sao chép";
  setTimeout(() => (btn.textContent = old), 1200);
};

$("btn-speak-all").onclick = () => {
  if (!state.transcript.length) return;
  speak(state.transcript.join(" ").toLowerCase(), true);
};

// ------------------------------------------------------------------ controls & mode
function sendControl(action, mode) {
  if (!ws || ws.readyState !== 1) return;
  ws.send(JSON.stringify({ type: "control", action, mode, device_id: state.device }));
}
document.querySelectorAll("#mode-switch button").forEach((b) => (b.onclick = () => sendControl("set_mode", b.dataset.mode)));
document.querySelectorAll("#spell-panel [data-action]").forEach((b) => (b.onclick = () => sendControl(b.dataset.action)));

function addTranscript(text) {
  state.transcript.push(text);
  if (state.transcript.length > 20) state.transcript.shift();
  renderTranscript();
}

function setMode(mode) {
  if (!mode || mode === state.mode) return;
  state.mode = mode;
  document.querySelectorAll("#mode-switch button").forEach((b) => {
    b.classList.toggle("active", b.dataset.mode === mode);
    b.disabled = state.modes[b.dataset.mode] ? !state.modes[b.dataset.mode].available : false;
  });
  const isLetters = mode === "letters";
  $("spell-panel").classList.toggle("hidden", !isLetters);
  $("ref-image").classList.toggle("hidden", !isLetters);
  $("words-ref-note").classList.toggle("hidden", isLetters);
  $("result-text").textContent = isLetters ? "Hãy đánh vần..." : "Đang chờ cử chỉ...";
  $("result-text").classList.add("muted");
  $("ribbon-mode").textContent = isLetters ? "Đánh vần Chữ cái" : "Cử chỉ Từ ngữ";
  renderGuide();
  renderModel();
}

function renderSpelling(sp) {
  if (!sp) return;
  $("spell-current").textContent = sp.current || "...";
}

function renderHold(hold) {
  const bar = $("hold-bar");
  if (!hold) {
    bar.style.width = "0%";
    bar.classList.remove("done");
    $("hold-text").textContent = "Giữ yên tay để nhận chữ";
    return;
  }
  bar.style.width = `${hold.progress * 100}%`;
  bar.classList.toggle("done", hold.progress >= 1);
  $("hold-text").innerHTML = hold.progress >= 1
    ? `Đã nhận: <b>${hold.text}</b>`
    : `Đang giữ yên: <b>${hold.text}</b> (${Math.round(hold.progress * 100)}%)`;
}

function setHoldMs(ms) {
  if (!ms) return;
  $("hold-ms").value = ms;
  $("hold-ms-val").textContent = `${(ms / 1000).toFixed(1)}s`;
}
$("hold-ms").oninput = (e) => ($("hold-ms-val").textContent = `${(e.target.value / 1000).toFixed(1)}s`);
$("hold-ms").onchange = (e) => {
  if (ws && ws.readyState === 1) {
    ws.send(JSON.stringify({ type: "control", action: "set_hold", value: Number(e.target.value), device_id: state.device }));
  }
};

function onMark(mark) {
  if (!mark) return;
  state.markUntil = Date.now() + 1500;
  $("result-label").innerHTML = `Đã ghép dấu: <span class="mark-chip">${mark.text}</span> (${(mark.confidence * 100).toFixed(0)}%)`;
  flashResult();
  if (state.speakLetters) speak(`dấu ${mark.text}`);
}

function renderTranscript() {
  if (!state.transcript.length) {
    $("transcript").innerHTML = `<em class="placeholder">Các từ sau khi chốt sẽ nối thành câu tại đây...</em>`;
    return;
  }
  $("transcript").innerHTML = state.transcript.map((t) => `<span>${escapeHtml(t)}</span>`).join("");
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ------------------------------------------------------------------ websocket
let ws;
function connect() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onopen = () => setPill("pill-ws", true, "WebSocket");
  ws.onclose = () => {
    setPill("pill-ws", false, "WebSocket");
    setTimeout(connect, 1500);
  };
  ws.onmessage = (ev) => handle(JSON.parse(ev.data));
}

function setPill(id, on, text) {
  const el = $(id);
  el.classList.toggle("on", on);
  el.innerHTML = `<i class="dot"></i>${escapeHtml(text)}`;
}

function setHandDetected(detected) {
  const txt = detected ? "Đã phát hiện tay" : "Chưa thấy tay";
  $("result-hand").textContent = txt;
  $("result-hand").classList.toggle("on", !!detected);
  $("ribbon-hand").textContent = txt;
  $("ribbon-hand").classList.toggle("ok", !!detected);
}

function handle(msg) {
  if (msg.type === "hello") {
    state.gestures = msg.gestures || {};
    state.modes = msg.modes || {};
    state.models = msg.model || {};
    state.device = state.device || msg.default_device;
    state.mode = null;
    setMode((msg.device_modes || {})[state.device] || msg.default_mode);
    setHoldMs(msg.hold_ms);
    renderMqtt(msg.mqtt);
  } else if (msg.type === "state") {
    if (msg.device_id !== state.device) return;
    setMode(msg.mode);
    setHoldMs(msg.hold_ms);
    renderSpelling(msg.spelling);
    onMark(msg.mark);
    if (msg.source === "gesture" && msg.event === "backspace") {
      state.markUntil = Date.now() + 1000;
      $("result-label").innerHTML = `<span class="mark-chip">Đã xóa 1 chữ bằng cử chỉ</span>`;
    }
    if (msg.word) {
      addTranscript(msg.word);
      $("result-text").textContent = msg.word;
      $("result-text").classList.remove("muted");
      flashResult();
      speak(msg.word.toLowerCase());
    }
  } else if (msg.type === "prediction") {
    if (msg.device_id.startsWith("bench") || msg.device_id === "live_test") return;
    state.device = msg.device_id;
    setMode(msg.mode);
    const e2e = Math.max(0, Date.now() - msg.edge_timestamp_ms);
    msg.e2e_ms = e2e;
    if (ws.readyState === 1) {
      ws.send(JSON.stringify({ type: "ack", device_id: msg.device_id, seq_id: msg.seq_id, e2e_ms: e2e }));
    }
    state.e2e.push(e2e);
    if (state.e2e.length > 500) state.e2e.shift();
    state.latest = msg;
    state.lastSeen = Date.now();
    state.dirty = true;
    setHandDetected(true);
    if (msg.mode === "letters") renderHold(msg.hold);
    if (msg.spelling) {
      renderSpelling(msg.spelling);
      if (msg.spelling.added) {
        flashResult();
        if (state.speakLetters) speak(msg.spelling.added);
      }
      onMark(msg.spelling.mark);
    }
    if (msg.speak && msg.speak_text) {
      addTranscript(msg.speak_text);
      flashResult();
      speak(msg.speak_text);
    }
  } else if (msg.type === "metrics") {
    renderMqtt(msg);
  } else if (msg.type === "edge_status") {
    setPill("pill-edge", true, `Edge: ${msg.source} · ${msg.fps} fps`);
    setHandDetected(!!msg.hand_detected);
  }
}

function flashResult() {
  const el = $("result-text");
  el.classList.remove("flash");
  void el.offsetWidth;
  el.classList.add("flash");
}

// ------------------------------------------------------------------ realtime render loop
const fmt = (v, d = 1) => (v === null || v === undefined ? "-" : Number(v).toFixed(d));

function percentile(arr, p) {
  if (!arr.length) return null;
  const s = [...arr].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor((p / 100) * s.length))];
}

function render() {
  requestAnimationFrame(render);
  if (!state.dirty) return;
  state.dirty = false;
  const m = state.latest;

  const resultEl = $("result-text");
  if (m.stable_text) {
    resultEl.textContent = m.stable_text;
    resultEl.classList.remove("muted");
  }
  const pct = Math.round(m.confidence * 100);
  $("conf-pct").textContent = `${pct}%`;
  $("conf-bar").style.width = `${pct}%`;
  $("conf-bar").style.background = m.confidence >= 0.7 ? "var(--ok)" : "var(--warn)";
  if (!(state.markUntil > Date.now())) {
    $("result-label").textContent = `Dự đoán tức thời: ${m.text} (${pct}%)`;
  }
  $("top3").innerHTML = m.top3.map((t) => `<span>${escapeHtml(t.text)} ${(t.confidence * 100).toFixed(0)}%</span>`).join("");
  document.querySelectorAll("#guide li").forEach((li) => li.classList.toggle("active", li.dataset.label === m.stable_label));

  // Latency KPIs + Ribbon
  const e2eTxt = `${fmt(m.e2e_ms, 0)} ms`;
  $("e2e-now").textContent = fmt(m.e2e_ms, 0);
  $("ribbon-e2e").textContent = e2eTxt;
  const okSla = m.e2e_ms < TARGET_MS;
  $("ribbon-sla").className = `kpi-tag ${okSla ? "ok" : "bad"}`;
  $("ribbon-sla").textContent = okSla ? "Đạt SLA < 500 ms" : "Vượt ngưỡng 500 ms";

  const badge = $("e2e-badge");
  badge.className = `badge ${okSla ? "ok" : "bad"}`;
  badge.textContent = okSla ? "Đạt mục tiêu < 500 ms" : "Vượt ngưỡng 500 ms";

  const arr = state.e2e;
  $("e2e-avg").textContent = `${fmt(arr.reduce((a, b) => a + b, 0) / arr.length)} ms`;
  $("e2e-p95").textContent = `${fmt(percentile(arr, 95))} ms`;
  $("e2e-max").textContent = `${fmt(Math.max(...arr))} ms`;
  $("e2e-under").textContent = `${fmt((arr.filter((v) => v < TARGET_MS).length / arr.length) * 100)} %`;
  $("lat-edge").textContent = `${fmt(m.edge_processing_ms)} ms`;
  $("lat-net").textContent = `${fmt(m.network_latency_ms)} ms`;
  $("lat-ai").textContent = `${fmt(m.ai_inference_ms, 2)} ms`;
  $("lat-cloud").textContent = `${fmt(m.cloud_total_ms, 2)} ms`;

  const f = m.sensors.flex, imu = m.sensors.imu;
  push(flexChart, [f.thumb, f.index, f.middle, f.ring, f.pinky]);
  push(imuChart, [imu.roll, imu.pitch, imu.speed, Math.hypot(imu.ax, imu.ay, imu.az) / 10]);
  push(latencyChart, [m.e2e_ms, m.network_latency_ms, TARGET_MS]);
  if (state.activeTab === "telemetry") {
    flexChart.update("none");
    imuChart.update("none");
    latencyChart.update("none");
  }
}
requestAnimationFrame(render);

function renderMqtt(s) {
  if (!s) return;
  setPill("pill-mqtt", s.connected, `MQTT ${s.connected ? "Online" : "Offline"}`);
  $("pill-rate").textContent = `${fmt(s.msg_rate)} msg/s`;
  $("mq-total").textContent = (s.msgs_total ?? 0).toLocaleString("vi-VN");
  $("mq-lost").textContent = s.msgs_lost ?? 0;
  $("mq-pdr").textContent = `${fmt(s.pdr, 2)} %`;
  $("mq-reconnect").textContent = s.reconnects ?? 0;
  $("mq-broker").textContent = `Broker: ${s.broker}`;

  $("ribbon-pdr").textContent = `${fmt(s.pdr, 1)} %`;
  $("ribbon-packets").textContent = `${(s.msgs_total ?? 0).toLocaleString("vi-VN")} gói · mất ${s.msgs_lost ?? 0}`;

  if (s.ts_ms) {
    push(rateChart, [s.msg_rate]);
    if (state.activeTab === "telemetry") rateChart.update("none");
  }
  const online = Object.values(s.devices || {}).some((d) => d.online);
  if (!online) setPill("pill-edge", false, "Edge offline");
}

// ------------------------------------------------------------------ AI models & stats
function renderModel() {
  const model = state.models[state.mode];
  const name = state.modes[state.mode] ? state.modes[state.mode].name : state.mode;
  if (!model) {
    $("acc-model-name").textContent = `${name}: chưa có dữ liệu`;
    $("acc-model").textContent = "-";
    $("ribbon-acc").textContent = "-";
    $("acc-eval").textContent = state.mode === "letters"
      ? "Chưa train chữ cái: chạy collect_data.py camera rồi train_model.py"
      : "-";
  } else {
    $("acc-model-name").textContent = `${name}: ${model.model_name}`;
    const accStr = `${fmt(model.accuracy * 100, 2)} %`;
    $("acc-model").textContent = accStr;
    $("ribbon-acc").textContent = accStr;
    $("ribbon-model-sub").textContent = `${model.model_name} · ${model.labels.length} nhãn (${ fmt(model.inference_ms, 2) } ms)`;

    const missing = (model.missing_labels || []).length;
    const label = (l) => state.gestures[l] || l;
    let text = `${model.labels.length} nhãn · tập test ${model.n_test} mẫu (${model.evaluation})`
      + (missing ? ` · thiếu dữ liệu: ${model.missing_labels.map(label).join(", ")}` : "");
    if (state.mode === "letters") {
      const mk = state.markModel;
      text += mk
        ? ` · Dấu động: ${mk.model_name} ${fmt(mk.accuracy * 100, 1)}% (${mk.n_test} nét test)`
        : " · Dấu động: chưa train";
    }
    $("acc-eval").textContent = text;
  }
  renderAllModelCards();
}

function renderAllModelCards() {
  const items = [
    { key: "letters", title: "1. Mô hình Chữ cái, Dấu tĩnh & Điều khiển", m: state.models.letters },
    { key: "marks", title: "2. Mô hình Quỹ đạo Dấu thanh & Dấu trăng", m: state.markModel },
    { key: "words", title: "3. Mô hình Cử chỉ Từ ngữ giao tiếp", m: state.models.words },
  ];
  $("all-models-cards").innerHTML = items.map(({ title, m }) => {
    if (!m) {
      return `<div class="model-item"><div class="model-item-head"><b>${title}</b><span class="badge warn">Chưa huấn luyện</span></div><p>Chưa có file mô hình trên Cloud.</p></div>`;
    }
    return `
      <div class="model-item">
        <div class="model-item-head">
          <b>${title}</b>
          <span class="badge ok">${m.model_name} · ${fmt(m.accuracy * 100, 2)}%</span>
        </div>
        <p>Số nhãn: <b>${m.labels.length}</b> · Train: <b>${(m.n_train || 0).toLocaleString("vi-VN")}</b> · Test độc lập: <b>${m.n_test}</b> · Suy luận: <b>${fmt(m.inference_ms, 2)} ms/mẫu</b></p>
      </div>`;
  }).join("");
}

$("btn-reload-model").onclick = async () => {
  const btn = $("btn-reload-model");
  btn.disabled = true;
  btn.textContent = "Đang nạp...";
  await fetch("/api/reload_model", { method: "POST" });
  await refreshStats();
  btn.disabled = false;
  btn.textContent = "Đã nạp lại mô hình";
  setTimeout(() => (btn.textContent = "Nạp lại mô hình vừa train"), 1500);
};

// ------------------------------------------------------------------ CRM Data & History
async function refreshStats() {
  try {
    const limit = Number($("crm-limit")?.value || 150);
    const [stats, history] = await Promise.all([
      fetch("/api/stats").then((r) => r.json()),
      fetch(`/api/history?limit=${limit}`).then((r) => r.json()),
    ]);
    state.models = stats.model || {};
    state.markModel = stats.mark_model;
    state.historyRows = history || [];
    renderModel();

    const acc = stats.stream_accuracy;
    $("acc-stream").textContent = acc.accuracy === null ? "-" : `${fmt(acc.accuracy, 2)} %`;
    $("acc-samples").textContent = acc.samples;
    $("acc-per-class").innerHTML = Object.entries(acc.per_class)
      .map(([k, v]) => `<div>${escapeHtml(k)}<b>${fmt(v, 1)}%</b></div>`).join("");

    const totalTranslated = (stats.translations || []).reduce((s, c) => s + c.n, 0);
    $("badge-total-records").textContent = totalTranslated.toLocaleString("vi-VN");
    $("db-total").textContent = `Tổng log cảm biến: ${stats.total_logs.toLocaleString("vi-VN")} · Bản dịch: ${totalTranslated}`;

    const counts = stats.translations || [];
    freqChart.data.labels = counts.map((c) => (c.label === "SPELL" ? "Từ ghép đánh vần" : state.gestures[c.label] || c.label));
    freqChart.data.datasets[0].data = counts.map((c) => c.n);
    if (state.activeTab === "ai") freqChart.update("none");

    renderMiniHistory();
    renderCrmTranslations();
  } catch (e) {
    console.warn("stats", e);
  }
}

function renderMiniHistory() {
  const rows = state.historyRows.slice(0, 8);
  if (!rows.length) {
    $("mini-history").innerHTML = `<li class="empty">Chưa có bản ghi phiên dịch nào.</li>`;
    return;
  }
  $("mini-history").innerHTML = rows.map((h) => `
    <li>
      <div>
        <div class="word">${escapeHtml(h.text)}</div>
        <div class="meta">${h.label === "SPELL" ? "Đánh vần" : escapeHtml(h.label)} · ${new Date(h.ts_ms).toLocaleTimeString("vi-VN")}</div>
      </div>
      <button class="btn ghost xs" data-say="${escapeHtml(h.text)}">Nghe</button>
    </li>`).join("");
  $("mini-history").querySelectorAll("[data-say]").forEach((b) => {
    b.onclick = () => speak(b.dataset.say.toLowerCase(), true);
  });
}

function getFilteredTranslations() {
  const q = ($("crm-search")?.value || "").trim().toLowerCase();
  const type = $("crm-type-filter")?.value || "all";
  return state.historyRows.filter((h) => {
    if (type === "spell" && h.label !== "SPELL") return false;
    if (type === "word" && h.label === "SPELL") return false;
    if (!q) return true;
    return `${h.text} ${h.label} ${h.device_id}`.toLowerCase().includes(q);
  });
}

function renderCrmTranslations() {
  const rows = getFilteredTranslations();
  $("crm-trans-count").textContent = `${rows.length} / ${state.historyRows.length} bản ghi`;
  if (!rows.length) {
    $("history").innerHTML = `<tr><td colspan="8" style="text-align:center;color:var(--muted);padding:22px">Không có bản ghi phù hợp bộ lọc.</td></tr>`;
    return;
  }
  $("history").innerHTML = rows.map((h) => {
    const isSpell = h.label === "SPELL";
    const badge = isSpell
      ? `<span class="type-pill spell">Từ đánh vần</span>`
      : `<span class="type-pill word">${escapeHtml(h.label)}</span>`;
    return `
      <tr>
        <td>#${h.id ?? "-"}</td>
        <td>${new Date(h.ts_ms).toLocaleString("vi-VN")}</td>
        <td><code>${escapeHtml(h.device_id)}</code></td>
        <td>${badge}</td>
        <td><b style="font-size:14px;color:#072236">${escapeHtml(h.text)}</b></td>
        <td>${(h.confidence * 100).toFixed(0)}%</td>
        <td>${h.latency_ms ? `${fmt(h.latency_ms)} ms` : "-"}</td>
        <td><button class="btn ghost xs" data-say="${escapeHtml(h.text)}">Nghe</button></td>
      </tr>`;
  }).join("");
  $("history").querySelectorAll("[data-say]").forEach((b) => {
    b.onclick = () => speak(b.dataset.say.toLowerCase(), true);
  });
}

async function fetchRawLogs() {
  try {
    state.rawLogs = await fetch("/api/logs?limit=200").then((r) => r.json());
    renderRawLogs();
  } catch (e) {
    console.warn("logs", e);
  }
}

function renderRawLogs() {
  const q = ($("logs-search")?.value || "").trim().toLowerCase();
  const rows = state.rawLogs.filter((r) => !q || `${r.label} ${r.stable_label || ""} ${r.device_id} ${r.seq_id}`.toLowerCase().includes(q));
  $("crm-logs-count").textContent = `${rows.length} khung mẫu gần nhất`;
  $("raw-logs-body").innerHTML = rows.map((r) => `
    <tr>
      <td>#${r.seq_id}</td>
      <td><code>${escapeHtml(r.device_id)}</code></td>
      <td><b>${escapeHtml(r.label)}</b></td>
      <td>${r.stable_label ? `<span class="type-pill control">${escapeHtml(r.stable_label)}</span>` : "-"}</td>
      <td>${((r.confidence || 0) * 100).toFixed(0)}%</td>
      <td>${fmt(r.edge_processing_ms)}</td>
      <td>${fmt(r.network_latency_ms)}</td>
      <td>${fmt(r.ai_inference_ms, 2)}</td>
      <td>${fmt(r.e2e_latency_ms)}</td>
    </tr>`).join("");
}

async function fetchDatasetSummary() {
  try {
    state.dataset = await fetch("/api/dataset_summary").then((r) => r.json());
    renderDatasetSummary();
  } catch (e) {
    console.warn("dataset", e);
  }
}

function renderDatasetSummary() {
  const ds = state.dataset;
  if (!ds) return;
  $("ds-sessions").textContent = ds.camera_sessions.toLocaleString("vi-VN");
  $("ds-frames").textContent = ds.camera_frames.toLocaleString("vi-VN");
  $("ds-images").textContent = ds.total_images.toLocaleString("vi-VN");
  $("ds-syn").textContent = ds.synthetic_rows.toLocaleString("vi-VN");

  const kindFilter = $("ds-kind-filter")?.value || "all";
  const kindNames = { letter: "Chữ cái", mark: "Dấu", control: "Điều khiển", word: "Từ ngữ" };
  const items = ds.items.filter((it) => kindFilter === "all" || it.kind === kindFilter);

  $("ds-table-body").innerHTML = items.map((it) => {
    let status = `<span class="badge warn">Chưa thu camera</span>`;
    if (it.kind === "word" && it.sessions === 0) {
      status = `<span class="badge">Dùng mẫu Bàn tay ảo</span>`;
    } else if (it.sessions >= 3) {
      status = `<span class="badge ok">Đủ chuẩn (${it.sessions} phiên)</span>`;
    } else if (it.sessions > 0) {
      status = `<span class="badge warn">Ít mẫu (${it.sessions} phiên)</span>`;
    }
    return `
      <tr>
        <td><code>${escapeHtml(it.label)}</code></td>
        <td><b>${escapeHtml(it.text)}</b></td>
        <td>${kindNames[it.kind] || it.kind}</td>
        <td><b>${it.sessions}</b></td>
        <td>${it.frames.toLocaleString("vi-VN")}</td>
        <td>${it.images}</td>
        <td>${status}</td>
      </tr>`;
  }).join("");

  const imgs = ds.recent_images || [];
  if (!imgs.length) {
    $("ds-gallery").innerHTML = `<p class="hint">Chưa có ảnh chụp trong dataset/images.</p>`;
  } else {
    $("ds-gallery").innerHTML = imgs.slice(0, 12).map((im) => `
      <a class="gallery-card" href="/captures/${encodeURI(im.file)}" target="_blank" title="Bấm để xem ảnh gốc">
        <img src="/captures/${encodeURI(im.file)}" loading="lazy" alt="${escapeHtml(im.text)}">
        <div><b>${escapeHtml(im.text)}</b> · ${escapeHtml(im.kind)}</div>
      </a>`).join("");
  }
}

function refreshCrmData() {
  refreshStats();
  if (state.activeSubtab.crm === "crm-sensor-logs") fetchRawLogs();
  if (state.activeSubtab.crm === "crm-dataset") fetchDatasetSummary();
}

$("crm-search").oninput = renderCrmTranslations;
$("crm-type-filter").onchange = renderCrmTranslations;
$("crm-limit").onchange = refreshStats;
$("logs-search").oninput = renderRawLogs;
$("ds-kind-filter").onchange = renderDatasetSummary;
$("btn-refresh-crm").onclick = refreshCrmData;

$("btn-export-csv").onclick = () => {
  const sub = state.activeSubtab.crm;
  let headers = [], rows = [], name = "export.csv";
  if (sub === "crm-translations") {
    headers = ["id", "time", "device_id", "label", "text", "confidence", "latency_ms"];
    rows = getFilteredTranslations().map((h) => [
      h.id, new Date(h.ts_ms).toISOString(), h.device_id, h.label, h.text, h.confidence, h.latency_ms ?? "",
    ]);
    name = "lich_su_phien_dich.csv";
  } else if (sub === "crm-sensor-logs") {
    headers = ["seq_id", "device_id", "label", "stable_label", "confidence", "edge_ms", "mqtt_ms", "ai_ms", "e2e_ms"];
    rows = state.rawLogs.map((r) => [
      r.seq_id, r.device_id, r.label, r.stable_label ?? "", r.confidence, r.edge_processing_ms, r.network_latency_ms, r.ai_inference_ms, r.e2e_latency_ms ?? "",
    ]);
    name = "nhat_ky_cam_bien.csv";
  } else if (sub === "crm-dataset" && state.dataset) {
    headers = ["label", "text", "kind", "sessions", "frames", "images"];
    rows = state.dataset.items.map((i) => [i.label, i.text, i.kind, i.sessions, i.frames, i.images]);
    name = "thong_ke_dataset.csv";
  }
  const csv = "\uFEFF" + [headers, ...rows]
    .map((r) => r.map((v) => `"${String(v ?? "").replace(/"/g, '""')}"`).join(","))
    .join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
};

$("btn-reset").onclick = async () => {
  if (!confirm("Xóa toàn bộ lịch sử phiên dịch và log cảm biến trong Cloud SQLite?")) return;
  await fetch("/api/reset", { method: "POST" });
  state.e2e = [];
  refreshCrmData();
};

// ------------------------------------------------------------------ gesture dictionary
async function loadGuide() {
  state.guide = await fetch("/api/gestures").then((r) => r.json());
  renderGuide();
}

function renderGuide() {
  const q = ($("guide-search")?.value || "").trim().toLowerCase();
  const list = state.guide.filter((g) => {
    if (g.kind === "pose") return false;
    if (!q && g.mode !== state.mode) return false;
    if (!q) return true;
    return `${g.text} ${g.label} ${g.description}`.toLowerCase().includes(q);
  });
  const name = state.modes[state.mode] ? state.modes[state.mode].name : "";
  const item = (g) => `<li data-label="${g.label}"><b>${escapeHtml(g.text)}</b><span>${escapeHtml(g.description)}</span></li>`;
  const main = list.filter((g) => g.kind !== "mark" && g.kind !== "control");
  const marks = list.filter((g) => g.kind === "mark");
  const controls = list.filter((g) => g.kind === "control");
  $("guide-title").textContent = q
    ? `Kết quả tìm kiếm (${list.length} cử chỉ)`
    : `Danh mục ${main.length} cử chỉ ${name.toLowerCase()}` + (marks.length ? ` + ${marks.length} dấu` : "");
  $("guide").innerHTML = main.map(item).join("")
    + (controls.length ? `<li class="group">Cử chỉ điều khiển</li>` + controls.map(item).join("") : "")
    + (marks.length ? `<li class="group">Dấu phụ và dấu thanh</li>` + marks.map(item).join("") : "");
}
$("guide-search").oninput = renderGuide;

setInterval(() => {
  if (state.lastSeen && Date.now() - state.lastSeen > 1500) {
    setHandDetected(false);
  }
}, 500);

// ------------------------------------------------------------------ start / stop Edge
let edgeError = null;
function renderEdge(s) {
  const btn = $("btn-edge");
  btn.disabled = false;
  btn.classList.toggle("stop", s.running);
  btn.textContent = s.running ? "Tắt Edge" : "Bật Edge";
  $("edge-source").disabled = s.running;
  if (s.running && s.source) $("edge-source").value = s.source;
  if (s.error && s.error !== edgeError) {
    edgeError = s.error;
    alert(`Edge đã dừng do lỗi:\n${s.error}\n\nNếu camera đang bị chương trình khác dùng (Zalo, Zoom, công cụ thu mẫu...), hãy tắt chương trình đó rồi bật lại.`);
  }
}

async function refreshEdge() {
  try {
    renderEdge(await fetch("/api/edge/status").then((r) => r.json()));
  } catch (e) { /* server restarting */ }
}

$("btn-edge").onclick = async () => {
  const btn = $("btn-edge");
  const running = btn.classList.contains("stop");
  btn.disabled = true;
  btn.textContent = running ? "Đang tắt..." : "Đang bật...";
  const url = running ? "/api/edge/stop" : `/api/edge/start?source=${$("edge-source").value}`;
  renderEdge(await fetch(url, { method: "POST" }).then((r) => r.json()));
  if (!running) setTimeout(refreshEdge, 4000);
};

// ------------------------------------------------------------------ boot
refreshEdge();
setInterval(refreshEdge, 2000);
loadGuide();
refreshStats();
setInterval(refreshStats, 3500);
connect();
