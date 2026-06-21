/* ============================================================
   SkillScope ID — Application Logic
   ONNX Runtime Web inference with premium UI interactions
   ============================================================ */

const MODEL_BASE = "./model";

const SAMPLE_TEXT = `PT Nusantara Data sedang mencari Data Analyst yang menguasai Python, SQL, dan Microsoft Excel. Kandidat diharapkan memiliki kemampuan komunikasi yang baik, teliti, mampu membuat dashboard, serta familiar dengan Tableau atau Power BI. Pengalaman dalam analisis data dan machine learning menjadi nilai tambah. Kemampuan bekerja dalam tim dan problem solving yang kuat sangat diutamakan.`;

/* ===== STATE ===== */
const state = {
  session: null,
  vocab: null,
  id2label: {},
  activeFilter: "all",
  entities: [],
  lastText: "",
};

/* ===== DOM ELEMENTS ===== */
const el = {
  // Loading
  overlay: document.getElementById("loadingOverlay"),
  loadingText: document.getElementById("loadingText"),
  loadingBar: document.getElementById("loadingBar"),

  // Header
  status: document.getElementById("modelStatus"),
  statusText: document.getElementById("modelStatusText"),

  // Input
  jobText: document.getElementById("jobText"),
  charCounter: document.getElementById("charCounter"),
  extractBtn: document.getElementById("extractButton"),
  clearBtn: document.getElementById("clearButton"),
  sampleBtn: document.getElementById("loadSampleButton"),
  helperText: document.getElementById("helperText"),

  // Output
  highlightBox: document.getElementById("highlightBox"),
  filterChips: [...document.querySelectorAll(".filter-chip")],

  // Metrics
  hardSkillCount: document.getElementById("hardSkillCount"),
  techCount: document.getElementById("techCount"),
  softSkillCount: document.getElementById("softSkillCount"),
  avgConfidence: document.getElementById("avgConfidence"),

  // Entity lists
  hardSkillList: document.getElementById("hardSkillList"),
  techList: document.getElementById("techList"),
  softSkillList: document.getElementById("softSkillList"),

  // Export
  exportJsonBtn: document.getElementById("exportJsonButton"),
  exportCsvBtn: document.getElementById("exportCsvButton"),

  // Footer
  footerModelInfo: document.getElementById("footerModelInfo"),
};

/* ===== UTILITIES ===== */
function setStatus(kind, message) {
  el.status.dataset.state = kind;
  el.statusText.textContent = message;
}

function setLoadingText(text) {
  if (el.loadingText) el.loadingText.textContent = text;
}

function hideOverlay() {
  if (el.overlay) el.overlay.classList.add("hidden");
}

function showOverlay() {
  if (el.overlay) el.overlay.classList.remove("hidden");
}

async function fetchText(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Gagal memuat ${path}`);
  return res.text();
}

async function fetchJson(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Gagal memuat ${path}`);
  return res.json();
}

function escapeHtml(str) {
  return str
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

/* ===== TOKENIZER ===== */
function parseVocab(vocabText) {
  const vocab = new Map();
  vocabText.split(/\r?\n/).forEach((token, i) => {
    if (token) vocab.set(token, i);
  });
  return vocab;
}

function basicTokenize(text) {
  return [...text.matchAll(/[A-Za-z0-9_+#.\-]+|[^\sA-Za-z0-9_+#.\-]/g)].map((m) => ({
    token: m[0],
    start: m.index,
    end: m.index + m[0].length,
  }));
}

function wordpieceTokenize(word, vocab) {
  if (vocab.has(word)) return [word];
  const lower = word.toLowerCase();
  if (vocab.has(lower)) return [lower];

  const pieces = [];
  let start = 0;

  while (start < lower.length) {
    let end = lower.length;
    let found = null;

    while (start < end) {
      const slice = lower.slice(start, end);
      const candidate = start === 0 ? slice : `##${slice}`;
      if (vocab.has(candidate)) {
        found = candidate;
        break;
      }
      end -= 1;
    }

    if (!found) return ["[UNK]"];
    pieces.push(found);
    start = end;
  }

  return pieces;
}

function tokenizeForBert(text, vocab, maxLength = 128) {
  const words = basicTokenize(text);
  const tokens = ["[CLS]"];
  const offsets = [{ start: -1, end: -1 }];

  for (const word of words) {
    const pieces = wordpieceTokenize(word.token, vocab);
    for (const piece of pieces) {
      if (tokens.length >= maxLength - 1) break;
      tokens.push(piece);
      offsets.push({ start: word.start, end: word.end });
    }
    if (tokens.length >= maxLength - 1) break;
  }

  tokens.push("[SEP]");
  offsets.push({ start: -1, end: -1 });

  const unkId = vocab.get("[UNK]") ?? 1;
  const padId = vocab.get("[PAD]") ?? 0;
  const inputIds = tokens.map((t) => vocab.get(t) ?? unkId);
  const attentionMask = inputIds.map(() => 1);
  const tokenTypeIds = inputIds.map(() => 0);

  while (inputIds.length < maxLength) {
    inputIds.push(padId);
    attentionMask.push(0);
    tokenTypeIds.push(0);
    offsets.push({ start: -1, end: -1 });
  }

  return { tokens, offsets, inputIds, attentionMask, tokenTypeIds, maxLength };
}

/* ===== INFERENCE ===== */
function softmax(values) {
  const max = Math.max(...values);
  const exps = values.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

function createInt64Tensor(values, shape) {
  return new ort.Tensor("int64", BigInt64Array.from(values.map(BigInt)), shape);
}

async function runInference(text) {
  const encoded = tokenizeForBert(text, state.vocab, 128);
  const feeds = {
    input_ids: createInt64Tensor(encoded.inputIds, [1, encoded.maxLength]),
    attention_mask: createInt64Tensor(encoded.attentionMask, [1, encoded.maxLength]),
  };

  const inputNames = state.session.inputNames || [];
  if (inputNames.includes("token_type_ids")) {
    feeds.token_type_ids = createInt64Tensor(encoded.tokenTypeIds, [1, encoded.maxLength]);
  }

  const output = await state.session.run(feeds);
  const outputName = state.session.outputNames[0];
  const logits = output[outputName].data;
  const numLabels = Object.keys(state.id2label).length;

  const preds = [];
  for (let pos = 0; pos < encoded.maxLength; pos++) {
    const start = pos * numLabels;
    const scores = Array.from(logits.slice(start, start + numLabels));
    const probs = softmax(scores);
    const labelId = probs.indexOf(Math.max(...probs));
    preds.push({
      label: state.id2label[String(labelId)] || "O",
      confidence: probs[labelId],
      offset: encoded.offsets[pos],
    });
  }

  return mergeBioEntities(preds, text);
}

function normalizeEntityType(label) {
  return label.replace(/^[BI]-/, "");
}

function mergeBioEntities(preds, text) {
  const entities = [];
  let cur = null;

  for (const p of preds) {
    if (!p.offset || p.offset.start < 0 || p.label === "O") {
      if (cur) { entities.push(cur); cur = null; }
      continue;
    }

    const type = normalizeEntityType(p.label);
    const isB = p.label.startsWith("B-");
    const startNew = !cur || isB || cur.type !== type || p.offset.start > cur.end + 2;

    if (startNew) {
      if (cur) entities.push(cur);
      cur = { type, start: p.offset.start, end: p.offset.end, scores: [p.confidence] };
    } else {
      cur.end = Math.max(cur.end, p.offset.end);
      cur.scores.push(p.confidence);
    }
  }

  if (cur) entities.push(cur);

  return dedupeEntities(
    entities.map((e) => ({
      ...e,
      text: text.slice(e.start, e.end),
      confidence: e.scores.reduce((a, b) => a + b, 0) / e.scores.length,
    }))
  );
}

function dedupeEntities(entities) {
  const seen = new Set();
  return entities.filter((e) => {
    const key = `${e.type}:${e.start}:${e.end}:${e.text.toLowerCase()}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return e.text.trim().length > 0;
  });
}

/* ===== RENDERING ===== */
function renderHighlights(text, entities) {
  const visible =
    state.activeFilter === "all" ? entities : entities.filter((e) => e.type === state.activeFilter);

  if (!text.trim()) {
    el.highlightBox.innerHTML = `
      <div class="empty-state">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.3"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
        <p>Hasil highlight akan muncul di sini setelah ekstraksi.</p>
      </div>`;
    return;
  }

  if (!visible.length) {
    el.highlightBox.innerHTML = `<p>${escapeHtml(text)}</p>`;
    return;
  }

  let html = "";
  let cursor = 0;

  for (const e of [...visible].sort((a, b) => a.start - b.start)) {
    html += escapeHtml(text.slice(cursor, e.start));
    html += `<span class="entity ${e.type}" title="${e.type} — ${(e.confidence * 100).toFixed(1)}%">${escapeHtml(text.slice(e.start, e.end))}</span>`;
    cursor = e.end;
  }

  html += escapeHtml(text.slice(cursor));
  el.highlightBox.innerHTML = `<p>${html}</p>`;
}

function renderEntityList(listEl, entities) {
  if (!entities.length) {
    listEl.innerHTML = `<li><span style="color:var(--text-muted)">Belum ada entitas</span><span class="confidence">—</span></li>`;
    return;
  }

  listEl.innerHTML = entities
    .map(
      (e, i) => `
      <li style="animation-delay: ${i * 0.05}s">
        <span>${escapeHtml(e.text)}</span>
        <span class="confidence">${(e.confidence * 100).toFixed(1)}%</span>
      </li>`
    )
    .join("");
}

function updateMetrics(entities) {
  const hskill = entities.filter((e) => e.type === "HSkill");
  const tech = entities.filter((e) => e.type === "Tech");
  const sskill = entities.filter((e) => e.type === "SSkill");
  const avg =
    entities.length === 0
      ? "—"
      : `${((entities.reduce((s, e) => s + e.confidence, 0) / entities.length) * 100).toFixed(1)}%`;

  animateCount(el.hardSkillCount, hskill.length);
  animateCount(el.techCount, tech.length);
  animateCount(el.softSkillCount, sskill.length);
  el.avgConfidence.textContent = avg;

  renderEntityList(el.hardSkillList, hskill);
  renderEntityList(el.techList, tech);
  renderEntityList(el.softSkillList, sskill);
}

function animateCount(element, target) {
  const current = parseInt(element.textContent) || 0;
  if (current === target) {
    element.textContent = target;
    return;
  }
  const duration = 400;
  const start = performance.now();

  function step(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    element.textContent = Math.round(current + (target - current) * eased);
    if (progress < 1) requestAnimationFrame(step);
  }

  requestAnimationFrame(step);
}

function renderAll() {
  renderHighlights(state.lastText, state.entities);
  updateMetrics(state.entities);
}

/* ===== EXPORT ===== */
function downloadFile(name, content, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

function exportJson() {
  downloadFile(
    "skillscope-results.json",
    JSON.stringify({ text: state.lastText, entities: state.entities }, null, 2),
    "application/json"
  );
}

function exportCsv() {
  const header = "type,text,confidence,start,end";
  const rows = state.entities.map((e) =>
    [e.type, `"${e.text.replaceAll('"', '""')}"`, e.confidence.toFixed(4), e.start, e.end].join(",")
  );
  downloadFile("skillscope-results.csv", [header, ...rows].join("\n"), "text/csv");
}

/* ===== MODEL INITIALIZATION ===== */
async function initializeModel() {
  try {
    if (!window.ort) {
      throw new Error("ONNX Runtime Web tidak tersedia.");
    }

    setLoadingText("Mengunduh konfigurasi model...");
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

    const [labelMap, vocabText] = await Promise.all([
      fetchJson(`${MODEL_BASE}/label_map.json`),
      fetchText(`${MODEL_BASE}/vocab.txt`),
    ]);

    state.id2label = labelMap.id2label;
    state.vocab = parseVocab(vocabText);

    setLoadingText("Memuat model ONNX ke browser (mungkin 20-60 detik)...");

    state.session = await ort.InferenceSession.create(`${MODEL_BASE}/model.onnx`, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });

    setStatus("ready", "Model siap");
    el.helperText.textContent = "Model siap. Tempel teks lowongan kerja, lalu klik Extract Skills.";

    // Update footer with model info
    if (el.footerModelInfo) {
      const modelName = labelMap.model_name || "NER Transformer";
      el.footerModelInfo.innerHTML = `
        <span class="meta-tag">Model: ${escapeHtml(modelName)}</span>
        <span class="meta-tag">ONNX Quantized INT8</span>
        <span class="meta-tag">Browser Inference</span>
      `;
    }

    // Hide loading overlay with smooth transition
    setTimeout(hideOverlay, 300);
  } catch (error) {
    setStatus("error", "Model belum tersedia");
    el.helperText.textContent =
      "Salin file model dari output Kaggle ke folder web/model, lalu buka via HTTP server.";
    console.error("Model load error:", error);
    hideOverlay();
  }
}

/* ===== EVENT LISTENERS ===== */

// Extract button
el.extractBtn.addEventListener("click", async () => {
  const text = el.jobText.value.trim();
  if (!text) {
    el.helperText.textContent = "Input masih kosong. Tempel teks lowongan atau gunakan contoh.";
    return;
  }

  if (!state.session || !state.vocab) {
    el.helperText.textContent = "Model belum siap. Pastikan file model tersedia di web/model/.";
    return;
  }

  try {
    el.extractBtn.disabled = true;
    el.extractBtn.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="spin-icon"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>
      Processing...`;

    state.lastText = text;
    state.entities = await runInference(text);
    renderAll();

    const count = state.entities.length;
    el.helperText.textContent = count > 0
      ? `✅ ${count} entitas ditemukan dalam ${text.split(/\s+/).length} kata.`
      : "Tidak ada entitas terdeteksi. Coba teks lowongan yang lebih detail.";
  } catch (err) {
    console.error("Inference error:", err);
    el.helperText.textContent = "Inference gagal. Lihat console browser untuk detail.";
  } finally {
    el.extractBtn.disabled = false;
    el.extractBtn.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
      Extract Skills`;
  }
});

// Clear button
el.clearBtn.addEventListener("click", () => {
  el.jobText.value = "";
  state.lastText = "";
  state.entities = [];
  renderAll();
  updateCharCounter();
  el.helperText.textContent = "Input dibersihkan.";
});

// Sample button
el.sampleBtn.addEventListener("click", () => {
  el.jobText.value = SAMPLE_TEXT;
  updateCharCounter();
  el.helperText.textContent = "Contoh lowongan dimuat. Klik Extract Skills untuk memulai.";
});

// Filter chips
el.filterChips.forEach((chip) => {
  chip.addEventListener("click", () => {
    el.filterChips.forEach((c) => c.classList.remove("active"));
    chip.classList.add("active");
    state.activeFilter = chip.dataset.filter;
    renderHighlights(state.lastText, state.entities);
  });
});

// Export buttons
el.exportJsonBtn.addEventListener("click", exportJson);
el.exportCsvBtn.addEventListener("click", exportCsv);

// Character counter
function updateCharCounter() {
  const len = el.jobText.value.length;
  el.charCounter.textContent = `${len.toLocaleString("id-ID")} karakter`;
}

el.jobText.addEventListener("input", updateCharCounter);

// Keyboard shortcut: Ctrl+Enter = Extract
document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    e.preventDefault();
    el.extractBtn.click();
  }
});

/* ===== INIT ===== */
renderAll();
updateCharCounter();
initializeModel();
