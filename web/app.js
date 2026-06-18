const MODEL_BASE = "./model";
const SAMPLE_TEXT = `PT Nusantara Data sedang mencari Data Analyst yang menguasai Python, SQL, dan Microsoft Excel. Kandidat diharapkan memiliki kemampuan komunikasi yang baik, teliti, mampu membuat dashboard, serta familiar dengan Tableau atau Power BI. Pengalaman dalam analisis data dan machine learning menjadi nilai tambah.`;

const state = {
  session: null,
  vocab: null,
  id2label: {},
  activeFilter: "all",
  entities: [],
  lastText: "",
};

const elements = {
  status: document.getElementById("modelStatus"),
  statusText: document.getElementById("modelStatusText"),
  jobText: document.getElementById("jobText"),
  extractButton: document.getElementById("extractButton"),
  clearButton: document.getElementById("clearButton"),
  loadSampleButton: document.getElementById("loadSampleButton"),
  helperText: document.getElementById("helperText"),
  highlightBox: document.getElementById("highlightBox"),
  hardSkillList: document.getElementById("hardSkillList"),
  techList: document.getElementById("techList"),
  softSkillList: document.getElementById("softSkillList"),
  hardSkillCount: document.getElementById("hardSkillCount"),
  techCount: document.getElementById("techCount"),
  softSkillCount: document.getElementById("softSkillCount"),
  avgConfidence: document.getElementById("avgConfidence"),
  exportJsonButton: document.getElementById("exportJsonButton"),
  exportCsvButton: document.getElementById("exportCsvButton"),
  filterTabs: [...document.querySelectorAll(".filter-tab")],
};

function setStatus(kind, message) {
  elements.status.dataset.state = kind;
  elements.statusText.textContent = message;
}

async function fetchText(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Gagal memuat ${path}`);
  }
  return response.text();
}

async function fetchJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Gagal memuat ${path}`);
  }
  return response.json();
}

function parseVocab(vocabText) {
  const vocab = new Map();
  vocabText.split(/\r?\n/).forEach((token, index) => {
    if (token) vocab.set(token, index);
  });
  return vocab;
}

function basicTokenize(text) {
  const matches = text.matchAll(/[A-Za-z0-9_+#.-]+|[^\sA-Za-z0-9_+#.-]/g);
  return [...matches].map((match) => ({
    token: match[0],
    start: match.index,
    end: match.index + match[0].length,
  }));
}

function wordpieceTokenize(word, vocab) {
  if (vocab.has(word)) return [word];

  const lowerWord = word.toLowerCase();
  if (vocab.has(lowerWord)) return [lowerWord];

  const pieces = [];
  let start = 0;

  while (start < lowerWord.length) {
    let end = lowerWord.length;
    let current = null;

    while (start < end) {
      const piece = lowerWord.slice(start, end);
      const candidate = start === 0 ? piece : `##${piece}`;
      if (vocab.has(candidate)) {
        current = candidate;
        break;
      }
      end -= 1;
    }

    if (!current) return ["[UNK]"];

    pieces.push(current);
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
  const inputIds = tokens.map((token) => vocab.get(token) ?? unkId);
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

function softmax(values) {
  const max = Math.max(...values);
  const exps = values.map((value) => Math.exp(value - max));
  const total = exps.reduce((sum, value) => sum + value, 0);
  return exps.map((value) => value / total);
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
  const labelCount = Object.keys(state.id2label).length;

  const tokenPredictions = [];
  for (let position = 0; position < encoded.maxLength; position += 1) {
    const start = position * labelCount;
    const scores = Array.from(logits.slice(start, start + labelCount));
    const probabilities = softmax(scores);
    const labelId = probabilities.indexOf(Math.max(...probabilities));
    const label = state.id2label[String(labelId)] || "O";
    tokenPredictions.push({
      label,
      confidence: probabilities[labelId],
      offset: encoded.offsets[position],
    });
  }

  return mergeBioEntities(tokenPredictions, text);
}

function normalizeEntityType(label) {
  return label.replace(/^B-/, "").replace(/^I-/, "");
}

function mergeBioEntities(predictions, text) {
  const entities = [];
  let current = null;

  for (const prediction of predictions) {
    const { label, confidence, offset } = prediction;
    if (!offset || offset.start < 0 || label === "O") {
      if (current) {
        entities.push(current);
        current = null;
      }
      continue;
    }

    const type = normalizeEntityType(label);
    const isBegin = label.startsWith("B-");
    const shouldStartNew = !current || isBegin || current.type !== type || offset.start > current.end + 2;

    if (shouldStartNew) {
      if (current) entities.push(current);
      current = {
        type,
        start: offset.start,
        end: offset.end,
        scores: [confidence],
      };
    } else {
      current.end = Math.max(current.end, offset.end);
      current.scores.push(confidence);
    }
  }

  if (current) entities.push(current);

  return dedupeEntities(
    entities.map((entity) => ({
      ...entity,
      text: text.slice(entity.start, entity.end),
      confidence: entity.scores.reduce((sum, score) => sum + score, 0) / entity.scores.length,
    }))
  );
}

function dedupeEntities(entities) {
  const seen = new Set();
  return entities.filter((entity) => {
    const key = `${entity.type}:${entity.start}:${entity.end}:${entity.text.toLowerCase()}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return entity.text.trim().length > 0;
  });
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderHighlights(text, entities) {
  const visibleEntities =
    state.activeFilter === "all" ? entities : entities.filter((entity) => entity.type === state.activeFilter);

  if (!text.trim()) {
    elements.highlightBox.innerHTML = '<p class="empty-state">Hasil highlight akan muncul di sini.</p>';
    return;
  }

  if (!visibleEntities.length) {
    elements.highlightBox.innerHTML = `<p>${escapeHtml(text)}</p>`;
    return;
  }

  let html = "";
  let cursor = 0;

  for (const entity of visibleEntities.sort((a, b) => a.start - b.start)) {
    html += escapeHtml(text.slice(cursor, entity.start));
    html += `<span class="entity ${entity.type}" title="${entity.type} - ${(entity.confidence * 100).toFixed(1)}%">${escapeHtml(text.slice(entity.start, entity.end))}</span>`;
    cursor = entity.end;
  }

  html += escapeHtml(text.slice(cursor));
  elements.highlightBox.innerHTML = `<p>${html}</p>`;
}

function renderEntityList(element, entities) {
  if (!entities.length) {
    element.innerHTML = '<li><span>Belum ada entitas</span><span class="confidence">-</span></li>';
    return;
  }

  element.innerHTML = entities
    .map(
      (entity) => `
        <li>
          <span>${escapeHtml(entity.text)}</span>
          <span class="confidence">${(entity.confidence * 100).toFixed(1)}%</span>
        </li>
      `
    )
    .join("");
}

function updateSummary(entities) {
  const hardSkills = entities.filter((entity) => entity.type === "HSkill");
  const tech = entities.filter((entity) => entity.type === "Tech");
  const softSkills = entities.filter((entity) => entity.type === "SSkill");
  const average =
    entities.length === 0
      ? "-"
      : `${((entities.reduce((sum, entity) => sum + entity.confidence, 0) / entities.length) * 100).toFixed(1)}%`;

  elements.hardSkillCount.textContent = hardSkills.length;
  elements.techCount.textContent = tech.length;
  elements.softSkillCount.textContent = softSkills.length;
  elements.avgConfidence.textContent = average;

  renderEntityList(elements.hardSkillList, hardSkills);
  renderEntityList(elements.techList, tech);
  renderEntityList(elements.softSkillList, softSkills);
}

function renderAll() {
  renderHighlights(state.lastText, state.entities);
  updateSummary(state.entities);
}

function downloadFile(fileName, content, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = fileName;
  anchor.click();
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
  const rows = state.entities.map((entity) =>
    [entity.type, `"${entity.text.replaceAll('"', '""')}"`, entity.confidence.toFixed(4), entity.start, entity.end].join(",")
  );
  downloadFile("skillscope-results.csv", [header, ...rows].join("\n"), "text/csv");
}

async function initializeModel() {
  try {
    if (!window.ort) {
      throw new Error("ONNX Runtime Web tidak tersedia. Periksa koneksi CDN atau gunakan file lokal.");
    }

    setStatus("loading", "Memuat model");
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

    const [labelMap, vocabText] = await Promise.all([
      fetchJson(`${MODEL_BASE}/label_map.json`),
      fetchText(`${MODEL_BASE}/vocab.txt`),
    ]);

    state.id2label = labelMap.id2label;
    state.vocab = parseVocab(vocabText);
    state.session = await ort.InferenceSession.create(`${MODEL_BASE}/model.onnx`, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });

    setStatus("ready", "Model siap");
    elements.helperText.textContent = "Model siap. Tempel teks lowongan, lalu jalankan ekstraksi.";
  } catch (error) {
    setStatus("error", "Model belum terpasang");
    elements.helperText.textContent =
      "Salin model.onnx dan vocab.txt dari output Kaggle ke folder web/model, lalu jalankan web via local server.";
    console.error(error);
  }
}

elements.extractButton.addEventListener("click", async () => {
  const text = elements.jobText.value.trim();
  if (!text) {
    elements.helperText.textContent = "Input masih kosong. Tempel lowongan kerja atau gunakan contoh.";
    return;
  }

  if (!state.session || !state.vocab) {
    elements.helperText.textContent = "Model belum siap. Pastikan web/model/model.onnx dan web/model/vocab.txt sudah tersedia.";
    return;
  }

  try {
    elements.extractButton.disabled = true;
    elements.extractButton.textContent = "Processing...";
    state.lastText = text;
    state.entities = await runInference(text);
    renderAll();
    elements.helperText.textContent = `${state.entities.length} entitas ditemukan.`;
  } catch (error) {
    console.error(error);
    elements.helperText.textContent = "Inference gagal. Cek console browser untuk detail error.";
  } finally {
    elements.extractButton.disabled = false;
    elements.extractButton.textContent = "Extract Skills";
  }
});

elements.clearButton.addEventListener("click", () => {
  elements.jobText.value = "";
  state.lastText = "";
  state.entities = [];
  renderAll();
  elements.helperText.textContent = "Input dibersihkan.";
});

elements.loadSampleButton.addEventListener("click", () => {
  elements.jobText.value = SAMPLE_TEXT;
  elements.helperText.textContent = "Contoh lowongan dimuat.";
});

elements.exportJsonButton.addEventListener("click", exportJson);
elements.exportCsvButton.addEventListener("click", exportCsv);

elements.filterTabs.forEach((button) => {
  button.addEventListener("click", () => {
    elements.filterTabs.forEach((tab) => tab.classList.remove("active"));
    button.classList.add("active");
    state.activeFilter = button.dataset.filter;
    renderHighlights(state.lastText, state.entities);
  });
});

renderAll();
initializeModel();
