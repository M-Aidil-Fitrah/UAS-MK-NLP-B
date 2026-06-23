/* ============================================================
   SkillScope ID — Application Logic
   ONNX Runtime Web inference with premium UI interactions
   ============================================================ */

const MODEL_BASE = "./model";

const SAMPLE_TEXT = `PT Nusantara Data sedang mencari Data Analyst yang menguasai Python, SQL, dan Microsoft Excel. Kandidat diharapkan memiliki kemampuan komunikasi yang baik, teliti, mampu membuat dashboard, serta familiar dengan Tableau atau Power BI. Pengalaman dalam analisis data dan machine learning menjadi nilai tambah. Kemampuan bekerja dalam tim dan problem solving yang kuat sangat diutamakan.`;

const MATCH_WEIGHTS = {
  Tech: 0.4,
  HSkill: 0.4,
  SSkill: 0.2,
};

const SOFT_SKILL_FALLBACKS = [
  "adaptif",
  "analitis",
  "beradaptasi",
  "berinisiatif",
  "berkomunikasi",
  "berpikir kritis",
  "bekerja dalam tim",
  "disiplin",
  "jujur",
  "kemampuan belajar",
  "kemauan belajar",
  "kerja sama",
  "kerja sama tim",
  "kepemimpinan",
  "komunikasi",
  "komunikasi yang baik",
  "kreatif",
  "mandiri",
  "pemecahan masalah",
  "problem solving",
  "teliti",
  "teamwork",
];

/* ===== STATE ===== */
const state = {
  session: null,
  vocab: null,
  id2label: {},
  activeFilter: "all",
  activeMode: "extractor",
  entities: [],
  lastText: "",
  cvText: "",
  matchJobText: "",
  cvEntities: [],
  jobMatchEntities: [],
  matchResult: null,
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

  // Mode switcher
  extractorModeBtn: document.getElementById("extractorModeButton"),
  matchModeBtn: document.getElementById("matchModeButton"),
  extractorWorkspace: document.getElementById("extractorWorkspace"),
  matchWorkspace: document.getElementById("matchWorkspace"),

  // CV match
  cvFileInput: document.getElementById("cvFileInput"),
  cvFileStatus: document.getElementById("cvFileStatus"),
  cvText: document.getElementById("cvText"),
  cvCharCounter: document.getElementById("cvCharCounter"),
  clearCvBtn: document.getElementById("clearCvButton"),
  matchJobText: document.getElementById("matchJobText"),
  matchJobCharCounter: document.getElementById("matchJobCharCounter"),
  loadMatchSampleBtn: document.getElementById("loadMatchSampleButton"),
  analyzeMatchBtn: document.getElementById("analyzeMatchButton"),
  matchProgress: document.getElementById("matchProgress"),
  matchProgressLabel: document.getElementById("matchProgressLabel"),
  matchProgressValue: document.getElementById("matchProgressValue"),
  matchProgressBar: document.getElementById("matchProgressBar"),
  matchHelperText: document.getElementById("matchHelperText"),
  matchScoreRing: document.getElementById("matchScoreRing"),
  matchScore: document.getElementById("matchScore"),
  matchedSkillCount: document.getElementById("matchedSkillCount"),
  missingSkillCount: document.getElementById("missingSkillCount"),
  extraSkillCount: document.getElementById("extraSkillCount"),
  categoryBreakdown: document.getElementById("categoryBreakdown"),
  matchedSkillList: document.getElementById("matchedSkillList"),
  missingSkillList: document.getElementById("missingSkillList"),
  extraSkillList: document.getElementById("extraSkillList"),
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

  return applySoftSkillFallbacks(text, mergeBioEntities(preds, text));
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

function escapeRegex(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function rangesOverlap(aStart, aEnd, bStart, bEnd) {
  return aStart < bEnd && bStart < aEnd;
}

function hasOverlappingEntity(entities, start, end, type = null) {
  return entities.some((entity) => {
    const sameType = !type || entity.type === type;
    return sameType && rangesOverlap(entity.start, entity.end, start, end);
  });
}

function applySoftSkillFallbacks(text, entities) {
  if (!text.trim()) return entities;

  const additions = [];
  const seen = new Set(entities.map((entity) => `${entity.type}:${canonicalSkillText(entity.text)}`));

  const sortedFallbacks = [...SOFT_SKILL_FALLBACKS].sort((a, b) => b.length - a.length);
  for (const phrase of sortedFallbacks) {
    const pattern = new RegExp(`(^|[^\\p{L}\\p{N}])(${escapeRegex(phrase)})(?=$|[^\\p{L}\\p{N}])`, "giu");
    for (const match of text.matchAll(pattern)) {
      const matchedText = match[2];
      const start = match.index + match[1].length;
      const end = start + matchedText.length;
      const key = `SSkill:${canonicalSkillText(matchedText)}`;

      if (seen.has(key)) continue;
      if (hasOverlappingEntity(entities, start, end) || hasOverlappingEntity(additions, start, end)) continue;

      seen.add(key);
      additions.push({
        type: "SSkill",
        start,
        end,
        text: text.slice(start, end),
        confidence: 0.72,
        source: "soft_skill_fallback",
      });
    }
  }

  return dedupeEntities([...entities, ...additions]).sort((a, b) => a.start - b.start || a.end - b.end);
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
        <span class="confidence">${e.source === "soft_skill_fallback" ? "fallback" : `${(e.confidence * 100).toFixed(1)}%`}</span>
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

/* ===== CV MATCH: DOCUMENT PARSING ===== */
function getFileExtension(file) {
  return (file.name.split(".").pop() || "").toLowerCase();
}

function cleanExtractedText(text) {
  return text
    .replace(/\u0000/g, "")
    .replace(/[ \t]+/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

async function extractPdfText(file) {
  if (!window.pdfjsLib) {
    throw new Error("PDF.js belum termuat. Cek koneksi CDN atau gunakan paste manual.");
  }

  pdfjsLib.GlobalWorkerOptions.workerSrc =
    "https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/build/pdf.worker.min.js";

  const arrayBuffer = await file.arrayBuffer();
  const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
  const pages = [];

  for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber += 1) {
    const page = await pdf.getPage(pageNumber);
    const content = await page.getTextContent();
    const pageText = content.items.map((item) => item.str || "").join(" ");
    pages.push(pageText);
  }

  return cleanExtractedText(pages.join("\n"));
}

async function extractDocxText(file) {
  if (!window.mammoth) {
    throw new Error("Mammoth.js belum termuat. Cek koneksi CDN atau gunakan paste manual.");
  }

  const arrayBuffer = await file.arrayBuffer();
  const result = await mammoth.extractRawText({ arrayBuffer });
  return cleanExtractedText(result.value || "");
}

async function extractTxtText(file) {
  return cleanExtractedText(await file.text());
}

async function extractDocumentText(file) {
  const extension = getFileExtension(file);
  const mime = file.type;

  if (extension === "pdf" || mime === "application/pdf") return extractPdfText(file);
  if (
    extension === "docx" ||
    mime === "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
  ) {
    return extractDocxText(file);
  }
  if (extension === "txt" || mime.startsWith("text/")) return extractTxtText(file);

  throw new Error("Format tidak didukung. Gunakan PDF, DOCX, TXT, atau paste manual.");
}

/* ===== CV MATCH: SCORING ===== */
function canonicalSkillText(text) {
  return text
    .toLowerCase()
    .replace(/\breact\s*\.?\s*js\b/g, "react")
    .replace(/\bnext\s*\.?\s*js\b/g, "next.js")
    .replace(/\bexpress\s*\.?\s*js\b/g, "express")
    .replace(/\bnode\s*\.?\s*js\b/g, "node.js")
    .replace(/\bgithub\b/g, "git")
    .replace(/\bmicrosoft\s+excel\b/g, "excel")
    .replace(/\bms\s+excel\b/g, "excel")
    .replace(/\bpowerbi\b/g, "power bi")
    .replace(/\bpostgre\s*sql\b/g, "postgresql")
    .replace(/\btailwind\s+css\b/g, "tailwind")
    .replace(/\bvue\s+js\b/g, "vue")
    .replace(/[^\p{L}\p{N}+#.]+/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function canonicalTextForSearch(text) {
  return ` ${canonicalSkillText(text)} `;
}

function searchContainsSkill(normalizedHaystack, normalizedNeedle) {
  if (!normalizedNeedle || normalizedNeedle.length < 2) return false;
  return normalizedHaystack.includes(` ${normalizedNeedle} `);
}

function uniqueEntitiesForMatch(entities) {
  const map = new Map();

  for (const entity of entities) {
    const key = `${entity.type}:${canonicalSkillText(entity.text)}`;
    const normalized = canonicalSkillText(entity.text);
    if (!normalized) continue;

    const existing = map.get(key);
    if (!existing || entity.confidence > existing.confidence) {
      map.set(key, { ...entity, normalized });
    }
  }

  return [...map.values()].sort((a, b) => a.type.localeCompare(b.type) || a.normalized.localeCompare(b.normalized));
}

function skillTextsMatch(a, b) {
  if (a === b) return true;
  if (a.length < 4 || b.length < 4) return false;

  const aWords = a.split(" ");
  const bWords = b.split(" ");

  if (aWords.length === 1 && bWords.length > 1) return bWords.includes(a);
  if (bWords.length === 1 && aWords.length > 1) return aWords.includes(b);

  return false;
}

function findEntityMatch(target, candidates, normalizedCvText) {
  const sameTypeMatch = candidates.find(
    (candidate) => target.type === candidate.type && skillTextsMatch(target.normalized, candidate.normalized)
  );
  if (sameTypeMatch) return { entity: sameTypeMatch, source: "ner" };

  const crossTypeMatch = candidates.find((candidate) => skillTextsMatch(target.normalized, candidate.normalized));
  if (crossTypeMatch) return { entity: crossTypeMatch, source: "ner_cross_label" };

  if (searchContainsSkill(normalizedCvText, target.normalized)) {
    return {
      entity: {
        type: target.type,
        text: target.text,
        normalized: target.normalized,
        confidence: 0.75,
      },
      source: "cv_text",
    };
  }

  return null;
}

function compareCvToJob(cvEntities, jobEntities, rawCvText = "") {
  const cvUnique = uniqueEntitiesForMatch(cvEntities);
  const jobUnique = uniqueEntitiesForMatch(jobEntities);
  const normalizedCvText = canonicalTextForSearch(rawCvText);

  const matched = [];
  const missing = [];
  const usedCvKeys = new Set();

  for (const jobEntity of jobUnique) {
    const match = findEntityMatch(jobEntity, cvUnique, normalizedCvText);
    if (match) {
      const cvMatch = match.entity;
      usedCvKeys.add(`${cvMatch.type}:${cvMatch.normalized}`);
      matched.push({
        type: jobEntity.type,
        text: jobEntity.text,
        cvText: cvMatch.text,
        confidence: Math.min(jobEntity.confidence, cvMatch.confidence),
        source: match.source,
      });
    } else {
      missing.push(jobEntity);
    }
  }

  const extra = cvUnique.filter((cvEntity) => !usedCvKeys.has(`${cvEntity.type}:${cvEntity.normalized}`));
  const breakdown = {};

  for (const type of Object.keys(MATCH_WEIGHTS)) {
    const jobCount = jobUnique.filter((entity) => entity.type === type).length;
    const matchedCount = matched.filter((entity) => entity.type === type).length;
    breakdown[type] = {
      jobCount,
      matchedCount,
      score: jobCount === 0 ? null : matchedCount / jobCount,
    };
  }

  const activeTypes = Object.entries(breakdown).filter(([, item]) => item.jobCount > 0);
  const activeWeightTotal = activeTypes.reduce((sum, [type]) => sum + MATCH_WEIGHTS[type], 0);
  const overall =
    activeWeightTotal === 0
      ? 0
      : activeTypes.reduce((sum, [type, item]) => sum + item.score * (MATCH_WEIGHTS[type] / activeWeightTotal), 0);

  return {
    overall,
    matched,
    missing,
    extra,
    breakdown,
    cvEntityCount: cvUnique.length,
    jobEntityCount: jobUnique.length,
    createdAt: new Date().toISOString(),
  };
}

function renderMatchList(listEl, items, emptyText, variant = "default") {
  if (!items.length) {
    listEl.innerHTML = `<li class="empty-row">${escapeHtml(emptyText)}</li>`;
    return;
  }

  listEl.innerHTML = items
    .map((item) => {
      const secondary =
        variant === "matched" && item.cvText && item.cvText.toLowerCase() !== item.text.toLowerCase()
          ? `<span class="match-secondary">CV: ${escapeHtml(item.cvText)}</span>`
          : `<span class="match-secondary">${item.source === "cv_text" ? "text match" : `${(item.confidence * 100).toFixed(1)}%`}</span>`;

      return `
        <li>
          <span>
            <strong>${escapeHtml(item.text)}</strong>
            <em>${escapeHtml(item.type)}</em>
          </span>
          ${secondary}
        </li>`;
    })
    .join("");
}

function renderCategoryBreakdown(result) {
  const rows = Object.entries(result.breakdown)
    .map(([type, item]) => {
      const pct = item.score === null ? 0 : Math.round(item.score * 100);
      const label = item.score === null ? "Tidak ada requirement" : `${item.matchedCount}/${item.jobCount} cocok`;
      return `
        <div class="breakdown-row">
          <div class="breakdown-top">
            <span>${type}</span>
            <strong>${item.score === null ? "—" : `${pct}%`}</strong>
          </div>
          <div class="breakdown-track">
            <span style="width:${pct}%"></span>
          </div>
          <small>${label}</small>
        </div>`;
    })
    .join("");

  el.categoryBreakdown.innerHTML = rows;
}

function renderMatchResult(result) {
  const score = Math.round(result.overall * 100);
  el.matchScore.textContent = `${score}%`;
  el.matchScoreRing.style.setProperty("--score", `${score}%`);
  el.matchedSkillCount.textContent = result.matched.length;
  el.missingSkillCount.textContent = result.missing.length;
  el.extraSkillCount.textContent = result.extra.length;

  renderCategoryBreakdown(result);
  renderMatchList(el.matchedSkillList, result.matched, "Belum ada skill yang cocok.", "matched");
  renderMatchList(el.missingSkillList, result.missing, "Tidak ada skill yang hilang.");
  renderMatchList(el.extraSkillList, result.extra, "Tidak ada extra skill dari CV.");
}

function setMode(mode) {
  state.activeMode = mode;
  const isMatch = mode === "match";

  el.extractorModeBtn.classList.toggle("active", !isMatch);
  el.matchModeBtn.classList.toggle("active", isMatch);
  el.extractorWorkspace.classList.toggle("active", !isMatch);
  el.matchWorkspace.classList.toggle("active", isMatch);

  if (isMatch && !el.matchJobText.value.trim() && el.jobText.value.trim()) {
    el.matchJobText.value = el.jobText.value;
    updateMatchJobCounter();
  }
}

function updateCvCounter() {
  el.cvCharCounter.textContent = `${el.cvText.value.length.toLocaleString("id-ID")} karakter`;
}

function updateMatchJobCounter() {
  el.matchJobCharCounter.textContent = `${el.matchJobText.value.length.toLocaleString("id-ID")} karakter`;
}

function setMatchButtonLoading(isLoading) {
  el.analyzeMatchBtn.disabled = isLoading;
  el.analyzeMatchBtn.innerHTML = isLoading
    ? `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="spin-icon"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>Analyzing...`
    : `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 11l3 3L22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg>Analyze Match`;
}

function setMatchProgress(percent, label, status = "running") {
  const safePercent = Math.max(0, Math.min(100, Math.round(percent)));
  el.matchProgress.dataset.state = status;
  el.matchProgressLabel.textContent = label;
  el.matchProgressValue.textContent = `${safePercent}%`;
  el.matchProgressBar.style.width = `${safePercent}%`;
}

function resetMatchProgress() {
  setMatchProgress(0, "Menunggu analisis", "idle");
}

async function analyzeCvMatch() {
  resetMatchProgress();
  const cvText = el.cvText.value.trim();
  const jobText = el.matchJobText.value.trim();

  if (!cvText) {
    setMatchProgress(0, "CV belum diisi", "error");
    el.matchHelperText.textContent = "CV masih kosong. Upload file atau paste teks CV terlebih dahulu.";
    return;
  }

  if (!jobText) {
    setMatchProgress(0, "Lowongan belum diisi", "error");
    el.matchHelperText.textContent = "Lowongan target masih kosong.";
    return;
  }

  if (!state.session || !state.vocab) {
    setMatchProgress(0, "Model belum siap", "error");
    el.matchHelperText.textContent = "Model belum siap. Tunggu model selesai dimuat.";
    return;
  }

  try {
    setMatchButtonLoading(true);
    setMatchProgress(8, "Memvalidasi input...");
    el.matchHelperText.textContent = "Menyiapkan analisis CV dan lowongan...";

    state.cvText = cvText;
    state.matchJobText = jobText;

    setMatchProgress(18, "Mengekstrak skill dari CV...");
    state.cvEntities = await runInference(cvText);

    setMatchProgress(52, "Mengekstrak requirement lowongan...");
    state.jobMatchEntities = await runInference(jobText);

    setMatchProgress(76, "Mencocokkan skill dan requirement...");
    state.matchResult = compareCvToJob(state.cvEntities, state.jobMatchEntities, cvText);

    setMatchProgress(92, "Merender hasil analisis...");
    renderMatchResult(state.matchResult);
    setMatchProgress(100, "Analisis selesai", "done");
    el.matchHelperText.textContent = `Analisis selesai: ${state.matchResult.matched.length} cocok, ${state.matchResult.missing.length} belum ditemukan di CV.`;
  } catch (error) {
    console.error("CV match error:", error);
    setMatchProgress(100, "Analisis gagal", "error");
    el.matchHelperText.textContent = "Analisis CV gagal. Cek console browser untuk detail.";
  } finally {
    setMatchButtonLoading(false);
  }
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
    if (state.activeMode === "match") {
      el.analyzeMatchBtn.click();
    } else {
      el.extractBtn.click();
    }
  }
});

// Mode switcher
el.extractorModeBtn.addEventListener("click", () => setMode("extractor"));
el.matchModeBtn.addEventListener("click", () => setMode("match"));

// CV file upload
el.cvFileInput.addEventListener("change", async () => {
  const file = el.cvFileInput.files?.[0];
  if (!file) return;

  try {
    el.cvFileStatus.textContent = `Membaca ${file.name}...`;
    const text = await extractDocumentText(file);

    if (!text) {
      el.cvFileStatus.textContent =
        "File berhasil dibaca, tetapi teks kosong. Jika PDF berupa scan/foto, paste teks CV secara manual.";
      return;
    }

    el.cvText.value = text;
    updateCvCounter();
    el.cvFileStatus.textContent = `${file.name} berhasil dibaca (${text.length.toLocaleString("id-ID")} karakter).`;
  } catch (error) {
    console.error("CV file parse error:", error);
    el.cvFileStatus.textContent = error.message || "Gagal membaca file. Gunakan paste manual.";
  } finally {
    el.cvFileInput.value = "";
  }
});

// CV match controls
el.cvText.addEventListener("input", updateCvCounter);
el.matchJobText.addEventListener("input", updateMatchJobCounter);

el.clearCvBtn.addEventListener("click", () => {
  el.cvText.value = "";
  state.cvText = "";
  state.cvEntities = [];
  updateCvCounter();
  el.cvFileStatus.textContent = "CV dibersihkan. Upload file baru atau paste teks CV.";
});

el.loadMatchSampleBtn.addEventListener("click", () => {
  el.matchJobText.value = SAMPLE_TEXT;
  updateMatchJobCounter();
  el.matchHelperText.textContent = "Contoh lowongan dimuat untuk mode CV Match.";
});

el.analyzeMatchBtn.addEventListener("click", analyzeCvMatch);

/* ===== INIT ===== */
renderAll();
updateCharCounter();
updateCvCounter();
updateMatchJobCounter();
initializeModel();
