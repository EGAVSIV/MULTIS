const SCANNERS_JSON = "data/scanners.json";
const TIMEFRAMES_JSON = "data/timeframes.json";
const LAST_CANDLES_JSON = "data/last-candles.json";

const state = { rawRows: [], scanner: null, scanners: [] };
const $ = id => document.getElementById(id);

function escapeHTML(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function setStatus(message, type = "") {
  const el = $("status");
  el.className = `status ${type}`;
  el.textContent = message;
}

function setConnection(text) {
  const el = $("connectionText");
  if (el) el.textContent = text;
}

async function fetchJSON(path) {
  const r = await fetch(path, { cache: "no-store" });
  if (!r.ok) throw new Error(`Missing data file: ${path}`);
  return r.json();
}

function safeFilename(name) {
  return name
    .replace(/\//g, "-")
    .replace(/ /g, "_")
    .replace(/→/g, "to")
    .replace(/–/g, "-");
}

function fillSelect(id, values, selected) {
  const el = $(id);
  el.innerHTML = "";
  values.forEach(v => {
    const o = document.createElement("option");
    o.value = v;
    o.textContent = v;
    o.selected = v === selected;
    el.appendChild(o);
  });
}

function renderTabs() {
  const box = $("scannerTabs");
  box.innerHTML = state.scanners.map(s =>
    `<button class="scanner-tab ${s.name === state.scanner ? "active" : ""}" data-name="${encodeURIComponent(s.name)}" style="--tab-color:${s.color}">${escapeHTML(s.name)}</button>`
  ).join("");
  box.querySelectorAll("button").forEach(b => b.onclick = () => {
    state.scanner = decodeURIComponent(b.dataset.name);
    $("selectedScannerName").textContent = state.scanner;
    renderTabs();
  });
  $("scannerCount").textContent = state.scanners.length;
}

async function loadSymbols() {
  const tf = $("timeframe").value.replace(/ /g, "_");
  $("symbol").innerHTML = "<option>Loading…</option>";
  const symbols = await fetchJSON(`data/symbols/${tf}.json`);
  fillSelect("symbol", symbols);
}

async function loadLastCandles() {
  const data = await fetchJSON(LAST_CANDLES_JSON);
  $("lastCandles").innerHTML = Object.entries(data).map(([tf, v]) =>
    `<div class="candle"><b>${escapeHTML(tf)}</b><br>${v ? new Date(v).toLocaleString() : "NA"}</div>`
  ).join("");
}

function renderTableData(rows) {
  const t = $("resultsTable"), e = $("emptyResults");
  if (!rows || !rows.length) {
    t.querySelector("thead").innerHTML = "";
    t.querySelector("tbody").innerHTML = "";
    e.style.display = "block";
    return;
  }
  e.style.display = "none";
  const cols = Object.keys(rows[0]);
  t.querySelector("thead").innerHTML = `<tr>${cols.map(c => `<th>${escapeHTML(c)}</th>`).join("")}</tr>`;
  t.querySelector("tbody").innerHTML = rows.map(row =>
    `<tr>${cols.map(c => {
      const v = row[c] ?? "";
      if (c === "TV_Link" && String(v).includes("http")) {
        const u = String(v).match(/\((.*?)\)/)?.[1] || v;
        return `<td><a href="${escapeHTML(u)}" target="_blank" rel="noopener">TV</a></td>`;
      }
      return `<td>${escapeHTML(v)}</td>`;
    }).join("")}</tr>`
  ).join("");
}

function setResults(rows) {
  state.rawRows = rows || [];
  renderTableData(state.rawRows);
}

function renderZones(z) {
  const x = Object.entries(z || {});
  $("zoneCard").classList.toggle("hidden", !x.length);
  $("zones").innerHTML = x.map(([n, c]) => `<div class="zone">${escapeHTML(n)}: ${escapeHTML(c)}</div>`).join("");
}

async function runScanner() {
  try {
    setStatus(`Loading ${state.scanner}…`);
    $("runBtn").disabled = true;
    const tf = $("timeframe").value.replace(/ /g, "_");
    const file = safeFilename(state.scanner);
    const result = await fetchJSON(`data/scan/${tf}/${file}.json`);
    $("summary").textContent = `${result.total_matches} matches • ${result.timeframe} • ${result.analysis_date}`;
    setResults(result.results || []);
    renderZones(result.zones || {});
    setStatus(`✓ ${state.scanner} loaded (as of ${result.analysis_date})`, "ok");
  } catch (e) {
    setStatus(`🔴 ${e.message}`, "error");
  } finally {
    $("runBtn").disabled = false;
  }
}

async function runMatrix() {
  try {
    setStatus("Loading scanner matrix…");
    $("matrixBtn").disabled = true;
    const tf = $("timeframe").value.replace(/ /g, "_");
    const symbol = $("symbol").value;
    const r = await fetchJSON(`data/matrix/${tf}/${symbol}.json`);
    $("matrixSummary").textContent = `${r.symbol} • ${r.timeframe} • ${r.analysis_date}`;
    $("matrixTable").querySelector("thead").innerHTML = "<tr><th>Scanner</th><th>Result</th></tr>";
    $("matrixTable").querySelector("tbody").innerHTML = (r.results || []).map(x =>
      `<tr><td>${escapeHTML(x.Scanner)}</td><td class="${x.Result ? "yes" : "no"}">${x.Result ? "🟢 YES" : "🔴 NO"}</td></tr>`
    ).join("");
    setStatus("✓ Scanner matrix loaded", "ok");
  } catch (e) {
    setStatus(`🔴 Matrix error: ${e.message}`, "error");
  } finally {
    $("matrixBtn").disabled = false;
  }
}

function filterTable() {
  const q = $("search").value.trim().toUpperCase();
  const filtered = q
    ? state.rawRows.filter(r => String(r.Symbol || "").toUpperCase().includes(q))
    : state.rawRows;
  renderTableData(filtered);
}

function downloadCSV() {
  if (!state.rawRows.length) return;
  const c = Object.keys(state.rawRows[0]), esc = v => `"${String(v ?? "").replaceAll('"', '""')}"`;
  const csv = [c.join(","), ...state.rawRows.map(r => c.map(k => esc(r[k])).join(","))].join("\n");
  const b = new Blob([csv], { type: "text/csv" });
  const u = URL.createObjectURL(b);
  const a = document.createElement("a");
  a.href = u;
  a.download = `${state.scanner}_${$("timeframe").value}.csv`.replaceAll(" ", "_");
  a.click();
  URL.revokeObjectURL(u);
}

async function refreshData() {
  try {
    setStatus("Re-checking latest committed scan data…");
    $("refreshBtn").disabled = true;
    await Promise.all([loadSymbols(), loadLastCandles()]);
    setStatus("✓ Refreshed from latest committed data. (New scans run automatically via GitHub Actions.)", "ok");
  } catch (e) {
    setStatus(`🔴 Refresh error: ${e.message}`, "error");
  } finally {
    $("refreshBtn").disabled = false;
  }
}

async function init() {
  $("runBtn").onclick = runScanner;
  $("matrixBtn").onclick = runMatrix;
  $("refreshBtn").onclick = refreshData;
  $("search").oninput = filterTable;
  $("csvBtn").onclick = downloadCSV;

  try {
    setConnection("LOADING");
    state.scanners = await fetchJSON(SCANNERS_JSON);
    state.scanner = state.scanners[0]?.name || null;
    $("selectedScannerName").textContent = state.scanner || "—";
    renderTabs();

    const tfs = await fetchJSON(TIMEFRAMES_JSON);
    fillSelect("timeframe", tfs, "Daily");
    await Promise.all([loadSymbols(), loadLastCandles()]);

    setConnection("STATIC DATA");
    setStatus("🟢 Loaded from static GitHub-committed scan data", "ok");
    $("timeframe").onchange = () => loadSymbols().catch(e => setStatus(`🔴 ${e.message}`, "error"));
  } catch (e) {
    setConnection("ERROR");
    setStatus(`🔴 Setup error: ${e.message} — has the "Run Scanners" workflow run yet?`, "error");
  }
}

init();
