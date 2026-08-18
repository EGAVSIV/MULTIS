// ============================================================
// IMPORTANT
// Replace this value with the public URL where backend/api.py runs.
// GitHub Pages cannot execute Python itself.
// Example: https://api.raosab.in
// ============================================================
const API = localStorage.getItem("SCANNER_API") || "http://127.0.0.1:8000";

const state = { rows: [], scanners: [] };

const $ = id => document.getElementById(id);

function setStatus(message, type="") {
  const el = $("status");
  el.className = `status ${type}`;
  el.textContent = message;
}

async function request(path, options={}) {
  const response = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options
  });

  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      message = body.detail || message;
    } catch (_) {}
    throw new Error(message);
  }

  return response.json();
}

function fillSelect(id, values, selected=null) {
  const el = $(id);
  el.innerHTML = "";
  values.forEach(value => {
    const opt = document.createElement("option");
    opt.value = typeof value === "string" ? value : value.name;
    opt.textContent = typeof value === "string" ? value : value.name;
    if (opt.value === selected) opt.selected = true;
    el.appendChild(opt);
  });
}

async function loadSymbols() {
  const tf = $("timeframe").value;
  $("symbol").innerHTML = `<option>Loading…</option>`;
  const symbols = await request(`/api/symbols?timeframe=${encodeURIComponent(tf)}`);
  fillSelect("symbol", symbols);
}

async function loadLastCandles() {
  const data = await request("/api/last-candles");
  $("lastCandles").innerHTML = Object.entries(data)
    .map(([tf, value]) => `<div class="candle"><b>${tf}</b><br>${value ? new Date(value).toLocaleString() : "NA"}</div>`)
    .join("");
}

function renderTiles() {
  const selected = $("scanner").value;
  $("scannerTiles").innerHTML = state.scanners.map(item =>
    `<button class="tile ${item.name === selected ? "active" : ""}"
      data-scanner="${encodeURIComponent(item.name)}"
      style="background:${item.color}">
      ${item.name}
    </button>`
  ).join("");

  document.querySelectorAll(".tile").forEach(button => {
    button.onclick = () => {
      $("scanner").value = decodeURIComponent(button.dataset.scanner);
      renderTiles();
    };
  });
}

function renderResults(rows) {
  state.rows = rows;
  const table = $("resultsTable");
  const empty = $("emptyResults");

  if (!rows.length) {
    table.querySelector("thead").innerHTML = "";
    table.querySelector("tbody").innerHTML = "";
    empty.textContent = "No stocks matched this scanner.";
    empty.style.display = "block";
    return;
  }

  empty.style.display = "none";
  const cols = Object.keys(rows[0]);

  table.querySelector("thead").innerHTML = `<tr>${cols.map(c => `<th>${c}</th>`).join("")}</tr>`;
  table.querySelector("tbody").innerHTML = rows.map(row =>
    `<tr>${cols.map(c => {
      const v = row[c] ?? "";
      if (c === "TV_Link" && typeof v === "string" && v.includes("http")) {
        const url = v.match(/\((.*?)\)/)?.[1] || v;
        return `<td><a href="${url}" target="_blank">TV</a></td>`;
      }
      return `<td>${String(v)}</td>`;
    }).join("")}</tr>`
  ).join("");
}

function renderZones(zones) {
  const entries = Object.entries(zones || {});
  $("zoneCard").classList.toggle("hidden", !entries.length);
  $("zones").innerHTML = entries.map(([name, count]) =>
    `<div class="zone">${name}: ${count}</div>`
  ).join("");
}

async function runScanner() {
  try {
    setStatus("Running scanner and loading latest data…");
    $("runBtn").disabled = true;

    const payload = {
      scanner: $("scanner").value,
      timeframe: $("timeframe").value,
      analysis_date: $("analysisDate").value || null
    };

    const result = await request("/api/scan", {
      method: "POST",
      body: JSON.stringify(payload)
    });

    $("summary").textContent =
      `${result.total_matches} matches • ${result.timeframe} • ${result.analysis_date}`;

    renderResults(result.results || []);
    renderZones(result.zones || {});
    setStatus(`✓ ${result.scanner} completed successfully`, "ok");
  } catch (error) {
    setStatus(`🔴 Scanner error: ${error.message}`, "error");
  } finally {
    $("runBtn").disabled = false;
  }
}

async function runMatrix() {
  try {
    setStatus("Running single-stock scanner matrix…");
    $("matrixBtn").disabled = true;

    const payload = {
      symbol: $("symbol").value,
      timeframe: $("timeframe").value,
      analysis_date: $("analysisDate").value || null
    };

    const result = await request("/api/matrix", {
      method: "POST",
      body: JSON.stringify(payload)
    });

    const rows = result.results || [];
    $("matrixSummary").textContent =
      `${result.symbol} • ${result.timeframe} • ${result.analysis_date}`;

    $("matrixTable").querySelector("thead").innerHTML =
      `<tr><th>Scanner</th><th>Result</th></tr>`;

    $("matrixTable").querySelector("tbody").innerHTML = rows.map(row =>
      `<tr><td>${row.Scanner}</td><td class="${row.Result ? "yes" : "no"}">${row.Result ? "🟢 YES" : "🔴 NO"}</td></tr>`
    ).join("");

    setStatus("✓ Scanner matrix completed", "ok");
  } catch (error) {
    setStatus(`🔴 Matrix error: ${error.message}`, "error");
  } finally {
    $("matrixBtn").disabled = false;
  }
}

function filterTable() {
  const q = $("search").value.trim().toUpperCase();
  renderResults(q ? state.rows.filter(row =>
    String(row.Symbol || "").toUpperCase().includes(q)
  ) : state.rows);
}

function downloadCSV() {
  if (!state.rows.length) return;

  const cols = Object.keys(state.rows[0]);
  const esc = value => `"${String(value ?? "").replaceAll('"', '""')}"`;
  const csv = [cols.join(","), ...state.rows.map(row => cols.map(c => esc(row[c])).join(","))].join("\n");

  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${$("scanner").value}_${$("timeframe").value}.csv`.replaceAll(" ", "_");
  a.click();
  URL.revokeObjectURL(url);
}

async function refreshData() {
  try {
    setStatus("Refreshing data from Data-Collector repository…");
    $("refreshBtn").disabled = true;
    const result = await request("/api/refresh-data", { method: "POST" });
    await loadLastCandles();
    await loadSymbols();
    setStatus(`✓ Data refreshed successfully`, "ok");
  } catch (error) {
    setStatus(`🔴 Refresh error: ${error.message}`, "error");
  } finally {
    $("refreshBtn").disabled = false;
  }
}

async function init() {
  try {
    const health = await request("/api/health");
    const [timeframes, scanners] = await Promise.all([
      request("/api/timeframes"),
      request("/api/scanners")
    ]);

    state.scanners = scanners;
    fillSelect("timeframe", timeframes, "Daily");
    fillSelect("scanner", scanners);
    renderTiles();

    await Promise.all([loadSymbols(), loadLastCandles()]);
    setStatus(`✓ ${health.service} connected`, "ok");

    $("timeframe").onchange = async () => {
      try { await loadSymbols(); } catch (error) { setStatus(`🔴 ${error.message}`, "error"); }
    };
    $("scanner").onchange = renderTiles;
    $("runBtn").onclick = runScanner;
    $("matrixBtn").onclick = runMatrix;
    $("refreshBtn").onclick = refreshData;
    $("search").oninput = filterTable;
    $("csvBtn").onclick = downloadCSV;

  } catch (error) {
    setStatus(`🔴 API connection error: ${error.message}`, "error");
  }
}

init();
