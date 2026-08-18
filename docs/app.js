const $ = id => document.getElementById(id);
const ROOT = "../SCANNER_RESULTS";

const SCANNERS = [
"RSI Market Pulse","Volume Shocker","NRB-7 Breakout","Counter Attack","Breakaway Gaps",
"RSI + ADX","MACD Market Pulse","MACD Normal Divergence","Trend Alignment (EMA)",
"Pullback to EMA","High Probability Confluence","MACD Hook Up","MACD Hook Down",
"MACD Histogram Divergence","EMA50 + Stoch Oversold","Dark Cloud Cover","Morning Star (Bottom)",
"Evening Star (Top)","Bullish GSAS","Bearish GSAS","50 EMA Fake Breakdown","50 EMA Fake Breakout",
"KDJ BUY (Oversold)","KDJ SELL (Overbought)","Probable Momentum (Consecutive Close)",
"Camarilla Breakout / Breakdown","CPR Breakout / Breakdown","Inside Bar Breakout",
"ADX Expansion (Trend Ignition)","Range Expansion Day","Failed Breakout / Breakdown",
"EMA Compression → Expansion","Top 10 by ATR %","Liquidity Sweep Reversal","Island Reversal",
"Wyckoff Spring / Upthrust","Smart Money Trap","Bump & Run Reversal","Exhaustion Bar",
"Shakeout / Trap","Hidden Pivot Reversal","Springer Reversal","RSI + MACD Cross Swing","RSI Swing"
];

let state = {scanner: SCANNERS[0], rows: [], manifest: null};

function slug(s){ return String(s).replace(/[^A-Za-z0-9]+/g,"_").replace(/^_|_$/g,"").toLowerCase(); }

async function getJSON(url){
  const r = await fetch(url + "?v=" + Date.now(), {cache:"no-store"});
  if(!r.ok) throw new Error(`Cannot load ${url} (${r.status})`);
  return r.json();
}

function renderTabs(){
  const host = $("scannerTabs");
  if(!host) return;
  host.innerHTML = SCANNERS.map(s =>
    `<button class="scanner-tab ${s===state.scanner?"active":""}" data-scanner="${s}">${s}</button>`
  ).join("");
  host.onclick = e => {
    const s = e.target.dataset.scanner;
    if(!s) return;
    state.scanner=s;
    if($("selectedScannerName")) $("selectedScannerName").textContent=s;
    renderTabs();
    loadSelected().catch(showError);
  };
  if($("scannerCount")) $("scannerCount").textContent=SCANNERS.length;
}

function fillTimeframes(){
  const sel=$("timeframe");
  if(!sel) return;
  const tfs = state.manifest ? Object.keys(state.manifest.timeframes||{}) : ["Daily"];
  sel.innerHTML=tfs.map(tf=>`<option value="${tf}" ${tf==="Daily"?"selected":""}>${tf}</option>`).join("");
  sel.onchange=()=>loadSelected().catch(showError);
}

function renderTable(rows, id="resultsTable"){
  const table=$(id);
  if(!table) return;
  const thead=table.querySelector("thead"), tbody=table.querySelector("tbody");
  thead.innerHTML=""; tbody.innerHTML="";
  if(!rows.length){
    if($("emptyResults")) $("emptyResults").style.display="block";
    return;
  }
  if($("emptyResults")) $("emptyResults").style.display="none";
  const cols=[...new Set(rows.flatMap(r=>Object.keys(r)))];
  thead.innerHTML=`<tr>${cols.map(c=>`<th>${c}</th>`).join("")}</tr>`;
  tbody.innerHTML=rows.map(r=>`<tr>${cols.map(c=>`<td>${r[c] ?? ""}</td>`).join("")}</tr>`).join("");
}

async function loadSelected(){
  const tf=$("timeframe")?.value || "Daily";
  const file=`${ROOT}/${slug(state.scanner)}/${slug(tf)}.json`;
  const data=await getJSON(file);
  state.rows=data.rows||[];
  renderTable(state.rows);
  if($("summary")) $("summary").textContent=
    `${data.match_count||0} matches • ${data.symbols_scanned||0} stocks scanned • Updated ${new Date(data.generated_at).toLocaleString()}`;
}

function showError(e){
  console.error(e);
  if($("summary")) $("summary").textContent="Results not available yet. Run RUN_ALL_SCANNERS.bat first.";
  renderTable([]);
}

function setupButtons(){
  if($("runBtn")) $("runBtn").onclick=()=>loadSelected().catch(showError);
  if($("refreshBtn")) $("refreshBtn").onclick=async()=>{
    try{
      state.manifest=await getJSON(`${ROOT}/manifest.json`);
      fillTimeframes();
      await loadSelected();
    }catch(e){showError(e);}
  };
  if($("csvBtn")) $("csvBtn").onclick=()=>{
    if(!state.rows.length) return;
    const cols=[...new Set(state.rows.flatMap(r=>Object.keys(r)))];
    const csv=[cols.join(","),...state.rows.map(r=>cols.map(c=>JSON.stringify(r[c]??"")).join(","))].join("\n");
    const a=document.createElement("a");
    a.href=URL.createObjectURL(new Blob([csv],{type:"text/csv"}));
    a.download=`${slug(state.scanner)}_${slug($("timeframe")?.value||"Daily")}.csv`;
    a.click();
  };
  if($("search")) $("search").oninput=e=>{
    const q=e.target.value.toLowerCase();
    renderTable(state.rows.filter(r=>JSON.stringify(r).toLowerCase().includes(q)));
  };
}

async function init(){
  renderTabs();
  setupButtons();
  try{
    state.manifest=await getJSON(`${ROOT}/manifest.json`);
    fillTimeframes();
    await loadSelected();
  }catch(e){showError(e);}
}
init();
