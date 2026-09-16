/**
 * Main Frontend Application JavaScript for IDS Phase 3.
 * Manages Dashboard switching, Dataset Upload/Prediction, Live Monitoring, WebSockets, & AI Agent.
 */

let activeFileId = null;
let ws = null;
let liveRowsCount = 0;
let flowCache = {};

document.addEventListener("DOMContentLoaded", () => {
    fetchInterfaces();
});

// =========================================================
// DASHBOARD SWITCHING
// =========================================================
function switchDashboard(mode) {
    const tabDataset = document.getElementById("tab-dataset");
    const tabLive = document.getElementById("tab-live");
    const dashDataset = document.getElementById("dashboard-dataset");
    const dashLive = document.getElementById("dashboard-live");

    if (mode === "dataset") {
        tabDataset.className = "px-5 py-2 rounded-lg text-sm font-semibold transition-all duration-200 bg-cyan-600 text-white shadow-md shadow-cyan-600/30";
        tabLive.className = "px-5 py-2 rounded-lg text-sm font-semibold text-slate-400 hover:text-white hover:bg-slate-900 transition-all duration-200";
        dashDataset.classList.remove("hidden");
        dashLive.classList.add("hidden");
    } else {
        tabLive.className = "px-5 py-2 rounded-lg text-sm font-semibold transition-all duration-200 bg-cyan-600 text-white shadow-md shadow-cyan-600/30";
        tabDataset.className = "px-5 py-2 rounded-lg text-sm font-semibold text-slate-400 hover:text-white hover:bg-slate-900 transition-all duration-200";
        dashLive.classList.remove("hidden");
        dashDataset.classList.add("hidden");
        initWebSocket();
    }
}

// =========================================================
// DASHBOARD 1: DATASET ANALYSIS & PREDICTION
// =========================================================
async function loadBuiltinDataset() {
    try {
        const res = await fetch("/api/dataset/builtin-validation");
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Validation failed");
        
        activeFileId = data.filename;
        renderValidationBadge(data);
    } catch (err) {
        alert(`Error loading built-in dataset: ${err.message}`);
    }
}

async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch("/api/dataset/upload", {
            method: "POST",
            body: formData,
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Upload failed");

        activeFileId = data.filename;
        renderValidationBadge(data);
    } catch (err) {
        alert(`Upload failed: ${err.message}`);
    }
}

function renderValidationBadge(data) {
    const valBadge = document.getElementById("val-badge");
    const valInfo = document.getElementById("val-info");

    valBadge.textContent = `${data.validation_status} (${data.row_count} rows, ${data.column_count} cols)`;
    if (data.validation_status === "VALID") {
        valBadge.className = "ml-2 font-bold px-2.5 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800";
        valInfo.innerHTML = `<span class="text-emerald-400 font-semibold"><i class="fa-solid fa-check-circle mr-1"></i>${data.message}</span><br>Labels Present: ${data.has_labels ? 'Yes (' + data.label_column + ')' : 'No (Prediction Only Mode)'}`;
    } else {
        valBadge.className = "ml-2 font-bold px-2.5 py-0.5 rounded bg-rose-950 text-rose-400 border border-rose-800";
        valInfo.innerHTML = `<span class="text-rose-400 font-semibold"><i class="fa-solid fa-circle-exclamation mr-1"></i>${data.message}</span>`;
    }
}

async function runPrediction(task) {
    if (!activeFileId) {
        alert("Please select the built-in UNSW-NB15 dataset or upload a file first.");
        return;
    }

    const featureMode = document.getElementById("select-feature-mode").value;
    const modelName = document.getElementById("select-model").value;

    const formData = new FormData();
    formData.append("file_id", activeFileId);
    formData.append("feature_mode", featureMode);
    formData.append("model_name", modelName);
    formData.append("prediction_task", task);

    try {
        const res = await fetch("/api/dataset/predict", {
            method: "POST",
            body: formData,
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Prediction failed");

        renderDatasetResults(data);
    } catch (err) {
        alert(`Prediction Error: ${err.message}`);
    }
}

function renderDatasetResults(data) {
    const container = document.getElementById("dataset-results-container");
    container.classList.remove("hidden");

    // Stats
    document.getElementById("stat-total").textContent = data.summary.total_records.toLocaleString();
    document.getElementById("stat-normal").textContent = data.summary.normal_count.toLocaleString();
    document.getElementById("stat-attack").textContent = data.summary.attack_count.toLocaleString();
    document.getElementById("stat-pct").textContent = `${data.summary.attack_percentage}%`;

    // Ground Truth Evaluation
    const evalCard = document.getElementById("eval-metrics-card");
    if (data.evaluation) {
        evalCard.classList.remove("hidden");
        document.getElementById("metric-acc").textContent = `${(data.evaluation.accuracy * 100).toFixed(2)}%`;
        document.getElementById("metric-prec").textContent = data.evaluation.precision.toFixed(4);
        document.getElementById("metric-rec").textContent = data.evaluation.recall.toFixed(4);
        document.getElementById("metric-f1").textContent = data.evaluation.f1_score.toFixed(4);
        document.getElementById("metric-macro").textContent = data.evaluation.macro_f1.toFixed(4);
        document.getElementById("metric-weighted").textContent = data.evaluation.weighted_f1.toFixed(4);

        // Confusion Matrix
        const cmBox = document.getElementById("confusion-matrix-box");
        cmBox.innerHTML = `<pre class="text-cyan-300 font-mono">${JSON.stringify(data.evaluation.confusion_matrix, null, 2)}</pre>`;
    } else {
        evalCard.classList.add("hidden");
    }

    // Attack Category Distribution Box
    const distBox = document.getElementById("attack-dist-box");
    if (data.summary.attack_categories) {
        distBox.innerHTML = Object.entries(data.summary.attack_categories)
            .map(([cat, count]) => `<div class="flex justify-between border-b border-slate-800/60 py-1"><span>${cat}</span><span class="font-bold text-cyan-400">${count}</span></div>`)
            .join("");
    } else {
        distBox.innerHTML = `<div class="text-slate-500 italic">Binary mode output (Normal vs Attack only)</div>`;
    }

    // Sample Records Table
    const tbody = document.getElementById("sample-table-body");
    tbody.innerHTML = data.sample_predictions.map(rec => `
        <tr class="hover:bg-slate-800/40">
            <td class="py-2 px-4 font-mono text-slate-400">#${rec.index}</td>
            <td class="py-2 px-4">
                <span class="px-2 py-0.5 rounded text-[11px] font-bold ${rec.prediction === 'Attack' ? 'bg-rose-950 text-rose-400 border border-rose-800' : 'bg-emerald-950 text-emerald-400 border border-emerald-800'}">
                    ${rec.prediction}
                </span>
            </td>
            <td class="py-2 px-4">${rec.attack_category || (rec.prediction === 'Attack' ? 'Attack' : 'Normal')}</td>
            <td class="py-2 px-4 font-mono text-slate-400">${rec.confidence ? (rec.confidence * 100).toFixed(1) + '%' : 'N/A'}</td>
        </tr>
    `).join("");
}

// =========================================================
// DASHBOARD 2: LIVE NETWORK MONITORING
// =========================================================
async function fetchInterfaces() {
    try {
        const res = await fetch("/api/live/interfaces");
        const ifaces = await res.json();
        const select = document.getElementById("select-interface");
        select.innerHTML = ifaces.map(i => `<option value="${i.name}">${i.description} ${i.is_active ? '[Active]' : ''}</option>`).join("");
    } catch (err) {
        console.error("Failed to fetch network interfaces:", err);
    }
}

async function startLiveMonitoring() {
    const iface = document.getElementById("select-interface").value;
    const testMode = document.getElementById("check-test-mode").checked;

    try {
        const res = await fetch("/api/live/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ interface_name: iface, test_mode: testMode }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Failed to start monitoring");

        document.getElementById("btn-start-live").disabled = true;
        document.getElementById("btn-stop-live").disabled = false;
        
        const badge = document.getElementById("live-status-badge");
        if (testMode) {
            badge.className = "px-3 py-1.5 rounded-lg text-xs font-bold bg-amber-950 text-amber-400 border border-amber-800 flex items-center";
            badge.innerHTML = `<span class="w-2 h-2 rounded-full bg-amber-400 pulse-dot mr-2"></span>Controlled TEST MODE`;
        } else {
            badge.className = "px-3 py-1.5 rounded-lg text-xs font-bold bg-emerald-950 text-emerald-400 border border-emerald-800 flex items-center";
            badge.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-400 pulse-dot mr-2"></span>LIVE MONITORING`;
        }

        initWebSocket();
    } catch (err) {
        alert(`Monitoring Error: ${err.message}`);
    }
}

async function stopLiveMonitoring() {
    try {
        await fetch("/api/live/stop", { method: "POST" });
        document.getElementById("btn-start-live").disabled = false;
        document.getElementById("btn-stop-live").disabled = true;

        const badge = document.getElementById("live-status-badge");
        badge.className = "px-3 py-1.5 rounded-lg text-xs font-bold bg-slate-800 text-slate-400 flex items-center";
        badge.innerHTML = `<span class="w-2 h-2 rounded-full bg-slate-500 mr-2"></span>IDLE`;
    } catch (err) {
        console.error("Error stopping monitoring:", err);
    }
}

// WebSocket Stream Handler
function initWebSocket() {
    if (ws && ws.readyState === WebSocket.OPEN) return;

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${protocol}//${window.location.host}/api/live/ws`);

    const indicator = document.getElementById("ws-indicator");

    ws.onopen = () => {
        indicator.innerHTML = `<i class="fa-solid fa-circle-dot text-emerald-400 pulse-dot mr-1"></i>WebSocket Connected`;
    };

    ws.onmessage = (event) => {
        const flow = JSON.parse(event.data);
        handleLiveFlowEvent(flow);
    };

    ws.onclose = () => {
        indicator.innerHTML = `<i class="fa-solid fa-circle-dot text-slate-600 mr-1"></i>WebSocket Disconnected`;
    };
}

function handleLiveFlowEvent(flow) {
    flowCache[flow.flow_id] = flow;

    // Stat totals
    if (flow.system_totals) {
        document.getElementById("live-stat-packets").textContent = flow.system_totals.packet_count.toLocaleString();
        document.getElementById("live-stat-flows").textContent = flow.system_totals.flow_count.toLocaleString();
        document.getElementById("live-stat-normal").textContent = flow.system_totals.normal_count.toLocaleString();
        document.getElementById("live-stat-attack").textContent = flow.system_totals.attack_count.toLocaleString();
    }

    const tbody = document.getElementById("live-feed-body");
    const noRow = document.getElementById("no-live-data-row");
    if (noRow) noRow.remove();

    liveRowsCount++;
    const isAttack = flow.binary_label === 1;

    const tr = document.createElement("tr");
    tr.className = `hover:bg-slate-800/60 transition ${isAttack ? 'bg-rose-950/20' : ''}`;
    tr.innerHTML = `
        <td class="py-2.5 px-3 font-mono text-[11px] text-slate-400">${flow.timestamp.split(' ')[1]}</td>
        <td class="py-2.5 px-3 font-mono text-[11px] text-slate-200">${flow.flow_key}</td>
        <td class="py-2.5 px-3"><span class="px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-mono text-[10px]">${flow.service}</span></td>
        <td class="py-2.5 px-3">
            <span class="px-2 py-0.5 rounded text-[10px] font-bold ${isAttack ? 'bg-rose-950 text-rose-400 border border-rose-800' : 'bg-emerald-950 text-emerald-400 border border-emerald-800'}">
                ${flow.prediction}
            </span>
        </td>
        <td class="py-2.5 px-3 font-semibold ${isAttack ? 'text-rose-300' : 'text-slate-400'}">${flow.attack_category}</td>
        <td class="py-2.5 px-3 font-mono text-slate-400">${(flow.confidence * 100).toFixed(1)}%</td>
        <td class="py-2.5 px-3 text-right">
            <button onclick="openAiAgentModal('${flow.flow_id}')" class="bg-cyan-950 hover:bg-cyan-900 text-cyan-300 px-3 py-1 rounded text-[11px] font-semibold border border-cyan-800/80 transition">
                <i class="fa-solid fa-robot mr-1"></i>AI Advice
            </button>
        </td>
    `;

    tbody.insertBefore(tr, tbody.firstChild);

    if (tbody.children.length > 50) {
        tbody.removeChild(tbody.lastChild);
    }
}

// =========================================================
// AI SECURITY AGENT MODAL
// =========================================================
async function openAiAgentModal(flowId) {
    const flow = flowCache[flowId];
    if (!flow) return;

    const modal = document.getElementById("ai-modal");
    document.getElementById("ai-flow-id").textContent = `Flow ID: ${flow.flow_id} (${flow.flow_key})`;
    document.getElementById("ai-summary").textContent = "Analyzing threat via AI Security Agent...";
    document.getElementById("ai-explanation").textContent = "Loading technical assessment...";
    document.getElementById("ai-actions").innerHTML = "<li>Loading recommendations...</li>";
    document.getElementById("ai-guidance").innerHTML = "<li>Loading guidance...</li>";

    modal.classList.remove("hidden");

    try {
        const res = await fetch("/api/live/ai-recommendation", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                flow_id: flow.flow_id,
                src_ip: flow.src_ip,
                dst_ip: flow.dst_ip,
                service: flow.service,
                attack_category: flow.attack_category,
                confidence: flow.confidence,
                flow_info: flow.flow_stats,
            }),
        });
        const rec = await res.json();
        if (!res.ok) throw new Error(rec.detail || "AI Agent request failed");

        renderAiModalContent(rec);
    } catch (err) {
        alert(`AI Agent Error: ${err.message}`);
        closeAiModal();
    }
}

function renderAiModalContent(rec) {
    document.getElementById("ai-summary").textContent = rec.threat_summary;
    document.getElementById("ai-explanation").textContent = rec.explanation;

    const severityEl = document.getElementById("ai-severity");
    severityEl.textContent = rec.severity;
    if (rec.severity === "Critical") severityEl.className = "inline-block px-3 py-1 rounded font-bold text-xs bg-rose-950 text-rose-400 border border-rose-800";
    else if (rec.severity === "High") severityEl.className = "inline-block px-3 py-1 rounded font-bold text-xs bg-amber-950 text-amber-400 border border-amber-800";
    else severityEl.className = "inline-block px-3 py-1 rounded font-bold text-xs bg-cyan-950 text-cyan-400 border border-cyan-800";

    document.getElementById("ai-actions").innerHTML = rec.recommended_actions.map(a => `<li>${a}</li>`).join("");
    document.getElementById("ai-guidance").innerHTML = rec.investigation_guidance.map(g => `<li>${g}</li>`).join("");
}

function closeAiModal() {
    document.getElementById("ai-modal").classList.add("hidden");
}
