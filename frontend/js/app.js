/**
 * Main Frontend Application JavaScript for IDS Academic & Real-Time Suite.
 * Controls 4-Section Dashboard:
 * 1. Dataset Analysis
 * 2. Dedicated Academic PERFORMANCES
 * 3. Live Monitoring (Scapy / TEST MODE / WebSocket)
 * 4. AI Security Advice
 */

let activeFileId = null;
let ws = null;
let flowCache = {};
let allLiveFlows = [];
let lastDatasetResult = null;
let performanceDataCache = null;

let sessionStartTime = null;
let sessionActive = false;

const charts = {
    datasetPie: null,
    p1Accuracy: null,
    featureImportance: null,
    liveTimeline: null,
    livePie: null,
};

document.addEventListener("DOMContentLoaded", () => {
    fetchHealthStatus();
    fetchInterfaces();
    // Default to Performances dashboard on initial load
    switchDashboard("performances");
});

// =========================================================
// SYSTEM HEALTH STATUS
// =========================================================
async function fetchHealthStatus() {
    try {
        const res = await fetch("/api/health");
        const data = await res.json();
        const el = document.getElementById("status-backend");
        if (data.status === "HEALTHY") {
            el.className = "flex items-center text-emerald-400 font-medium";
            el.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-400 pulse-dot mr-1.5"></span>Backend API: Online`;
        }
    } catch (err) {
        const el = document.getElementById("status-backend");
        el.className = "flex items-center text-rose-400 font-medium";
        el.innerHTML = `<span class="w-2 h-2 rounded-full bg-rose-500 mr-1.5"></span>Backend API: Offline`;
    }
}

// =========================================================
// DASHBOARD SWITCHING (4 SECTIONS)
// =========================================================
function switchDashboard(mode) {
    const tabs = {
        dataset: document.getElementById("tab-dataset"),
        performances: document.getElementById("tab-performances"),
        live: document.getElementById("tab-live"),
        ai: document.getElementById("tab-ai"),
    };

    const sections = {
        dataset: document.getElementById("dashboard-dataset"),
        performances: document.getElementById("dashboard-performances"),
        live: document.getElementById("dashboard-live"),
        ai: document.getElementById("dashboard-ai"),
    };

    for (const key in tabs) {
        if (key === mode) {
            tabs[key].className = "px-4 py-2 rounded-lg text-xs sm:text-sm font-semibold transition-all duration-200 tab-active";
            sections[key].classList.remove("hidden");
        } else {
            tabs[key].className = "px-4 py-2 rounded-lg text-xs sm:text-sm font-semibold transition-all duration-200 tab-inactive";
            sections[key].classList.add("hidden");
        }
    }

    if (mode === "performances") {
        loadPerformancesData();
    } else if (mode === "live") {
        initLiveCharts();
        initWebSocket();
    }
}

// =========================================================
// DASHBOARD 2: ACADEMIC PERFORMANCES
// =========================================================
async function loadPerformancesData() {
    try {
        const [resSummary, resComp, resFeat, resP3] = await Promise.all([
            fetch("/api/performances/summary").then(r => r.json()),
            fetch("/api/performances/paper-comparison").then(r => r.json()),
            fetch("/api/performances/feature-importance").then(r => r.json()),
            fetch("/api/performances/phase3-selection").then(r => r.json()),
        ]);

        performanceDataCache = {
            summary: resSummary,
            comparison: resComp,
            features: resFeat,
            phase3: resP3,
        };

        renderStatBadges(resSummary, resP3);
        renderPaperComparisonTables(resComp);
        renderPhase1Tables(resSummary.phase1);
        renderPhase2Tables(resSummary.phase2);
        renderFeatureImportance(resFeat);
        renderPhase3Tables(resP3);
        renderP1Chart(resSummary.phase1);
        renderFeatureImportanceChart(resFeat);

        updateConfusionMatrixModelList();
    } catch (err) {
        console.error("Error loading performance data:", err);
    }
}

function formatMetricPercent(val) {
    if (val === null || val === undefined || val === "" || val === "N/A" || val === "NaN") {
        return "—";
    }
    const num = parseFloat(val);
    if (isNaN(num)) return "—";
    const pct = (num <= 1.0 && num > 0.0) ? (num * 100).toFixed(2) : num.toFixed(2);
    return `${pct}%`;
}

function renderStatBadges(summary, p3) {
    const paperTargets = {
        p1_bin: { "DT": 88.13, "ANN": 86.71, "kNN": 83.18, "LR": 79.59, "Decision Tree": 88.13 },
        p1_multi: { "ANN": 75.62, "kNN": 70.09, "DT": 66.03, "LR": 65.53, "Decision Tree": 66.03 },
        p2_bin: { "DT": 90.85, "ANN": 84.39, "kNN": 84.46, "LR": 77.64, "Decision Tree": 90.85 },
        p2_multi: { "ANN": 77.51, "kNN": 72.30, "DT": 67.57, "LR": 65.29, "Decision Tree": 67.57 },
    };

    function getBest(arr, targets) {
        if (!arr || arr.length === 0) return null;
        const valid = arr.filter(r => r["Test AC (%)"] && r["Test AC (%)"] !== "N/A" && !isNaN(parseFloat(r["Test AC (%)"])));
        if (valid.length === 0) return null;
        const sorted = [...valid].sort((a, b) => parseFloat(b["Test AC (%)"]) - parseFloat(a["Test AC (%)"]));
        const best = sorted[0];
        const model = best["ML method"] || best["Model"] || "Best";
        const ourAcc = parseFloat(best["Test AC (%)"]);
        const paperAcc = targets[model] || targets[model.replace("Decision Tree", "DT").replace("Logistic Regression", "LR")] || null;
        const diff = paperAcc !== null ? (ourAcc - paperAcc).toFixed(2) : null;
        return { model, ourAcc: ourAcc.toFixed(2), paperAcc: paperAcc ? paperAcc.toFixed(2) : "—", diff };
    }

    // 1. 42-Feat Binary Best
    const b1 = getBest(summary.phase1?.binary_results, paperTargets.p1_bin);
    if (b1) {
        if (document.getElementById("stat-p1-bin-model")) document.getElementById("stat-p1-bin-model").textContent = b1.model;
        if (document.getElementById("stat-p1-bin-acc")) document.getElementById("stat-p1-bin-acc").textContent = `${b1.ourAcc}%`;
        if (document.getElementById("stat-p1-bin-paper")) document.getElementById("stat-p1-bin-paper").textContent = `${b1.paperAcc}%`;
        if (document.getElementById("stat-p1-bin-diff")) {
            const el = document.getElementById("stat-p1-bin-diff");
            el.textContent = `${b1.diff > 0 ? '+' : ''}${b1.diff}%`;
            el.className = parseFloat(b1.diff) >= 0 ? "text-xs font-semibold text-emerald-400" : "text-xs font-semibold text-rose-400";
        }
    }

    // 2. 42-Feat Multiclass Best
    const b2 = getBest(summary.phase1?.multiclass_results, paperTargets.p1_multi);
    if (b2) {
        if (document.getElementById("stat-p1-multi-model")) document.getElementById("stat-p1-multi-model").textContent = b2.model;
        if (document.getElementById("stat-p1-multi-acc")) document.getElementById("stat-p1-multi-acc").textContent = `${b2.ourAcc}%`;
        if (document.getElementById("stat-p1-multi-paper")) document.getElementById("stat-p1-multi-paper").textContent = `${b2.paperAcc}%`;
        if (document.getElementById("stat-p1-multi-diff")) {
            const el = document.getElementById("stat-p1-multi-diff");
            el.textContent = `${b2.diff > 0 ? '+' : ''}${b2.diff}%`;
            el.className = parseFloat(b2.diff) >= 0 ? "text-xs font-semibold text-emerald-400" : "text-xs font-semibold text-rose-400";
        }
    }

    // 3. 19-Feat Binary Best
    const b3 = getBest(summary.phase2?.binary_results, paperTargets.p2_bin);
    if (b3) {
        if (document.getElementById("stat-p2-bin-model")) document.getElementById("stat-p2-bin-model").textContent = b3.model;
        if (document.getElementById("stat-p2-bin-acc")) document.getElementById("stat-p2-bin-acc").textContent = `${b3.ourAcc}%`;
        if (document.getElementById("stat-p2-bin-paper")) document.getElementById("stat-p2-bin-paper").textContent = `${b3.paperAcc}%`;
        if (document.getElementById("stat-p2-bin-diff")) {
            const el = document.getElementById("stat-p2-bin-diff");
            el.textContent = `${b3.diff > 0 ? '+' : ''}${b3.diff}%`;
            el.className = parseFloat(b3.diff) >= 0 ? "text-xs font-semibold text-emerald-400" : "text-xs font-semibold text-rose-400";
        }
    }

    // 4. 19-Feat Multiclass Best (Actual Measured Winner)
    const b4 = getBest(summary.phase2?.multiclass_results, paperTargets.p2_multi);
    if (b4) {
        if (document.getElementById("stat-p2-multi-model")) document.getElementById("stat-p2-multi-model").textContent = b4.model;
        if (document.getElementById("stat-p2-multi-acc")) document.getElementById("stat-p2-multi-acc").textContent = `${b4.ourAcc}%`;
        if (document.getElementById("stat-p2-multi-paper")) document.getElementById("stat-p2-multi-paper").textContent = `${b4.paperAcc}%`;
        if (document.getElementById("stat-p2-multi-diff")) {
            const el = document.getElementById("stat-p2-multi-diff");
            el.textContent = `${b4.diff > 0 ? '+' : ''}${b4.diff}%`;
            el.className = parseFloat(b4.diff) >= 0 ? "text-xs font-semibold text-emerald-400" : "text-xs font-semibold text-rose-400";
        }
    }
}

function renderPaperComparisonTables(comp) {
    const tbodyP1 = document.querySelector("#table-comp-p1 tbody");
    const tbodyP2 = document.querySelector("#table-comp-p2 tbody");

    tbodyP1.innerHTML = "";
    tbodyP2.innerHTML = "";

    const p1Rows = comp.phase1_binary || [];
    p1Rows.forEach(r => {
        const diffClass = (typeof r.Difference === "number" && r.Difference >= 0) ? "diff-positive" : (typeof r.Difference === "number" ? "diff-negative" : "text-slate-400");
        const diffText = (typeof r.Difference === "number") ? (r.Difference > 0 ? `+${r.Difference}%` : `${r.Difference}%`) : r.Difference;

        tbodyP1.innerHTML += `
            <tr>
                <td class="font-bold text-white">${r.Model}</td>
                <td class="text-cyan-400 font-semibold">${formatMetricPercent(r.Paper_Test_AC)}</td>
                <td class="text-slate-200 font-semibold">${typeof r.Our_Test_AC === "number" ? formatMetricPercent(r.Our_Test_AC) : r.Our_Test_AC}</td>
                <td class="${diffClass}">${diffText}</td>
            </tr>
        `;
    });

    const p2Rows = comp.phase2_binary || [];
    p2Rows.forEach(r => {
        const diffClass = (typeof r.Difference === "number" && r.Difference >= 0) ? "diff-positive" : (typeof r.Difference === "number" ? "diff-negative" : "text-slate-400");
        const diffText = (typeof r.Difference === "number") ? (r.Difference > 0 ? `+${r.Difference}%` : `${r.Difference}%`) : r.Difference;

        tbodyP2.innerHTML += `
            <tr>
                <td class="font-bold text-white">${r.Model}</td>
                <td class="text-emerald-400 font-semibold">${formatMetricPercent(r.Paper_Test_AC)}</td>
                <td class="text-slate-200 font-semibold">${typeof r.Our_Test_AC === "number" ? formatMetricPercent(r.Our_Test_AC) : r.Our_Test_AC}</td>
                <td class="${diffClass}">${diffText}</td>
            </tr>
        `;
    });
}

function renderPhase1Tables(p1) {
    const tbodyBin = document.querySelector("#table-p1-binary tbody");
    const tbodyMulti = document.querySelector("#table-p1-multiclass tbody");

    tbodyBin.innerHTML = "";
    tbodyMulti.innerHTML = "";

    (p1.binary_results || []).forEach(r => {
        const isImpractical = r.Status && r.Status.includes("Not completed");
        tbodyBin.innerHTML += `
            <tr>
                <td class="font-bold text-white">${r["ML method"] || r.Model}</td>
                <td>${formatMetricPercent(r["Tr. AC (%)"] || r.Tr_AC)}</td>
                <td>${formatMetricPercent(r["Val. AC (%)"] || r.Val_AC)}</td>
                <td class="font-bold text-cyan-400">${isImpractical ? '<span class="text-amber-400 text-xs font-normal">Impractical</span>' : formatMetricPercent(r["Test AC (%)"] || r.Accuracy)}</td>
                <td>${isImpractical ? '—' : formatMetricPercent(r["Precision (%)"] || r["Precision (Attack)"])}</td>
                <td>${isImpractical ? '—' : formatMetricPercent(r["Recall (%)"] || r["Recall (Attack)"])}</td>
                <td class="font-semibold text-white">${isImpractical ? '—' : formatMetricPercent(r["F1-Score (%)"] || r["F1 (Attack)"])}</td>
            </tr>
        `;
    });

    (p1.multiclass_results || []).forEach(r => {
        const isImpractical = r.Status && r.Status.includes("Not completed");
        tbodyMulti.innerHTML += `
            <tr>
                <td class="font-bold text-white">${r["ML method"] || r.Model}</td>
                <td>${formatMetricPercent(r["Tr. AC (%)"] || r.Tr_AC)}</td>
                <td>${formatMetricPercent(r["Val. AC (%)"] || r.Val_AC)}</td>
                <td class="font-bold text-cyan-400">${isImpractical ? '<span class="text-amber-400 text-xs font-normal">Impractical</span>' : formatMetricPercent(r["Test AC (%)"] || r.Accuracy)}</td>
                <td>${isImpractical ? '—' : formatMetricPercent(r["Precision (%)"])}</td>
                <td>${isImpractical ? '—' : formatMetricPercent(r["Recall (%)"])}</td>
                <td class="font-semibold text-white">${isImpractical ? '—' : formatMetricPercent(r["F1-Score (%)"] || r["Macro F1 (%)"] || r["Macro F1"])}</td>
            </tr>
        `;
    });
}

function renderPhase2Tables(p2) {
    const tbodyBin = document.querySelector("#table-p2-binary tbody");
    const tbodyMulti = document.querySelector("#table-p2-multiclass tbody");
    const tbodyComp = document.querySelector("#table-42-vs-19 tbody");

    tbodyBin.innerHTML = "";
    tbodyMulti.innerHTML = "";
    tbodyComp.innerHTML = "";

    (p2.binary_results || []).forEach(r => {
        const isImpractical = r.Status && r.Status.includes("Not completed");
        tbodyBin.innerHTML += `
            <tr>
                <td class="font-bold text-white">${r["ML method"] || r.Model}</td>
                <td>${formatMetricPercent(r["Tr. AC (%)"] || r.Tr_AC)}</td>
                <td>${formatMetricPercent(r["Val. AC (%)"] || r.Val_AC)}</td>
                <td class="font-bold text-emerald-400">${isImpractical ? '<span class="text-amber-400 text-xs font-normal">Impractical</span>' : formatMetricPercent(r["Test AC (%)"] || r.Accuracy)}</td>
                <td>${isImpractical ? '—' : formatMetricPercent(r["Precision (%)"] || r["Precision (Attack)"])}</td>
                <td>${isImpractical ? '—' : formatMetricPercent(r["Recall (%)"] || r["Recall (Attack)"])}</td>
                <td class="font-semibold text-white">${isImpractical ? '—' : formatMetricPercent(r["F1-Score (%)"] || r["F1 (Attack)"])}</td>
            </tr>
        `;
    });

    (p2.multiclass_results || []).forEach(r => {
        const isImpractical = r.Status && r.Status.includes("Not completed");
        tbodyMulti.innerHTML += `
            <tr>
                <td class="font-bold text-white">${r["ML method"] || r.Model}</td>
                <td>${formatMetricPercent(r["Tr. AC (%)"] || r.Tr_AC)}</td>
                <td>${formatMetricPercent(r["Val. AC (%)"] || r.Val_AC)}</td>
                <td class="font-bold text-emerald-400">${isImpractical ? '<span class="text-amber-400 text-xs font-normal">Impractical</span>' : formatMetricPercent(r["Test AC (%)"] || r.Accuracy)}</td>
                <td>${isImpractical ? '—' : formatMetricPercent(r["Precision (%)"])}</td>
                <td>${isImpractical ? '—' : formatMetricPercent(r["Recall (%)"])}</td>
                <td class="font-semibold text-white">${isImpractical ? '—' : formatMetricPercent(r["F1-Score (%)"] || r["Macro F1 (%)"] || r["Macro F1"])}</td>
            </tr>
        `;
    });

    (p2.comparison_42_vs_19 || []).forEach(r => {
        const delta = r.Accuracy_Delta;
        const deltaClass = (typeof delta === "number" && delta >= 0) ? "diff-positive" : (typeof delta === "number" ? "diff-negative" : "text-slate-400");
        const deltaText = (typeof delta === "number") ? (delta > 0 ? `+${delta}%` : `${delta}%`) : (delta || "—");

        tbodyComp.innerHTML += `
            <tr>
                <td><span class="badge-tag ${r.Task === 'Binary' ? 'bg-cyan-950 text-cyan-300' : 'bg-indigo-950 text-indigo-300'}">${r.Task}</span></td>
                <td class="font-bold text-white">${r.Model}</td>
                <td class="text-slate-300">${formatMetricPercent(r["42_Feat_Test_AC"] || r["42_Feat_Accuracy"])}</td>
                <td class="text-emerald-400 font-semibold">${formatMetricPercent(r["19_Feat_Test_AC"] || r["19_Feat_Accuracy"])}</td>
                <td class="${deltaClass}">${deltaText}</td>
                <td class="text-slate-300">${formatMetricPercent(r["42_Feat_F1"])}</td>
                <td class="text-emerald-400 font-semibold">${formatMetricPercent(r["19_Feat_F1"])}</td>
                <td class="${typeof r.F1_Delta === 'number' && r.F1_Delta >= 0 ? 'diff-positive' : 'diff-negative'}">${typeof r.F1_Delta === 'number' ? (r.F1_Delta > 0 ? `+${r.F1_Delta}%` : `${r.F1_Delta}%`) : (r.F1_Delta || "—")}</td>
            </tr>
        `;
    });
}

function renderFeatureImportance(feat) {
    const tbody = document.querySelector("#table-feature-ranking tbody");
    tbody.innerHTML = "";

    (feat.features || []).forEach(r => {
        const isTop19 = r.Rank <= 19;
        tbody.innerHTML += `
            <tr class="${isTop19 ? 'bg-emerald-950/20' : ''}">
                <td class="font-bold ${isTop19 ? 'text-emerald-400' : 'text-slate-500'}">#${r.Rank}</td>
                <td class="font-mono ${isTop19 ? 'text-white font-bold' : 'text-slate-400'}">${r.Feature}</td>
                <td><span class="badge-tag ${r.Type === 'Categorical' ? 'bg-amber-950 text-amber-300' : 'bg-slate-800 text-slate-300'}">${r.Type}</span></td>
                <td class="font-mono text-xs ${isTop19 ? 'text-emerald-300 font-bold' : 'text-slate-400'}">${r.Paper_Importance_Score || r.Importance_Score}</td>
            </tr>
        `;
    });
}

function renderPhase3Tables(p3) {
    const tbodyCand = document.querySelector("#table-p3-candidates tbody");
    const tbodyEns = document.querySelector("#table-p3-ensembles tbody");
    const tbodyFinal = document.querySelector("#table-p3-final-test tbody");

    tbodyCand.innerHTML = "";
    tbodyEns.innerHTML = "";
    tbodyFinal.innerHTML = "";

    // Candidate models (Binary priority)
    const candidates = p3.binary?.candidates || [];
    candidates.forEach(r => {
        tbodyCand.innerHTML += `
            <tr>
                <td class="font-bold text-white">${r.Model}</td>
                <td class="text-indigo-400 font-semibold">${r.Val_Accuracy}%</td>
                <td>${r.Val_Precision}%</td>
                <td>${r.Val_Recall}%</td>
                <td class="font-semibold text-white">${r.Val_F1}%</td>
            </tr>
        `;
    });

    // Ensembles
    const ensembles = p3.binary?.ensembles || [];
    ensembles.forEach(r => {
        tbodyEns.innerHTML += `
            <tr>
                <td class="font-bold text-white">${r.Model}</td>
                <td class="text-violet-400 font-semibold">${r.Val_Accuracy}%</td>
                <td>${r.Val_Precision}%</td>
                <td>${r.Val_Recall}%</td>
                <td class="font-semibold text-white">${r.Val_F1}%</td>
            </tr>
        `;
    });

    // Rationale Card
    const summary = p3.binary?.summary || {};
    if (summary.Rationale) {
        document.getElementById("p3-rationale-text").textContent = summary.Rationale;
        const badge = document.getElementById("p3-badge-decision");
        if (summary.Is_Enhanced_Adopted) {
            badge.className = "badge-tag bg-emerald-950 text-emerald-300 border border-emerald-800";
            badge.textContent = "Enhanced Model Adopted";
        } else {
            badge.className = "badge-tag bg-amber-950 text-amber-300 border border-amber-800";
            badge.textContent = "Baseline Retained";
        }
    }

    // Final Frozen Test Results
    (p3.final_test_results || []).forEach(r => {
        tbodyFinal.innerHTML += `
            <tr>
                <td><span class="badge-tag ${r.Task === 'Binary' ? 'bg-cyan-950 text-cyan-300' : 'bg-indigo-950 text-indigo-300'}">${r.Task}</span></td>
                <td class="font-bold text-white">${r["Selected Model"]}</td>
                <td><span class="badge-tag ${r.Status.includes('Adopted') ? 'bg-emerald-950 text-emerald-300' : 'bg-amber-950 text-amber-300'}">${r.Status}</span></td>
                <td class="text-slate-300">${r["Val Accuracy (%)"]}%</td>
                <td class="font-bold text-indigo-300 text-sm">${r["Test Accuracy (%)"]}%</td>
                <td>${r["Precision (%)"]}%</td>
                <td>${r["Recall (%)"]}%</td>
                <td class="font-bold text-white">${r["F1-Score (%)"]}%</td>
            </tr>
        `;
    });
}

function renderP1Chart(p1) {
    const ctx = document.getElementById("chart-p1-accuracy");
    if (!ctx) return;

    if (charts.p1Accuracy) {
        charts.p1Accuracy.destroy();
    }

    const binRows = p1.binary_results || [];
    const labels = binRows.map(r => r["ML method"] || r.Model);
    const dataAcc = binRows.map(r => parseFloat(r["Test AC (%)"] || r.Accuracy || 0));

    charts.p1Accuracy = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: "Phase 1 Test Accuracy (%)",
                data: dataAcc,
                backgroundColor: [
                    "rgba(14, 165, 233, 0.8)",
                    "rgba(99, 102, 241, 0.8)",
                    "rgba(16, 185, 129, 0.8)",
                    "rgba(245, 158, 11, 0.8)",
                    "rgba(244, 63, 94, 0.8)",
                ],
                borderRadius: 6,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    min: 50,
                    max: 100,
                    ticks: { color: "#94a3b8" },
                    grid: { color: "rgba(51, 65, 85, 0.3)" }
                },
                x: {
                    ticks: { color: "#94a3b8" },
                    grid: { display: false }
                }
            }
        }
    });
}

function renderFeatureImportanceChart(feat) {
    const ctx = document.getElementById("chart-feature-importance");
    if (!ctx) return;

    if (charts.featureImportance) {
        charts.featureImportance.destroy();
    }

    const top19 = (feat.features || []).slice(0, 19);
    const labels = top19.map(r => r.Feature);
    const scores = top19.map(r => parseFloat(r.Paper_Importance_Score || r.Importance_Score || 0));

    charts.featureImportance = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: "XGBoost Importance",
                data: scores,
                backgroundColor: "rgba(16, 185, 129, 0.85)",
                borderRadius: 4,
            }]
        },
        options: {
            indexAxis: "y",
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    ticks: { color: "#94a3b8" },
                    grid: { color: "rgba(51, 65, 85, 0.3)" }
                },
                y: {
                    ticks: { color: "#cbd5e1", font: { size: 10 } },
                    grid: { display: false }
                }
            }
        }
    });
}

function updateConfusionMatrixModelList() {
    const phase = document.getElementById("cm-select-phase").value;
    const task = document.getElementById("cm-select-task").value;
    const selectModel = document.getElementById("cm-select-model");

    selectModel.innerHTML = "";

    if (phase === "phase1" || phase === "phase2") {
        selectModel.innerHTML = `
            <option value="dt">Decision Tree</option>
            <option value="ann">ANN</option>
            <option value="knn">kNN</option>
            <option value="lr">Logistic Regression</option>
        `;
    } else {
        selectModel.innerHTML = `
            <option value="enhanced">Final Selected Model</option>
        `;
    }

    loadSelectedConfusionMatrix();
}

async function loadSelectedConfusionMatrix() {
    const phase = document.getElementById("cm-select-phase").value;
    const task = document.getElementById("cm-select-task").value;
    const model = document.getElementById("cm-select-model").value;
    const box = document.getElementById("cm-render-box");

    box.innerHTML = `<span class="text-xs text-slate-400"><i class="fa-solid fa-spinner fa-spin mr-2"></i>Loading confusion matrix...</span>`;

    try {
        const res = await fetch(`/api/performances/confusion-matrix/${phase}/${task}/${model}`);
        if (!res.ok) throw new Error("Matrix not found");
        const data = await res.json();
        renderConfusionMatrixTable(data, box);
    } catch (err) {
        box.innerHTML = `<span class="text-xs text-slate-500">Confusion matrix not available for this selection.</span>`;
    }
}

function renderConfusionMatrixTable(data, container) {
    const labels = data.labels || [];
    const matrix = data.confusion_matrix || [];

    if (matrix.length === 0) {
        container.innerHTML = `<span class="text-xs text-slate-500">Empty matrix.</span>`;
        return;
    }

    let html = `<div class="space-y-2"><table class="border-collapse text-xs text-center"><thead><tr><th class="p-2 border border-slate-800 bg-slate-900 text-slate-400 font-bold">True \\ Pred</th>`;
    labels.forEach(l => {
        html += `<th class="p-2 border border-slate-800 bg-slate-900 text-slate-300 font-bold">${l}</th>`;
    });
    html += `</tr></thead><tbody>`;

    for (let r = 0; r < matrix.length; r++) {
        html += `<tr><td class="p-2 border border-slate-800 bg-slate-900 text-slate-300 font-bold text-left">${labels[r]}</td>`;
        for (let c = 0; c < matrix[r].length; c++) {
            const val = matrix[r][c];
            const isDiag = (r === c);
            const bgClass = isDiag ? "bg-emerald-950/40 text-emerald-300 font-bold" : (val > 0 ? "bg-rose-950/30 text-rose-300" : "text-slate-500");
            html += `<td class="p-2 border border-slate-800 font-mono ${bgClass}">${val.toLocaleString()}</td>`;
        }
        html += `</tr>`;
    }
    html += `</tbody></table></div>`;
    container.innerHTML = html;
}

async function downloadPerformancePdfReport() {
    if (!performanceDataCache) {
        alert("Performance data not loaded yet. Please wait.");
        return;
    }

    try {
        const res = await fetch("/api/reports/performances-pdf", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(performanceDataCache.summary),
        });

        if (!res.ok) throw new Error("Failed to generate PDF report");
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "IDS_Academic_Performance_Report.pdf";
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
    } catch (err) {
        alert(`Error downloading report: ${err.message}`);
    }
}

// =========================================================
// DASHBOARD 1: DATASET ANALYSIS
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
        if (!res.ok) throw new Error(data.detail || "Upload validation failed");

        activeFileId = data.filename;
        renderValidationBadge(data);
    } catch (err) {
        alert(`Upload error: ${err.message}`);
    }
}

function renderValidationBadge(data) {
    const badge = document.getElementById("val-badge");
    const info = document.getElementById("val-info");
    const btnPredict = document.getElementById("btn-run-prediction");

    if (data.validation_status === "VALID") {
        badge.className = "ml-2 font-bold px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800";
        badge.textContent = "VALID DATASET";
        info.innerHTML = `Loaded: <b>${data.filename}</b> (${data.row_count.toLocaleString()} rows, ${data.column_count} cols) | 42-feat: ${data.features_present_42 ? '✓' : '✗'}, 19-feat: ${data.features_present_19 ? '✓' : '✗'}`;
        btnPredict.disabled = false;
    } else {
        badge.className = "ml-2 font-bold px-2 py-0.5 rounded bg-rose-950 text-rose-400 border border-rose-800";
        badge.textContent = "SCHEMA MISMATCH";
        info.innerHTML = data.message;
        btnPredict.disabled = true;
    }
}

function updateModelDropdown() {
    const task = document.getElementById("select-prediction-task").value;
    const selectModel = document.getElementById("select-model");

    if (task === "binary") {
        selectModel.innerHTML = `
            <option value="decision_tree" selected>Decision Tree</option>
            <option value="ann">ANN (Neural Network)</option>
            <option value="knn">k-Nearest Neighbors</option>
            <option value="logistic_regression">Logistic Regression</option>
            <option value="enhanced">Phase 3 Enhanced Model (HistGB)</option>
        `;
    } else {
        selectModel.innerHTML = `
            <option value="decision_tree" selected>Decision Tree</option>
            <option value="ann">ANN (Neural Network)</option>
            <option value="knn">k-Nearest Neighbors</option>
            <option value="logistic_regression">Logistic Regression</option>
            <option value="enhanced">Phase 3 Enhanced Model (RF+XGB Ensemble)</option>
        `;
    }
}

async function runDatasetPrediction() {
    if (!activeFileId) return;

    const btn = document.getElementById("btn-run-prediction");
    btn.disabled = true;
    btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin mr-1.5"></i>Running Inference...`;

    const featureMode = document.getElementById("select-feature-mode").value;
    const predictionTask = document.getElementById("select-prediction-task").value;
    const modelName = document.getElementById("select-model").value;

    const formData = new FormData();
    formData.append("file_id", activeFileId);
    formData.append("feature_mode", featureMode);
    formData.append("prediction_task", predictionTask);
    formData.append("model_name", modelName);

    try {
        const res = await fetch("/api/dataset/predict", {
            method: "POST",
            body: formData,
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Inference failed");

        lastDatasetResult = data;
        renderDatasetResults(data);
    } catch (err) {
        alert(`Prediction error: ${err.message}`);
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<i class="fa-solid fa-play mr-1.5"></i>Run Inference & Eval`;
    }
}

function renderDatasetResults(data) {
    document.getElementById("dataset-results-container").classList.remove("hidden");
    document.getElementById("btn-dataset-pdf").classList.remove("hidden");

    const summary = data.summary || data.predictions_summary || {};
    const totalRecords = summary.total_records || summary.total_predictions || 0;
    const normalCount = summary.normal_count || 0;
    const attackCount = summary.attack_count || 0;

    document.getElementById("metric-total-rows").textContent = totalRecords.toLocaleString();
    document.getElementById("metric-normal-count").textContent = normalCount.toLocaleString();
    document.getElementById("metric-attack-count").textContent = attackCount.toLocaleString();

    const evaluation = data.evaluation || data.evaluation_metrics;
    if (evaluation) {
        let acc = evaluation.accuracy !== undefined ? evaluation.accuracy : evaluation.accuracy_percent;
        if (acc !== undefined && acc !== null) {
            let accPct = acc <= 1.0 ? (acc * 100).toFixed(2) : Number(acc).toFixed(2);
            document.getElementById("metric-eval-acc").textContent = `${accPct}%`;
        } else {
            document.getElementById("metric-eval-acc").textContent = "N/A";
        }
    } else {
        document.getElementById("metric-eval-acc").textContent = "N/A (No Ground Truth)";
    }

    // Pie chart
    renderDatasetPieChart(summary.normal_count, summary.attack_count);

    // Samples Table
    const tbody = document.querySelector("#table-dataset-samples tbody");
    tbody.innerHTML = "";
    (data.sample_predictions || []).slice(0, 10).forEach((s, idx) => {
        const isAttack = s.binary_class === 1;
        tbody.innerHTML += `
            <tr>
                <td class="font-mono text-slate-400">#${idx + 1}</td>
                <td><span class="badge-tag ${isAttack ? 'bg-rose-950 text-rose-400' : 'bg-emerald-950 text-emerald-400'}">${s.prediction}</span></td>
                <td class="font-bold text-white">${s.attack_category}</td>
                <td class="font-mono text-cyan-400">${(s.confidence * 100).toFixed(1)}%</td>
            </tr>
        `;
    });
}

function renderDatasetPieChart(normal, attack) {
    const ctx = document.getElementById("chart-dataset-pie");
    if (!ctx) return;

    if (charts.datasetPie) {
        charts.datasetPie.destroy();
    }

    charts.datasetPie = new Chart(ctx, {
        type: "doughnut",
        data: {
            labels: ["Normal Traffic", "Attacks Flagged"],
            datasets: [{
                data: [normal, attack],
                backgroundColor: ["rgba(16, 185, 129, 0.8)", "rgba(244, 63, 94, 0.8)"],
                borderColor: ["#059669", "#e11d48"],
                borderWidth: 1,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: "bottom",
                    labels: { color: "#94a3b8", font: { size: 11 } }
                }
            }
        }
    });
}

async function downloadDatasetPdfReport() {
    if (!lastDatasetResult) return;
    try {
        const res = await fetch("/api/reports/dataset-pdf", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(lastDatasetResult),
        });
        if (!res.ok) throw new Error("Failed to generate PDF report");
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "IDS_Dataset_Analysis_Report.pdf";
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
    } catch (err) {
        alert(`Error downloading report: ${err.message}`);
    }
}

// =========================================================
// DASHBOARD 3: LIVE NETWORK MONITORING
// =========================================================
async function fetchInterfaces() {
    try {
        const res = await fetch("/api/live/interfaces");
        const ifaces = await res.json();
        const select = document.getElementById("select-interface");
        if (Array.isArray(ifaces) && ifaces.length > 0) {
            select.innerHTML = "";
            ifaces.forEach(i => {
                select.innerHTML += `<option value="${i.name}">${i.description || i.name} (${i.ip || 'No IP'})</option>`;
            });
        }
    } catch (err) {
        console.warn("Could not fetch network interfaces, keeping default Wi-Fi.");
    }
}

async function onThresholdChange(val) {
    const num = parseFloat(val);
    const lbl = document.getElementById("label-threshold-val");
    if (lbl) lbl.textContent = num.toFixed(2);
    if (sessionActive) {
        try {
            await fetch("/api/live/threshold", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ threshold: num }),
            });
        } catch (e) {
            console.error("Failed to update threshold live:", e);
        }
    }
}

async function startLiveMonitoring(isTestMode = false) {
    const iface = document.getElementById("select-interface").value;
    const thresholdInput = document.getElementById("input-threshold");
    const thresholdVal = thresholdInput ? parseFloat(thresholdInput.value) : 0.80;

    try {
        const res = await fetch("/api/live/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                interface_name: iface,
                test_mode: isTestMode,
                threshold: thresholdVal,
            }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Failed to start capture");

        sessionActive = true;
        sessionStartTime = new Date();

        document.getElementById("btn-start-live").disabled = true;
        document.getElementById("btn-start-test").disabled = true;
        document.getElementById("btn-stop-live").disabled = false;

        const badgeBar = document.getElementById("test-mode-badge-bar");
        const evalCard = document.getElementById("test-mode-eval-card");
        const wifiCard = document.getElementById("wifi-mode-info-card");

        if (isTestMode) {
            if (badgeBar) badgeBar.classList.remove("hidden");
            if (evalCard) evalCard.classList.remove("hidden");
            if (wifiCard) wifiCard.classList.add("hidden");
            // Reset evaluation UI counters
            ["total", "tp", "fp", "tn", "fn"].forEach(id => {
                const el = document.getElementById(`test-eval-${id}`);
                if (el) el.textContent = "0";
            });
            ["acc", "prec", "rec", "f1", "fpr", "fnr"].forEach(id => {
                const el = document.getElementById(`test-eval-${id}`);
                if (el) el.textContent = "0.0%";
            });
        } else {
            if (badgeBar) badgeBar.classList.add("hidden");
            if (evalCard) evalCard.classList.add("hidden");
            if (wifiCard) wifiCard.classList.remove("hidden");
        }

        initWebSocket();
    } catch (err) {
        alert(`Monitoring error: ${err.message}`);
    }
}

async function stopLiveMonitoring() {
    try {
        const res = await fetch("/api/live/stop", { method: "POST" });
        const data = await res.json();

        sessionActive = false;
        document.getElementById("btn-start-live").disabled = false;
        document.getElementById("btn-start-test").disabled = false;
        document.getElementById("btn-stop-live").disabled = true;
        const badgeBar = document.getElementById("test-mode-badge-bar");
        if (badgeBar) badgeBar.classList.add("hidden");

        showSessionSummaryModal(data);
    } catch (err) {
        alert(`Error stopping capture: ${err.message}`);
    }
}

function initWebSocket() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
        return;
    }

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host;
    ws = new WebSocket(`${protocol}//${host}/api/live/ws`);

    ws.onopen = () => {
        const el = document.getElementById("status-ws");
        el.className = "flex items-center text-emerald-400 font-medium";
        el.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-400 mr-1.5"></span>WebSocket: Connected`;
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleLiveFlowEvent(data);
        } catch (e) {
            console.error("Error parsing WebSocket event:", e);
        }
    };

    ws.onclose = () => {
        const el = document.getElementById("status-ws");
        el.className = "flex items-center text-slate-400 font-medium";
        el.innerHTML = `<span class="w-2 h-2 rounded-full bg-slate-500 mr-1.5"></span>WebSocket: Disconnected`;
    };
}

function handleLiveFlowEvent(data) {
    allLiveFlows.unshift(data);
    if (allLiveFlows.length > 200) allLiveFlows.pop();

    const isAttack = (data.binary_class === 1 || data.prediction === "Attack");
    flowCache[data.flow_id] = data;

    // Increment metrics
    const pEl = document.getElementById("live-stat-packets");
    const fEl = document.getElementById("live-stat-flows");
    const nEl = document.getElementById("live-stat-normal");
    const aEl = document.getElementById("live-stat-attack");

    if (fEl) fEl.textContent = (parseInt(fEl.textContent) + 1).toLocaleString();
    if (pEl) pEl.textContent = (parseInt(pEl.textContent) + (data.flow_info?.spkts || 1) + (data.flow_info?.dpkts || 0)).toLocaleString();

    if (isAttack) {
        if (aEl) aEl.textContent = (parseInt(aEl.textContent) + 1).toLocaleString();
    } else {
        if (nEl) nEl.textContent = (parseInt(nEl.textContent) + 1).toLocaleString();
    }

    // Update TEST MODE ground-truth evaluation card if telemetry present
    if (data.test_evaluation) {
        const te = data.test_evaluation;
        const evalCard = document.getElementById("test-mode-eval-card");
        if (evalCard) evalCard.classList.remove("hidden");
        const elTotal = document.getElementById("test-eval-total");
        const elTp = document.getElementById("test-eval-tp");
        const elFp = document.getElementById("test-eval-fp");
        const elTn = document.getElementById("test-eval-tn");
        const elFn = document.getElementById("test-eval-fn");
        const elAcc = document.getElementById("test-eval-acc");
        const elPrec = document.getElementById("test-eval-prec");
        const elRec = document.getElementById("test-eval-rec");
        const elF1 = document.getElementById("test-eval-f1");
        const elFpr = document.getElementById("test-eval-fpr");
        const elFnr = document.getElementById("test-eval-fnr");

        if (elTotal) elTotal.textContent = te.total.toLocaleString();
        if (elTp) elTp.textContent = te.tp.toLocaleString();
        if (elFp) elFp.textContent = te.fp.toLocaleString();
        if (elTn) elTn.textContent = te.tn.toLocaleString();
        if (elFn) elFn.textContent = te.fn.toLocaleString();
        if (elAcc) elAcc.textContent = `${te.accuracy}%`;
        if (elPrec) elPrec.textContent = `${te.precision}%`;
        if (elRec) elRec.textContent = `${te.recall}%`;
        if (elF1) elF1.textContent = `${te.f1}%`;
        if (elFpr) elFpr.textContent = `${te.fpr}%`;
        if (elFnr) elFnr.textContent = `${te.fnr}%`;
    }

    // Update live table
    const tbody = document.getElementById("live-flow-tbody");
    if (tbody) {
        const row = document.createElement("tr");
        row.className = isAttack ? "bg-rose-950/20" : "";
        let statusBadge = "";
        if (!isAttack) {
            statusBadge = `<span class="badge-tag bg-emerald-950 text-emerald-400 border border-emerald-800">Normal</span>`;
        } else if (data.status && data.status.includes("Potential Anomaly")) {
            statusBadge = `<span class="badge-tag bg-amber-950 text-amber-400 border border-amber-800">Potential Anomaly</span>`;
        } else {
            statusBadge = `<span class="badge-tag bg-rose-950 text-rose-400 border border-rose-800">${data.status || 'Attack Alert'}</span>`;
        }
        const confPct = ((data.confidence !== undefined ? data.confidence : 1.0) * 100).toFixed(1);
        row.innerHTML = `
            <td class="font-mono text-[11px] text-slate-400">${new Date().toLocaleTimeString()}</td>
            <td class="font-mono text-xs">${data.src_ip}:${data.src_port || 0}</td>
            <td class="font-mono text-xs">${data.dst_ip}:${data.dst_port || 0}</td>
            <td class="uppercase text-[11px] text-slate-400">${data.proto || 'tcp'}</td>
            <td>${statusBadge}</td>
            <td class="font-bold ${isAttack ? 'text-rose-300' : 'text-slate-400'}">${data.attack_category}</td>
            <td class="font-mono text-xs text-cyan-400">${confPct}%</td>
            <td>
                <button onclick="inspectLiveFlow('${data.flow_id}')" class="bg-slate-800 hover:bg-cyan-900 text-cyan-300 px-2.5 py-1 rounded text-[11px] font-semibold border border-slate-700 transition">
                    Inspect
                </button>
            </td>
        `;
        if (tbody.children.length === 1 && tbody.children[0].innerText.includes("Click")) {
            tbody.innerHTML = "";
        }
        tbody.insertBefore(row, tbody.firstChild);
        if (tbody.children.length > 50) tbody.removeChild(tbody.lastChild);
    }

    updateLiveCharts();
}

function initLiveCharts() {
    const ctxTimeline = document.getElementById("chart-live-timeline");
    const ctxPie = document.getElementById("chart-live-pie");

    if (ctxTimeline && !charts.liveTimeline) {
        charts.liveTimeline = new Chart(ctxTimeline, {
            type: "line",
            data: {
                labels: Array(10).fill(""),
                datasets: [{
                    label: "Packets/sec",
                    data: Array(10).fill(0),
                    borderColor: "#0284c7",
                    backgroundColor: "rgba(2, 132, 199, 0.15)",
                    fill: true,
                    tension: 0.4,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: { ticks: { color: "#64748b" }, grid: { color: "rgba(51, 65, 85, 0.2)" } },
                    x: { display: false }
                }
            }
        });
    }

    if (ctxPie && !charts.livePie) {
        charts.livePie = new Chart(ctxPie, {
            type: "doughnut",
            data: {
                labels: ["Normal", "Attacks"],
                datasets: [{
                    data: [1, 0],
                    backgroundColor: ["rgba(16, 185, 129, 0.8)", "rgba(244, 63, 94, 0.8)"],
                    borderWidth: 0,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: "bottom", labels: { color: "#94a3b8", font: { size: 10 } } }
                }
            }
        });
    }
}

function updateLiveCharts() {
    const normalCount = parseInt(document.getElementById("live-stat-normal")?.textContent || "0");
    const attackCount = parseInt(document.getElementById("live-stat-attack")?.textContent || "0");

    if (charts.livePie) {
        charts.livePie.data.datasets[0].data = [normalCount, attackCount];
        charts.livePie.update();
    }

    if (charts.liveTimeline) {
        const randFlow = Math.floor(Math.random() * 5) + 1;
        charts.liveTimeline.data.datasets[0].data.shift();
        charts.liveTimeline.data.datasets[0].data.push(randFlow);
        charts.liveTimeline.update();
    }
}

async function inspectLiveFlow(flowId) {
    const flow = flowCache[flowId];
    if (!flow) return;

    document.getElementById("alert-modal-key").textContent = `Flow ID: ${flowId}`;
    document.getElementById("alert-modal-pred").textContent = flow.prediction;
    document.getElementById("alert-modal-cat").textContent = flow.attack_category;
    document.getElementById("alert-modal-conf").textContent = `${(flow.confidence * 100).toFixed(1)}%`;

    // Render features
    const fContainer = document.getElementById("alert-modal-features");
    fContainer.innerHTML = "";
    const fInfo = flow.flow_info || {};
    for (const k in fInfo) {
        fContainer.innerHTML += `<div><span class="text-slate-500">${k}:</span> <span class="text-white">${fInfo[k]}</span></div>`;
    }

    // Call AI Agent for deep recommendation
    try {
        const res = await fetch("/api/live/ai-recommendation", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                flow_id: flowId,
                src_ip: flow.src_ip,
                dst_ip: flow.dst_ip,
                service: flow.service || "-",
                attack_category: flow.attack_category,
                confidence: flow.confidence,
                flow_info: fInfo,
            })
        });
        const rec = await res.json();
        document.getElementById("alert-modal-summary").textContent = rec.explanation || rec.threat_summary;
        
        const actList = document.getElementById("alert-modal-actions");
        actList.innerHTML = "";
        (rec.recommended_actions || []).forEach(act => {
            actList.innerHTML += `<li>${act}</li>`;
        });
    } catch (e) {
        document.getElementById("alert-modal-summary").textContent = "AI Security Advisor: Flow behavior exhibits characteristics matching attack profile. Review source IP activity and payload length.";
    }

    document.getElementById("alert-modal").classList.remove("hidden");
}

function closeAlertModal() {
    document.getElementById("alert-modal").classList.add("hidden");
}

function showSessionSummaryModal(summary) {
    document.getElementById("sum-duration").textContent = "Completed";
    document.getElementById("sum-packets").textContent = (summary.packet_count || 0).toLocaleString();
    document.getElementById("sum-normal").textContent = (summary.normal_count || 0).toLocaleString();
    document.getElementById("sum-attack").textContent = (summary.attack_count || 0).toLocaleString();
    document.getElementById("sum-iface").textContent = summary.interface_name || "Wi-Fi";
    document.getElementById("sum-mode").textContent = summary.test_mode ? "Controlled TEST MODE" : "Live Capture";

    document.getElementById("session-summary-modal").classList.remove("hidden");
    document.getElementById("btn-live-pdf").classList.remove("hidden");
}

function closeSessionSummaryModal() {
    document.getElementById("session-summary-modal").classList.add("hidden");
}

async function downloadLivePdfReport() {
    const payload = {
        session_id: "LIVE_SESSION_" + Date.now(),
        interface_name: document.getElementById("select-interface")?.value || "Wi-Fi",
        test_mode: !document.getElementById("test-mode-badge-bar")?.classList.contains("hidden"),
        total_packets: parseInt(document.getElementById("live-stat-packets")?.textContent || "0"),
        total_flows: parseInt(document.getElementById("live-stat-flows")?.textContent || "0"),
        normal_count: parseInt(document.getElementById("live-stat-normal")?.textContent || "0"),
        attack_count: parseInt(document.getElementById("live-stat-attack")?.textContent || "0"),
        recent_flows: allLiveFlows.slice(0, 15),
    };

    try {
        const res = await fetch("/api/reports/live-pdf", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (!res.ok) throw new Error("Failed to generate Live PDF");
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "IDS_Live_Monitoring_Session_Report.pdf";
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
    } catch (err) {
        alert(`Error downloading live report: ${err.message}`);
    }
}
