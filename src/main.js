import { World } from "./sim/world.js";
import { CanvasRenderer } from "./render/canvas.js";
import { resetReactionCounter } from "./sim/bio.js";
import { ELEMENTS } from "./sim/chem.js";

const world = new World(160, 120);

let simTime = 0.0;
window.SIM_TIME = simTime;

let selectedLineageId = null;
const lineageEvents = new Map();
let lineageRateLastSnapshot = new Map();
let lineageRateLastSampleSim = 0;
const lineageRateCached = new Map();
const LINEAGE_RATE_WINDOW_SEC = 2.0;

const ensureLineageEvents = (lineageId) => {
    if (!lineageEvents.has(lineageId)) lineageEvents.set(lineageId, { births: 0, deaths: 0 });
    return lineageEvents.get(lineageId);
};

const recordLineageBirth = (lineageId) => {
    ensureLineageEvents(lineageId).births++;
};

const recordLineageDeath = (lineageId) => {
    ensureLineageEvents(lineageId).deaths++;
};

window.__recordLineageBirth = recordLineageBirth;
window.__recordLineageDeath = recordLineageDeath;

const refreshLineageRateSnapshots = () => {
    const elapsed = simTime - lineageRateLastSampleSim;
    if (elapsed < LINEAGE_RATE_WINDOW_SEC) return;
    for (const [id, current] of lineageEvents.entries()) {
        const prev = lineageRateLastSnapshot.get(id) || { births: 0, deaths: 0 };
        const birthsPerSec = (current.births - prev.births) / elapsed;
        const deathsPerSec = (current.deaths - prev.deaths) / elapsed;
        lineageRateCached.set(id, {
            birthsPerSec,
            deathsPerSec,
            netGrowthPerSec: birthsPerSec - deathsPerSec
        });
    }
    lineageRateLastSampleSim = simTime;
    lineageRateLastSnapshot = new Map();
    for (const [id, counters] of lineageEvents.entries()) {
        lineageRateLastSnapshot.set(id, { ...counters });
    }
};

const computeLineageRates = (lineageId) => {
    const totals = lineageEvents.get(lineageId) || { births: 0, deaths: 0 };
    const cached = lineageRateCached.get(lineageId) || { birthsPerSec: 0, deathsPerSec: 0, netGrowthPerSec: 0 };
    return {
        birthsPerSec: cached.birthsPerSec,
        deathsPerSec: cached.deathsPerSec,
        netGrowthPerSec: cached.netGrowthPerSec,
        totals
    };
};

const currentAverageTemperature = () => {
    let sum = 0;
    for (let x = 0; x < world.width; x++) {
        for (let y = 0; y < world.height; y++) {
            sum += world.grid[x][y].temperature;
        }
    }
    const count = world.width * world.height;
    if (count === 0) return world.baseTemperature ?? 0.5;
    return sum / count;
};

const randomGenome = () => {
    const avgT = currentAverageTemperature();
    const opt = Math.max(0, Math.min(1, avgT + (Math.random() - 0.5) * 0.10));

    return {
        optimalTemp: opt,
        enzymes: [
            {
                type: "anabolase",
                affinity: { A: 1, B: 0.5, C: 0.3 },
                tOpt: opt,
                pHOpt: Math.random(),
                bondMultiplier: 1.15,
                secretionProb: 0.12
            },
            {
                type: "catabolase",
                affinity: { D: 1, E: 1 },
                tOpt: opt,
                pHOpt: Math.random(),
                transmuteProb: 0.35,
                harvestFraction: 0.8
            }
        ],
        reproThreshold: 2 + Math.random() * 6,
        initialEnergy: 1.2 + Math.random() * 1.8,
        decayTime: 700 + Math.random() * 2000,
        defaultSecretionProb: 0.15,
        mutationRate: 0.06,
        postDivideMortality: 0.0,
        desiredElementReserve: 2,
        tempStressFactor: 0.02,
        lineageId: Math.floor(Math.random() * 1e9)
    };
};

const canvas = document.getElementById("canvas");
const renderer = new CanvasRenderer(canvas, world);

let fpsFrames = 0;
let fpsLastSampleMs = performance.now();
let fpsValue = 0;
let tpsTicks = 0;
let tpsLastSampleSim = simTime;
let tpsValue = 0;

let lastHudUpdateMs = performance.now();
const HUD_UPDATE_INTERVAL_MS = 200;
let lastInfoUpdateMs = performance.now();
const INFO_UPDATE_INTERVAL_MS = 100;

const hud = {
    fps: document.getElementById("statFPS"),
    tps: document.getElementById("statTPS"),
    population: document.getElementById("statPopulation"),
    avgEnergy: document.getElementById("statAvgEnergy"),
    lineages: document.getElementById("statLineages"),
    enzymeDiversity: document.getElementById("statEnzymeDiversity"),
    selectedLineage: document.getElementById("statSelectedLineage")
};

let elementsDiv = document.getElementById("statElements");
if (!elementsDiv) {
    elementsDiv = document.createElement("div");
    elementsDiv.id = "statElements";
    elementsDiv.className = "statItem";
    hud.population.parentElement.insertBefore(elementsDiv, hud.population.nextSibling);
}

let selectedCell = null;
let selectedTile = null;

renderer.onCellClick = (x, y, tile, topCell) => {
    selectedCell = topCell;
    selectedTile = tile;
    updateInfoPanel();
    lastInfoUpdateMs = performance.now();
};

renderer.onCellRightClick = (x, y, tile, topCell) => {
    if (!topCell) {
        selectedLineageId = null;
        updateInfoPanel();
        lastInfoUpdateMs = performance.now();
        return;
    }
    selectedLineageId = topCell.lineageId;
    updateInfoPanel();
    lastInfoUpdateMs = performance.now();
};

let paused = false;
const createPlayPauseButton = () => {
    const btn = document.createElement("button");
    btn.id = "playPauseBtn";
    btn.textContent = "Pause";
    btn.style.position = "fixed";
    btn.style.top = "10px";
    btn.style.right = "10px";
    btn.style.zIndex = 9999;
    btn.style.padding = "6px 10px";
    btn.style.background = "#fff";
    btn.style.border = "1px solid rgba(0,0,0,0.12)";
    btn.style.borderRadius = "6px";
    btn.style.cursor = "pointer";
    document.body.appendChild(btn);

    btn.addEventListener("click", () => {
        paused = !paused;
        btn.textContent = paused ? "Play" : "Pause";
    });
};
createPlayPauseButton();

for (let i = 0; i < 32; i++) world.spawnRandomCell(randomGenome);

const computeElementTotals = () => {
    const totals = {};
    for (const k in ELEMENTS) totals[k] = 0;
    for (let x = 0; x < world.width; x++) {
        for (let y = 0; y < world.height; y++) {
            const tile = world.grid[x][y];
            for (const m of tile.molecules) {
                for (const el in m.composition) totals[el] = (totals[el] || 0) + m.composition[el];
            }
            for (const c of tile.cells) {
                for (const m of c.molecules) {
                    for (const el in m.composition) totals[el] = (totals[el] || 0) + m.composition[el];
                }
            }
        }
    }
    return totals;
};

const enzymeSignature = (e) => {
    const type = e.type || "any";
    const affinityKeys = e.affinity ? Object.keys(e.affinity).sort().join(",") : "any";
    const bm = (typeof e.bondMultiplier === "number") ? e.bondMultiplier.toFixed(2) : "bmX";
    const tp = (typeof e.transmuteProb === "number") ? e.transmuteProb.toFixed(2) : "tpX";
    return `${type}|aff:${affinityKeys}|bm:${bm}|tp:${tp}`;
};

const computeStats = () => {
    const energies = [];
    const lineageIds = [];
    const enzymeSigs = [];

    for (let x = 0; x < world.width; x++) {
        for (let y = 0; y < world.height; y++) {
            const tile = world.grid[x][y];
            for (const c of tile.cells) {
                energies.push(c.energy);
                lineageIds.push(c.lineageId);
                for (const e of c.genome.enzymes) enzymeSigs.push(enzymeSignature(e));
            }
        }
    }

    const population = energies.length;
    const avgEnergy = population > 0 ? Chalkboard.stat.mean(energies).toFixed(2) : "—";
    const lineageCount = Chalkboard.stat.unique(lineageIds).length;
    const enzymeFunctionalDiversity = Chalkboard.stat.unique(enzymeSigs).length;

    return { population, avgEnergy, lineageCount, enzymeFunctionalDiversity };
};

const computeLineageSnapshot = (lineageId) => {
    let count = 0;
    let totalEnergy = 0;
    let totalAge = 0;
    let maxAge = 0;

    const enzymeSet = new Set();
    const enzymeTypeCounts = new Map();

    for (let x = 0; x < world.width; x++) {
        for (let y = 0; y < world.height; y++) {
            for (const c of world.grid[x][y].cells) {
                if (c.lineageId !== lineageId) continue;
                count++;
                totalEnergy += (c.energy || 0);

                const age = c.getAgeSecSim ? c.getAgeSecSim() : 0;
                totalAge += age;
                if (age > maxAge) maxAge = age;

                if (c.genome && c.genome.enzymes) {
                    for (const e of c.genome.enzymes) {
                        enzymeSet.add(enzymeSignature(e));
                        enzymeTypeCounts.set(e.type, (enzymeTypeCounts.get(e.type) || 0) + 1);
                    }
                }
            }
        }
    }

    const topEnzymeTypes = [...enzymeTypeCounts.entries()]
        .sort((a, b) => b[1] - a[1])
        .slice(0, 4);

    return {
        lineageId,
        population: count,
        avgEnergy: count > 0 ? totalEnergy / count : 0,
        avgAgeSec: count > 0 ? totalAge / count : 0,
        maxAgeSec: maxAge,
        enzymeFunctionalDiversity: enzymeSet.size,
        topEnzymeTypes
    };
};

const updateHud = () => {
    const s = computeStats();
    const totals = computeElementTotals();

    if (hud.fps) hud.fps.textContent = `FPS: ${fpsValue.toFixed(0)}`;
    if (hud.tps) hud.tps.textContent = `TPS: ${tpsValue.toFixed(0)}`;
    hud.population.textContent = `Population: ${s.population}`;
    hud.avgEnergy.textContent = `Avg energy: ${s.avgEnergy}`;
    hud.lineages.textContent = `Lineages: ${s.lineageCount}`;
    hud.enzymeDiversity.textContent = `Enzyme diversity: ${s.enzymeFunctionalDiversity}`;

    const parts = [];
    for (const k in totals) {
        const total = totals[k];
        const avgPerTile = (total / (world.width * world.height)).toFixed(2);
        parts.push(`${k}:${total} (avg:${avgPerTile})`);
    }
    let tsum = 0;
    for (let x = 0; x < world.width; x++) for (let y = 0; y < world.height; y++) tsum += world.grid[x][y].temperature;
    const avgTemp = (tsum / (world.width * world.height)).toFixed(3);
    parts.push(`T_avg:${avgTemp}`);
    elementsDiv.textContent = parts.join("  •  ");
};

const enzymeEquationString = (enzyme) => {
    const keys = enzyme.affinity ? Object.keys(enzyme.affinity).sort() : [];
    if (keys.length === 0) {
        if (enzyme.type === "anabolase") return "a + b → ab";
        if (enzyme.type === "catabolase") return "AB → A + B";
        if (enzyme.type === "transportase") return "import D/E";
        return `${enzyme.type}: ? → ?`;
    }
    if (enzyme.type === "anabolase") {
        const left = keys.slice(0, enzyme.maxInputs || 2).join(" + ");
        const prod = keys.slice(0, enzyme.maxInputs || 2).join("");
        return `${left} → ${prod}`;
    }
    if (enzyme.type === "catabolase") {
        const a = keys[0];
        return `${a} → fragments`;
    }
    if (enzyme.type === "transportase") return `transport ${keys.slice(0, 2).join(",")}`;
    const left = keys.slice(0, 2).join(" + ");
    const prod = keys.slice(0, 2).join("");
    return `${left} → ${prod}`;
};

const formatReactionExpression = (ev) => {
    const left = ev.substrates.length === 0 ? "—" : ev.substrates.join(" + ");
    let right;
    if (ev.product && ev.product !== "—") {
        right = ev.product;
        if (ev.byproducts.length > 0) right += " (+ " + ev.byproducts.join(", ") + ")";
    } else {
        if (ev.byproducts.length === 0) right = "—";
        else if (ev.byproducts.length === 1) right = ev.byproducts[0];
        else right = ev.byproducts.join(" + ");
    }
    return `${left} → ${right}`;
};

const buildCellInfoHtml = (cell) => {
    if (!cell) {
        return {
            title: "No cell selected",
            body: `<div class="smallNote">Left-click a cell to inspect it. Right-click a cell to inspect its lineage.</div>`
        };
    }

    const birthSim = cell.birthSimTime ?? 0;
    const deathSim = cell.deathSimTime ?? null;
    const ageS = deathSim !== null ? (deathSim - birthSim) : ((window.SIM_TIME || 0) - birthSim);
    const ageSstr = Number(ageS).toFixed(2);
    const internalMolEnergy = cell.molecules.reduce((s, m) => s + (m.energy || 0), 0);
    const totalStoredEnergy = (cell.energy || 0) + internalMolEnergy;

    const dom = cell.getDominantElement ? (cell.getDominantElement() || "none") : "none";

    let moleculesHtml = "";
    if (!cell.molecules || cell.molecules.length === 0) {
        moleculesHtml = "<div class='smallNote'>No internal molecules</div>";
    } else {
        moleculesHtml = "<ul style='margin:4px 0 8px 18px;padding:0;'>";
        for (const m of cell.molecules) {
            const comp = Object.entries(m.composition).map(([k, v]) => `${k}${v}`).join(", ");
            moleculesHtml += `<li>{ ${comp} } — size:${m.size} pol:${m.polarity.toFixed(2)} energy:${m.energy.toFixed(2)}</li>`;
        }
        moleculesHtml += "</ul>";
    }

    let enzymesHtml = "<ul style='margin:4px 0 8px 18px;padding:0;'>";
    for (const e of cell.genome.enzymes) {
        const eq = enzymeEquationString(e);
        enzymesHtml += `<li><strong>${e.type}</strong> — <code>${eq}</code> (tOpt:${(e.tOpt ?? 0.5).toFixed(2)}, pH:${(e.pHOpt ?? 0.5).toFixed(2)}, sec:${(e.secretionProb ?? cell.genome.defaultSecretionProb).toFixed(2)})</li>`;
    }
    enzymesHtml += "</ul>";

    let reactionHtml = "";
    if (!cell.reactionLog || cell.reactionLog.length === 0) {
        reactionHtml = "<div class='smallNote'>No recent reactions logged for this cell.</div>";
    } else {
        reactionHtml = "<table style='width:100%;font-size:12px;border-collapse:collapse;'><thead><tr style='text-align:left;'><th>t@evt(s)</th><th>Reaction</th><th>ΔE</th><th>ΔT</th><th>Es</th><th>Ep</th><th>rawΔ</th></tr></thead><tbody>";
        const slice = cell.reactionLog.slice(0, 12);
        for (const ev of slice) {
            const tAtEvt = (ev.ageAtEventSec !== undefined) ? Number(ev.ageAtEventSec).toFixed(3) : "—";
            const reactionStr = formatReactionExpression(ev);
            reactionHtml += `<tr><td style='padding:4px 6px;border-bottom:1px solid rgba(0,0,0,0.06)'>${tAtEvt}</td><td style='padding:4px 6px;border-bottom:1px solid rgba(0,0,0,0.06)'><code>${reactionStr}</code></td><td style='padding:4px 6px;border-bottom:1px solid rgba(0,0,0,0.06)'>${ev.deltaE}</td><td style='padding:4px 6px;border-bottom:1px solid rgba(0,0,0,0.06)'>${ev.deltaT}</td><td style='padding:4px 6px;border-bottom:1px solid rgba(0,0,0,0.06)'>${ev.substrateAtomEnergy ?? "—"}</td><td style='padding:4px 6px;border-bottom:1px solid rgba(0,0,0,0.06)'>${ev.productAtomEnergy ?? "—"}</td><td style='padding:4px 6px;border-bottom:1px solid rgba(0,0,0,0.06)'>${ev.rawDelta ?? "—"}</td></tr>`;
        }
        reactionHtml += "</tbody></table>";
    }

    const deathNote = (cell.deathSimTime !== null && cell.deathSimTime !== undefined) ? " (dead)" : "";

    return {
        title: `Cell — lineage ${cell.lineageId}`,
        body: `
            <div><strong>Age:</strong> ${ageSstr}s${deathNote}</div>
            <div><strong>Energy:</strong> ${cell.energy.toFixed(3)} (total stored: ${totalStoredEnergy.toFixed(3)})</div>
            <div><strong>Dominant element:</strong> ${dom}</div>
            <div style="margin-top:6px;"><strong>Internal molecules</strong>${moleculesHtml}</div>
            <div style="margin-top:6px;"><strong>Enzymes</strong>${enzymesHtml}</div>
            <div style="margin-top:6px;"><strong>Recent reactions</strong>${reactionHtml}</div>
        `
    };
};

const buildLineageInfoHtml = (lineageId) => {
    if (lineageId == null) {
        return "";
    }

    const snap = computeLineageSnapshot(lineageId);
    const rates = computeLineageRates(lineageId);

    const topTypes = snap.topEnzymeTypes.length
        ? snap.topEnzymeTypes.map(([t, n]) => `${t}(${n})`).join(", ")
        : "—";

    const birthsPerSec = rates.birthsPerSec;
    const deathsPerSec = rates.deathsPerSec;
    const netPerSec = rates.netGrowthPerSec;

    return `
        <hr style="border:none;border-top:1px solid rgba(0,0,0,0.08);margin:10px 0;">
        <div style="margin-bottom:6px;"><strong>Lineage:</strong> ${lineageId}</div>

        <div><strong>Population:</strong> ${snap.population}</div>
        <div><strong>Avg energy:</strong> ${snap.avgEnergy.toFixed(3)}</div>
        <div><strong>Avg age:</strong> ${snap.avgAgeSec.toFixed(2)}s</div>
        <div><strong>Oldest age:</strong> ${snap.maxAgeSec.toFixed(2)}s</div>

        <div style="margin-top:6px;"><strong>Rates (per sim-second)</strong></div>
        <div>Births/s: ${birthsPerSec.toFixed(2)}</div>
        <div>Deaths/s: ${deathsPerSec.toFixed(2)}</div>
        <div>Net/s: ${netPerSec.toFixed(2)}</div>

        <div style="margin-top:6px;"><strong>Totals</strong></div>
        <div>Births: ${rates.totals.births}</div>
        <div>Deaths: ${rates.totals.deaths}</div>

        <div style="margin-top:6px;"><strong>Genetics</strong></div>
        <div>Enzyme diversity (functional): ${snap.enzymeFunctionalDiversity}</div>
        <div>Top enzyme types: ${topTypes}</div>
    `;
};

const updateInfoPanel = () => {
    const titleEl = document.getElementById("infoTitle");
    const bodyEl = document.getElementById("infoBody");

    const cellSection = buildCellInfoHtml(selectedCell);
    titleEl.textContent = cellSection.title;

    bodyEl.innerHTML = cellSection.body + buildLineageInfoHtml(selectedLineageId);
};

const isSelectingInInfoPanel = () => {
    const sel = window.getSelection();
    if (!sel || sel.rangeCount === 0) return false;
    const range = sel.getRangeAt(0);
    const panel = document.getElementById("infoPanel");
    if (!panel) return false;
    return panel.contains(range.commonAncestorContainer);
};

const createCopyReactionsButton = () => {
    const btn = document.createElement("button");
    btn.id = "copyReactionsBtn";
    btn.textContent = "Copy reactions";
    btn.style.position = "fixed";
    btn.style.bottom = "10px";
    btn.style.right = "10px";
    btn.style.zIndex = 10000;
    btn.style.padding = "6px 10px";
    btn.style.background = "#fff";
    btn.style.border = "1px solid rgba(0,0,0,0.12)";
    btn.style.borderRadius = "6px";
    btn.style.cursor = "pointer";
    document.body.appendChild(btn);

    btn.addEventListener("click", async () => {
        if (!selectedCell || !selectedCell.reactionLog) return;
        const slice = selectedCell.reactionLog.slice(0, 20);
        const lines = slice.map((ev) => {
            const time = (ev.ageAtEventSec !== undefined) ? Number(ev.ageAtEventSec).toFixed(3) : "—";
            const reactionStr = formatReactionExpression(ev);
            return `[t=${time}s] ${reactionStr} | ΔE=${ev.deltaE} | ΔT=${ev.deltaT} | Es=${ev.substrateAtomEnergy} | Ep=${ev.productAtomEnergy} | rawΔ=${ev.rawDelta}`;
        }).join("\n");
        try {
            await navigator.clipboard.writeText(lines);
            btn.textContent = "Copied ✓";
            setTimeout(() => btn.textContent = "Copy reactions", 900);
        } catch (e) {
            console.warn("copy failed", e);
            alert("Copy failed — select and press Ctrl+C as fallback.");
        }
    });
};
createCopyReactionsButton();

const spawnChanceBasedOnPopulation = (pop) => {
    const carryingCapacity = world.width * world.height * 0.30;
    const baseSpawn = 0.005;
    const minSpawn = 0.0002;
    const frac = Math.min(1, pop / Math.max(1, carryingCapacity));
    return Math.max(minSpawn, baseSpawn * (1 - frac));
};

let lastWallTimeMs = performance.now();
let accumulatorMs = 0;
const main = () => {
    const nowMs = performance.now();
    let frameElapsedMs = nowMs - lastWallTimeMs;
    lastWallTimeMs = nowMs;

    if (frameElapsedMs > 200) frameElapsedMs = 200;

    if (paused) {
        renderer.render();
        if (nowMs - lastHudUpdateMs >= HUD_UPDATE_INTERVAL_MS) {
            updateHud();
            lastHudUpdateMs = nowMs;
            refreshLineageRateSnapshots();
        }
        if (!isSelectingInInfoPanel() && (nowMs - lastInfoUpdateMs >= INFO_UPDATE_INTERVAL_MS)) {
            updateInfoPanel();
            lastInfoUpdateMs = nowMs;
        }
        accumulatorMs = 0;
        requestAnimationFrame(main);
        return;
    }

    accumulatorMs += frameElapsedMs;

    const stepMs = world.dt;
    const MAX_ACCUMULATED_MS = stepMs * 4;
    if (accumulatorMs > MAX_ACCUMULATED_MS) accumulatorMs = MAX_ACCUMULATED_MS;
    const MAX_STEPS_PER_FRAME = 6;
    let steps = 0;
    while (accumulatorMs >= stepMs && steps < MAX_STEPS_PER_FRAME) {
        simTime += stepMs / 1000;
        window.SIM_TIME = simTime;

        const stats = computeStats();
        const spawnProb = spawnChanceBasedOnPopulation(stats.population);
        if (Math.random() < spawnProb) world.spawnRandomCell(randomGenome);

        resetReactionCounter();
        world.step();
        tpsTicks++;

        accumulatorMs -= stepMs;
        steps++;
    }

    const simElapsed = simTime - tpsLastSampleSim;
    if (simElapsed >= 0.5) {
        tpsValue = tpsTicks / simElapsed;
        tpsTicks = 0;
        tpsLastSampleSim = simTime;
    }

    if (accumulatorMs >= stepMs) accumulatorMs = 0;
    renderer.render();
    if (nowMs - lastHudUpdateMs >= HUD_UPDATE_INTERVAL_MS) {
        updateHud();
        lastHudUpdateMs = nowMs;
        refreshLineageRateSnapshots();
    }
    if (!isSelectingInInfoPanel() && (nowMs - lastInfoUpdateMs >= INFO_UPDATE_INTERVAL_MS)) {
        updateInfoPanel();
        lastInfoUpdateMs = nowMs;
    }

    fpsFrames++;
    const fpsNow = performance.now();
    const fpsElapsed = fpsNow - fpsLastSampleMs;
    if (fpsElapsed >= 500) {
        fpsValue = (fpsFrames * 1000) / fpsElapsed;
        fpsFrames = 0;
        fpsLastSampleMs = fpsNow;
    }

    requestAnimationFrame(main);
};
lastWallTimeMs = performance.now();
requestAnimationFrame(main);
