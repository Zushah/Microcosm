import { World } from "./sim/world.js";
import { CanvasRenderer } from "./render/canvas.js";
import { resetReactionCounter, reactionsThisTick } from "./sim/bio.js";
import { ELEMENTS } from "./sim/chem.js";

const world = new World(160, 120);

let simTime = 0.0;
window.SIM_TIME = simTime;

function randomGenome() {
    return {
        enzymes: [
            { type: "anabolase", affinity: { A: 1, B: 0.5 }, tOpt: Math.random(), pHOpt: Math.random(), bondMultiplier: 1.15, secretionProb: 0.12 },
            { type: "catabolase", affinity: { A: 1, B: 0.5 }, tOpt: Math.random(), pHOpt: Math.random(), transmuteProb: 0.35, harvestFraction: 0.8 }
        ],
        reproThreshold: 2 + Math.random() * 6,
        initialEnergy: 1.2 + Math.random() * 1.8,
        decayTime: 700 + Math.random() * 2000,
        defaultSecretionProb: 0.15,
        mutationRate: 0.06,
        postDivideMortality: 0.0,
        desiredElementReserve: 2,
        lineageId: Math.floor(Math.random() * 1e9)
    };
}

const canvas = document.getElementById("canvas");
const renderer = new CanvasRenderer(canvas, world);

const hud = {
    population: document.getElementById("statPopulation"),
    avgEnergy: document.getElementById("statAvgEnergy"),
    lineages: document.getElementById("statLineages"),
    enzymeDiversity: document.getElementById("statEnzymeDiversity")
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
let highlightedLineage = null;

renderer.onCellClick = (x, y, tile, topCell) => {
    selectedCell = topCell;
    selectedTile = tile;
    updateInfoPanel();
};

renderer.onCellRightClick = (x, y, tile, topCell) => {
    if (!topCell) {
        highlightedLineage = null;
        clearAllHighlights();
        return;
    }
    highlightedLineage = topCell.lineageId;
    markDescendants(highlightedLineage);
};

let paused = false;
function createPlayPauseButton() {
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
}
createPlayPauseButton();

for (let i = 0; i < 32; i++) world.spawnRandomCell(randomGenome);

function computeElementTotals() {
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
}

function computeStats() {
    let pop = 0;
    let totalEnergy = 0;
    const lineages = new Set();
    const functionalSet = new Set();

    for (let x = 0; x < world.width; x++) {
        for (let y = 0; y < world.height; y++) {
            const tile = world.grid[x][y];
            for (const c of tile.cells) {
                pop++;
                totalEnergy += c.energy;
                lineages.add(c.lineageId);
                for (const e of c.genome.enzymes) {
                    const keys = e.affinity ? Object.keys(e.affinity).sort() : ["any"];
                    functionalSet.add(keys.join("|"));
                }
            }
        }
    }

    const avgEnergy = pop > 0 ? (totalEnergy / pop).toFixed(2) : "—";
    return { population: pop, avgEnergy, lineageCount: lineages.size, enzymeFunctionalDiversity: functionalSet.size };
}

function updateHud() {
    const s = computeStats();
    const totals = computeElementTotals();

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
}

function enzymeEquationString(enzyme) {
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
    if (enzyme.type === "transportase") return `transport ${keys.slice(0,2).join(",")}`;
    const left = keys.slice(0, 2).join(" + ");
    const prod = keys.slice(0, 2).join("");
    return `${left} → ${prod}`;
}

function updateInfoPanel() {
    const title = document.getElementById("infoTitle");
    const body = document.getElementById("infoBody");
    if (!selectedCell) {
        title.textContent = "No cell selected";
        body.innerHTML = `<div class="smallNote">Click a pixel (cell) to inspect it. Right-click to highlight descendants.</div>`;
        return;
    }
    title.textContent = `Cell — lineage ${selectedCell.lineageId}`;

    const birthSim = selectedCell.birthSimTime ?? 0;
    const deathSim = selectedCell.deathSimTime ?? null;
    const ageS = deathSim !== null ? (deathSim - birthSim) : ((window.SIM_TIME || 0) - birthSim);
    const ageSstr = Number(ageS).toFixed(2);

    const dom = selectedCell.getDominantElement() || "none";

    let moleculesHtml = "";
    if (selectedCell.molecules.length === 0) {
        moleculesHtml = "<div class='smallNote'>No internal molecules</div>";
    } else {
        moleculesHtml = "<ul style='margin:4px 0 8px 18px;padding:0;'>";
        for (const m of selectedCell.molecules) {
            const comp = Object.entries(m.composition).map(([k, v]) => `${k}${v}`).join(", ");
            moleculesHtml += `<li>{ ${comp} } — size:${m.size} pol:${m.polarity.toFixed(2)} energy:${m.energy.toFixed(2)}</li>`;
        }
        moleculesHtml += "</ul>";
    }

    let enzymesHtml = "<ul style='margin:4px 0 8px 18px;padding:0;'>";
    for (const e of selectedCell.genome.enzymes) {
        const eq = enzymeEquationString(e);
        enzymesHtml += `<li><strong>${e.type}</strong> — <code>${eq}</code> (tOpt:${(e.tOpt ?? 0.5).toFixed(2)}, pH:${(e.pHOpt ?? 0.5).toFixed(2)}, sec:${(e.secretionProb ?? selectedCell.genome.defaultSecretionProb).toFixed(2)})</li>`;
    }
    enzymesHtml += "</ul>";

    let reactionHtml = "";
    if (!selectedCell.reactionLog || selectedCell.reactionLog.length === 0) reactionHtml = "<div class='smallNote'>No recent reactions logged for this cell.</div>";
    else {
        reactionHtml = "<table style='width:100%;font-size:12px;border-collapse:collapse;'><thead><tr style='text-align:left;'><th>t@evt(s)</th><th>Reaction</th><th>ΔE</th><th>ΔT</th><th>Es</th><th>Ep</th><th>rawΔ</th></tr></thead><tbody>";
        const slice = selectedCell.reactionLog.slice(0, 12);
        for (const ev of slice) {
            const tAtEvt = (ev.ageAtEventSec !== undefined) ? Number(ev.ageAtEventSec).toFixed(3) : "—";
            const left = ev.substrates.length === 0 ? "—" : ev.substrates.join(" + ");
            const byp = ev.byproducts.length === 0 ? "" : "  (+ " + ev.byproducts.join(", ") + ")";
            const reactionStr = `${left} → ${ev.product}${byp}`;
            reactionHtml += `<tr><td style='padding:4px 6px;border-bottom:1px solid rgba(0,0,0,0.06)'>${tAtEvt}</td><td style='padding:4px 6px;border-bottom:1px solid rgba(0,0,0,0.06)'><code>${reactionStr}</code></td><td style='padding:4px 6px;border-bottom:1px solid rgba(0,0,0,0.06)'>${ev.deltaE}</td><td style='padding:4px 6px;border-bottom:1px solid rgba(0,0,0,0.06)'>${ev.deltaT}</td><td style='padding:4px 6px;border-bottom:1px solid rgba(0,0,0,0.06)'>${ev.substrateAtomEnergy ?? "—"}</td><td style='padding:4px 6px;border-bottom:1px solid rgba(0,0,0,0.06)'>${ev.productAtomEnergy ?? "—"}</td><td style='padding:4px 6px;border-bottom:1px solid rgba(0,0,0,0.06)'>${ev.rawDelta ?? "—"}</td></tr>`;
        }
        reactionHtml += "</tbody></table>";
    }

    const deathNote = (selectedCell.deathSimTime !== null && selectedCell.deathSimTime !== undefined) ? " (dead)" : "";
    document.getElementById("infoBody").innerHTML = `
        <div><strong>Age:</strong> ${ageSstr} s${deathNote}</div>
        <div><strong>Energy:</strong> ${selectedCell.energy.toFixed(2)}</div>
        <div><strong>State:</strong> ${selectedCell.state}</div>
        <div><strong>Dominant internal element:</strong> ${dom}</div>
        <div style="margin-top:8px"><strong>Enzymes (approx reaction):</strong> ${enzymesHtml}</div>
        <div style="margin-top:4px"><strong>Internal molecules:</strong> ${moleculesHtml}</div>
        <div style="margin-top:8px"><strong>Recent reactions (newest first):</strong> ${reactionHtml}</div>
        <div class="smallNote" style="margin-top:6px">ΔE = usable energy gained by the cell. ΔT ≈ immediate local tile temperature change. Es/Ep = substrate/product atom energies. rawΔ = Es−Ep−enzymeCost.</div>
    `;
}

function markDescendants(lineageId) {
    for (let x = 0; x < world.width; x++) {
        for (let y = 0; y < world.height; y++) {
            for (const c of world.grid[x][y].cells) c._highlight = (c.lineageId === lineageId);
        }
    }
}

function clearAllHighlights() {
    for (let x = 0; x < world.width; x++) {
        for (let y = 0; y < world.height; y++) {
            for (const c of world.grid[x][y].cells) c._highlight = false;
        }
    }
}

function spawnChanceBasedOnPopulation(pop) {
    const carryingCapacity = world.width * world.height * 0.18;
    const baseSpawn = 0.02;
    const minSpawn = 0.0005;
    const frac = Math.min(1, pop / Math.max(1, carryingCapacity));
    return Math.max(minSpawn, baseSpawn * (1 - frac));
}

setInterval(() => {
    if (paused) {
        renderer.render();
        updateHud();
        updateInfoPanel();
        return;
    }
    simTime += world.dt / 1000;
    window.SIM_TIME = simTime;
    const stats = computeStats();
    const spawnProb = spawnChanceBasedOnPopulation(stats.population);
    if (Math.random() < spawnProb) world.spawnRandomCell(randomGenome);
    resetReactionCounter();
    world.step();
    renderer.render();
    updateHud();
    updateInfoPanel();
}, world.dt);
