import { World } from "./sim/world.js";
import { CanvasRenderer } from "./render/canvas.js";
import { ELEMENTS } from "./sim/chem.js";

const world = new World(200, 200);

function randomGenome() {
    return {
        enzymes: [
            {
                type: "ligase",
                affinity: { B: 1.0, C: 0.5 },
                tOpt: Math.random(),
                pHOpt: Math.random(),
                secretionProb: 0.15
            }
        ],
        reproThreshold: 6 + Math.random() * 6,
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

for (let i = 0; i < 40; i++) world.spawnRandomCell(randomGenome);

function computeElementTotals() {
    const totals = {};
    for (const k in ELEMENTS) totals[k] = 0;

    for (let x = 0; x < world.width; x++) {
        for (let y = 0; y < world.height; y++) {
            const tile = world.grid[x][y];
            for (const m of tile.molecules) {
                for (const el in m.composition) {
                    totals[el] = (totals[el] || 0) + m.composition[el];
                }
            }
        }
    }

    for (let x = 0; x < world.width; x++) {
        for (let y = 0; y < world.height; y++) {
            const tile = world.grid[x][y];
            for (const c of tile.cells) {
                for (const m of c.molecules) {
                    for (const el in m.composition) {
                        totals[el] = (totals[el] || 0) + m.composition[el];
                    }
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
    return {
        population: pop,
        avgEnergy,
        lineageCount: lineages.size,
        enzymeFunctionalDiversity: functionalSet.size
    };
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
    for (let x = 0; x < world.width; x++) {
        for (let y = 0; y < world.height; y++) {
            tsum += world.grid[x][y].temperature;
        }
    }
    const avgTemp = (tsum / (world.width * world.height)).toFixed(3);
    parts.push(`T_avg:${avgTemp}`);
    elementsDiv.textContent = parts.join("  •  ");
}

function enzymeEquationString(enzyme) {
    const keys = enzyme.affinity ? Object.keys(enzyme.affinity).sort() : [];
    if (keys.length === 0) {
        if (enzyme.type === "ligase") return "A + B → AB";
        if (enzyme.type === "hydrolase") return "A → A' + B' (fragments)";
        if (enzyme.type === "isomerase") return "A → A* (isomer)";
        return `${enzyme.type}: ? → ?`;
    }

    if (enzyme.type === "ligase") {
        const left = keys.slice(0, enzyme.maxInputs || 2).join(" + ");
        const prod = keys.slice(0, enzyme.maxInputs || 2).join("");
        return `${left} → ${prod}`;
    }

    if (enzyme.type === "hydrolase") {
        const a = keys[0];
        return `${a} → ${a}₁ + ${a}₂`;
    }

    if (enzyme.type === "isomerase") {
        const a = keys[0];
        return `${a} → ${a}*`;
    }

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
    const age = selectedCell.getAgeMs();
    const dom = selectedCell.getDominantElement() || "none";

    let moleculesHtml = "";
    if (selectedCell.molecules.length === 0) {
        moleculesHtml = "<div class='smallNote'>No internal molecules</div>";
    } else {
        moleculesHtml = "<ul style='margin:4px 0 8px 18px;padding:0;'>";
        for (const m of selectedCell.molecules) {
            const comp = Object.entries(m.composition).map(([k,v]) => `${k}${v}`).join(", ");
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

    body.innerHTML = `
        <div><strong>Age:</strong> ${(age/1000).toFixed(1)} s</div>
        <div><strong>Energy:</strong> ${selectedCell.energy.toFixed(2)}</div>
        <div><strong>State:</strong> ${selectedCell.state}</div>
        <div><strong>Dominant internal element:</strong> ${dom}</div>
        <div style="margin-top:8px"><strong>Enzymes (approx reaction):</strong> ${enzymesHtml}</div>
        <div style="margin-top:4px"><strong>Internal molecules:</strong> ${moleculesHtml}</div>
        <div class="smallNote" style="margin-top:6px">Right-click the cell to highlight its descendants in yellow.</div>
    `;
}

function markDescendants(lineageId) {
    for (let x = 0; x < world.width; x++) {
        for (let y = 0; y < world.height; y++) {
            const tile = world.grid[x][y];
            for (const c of tile.cells) {
                c._highlight = (c.lineageId === lineageId);
            }
        }
    }
}

function clearAllHighlights() {
    for (let x = 0; x < world.width; x++) {
        for (let y = 0; y < world.height; y++) {
            const tile = world.grid[x][y];
            for (const c of tile.cells) c._highlight = false;
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
    const stats = computeStats();
    const spawnProb = spawnChanceBasedOnPopulation(stats.population);
    if (Math.random() < spawnProb) world.spawnRandomCell(randomGenome);
    world.step();
    renderer.render();
    updateHud();
    updateInfoPanel();
}, world.dt);
