import { World } from "./sim/world.js";
import { CanvasRenderer } from "./render/canvas.js";

const world = new World(200, 200);

function randomGenome() {
    return {
        enzymes: [
            {
                type: "ligase",
                affinity: { B: 1.0, C: 0.5 },
                tOpt: Math.random(),
                pHOpt: Math.random(),
                secretionProb: 0.5
            }
        ],
        reproThreshold: 6 + Math.random() * 6,
        initialEnergy: 1.0 + Math.random() * 1.5,
        decayTime: 500 + Math.random() * 2000,
        dormancyBias: 0.6,
        defaultSecretionProb: 0.15,
        mutationRate: 0.06,
        postDivideMortality: 0.0,
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

let selectedCell = null;
let selectedTile = null;

renderer.onCellClick = (x, y, tile, topCell) => {
    selectedCell = topCell;
    selectedTile = tile;
    updateInfoPanel();
};

for (let i = 0; i < 40; i++) {
    world.spawnRandomCell(randomGenome);
}

function computeStats() {
    let pop = 0;
    let totalEnergy = 0;
    const lineages = new Set();
    const enzymeSet = new Set();

    for (let x = 0; x < world.width; x++) {
        for (let y = 0; y < world.height; y++) {
            const tile = world.grid[x][y];
            for (const c of tile.cells) {
                pop++;
                totalEnergy += c.energy;
                lineages.add(c.lineageId);
                for (const e of c.genome.enzymes) {
                    enzymeSet.add(e.type);
                }
            }
        }
    }

    const avgEnergy = pop > 0 ? (totalEnergy / pop).toFixed(2) : "—";
    return {
        population: pop,
        avgEnergy,
        lineageCount: lineages.size,
        enzymeDiversity: enzymeSet.size
    };
}

function updateHud() {
    const s = computeStats();
    hud.population.textContent = `Population: ${s.population}`;
    hud.avgEnergy.textContent = `Avg energy: ${s.avgEnergy}`;
    hud.lineages.textContent = `Lineages: ${s.lineageCount}`;
    hud.enzymeDiversity.textContent = `Enzyme diversity: ${s.enzymeDiversity}`;
}

function updateInfoPanel() {
    const title = document.getElementById("infoTitle");
    const body = document.getElementById("infoBody");
    if (!selectedCell) {
        title.textContent = "No cell selected";
        body.innerHTML = `<div class="smallNote">Click a white pixel (cell) to inspect it. Info updates live.</div>`;
        return;
    }
    title.textContent = `Cell — lineage ${selectedCell.lineageId}`;
    const age = selectedCell.getAgeMs();
    const dom = selectedCell.getDominantElement() || "none";
    let enzymesHtml = "<ul style='margin:4px 0 8px 18px;padding:0;'>";
    for (const e of selectedCell.genome.enzymes) {
        enzymesHtml += `<li><strong>${e.type}</strong> (tOpt:${(e.tOpt ?? 0.5).toFixed(2)}, pH:${(e.pHOpt ?? 0.5).toFixed(2)}, sec:${(e.secretionProb ?? selectedCell.genome.defaultSecretionProb).toFixed(2)})</li>`;
    }
    enzymesHtml += "</ul>";
    body.innerHTML = `
        <div><strong>Age:</strong> ${(age/1000).toFixed(1)} s</div>
        <div><strong>Energy:</strong> ${selectedCell.energy.toFixed(2)}</div>
        <div><strong>State:</strong> ${selectedCell.state}</div>
        <div><strong>Dominant internal element:</strong> ${dom}</div>
        <div style="margin-top:8px"><strong>Enzymes:</strong> ${enzymesHtml}</div>
        <div><strong>Internal molecule count:</strong> ${selectedCell.molecules.length}</div>
        <div style="margin-top:6px" class="smallNote">Click other cells to update. Metrics refresh automatically.</div>
    `;
}

setInterval(() => {
    if (Math.random() < 0.02) {
        world.spawnRandomCell(randomGenome);
    }
    world.step();
    renderer.render();
    updateHud();
    updateInfoPanel();
}, world.dt);
