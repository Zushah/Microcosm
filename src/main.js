import { World } from "./sim/world.js";
import { CanvasRenderer } from "./render/canvas.js";
import { ALL_ELEMENT_MASK, ELEMENTS, ELEMENT_MASKS, maskToString, normalizeSpecificityMask } from "./sim/chem.js";
import { chance, createRandomSeed, random, randomGaussian, randomInt, setSeed } from "./sim/rng.js";
import { isCombatEnzymeType, normalizeCombatLevel } from "./sim/eco.js";
import { applyEnzymeClassDefaults, defaultSpecificityMaskForType } from "./sim/cell.js";

const SEED_QUERY_PARAM = "seed";

const normalizeSeedValue = (value) => {
    const text = `${value ?? ""}`;
    return text.trim() !== "" ? text : createRandomSeed();
};

const seedUrl = (seed) => {
    const url = new URL(window.location.href);
    url.searchParams.set(SEED_QUERY_PARAM, seed);
    return url;
};

const currentRunSeed = setSeed(normalizeSeedValue(new URL(window.location.href).searchParams.get(SEED_QUERY_PARAM)));
window.MICROCOSM_SEED = currentRunSeed;
window.history.replaceState(null, "", seedUrl(currentRunSeed).toString());

const EDIT_BRUSH_TYPES = Object.freeze(["enval", "genome"]);
const GENOME_BRUSH_ENZYME_TYPES = Object.freeze(["anabolase", "catabolase", "transmutase", "defensase", "attackase"]);
const METABOLIC_GENOME_BRUSH_ENZYME_TYPES = Object.freeze(["anabolase", "catabolase", "transmutase"]);
const METABOLIC_GENOME_BRUSH_ENZYME_TYPE_SET = new Set(METABOLIC_GENOME_BRUSH_ENZYME_TYPES);
const DEFAULT_GENOME_BRUSH_ENZYME_CLASS = "anabolase";
const DEFAULT_GENOME_BRUSH_LEVEL = 100;
const DEFAULT_GENOME_BRUSH_ENVAL_SIGMA = 0.18;

const restartWithSeed = (value) => {
    const nextSeed = normalizeSeedValue(value);
    window.location.assign(seedUrl(nextSeed).toString());
};

const world = new World(320, 240);

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

const currentAverageEnval = () => {
    if (Number.isFinite(world.avgEnval)) return world.avgEnval;
    let sum = 0;
    for (let x = 0; x < world.width; x++) {
        for (let y = 0; y < world.height; y++) {
            sum += world.grid[x][y].enval;
        }
    }
    const count = world.width * world.height;
    if (count === 0) return world.baseEnval ?? 0;
    return sum / count;
};

const randomGenome = () => {
    const avgEnval = currentAverageEnval();
    const opt = avgEnval + (random() - 0.5) * 0.20;

    return {
        optimalEnval: opt,
        enzymes: [
            {
                type: "anabolase",
                specificityMask: ELEMENT_MASKS.A | ELEMENT_MASKS.B | ELEMENT_MASKS.C,
                bondMultiplier: 1.18,
                bondCostFraction: 0.70,
                secretionProb: 0.12,
                envalSigma: 0.16 + random() * 0.08,
                envalThroughput: 0.18
            },
            {
                type: "catabolase",
                specificityMask: ELEMENT_MASKS.A | ELEMENT_MASKS.B | ELEMENT_MASKS.C,
                bondHarvestFraction: 1.00,
                envalSigma: 0.16 + random() * 0.08,
                envalThroughput: 0.14
            },
            {
                type: "defensase",
                level: normalizeCombatLevel(normalizeCombatLevel(randomGaussian(100, 10), 100))
            }
        ],
        reproThreshold: 2 + random() * 6,
        initialEnergy: 1.2 + random() * 1.8,
        decayTime: 700 + random() * 2000,
        defaultSecretionProb: 0.15,
        mutationRate: 0.06,
        postDivideMortality: 0.0,
        desiredElementReserve: 2,
        envalStressFactor: 0.02,
        envalMutationFloor: 0.03,
        lineageId: randomInt(1e9)
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

const defaultGenomeBrushSpecificity = (type = DEFAULT_GENOME_BRUSH_ENZYME_CLASS) => {
    const normalizedType = GENOME_BRUSH_ENZYME_TYPES.includes(type) ? type : DEFAULT_GENOME_BRUSH_ENZYME_CLASS;
    const mask = defaultSpecificityMaskForType(normalizedType);
    const text = maskToString(mask);
    return text === "ABCDEF" ? "ALL" : text;
};

const interactionState = {
    mode: "explore",
    tileColorMode: "enval",
    brushWidth: 10,
    brushHeight: 10,
    brushType: "enval",
    brushIntensity: 0.00,
    genomeBrushEnzymeClass: DEFAULT_GENOME_BRUSH_ENZYME_CLASS,
    genomeBrushSpecificity: defaultGenomeBrushSpecificity(DEFAULT_GENOME_BRUSH_ENZYME_CLASS),
    genomeBrushLevel: DEFAULT_GENOME_BRUSH_LEVEL,
    genomeBrushEnvalSigma: DEFAULT_GENOME_BRUSH_ENVAL_SIGMA,
    panelOpen: true
};

const interactionUi = {
    panel: null,
    toggle: null,
    seedInput: null,
    seedSetButton: null,
    modeButtons: [],
    tileColorModeInput: null,
    editSettings: null,
    brushWidthInput: null,
    brushHeightInput: null,
    brushTypeInput: null,
    envalBrushSettings: null,
    brushIntensityInput: null,
    genomeBrushSettings: null,
    genomeBrushEnzymeClassInput: null,
    genomeBrushValueLabel: null,
    genomeBrushValueInput: null,
    genomeBrushValueNote: null,
    genomeBrushEnvalSigmaRow: null,
    genomeBrushEnvalSigmaInput: null,
    modeNote: null
};

const isGenomeBrushMetabolicType = (type) => METABOLIC_GENOME_BRUSH_ENZYME_TYPE_SET.has(type);

const normalizeBrushType = (value) => EDIT_BRUSH_TYPES.includes(value) ? value : "enval";

const normalizeGenomeBrushEnzymeClass = (value) => GENOME_BRUSH_ENZYME_TYPES.includes(value) ? value : DEFAULT_GENOME_BRUSH_ENZYME_CLASS;

const clampBrushSpan = (value, maxValue, fallback) => {
    const parsed = Math.round(Number(value));
    if (!Number.isFinite(parsed)) return fallback;
    return Math.max(1, Math.min(maxValue, parsed));
};

const parseBrushIntensity = (value, fallback) => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
};

const formatBrushIntensity = (value) => {
    if (!Number.isFinite(value)) return "0.00";
    if (value === 0) return "0.00";
    return String(value);
};

const parseGenomeBrushSpecificityMask = (value) => {
    const compact = `${value ?? ""}`.toUpperCase().replace(/[\s,]+/g, "");
    if (!compact) return null;
    if (compact === "ALL") return ALL_ELEMENT_MASK;
    if (!/^[A-F]+$/.test(compact)) return null;
    let mask = 0;
    for (let i = 0; i < compact.length; i++) mask |= ELEMENT_MASKS[compact[i]] || 0;
    return normalizeSpecificityMask(mask, ALL_ELEMENT_MASK);
};

const formatGenomeBrushSpecificity = (value, fallback = defaultGenomeBrushSpecificity(DEFAULT_GENOME_BRUSH_ENZYME_CLASS)) => {
    const mask = parseGenomeBrushSpecificityMask(value);
    if (!mask) return fallback;
    const text = maskToString(mask);
    return text === "ABCDEF" ? "ALL" : text;
};

const parseGenomeBrushLevel = (value) => {
    const parsed = Number(value);
    if (!Number.isFinite(parsed) || !Number.isInteger(parsed) || parsed < 1) return null;
    return normalizeCombatLevel(parsed);
};

const parseGenomeBrushEnvalSigma = (value) => {
    const parsed = Number(value);
    if (!Number.isFinite(parsed) || parsed <= 0) return null;
    return parsed;
};

const updateGenomeBrushValueValidity = () => {
    const input = interactionUi.genomeBrushValueInput;
    if (!input) return true;
    const enzymeClass = normalizeGenomeBrushEnzymeClass(interactionState.genomeBrushEnzymeClass);
    if (isGenomeBrushMetabolicType(enzymeClass)) {
        const specificityMask = parseGenomeBrushSpecificityMask(input.value);
        input.setCustomValidity(specificityMask ? "" : "Use letters A-F only, or enter ALL.");
        return Boolean(specificityMask);
    }
    const level = parseGenomeBrushLevel(input.value);
    input.setCustomValidity(level ? "" : "Enter a positive integer level.");
    return Boolean(level);
};

const updateGenomeBrushEnvalSigmaValidity = () => {
    const input = interactionUi.genomeBrushEnvalSigmaInput;
    if (!input) return true;
    if (input.disabled) {
        input.setCustomValidity("");
        return true;
    }
    const sigma = parseGenomeBrushEnvalSigma(input.value);
    input.setCustomValidity(sigma ? "" : "Enter an envalSigma greater than 0.");
    return Boolean(sigma);
};

const commitGenomeBrushValueInput = (reportValidity = false) => {
    if (!interactionUi.genomeBrushValueInput) return true;
    const enzymeClass = normalizeGenomeBrushEnzymeClass(interactionState.genomeBrushEnzymeClass);
    if (isGenomeBrushMetabolicType(enzymeClass)) {
        const specificityMask = parseGenomeBrushSpecificityMask(interactionUi.genomeBrushValueInput.value);
        if (!specificityMask) {
            updateGenomeBrushValueValidity();
            if (reportValidity) interactionUi.genomeBrushValueInput.reportValidity();
            syncInteractionUi();
            return false;
        }
        interactionState.genomeBrushSpecificity = formatGenomeBrushSpecificity(
            interactionUi.genomeBrushValueInput.value,
            defaultGenomeBrushSpecificity(enzymeClass)
        );
        syncInteractionUi();
        return true;
    }
    const level = parseGenomeBrushLevel(interactionUi.genomeBrushValueInput.value);
    if (!level) {
        updateGenomeBrushValueValidity();
        if (reportValidity) interactionUi.genomeBrushValueInput.reportValidity();
        syncInteractionUi();
        return false;
    }
    interactionState.genomeBrushLevel = level;
    syncInteractionUi();
    return true;
};

const commitGenomeBrushEnvalSigmaInput = (reportValidity = false) => {
    if (!interactionUi.genomeBrushEnvalSigmaInput || interactionUi.genomeBrushEnvalSigmaInput.disabled) return true;
    const sigma = parseGenomeBrushEnvalSigma(interactionUi.genomeBrushEnvalSigmaInput.value);
    if (!sigma) {
        updateGenomeBrushEnvalSigmaValidity();
        if (reportValidity) interactionUi.genomeBrushEnvalSigmaInput.reportValidity();
        syncInteractionUi();
        return false;
    }
    interactionState.genomeBrushEnvalSigma = sigma;
    syncInteractionUi();
    return true;
};

const buildGenomeBrushEnzyme = () => {
    const enzymeClass = normalizeGenomeBrushEnzymeClass(interactionUi.genomeBrushEnzymeClassInput ? interactionUi.genomeBrushEnzymeClassInput.value : interactionState.genomeBrushEnzymeClass);
    interactionState.genomeBrushEnzymeClass = enzymeClass;
    const enzyme = { type: enzymeClass };
    if (isGenomeBrushMetabolicType(enzymeClass)) {
        const specificitySource = interactionUi.genomeBrushValueInput ? interactionUi.genomeBrushValueInput.value : interactionState.genomeBrushSpecificity;
        const sigmaSource = interactionUi.genomeBrushEnvalSigmaInput ? interactionUi.genomeBrushEnvalSigmaInput.value : interactionState.genomeBrushEnvalSigma;
        const specificityMask = parseGenomeBrushSpecificityMask(specificitySource);
        const envalSigma = parseGenomeBrushEnvalSigma(sigmaSource);
        if (!specificityMask || !envalSigma) return null;
        interactionState.genomeBrushSpecificity = formatGenomeBrushSpecificity(specificitySource, defaultGenomeBrushSpecificity(enzymeClass));
        interactionState.genomeBrushEnvalSigma = envalSigma;
        enzyme.specificityMask = specificityMask;
        enzyme.envalSigma = envalSigma;
    } else {
        const levelSource = interactionUi.genomeBrushValueInput ? interactionUi.genomeBrushValueInput.value : interactionState.genomeBrushLevel;
        const level = parseGenomeBrushLevel(levelSource);
        if (!level) return null;
        interactionState.genomeBrushLevel = level;
        enzyme.level = level;
    }
    applyEnzymeClassDefaults(enzyme);
    return enzyme;
};

const syncInteractionUi = () => {
    renderer.setInteractionMode(interactionState.mode);
    renderer.setTileColorMode(interactionState.tileColorMode);
    renderer.setEditBrush(interactionState.brushWidth, interactionState.brushHeight, interactionState.brushIntensity, interactionState.brushType);
    const panel = interactionUi.panel;
    if (!panel) return;
    const isEditMode = interactionState.mode === "edit";
    const isGenomeBrush = interactionState.brushType === "genome";
    const enzymeClass = normalizeGenomeBrushEnzymeClass(interactionState.genomeBrushEnzymeClass);
    const isMetabolicGenomeBrush = isGenomeBrushMetabolicType(enzymeClass);
    panel.classList.toggle("isCollapsed", !interactionState.panelOpen);
    interactionUi.toggle.textContent = interactionState.panelOpen ? "❮" : "❯";
    interactionUi.toggle.setAttribute("aria-label", interactionState.panelOpen ? "Collapse controls" : "Expand controls");
    for (const button of interactionUi.modeButtons) {
        const active = button.dataset.mode === interactionState.mode;
        button.classList.toggle("isActive", active);
        button.setAttribute("aria-pressed", active ? "true" : "false");
    }
    interactionUi.editSettings.classList.toggle("isDisabled", !isEditMode);
    interactionUi.brushWidthInput.disabled = !isEditMode;
    interactionUi.brushHeightInput.disabled = !isEditMode;
    interactionUi.brushTypeInput.disabled = !isEditMode;
    interactionUi.brushIntensityInput.disabled = !isEditMode || isGenomeBrush;
    interactionUi.genomeBrushEnzymeClassInput.disabled = !isEditMode || !isGenomeBrush;
    interactionUi.genomeBrushValueInput.disabled = !isEditMode || !isGenomeBrush;
    interactionUi.genomeBrushEnvalSigmaInput.disabled = !isEditMode || !isGenomeBrush || !isMetabolicGenomeBrush;
    interactionUi.envalBrushSettings.hidden = isGenomeBrush;
    interactionUi.genomeBrushSettings.hidden = !isGenomeBrush;
    interactionUi.genomeBrushEnvalSigmaRow.hidden = !isMetabolicGenomeBrush;
    interactionUi.tileColorModeInput.value = interactionState.tileColorMode;
    interactionUi.brushWidthInput.value = String(interactionState.brushWidth);
    interactionUi.brushHeightInput.value = String(interactionState.brushHeight);
    interactionUi.brushTypeInput.value = interactionState.brushType;
    interactionUi.brushIntensityInput.value = formatBrushIntensity(interactionState.brushIntensity);
    interactionUi.genomeBrushEnzymeClassInput.value = enzymeClass;
    interactionUi.genomeBrushValueInput.type = isMetabolicGenomeBrush ? "text" : "number";
    interactionUi.genomeBrushValueInput.placeholder = isMetabolicGenomeBrush ? "ABC or ALL" : "100";
    interactionUi.genomeBrushValueInput.inputMode = isMetabolicGenomeBrush ? "text" : "numeric";
    interactionUi.genomeBrushValueInput.min = isMetabolicGenomeBrush ? "" : "1";
    interactionUi.genomeBrushValueInput.step = isMetabolicGenomeBrush ? "" : "1";
    interactionUi.genomeBrushValueLabel.textContent = isMetabolicGenomeBrush ? "Specificity" : "Level";
    interactionUi.genomeBrushValueNote.textContent = isMetabolicGenomeBrush ? "Use letters A-F only, or ALL. Example: ABC or BDF." : "Positive integer, minimum 1.";
    interactionUi.genomeBrushValueInput.value = isMetabolicGenomeBrush ? interactionState.genomeBrushSpecificity : String(interactionState.genomeBrushLevel);
    interactionUi.genomeBrushEnvalSigmaInput.value = String(interactionState.genomeBrushEnvalSigma);
    updateGenomeBrushValueValidity();
    updateGenomeBrushEnvalSigmaValidity();
    interactionUi.modeNote.textContent = isEditMode
        ? (isGenomeBrush
            ? "Right-click and drag to add the configured enzyme to every cell inside the rectangular brush footprint. Empty tiles are ignored."
            : "Right-click and drag to stamp the configured Δ enval across the rectangular brush footprint.")
        : "Left-drag pans, the mouse wheel zooms, left-click inspects a cell, and right-click inspects its lineage.";
};

const createInteractionPanel = () => {
    const panel = document.createElement("div");
    panel.id = "interactionPanel";
    panel.innerHTML = `
        <button id="interactionPanelToggle" type="button" aria-label="Collapse controls">❮</button>
        <h3>Interaction</h3>
        <div class="controlRow">
            <div class="seedRow">
                <label class="fieldLabel fieldLabelGrow">
                    <span>Seed</span>
                    <input id="seedInput" type="text" spellcheck="false" autocomplete="off" placeholder="Random seed">
                </label>
                <button id="seedSetButton" class="inlineActionButton" type="button">Set</button>
            </div>
        </div>
        <div class="controlCaption">Choose how the canvas responds to your mouse.</div>
        <div class="controlRow">
            <div class="controlLabel">Mode</div>
            <div class="modeSwitch">
                <button class="modeOption" data-mode="explore" type="button">Explore</button>
                <button class="modeOption" data-mode="edit" type="button">Edit</button>
            </div>
        </div>
        <div class="controlRow">
            <label class="fieldLabel fieldLabelFull">
                <span>Tile coloring</span>
                <select id="tileColorMode">
                    <option value="enval">Enval</option>
                    <option value="element:A">Element A concentration</option>
                    <option value="element:B">Element B concentration</option>
                    <option value="element:C">Element C concentration</option>
                    <option value="element:D">Element D concentration</option>
                    <option value="element:E">Element E concentration</option>
                    <option value="element:F">Element F concentration</option>
                </select>
            </label>
        </div>
        <div id="interactionEditSettings">
            <div class="controlRow">
                <div class="controlLabel">Brush size</div>
                <div class="brushGrid">
                    <label class="fieldLabel">
                        <span>X tiles</span>
                        <input id="editBrushWidth" type="number" min="1" max="${world.width}" step="1" value="${interactionState.brushWidth}">
                    </label>
                    <label class="fieldLabel">
                        <span>Y tiles</span>
                        <input id="editBrushHeight" type="number" min="1" max="${world.height}" step="1" value="${interactionState.brushHeight}">
                    </label>
                </div>
            </div>
            <div class="controlRow">
                <label class="fieldLabel fieldLabelFull">
                    <span>Brush type</span>
                    <select id="editBrushType">
                        <option value="enval">Enval</option>
                        <option value="genome">Genome</option>
                    </select>
                </label>
            </div>
            <div id="interactionEnvalBrushSettings">
                <div class="controlRow">
                    <label class="fieldLabel fieldLabelFull">
                        <span>Δ enval / stamp</span>
                        <input id="editBrushIntensity" type="number" step="0.01" value="${formatBrushIntensity(interactionState.brushIntensity)}">
                    </label>
                </div>
            </div>
            <div id="interactionGenomeBrushSettings" hidden>
                <div class="controlRow">
                    <label class="fieldLabel fieldLabelFull">
                        <span>Enzyme class</span>
                        <select id="editGenomeBrushEnzymeClass">
                            <option value="anabolase">anabolase</option>
                            <option value="catabolase">catabolase</option>
                            <option value="transmutase">transmutase</option>
                            <option value="defensase">defensase</option>
                            <option value="attackase">attackase</option>
                        </select>
                    </label>
                </div>
                <div class="controlRow">
                    <label class="fieldLabel fieldLabelFull">
                        <span id="editGenomeBrushValueLabel">Specificity</span>
                        <input id="editGenomeBrushValue" type="text" spellcheck="false" autocomplete="off" placeholder="ABC or ALL" value="${interactionState.genomeBrushSpecificity}">
                    </label>
                    <div class="controlCaption" id="editGenomeBrushValueNote">Use letters A-F only, or ALL. Example: ABC or BDF.</div>
                </div>
                <div class="controlRow" id="interactionGenomeBrushEnvalSigmaRow">
                    <label class="fieldLabel fieldLabelFull">
                        <span>envalSigma</span>
                        <input id="editGenomeBrushEnvalSigma" type="number" min="0.000001" step="0.01" value="${interactionState.genomeBrushEnvalSigma}">
                    </label>
                </div>
            </div>
        </div>
        <div class="controlCaption" id="interactionModeNote"></div>
    `;
    document.body.appendChild(panel);

    interactionUi.panel = panel;
    interactionUi.toggle = panel.querySelector("#interactionPanelToggle");
    interactionUi.seedInput = panel.querySelector("#seedInput");
    interactionUi.seedSetButton = panel.querySelector("#seedSetButton");
    interactionUi.modeButtons = [...panel.querySelectorAll(".modeOption")];
    interactionUi.tileColorModeInput = panel.querySelector("#tileColorMode");
    interactionUi.editSettings = panel.querySelector("#interactionEditSettings");
    interactionUi.brushWidthInput = panel.querySelector("#editBrushWidth");
    interactionUi.brushHeightInput = panel.querySelector("#editBrushHeight");
    interactionUi.brushTypeInput = panel.querySelector("#editBrushType");
    interactionUi.envalBrushSettings = panel.querySelector("#interactionEnvalBrushSettings");
    interactionUi.brushIntensityInput = panel.querySelector("#editBrushIntensity");
    interactionUi.genomeBrushSettings = panel.querySelector("#interactionGenomeBrushSettings");
    interactionUi.genomeBrushEnzymeClassInput = panel.querySelector("#editGenomeBrushEnzymeClass");
    interactionUi.genomeBrushValueLabel = panel.querySelector("#editGenomeBrushValueLabel");
    interactionUi.genomeBrushValueInput = panel.querySelector("#editGenomeBrushValue");
    interactionUi.genomeBrushValueNote = panel.querySelector("#editGenomeBrushValueNote");
    interactionUi.genomeBrushEnvalSigmaRow = panel.querySelector("#interactionGenomeBrushEnvalSigmaRow");
    interactionUi.genomeBrushEnvalSigmaInput = panel.querySelector("#editGenomeBrushEnvalSigma");
    interactionUi.modeNote = panel.querySelector("#interactionModeNote");

    interactionUi.seedInput.value = currentRunSeed;
    interactionUi.seedSetButton.addEventListener("click", () => {
        restartWithSeed(interactionUi.seedInput.value);
    });
    interactionUi.seedInput.addEventListener("keydown", (e) => {
        if (e.key !== "Enter") return;
        e.preventDefault();
        restartWithSeed(interactionUi.seedInput.value);
    });

    interactionUi.toggle.addEventListener("click", () => {
        interactionState.panelOpen = !interactionState.panelOpen;
        syncInteractionUi();
    });

    for (const button of interactionUi.modeButtons) {
        button.addEventListener("click", () => {
            interactionState.mode = button.dataset.mode === "edit" ? "edit" : "explore";
            syncInteractionUi();
        });
    }

    interactionUi.tileColorModeInput.addEventListener("change", () => {
        interactionState.tileColorMode = interactionUi.tileColorModeInput.value || "enval";
        syncInteractionUi();
    });

    interactionUi.brushWidthInput.addEventListener("change", () => {
        interactionState.brushWidth = clampBrushSpan(
            interactionUi.brushWidthInput.value,
            world.width,
            interactionState.brushWidth
        );
        syncInteractionUi();
    });
    interactionUi.brushWidthInput.addEventListener("blur", syncInteractionUi);

    interactionUi.brushHeightInput.addEventListener("change", () => {
        interactionState.brushHeight = clampBrushSpan(
            interactionUi.brushHeightInput.value,
            world.height,
            interactionState.brushHeight
        );
        syncInteractionUi();
    });
    interactionUi.brushHeightInput.addEventListener("blur", syncInteractionUi);

    interactionUi.brushTypeInput.addEventListener("change", () => {
        interactionState.brushType = normalizeBrushType(interactionUi.brushTypeInput.value);
        syncInteractionUi();
    });

    interactionUi.brushIntensityInput.addEventListener("change", () => {
        interactionState.brushIntensity = parseBrushIntensity(
            interactionUi.brushIntensityInput.value,
            interactionState.brushIntensity
        );
        syncInteractionUi();
    });
    interactionUi.brushIntensityInput.addEventListener("blur", syncInteractionUi);

    interactionUi.genomeBrushEnzymeClassInput.addEventListener("change", () => {
        const prevClass = normalizeGenomeBrushEnzymeClass(interactionState.genomeBrushEnzymeClass);
        const nextClass = normalizeGenomeBrushEnzymeClass(interactionUi.genomeBrushEnzymeClassInput.value);
        interactionState.genomeBrushEnzymeClass = nextClass;
        if (isGenomeBrushMetabolicType(nextClass) && !isGenomeBrushMetabolicType(prevClass)) {
            interactionState.genomeBrushSpecificity = defaultGenomeBrushSpecificity(nextClass);
        }
        syncInteractionUi();
    });

    interactionUi.genomeBrushValueInput.addEventListener("input", () => updateGenomeBrushValueValidity());
    interactionUi.genomeBrushValueInput.addEventListener("change", () => commitGenomeBrushValueInput(true));
    interactionUi.genomeBrushValueInput.addEventListener("blur", () => commitGenomeBrushValueInput(false));
    interactionUi.genomeBrushEnvalSigmaInput.addEventListener("input", () => updateGenomeBrushEnvalSigmaValidity());
    interactionUi.genomeBrushEnvalSigmaInput.addEventListener("change", () => commitGenomeBrushEnvalSigmaInput(true));
    interactionUi.genomeBrushEnvalSigmaInput.addEventListener("blur", () => commitGenomeBrushEnvalSigmaInput(false));

    syncInteractionUi();
};

createInteractionPanel();

renderer.onEditBrushStroke = (x, y) => {
    if (interactionState.mode !== "edit") return;
    if (interactionState.brushType === "genome") {
        const enzyme = buildGenomeBrushEnzyme();
        if (!enzyme) {
            if (!updateGenomeBrushValueValidity()) interactionUi.genomeBrushValueInput.reportValidity();
            else if (!updateGenomeBrushEnvalSigmaValidity()) interactionUi.genomeBrushEnvalSigmaInput.reportValidity();
            return;
        }
        const modifiedCellCount = world.applyGenomeBrush(x, y, interactionState.brushWidth, interactionState.brushHeight, enzyme);
        if (modifiedCellCount <= 0) return;
    } else world.applyEnvalBrush(x, y, interactionState.brushWidth, interactionState.brushHeight, interactionState.brushIntensity);
    lastHudUpdateMs = 0;
    lastInfoUpdateMs = 0;
};

renderer.onCellClick = (x, y, tile, topCell) => {
    selectedCell = topCell;
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

const elementTotals = (() => {
    const totals = {};
    for (const k in ELEMENTS) totals[k] = 0;
    return totals;
})();

const rebuildElementTotals = () => {
    for (const k in elementTotals) elementTotals[k] = 0;
    for (let x = 0; x < world.width; x++) {
        for (let y = 0; y < world.height; y++) {
            const tile = world.grid[x][y];
            const tileMolecules = tile.molecules;
            for (let i = 0; i < tileMolecules.length; i++) {
                const comp = tileMolecules[i].composition;
                for (const el in comp) elementTotals[el] = (elementTotals[el] || 0) + comp[el];
            }
            const cells = tile.cells;
            for (let ci = 0; ci < cells.length; ci++) {
                const cellMolecules = cells[ci].molecules;
                for (let mi = 0; mi < cellMolecules.length; mi++) {
                    const comp = cellMolecules[mi].composition;
                    for (const el in comp) elementTotals[el] = (elementTotals[el] || 0) + comp[el];
                }
            }
        }
    }
};

const applyElementDelta = (delta) => {
    if (!delta) return;
    for (const el in delta) {
        const d = delta[el] || 0;
        if (d !== 0) elementTotals[el] = (elementTotals[el] || 0) + d;
    }
};

window.__recordElementDelta = applyElementDelta;

rebuildElementTotals();

const computeElementTotals = () => elementTotals;

const normalizedSpecificityMask = (enzyme) => normalizeSpecificityMask(
    enzyme && enzyme.specificityMask,
    ALL_ELEMENT_MASK
);

const specificityLetters = (enzyme) => maskToString(normalizedSpecificityMask(enzyme));

const specificityLabel = (enzyme) => {
    const letters = specificityLetters(enzyme);
    return letters === "ABCDEF" ? "ALL" : letters;
};

const enzymeDisplayName = (enzyme) => {
    if (enzyme && isCombatEnzymeType(enzyme.type)) return `${normalizeCombatLevel(enzyme.level)}-${enzyme.type}`;
    return `${specificityLabel(enzyme)}-${enzyme.type || "enzyme"}`;
};

const enzymeSignature = (e) => {
    const type = e.type || "any";
    if (isCombatEnzymeType(type)) return `${type}|level:${normalizeCombatLevel(e.level)}`;
    const specificity = specificityLetters(e);
    const bm = (typeof e.bondMultiplier === "number") ? e.bondMultiplier.toFixed(2) : "bmX";
    const hb = (typeof e.bondHarvestFraction === "number") ? e.bondHarvestFraction.toFixed(2) : "hbX";
    const hd = (typeof e.downhillHarvestFraction === "number") ? e.downhillHarvestFraction.toFixed(2) : "hdX";
    const es = (typeof e.envalSigma === "number") ? e.envalSigma.toFixed(2) : "esX";
    const et = (typeof e.envalThroughput === "number") ? e.envalThroughput.toFixed(2) : "etX";
    return `${type}|spec:${specificity}|bm:${bm}|hb:${hb}|hd:${hd}|es:${es}|et:${et}`;
};

const snapshotEnzymesForStats = (enzymes) => Array.isArray(enzymes) ? enzymes.map((enzyme) => ({ ...enzyme })) : [];

const livingCells = new Set();
const lineageCounts = new Map();
const lineageCells = new Map();
const enzymeSignatureCounts = new Map();

const getOrCreateSet = (map, key) => {
    if (!map.has(key)) map.set(key, new Set());
    return map.get(key);
};

const bumpCounter = (map, key, delta) => {
    const next = (map.get(key) || 0) + delta;
    if (next <= 0) {
        map.delete(key);
        return 0;
    }
    map.set(key, next);
    return next;
};

const trackCellBirth = (cell) => {
    if (!cell) return;
    if (cell.state === "dead") return;
    if (livingCells.has(cell)) return;
    livingCells.add(cell);
    const lineageId = cell.lineageId;
    bumpCounter(lineageCounts, lineageId, 1);
    getOrCreateSet(lineageCells, lineageId).add(cell);
    const enzymeSnapshot = snapshotEnzymesForStats(cell.genome && cell.genome.enzymes);
    cell._statsEnzymeSnapshot = enzymeSnapshot;
    for (const e of enzymeSnapshot) {
        const sig = enzymeSignature(e);
        bumpCounter(enzymeSignatureCounts, sig, 1);
    }
};

const trackCellGenomeChange = (cell, previousEnzymes = null) => {
    if (!cell) return;
    if (!livingCells.has(cell)) return;

    const prior = Array.isArray(previousEnzymes)
        ? previousEnzymes
        : snapshotEnzymesForStats(cell._statsEnzymeSnapshot);
    for (const e of prior) {
        const sig = enzymeSignature(e);
        bumpCounter(enzymeSignatureCounts, sig, -1);
    }

    const nextSnapshot = snapshotEnzymesForStats(cell.genome && cell.genome.enzymes);
    cell._statsEnzymeSnapshot = nextSnapshot;
    for (const e of nextSnapshot) {
        const sig = enzymeSignature(e);
        bumpCounter(enzymeSignatureCounts, sig, 1);
    }
};

const trackCellDeath = (cell) => {
    if (!cell) return;
    if (!livingCells.has(cell)) return;
    livingCells.delete(cell);
    const lineageId = cell.lineageId;
    bumpCounter(lineageCounts, lineageId, -1);
    const set = lineageCells.get(lineageId);
    if (set) {
        set.delete(cell);
        if (set.size === 0) lineageCells.delete(lineageId);
    }
    const enzymeSnapshot = Array.isArray(cell._statsEnzymeSnapshot)
        ? cell._statsEnzymeSnapshot
        : snapshotEnzymesForStats(cell.genome && cell.genome.enzymes);
    for (const e of enzymeSnapshot) {
        const sig = enzymeSignature(e);
        bumpCounter(enzymeSignatureCounts, sig, -1);
    }
    cell._statsEnzymeSnapshot = [];
};

window.__recordCellBirth = trackCellBirth;
window.__recordCellDeath = trackCellDeath;
window.__recordCellGenomeChange = trackCellGenomeChange;

const rebuildStatsIndex = () => {
    livingCells.clear();
    lineageCounts.clear();
    lineageCells.clear();
    enzymeSignatureCounts.clear();
    for (let x = 0; x < world.width; x++) {
        for (let y = 0; y < world.height; y++) {
            const tile = world.grid[x][y];
            for (const c of tile.cells) trackCellBirth(c);
        }
    }
};

rebuildStatsIndex();

const computeStats = () => {
    const population = livingCells.size;
    let avgEnergy = "—";
    if (population > 0) {
        let sum = 0;
        for (const c of livingCells) {
            const e = (typeof c.energy === "number") ? c.energy : 0;
            sum += e;
        }
        avgEnergy = (sum / population).toFixed(2);
    }
    const lineageCount = lineageCounts.size;
    const enzymeFunctionalDiversity = enzymeSignatureCounts.size;
    return { population, avgEnergy, lineageCount, enzymeFunctionalDiversity };
};

const computeLineageSnapshot = (lineageId) => {
    const cells = lineageCells.get(lineageId);
    if (!cells || cells.size === 0) {
        return {
            lineageId,
            population: 0,
            avgEnergy: 0,
            avgAgeSec: 0,
            maxAgeSec: 0,
            enzymeFunctionalDiversity: 0,
            topEnzymeTypes: []
        };
    }
    let totalEnergy = 0;
    let totalAge = 0;
    let maxAge = 0;
    const enzymeSet = new Set();
    const enzymeTypeCounts = new Map();
    for (const c of cells) {
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
    const topEnzymeTypes = [...enzymeTypeCounts.entries()]
        .sort((a, b) => b[1] - a[1])
        .slice(0, 4);
    const count = cells.size;
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
    const avgEnval = Number.isFinite(world.avgEnval) ? world.avgEnval : (world.baseEnval ?? 0);
    parts.push(`enval_avg:${avgEnval.toFixed(3)}`);
    elementsDiv.textContent = parts.join("  •  ");
};

const compositionToString = (comp) => {
    if (!comp) return "—";
    return Object.entries(comp).map(([k, v]) => `${k}${v}`).join("");
};

const moleculeLikeToString = (v) => {
    if (v == null) return "—";
    if (typeof v === "string") return v;
    if (typeof v === "object") {
        if (v.composition) return compositionToString(v.composition);
        if (typeof v.product === "string") return v.product;
    }
    return "—";
};

const ensureArray = (v) => {
    if (Array.isArray(v)) return v;
    if (v == null) return [];
    return [v];
};

const normalizeReactionEventForDisplay = (ev) => {
    const src = ev || {};
    const substratesRaw = (src.substrates !== undefined) ? src.substrates : src.consumed;
    const byproductsRaw = (src.byproducts !== undefined) ? src.byproducts : src.byproduct;
    const substrates = ensureArray(substratesRaw).map((x) => moleculeLikeToString(x));
    const byproducts = ensureArray(byproductsRaw).map((x) => moleculeLikeToString(x));
    let product = "—";
    if (typeof src.product === "string") product = src.product;
    else if (src.produced !== undefined) product = moleculeLikeToString(src.produced);
    return { substrates, product, byproducts };
};

const formatReactionExpression = (ev) => {
    const norm = normalizeReactionEventForDisplay(ev);
    const left = norm.substrates.length === 0 ? "—" : norm.substrates.join(" + ");
    let right;
    if (norm.product && norm.product !== "—") {
        right = norm.product;
        if (norm.byproducts.length > 0) right += " (+ " + norm.byproducts.join(", ") + ")";
    } else {
        if (norm.byproducts.length === 0) right = "—";
        else if (norm.byproducts.length === 1) right = norm.byproducts[0];
        else right = norm.byproducts.join(" + ");
    }

    let expr = `${left} → ${right}`;
    if (ev && ev.enzymeType === "transmutase" && ev.transmutedFrom && ev.transmutedTo) {
        expr += ` [${ev.transmutedFrom}→${ev.transmutedTo}]`;
    }
    if (ev && ev.enzymeType === "attackase" && typeof ev.attackLevel === "number" && typeof ev.defenseLevel === "number") {
        expr += ` [${ev.attackLevel}>${ev.defenseLevel}]`;
    }
    return expr;
};

const formatEnvalExchange = (ev) => {
    const input = (typeof ev.envalInput === "number") ? ev.envalInput : 0;
    const output = (typeof ev.envalOutput === "number") ? ev.envalOutput : 0;
    const delta = (typeof ev.deltaEnval === "number") ? ev.deltaEnval : (output - input);
    return `${input.toFixed(3)}→${output.toFixed(3)} (${delta.toFixed(3)})`;
};

const formatEnergyTerm = (v) => {
    return (typeof v === "number" && Number.isFinite(v)) ? v.toFixed(3) : "—";
};

const enzymeParameterString = (enzyme, genome) => {
    if (enzyme && isCombatEnzymeType(enzyme.type)) {
        return `level:${normalizeCombatLevel(enzyme.level)}`;
    }
    const sigma = (enzyme.envalSigma ?? 0.18).toFixed(2);
    const throughput = (enzyme.envalThroughput ?? 0).toFixed(3);
    const secretion = (enzyme.secretionProb ?? genome.defaultSecretionProb ?? 0).toFixed(2);
    const parts = [`specificity:${specificityLabel(enzyme)}`, `envalSigma:${sigma}`, `throughput:${throughput}`, `secretion:${secretion}`];
    if (enzyme.type === "anabolase") {
        parts.unshift(`cost:${(enzyme.bondCostFraction ?? 0.90).toFixed(2)}`);
        parts.unshift(`multiplier:${(enzyme.bondMultiplier ?? 1.15).toFixed(2)}`);
    } else if (enzyme.type === "catabolase") {
        parts.unshift(`harvest:${(enzyme.bondHarvestFraction ?? 1.00).toFixed(2)}`);
    } else if (enzyme.type === "transmutase") {
        parts.unshift(`harvest:${(enzyme.downhillHarvestFraction ?? 0.18).toFixed(2)}`);
    }
    return parts.join(", ");
};

const buildCellInfoHtml = (cell) => {
    if (!cell) {
        return {
            title: "No cell selected",
            body: `<div class="smallNote">Left-click a cell to inspect it. In Explore mode, right-click a cell to inspect its lineage.</div>`
        };
    }

    const birthSim = cell.birthSimTime ?? 0;
    const deathSim = cell.deathSimTime ?? null;
    const ageS = deathSim !== null ? (deathSim - birthSim) : ((window.SIM_TIME || 0) - birthSim);
    const ageSstr = Number(ageS).toFixed(2);
    const internalMolEnergy = cell.molecules.reduce((s, m) => s + (m.energy || 0), 0);
    const totalStoredEnergy = (cell.energy || 0) + internalMolEnergy;

    const dom = cell.getDominantElement ? (cell.getDominantElement() || "none") : "none";
    const optimalEnval = (typeof cell.genome.optimalEnval === "number") ? cell.genome.optimalEnval : 0;
    const localEnval = (
        cell._worldRef &&
        typeof cell._tileX === "number" &&
        typeof cell._tileY === "number" &&
        typeof cell._worldRef.getLocalEnvalAverage === "function"
    )
        ? cell._worldRef.getLocalEnvalAverage(cell._tileX, cell._tileY, 2)
        : 0;
    const worldAvgEnval = (cell._worldRef && Number.isFinite(cell._worldRef.avgEnval))
        ? cell._worldRef.avgEnval
        : 0;

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
        const params = enzymeParameterString(e, cell.genome);
        enzymesHtml += `<li><strong>${enzymeDisplayName(e)}</strong> — <code>${params}</code></li>`;
    }
    enzymesHtml += "</ul>";

    let reactionHtml = "";
    if (!cell.reactionLog || cell.reactionLog.length === 0) {
        reactionHtml = "<div class='smallNote'>No recent reactions logged for this cell.</div>";
    } else {
        reactionHtml = "<table style='width:100%;font-size:12px;border-collapse:collapse;'><thead><tr style='text-align:left;'><th>t@evt(s)</th><th>Reaction</th><th>ΔE</th><th>enval</th><th>Eenv</th><th>Echem</th><th>Ebond</th><th>Etrans</th><th>Ecat</th></tr></thead><tbody>";
        const slice = cell.reactionLog.slice(0, 12);
        for (const ev of slice) {
            const tAtEvt = (ev.ageAtEventSec !== undefined) ? Number(ev.ageAtEventSec).toFixed(3) : "—";
            const reactionStr = formatReactionExpression(ev);
            const envalStr = formatEnvalExchange(ev);
            reactionHtml += `<tr><td style='padding:4px 6px;border-bottom:1px solid rgba(0,0,0,0.06)'>${tAtEvt}</td><td style='padding:4px 6px;border-bottom:1px solid rgba(0,0,0,0.06)'><code>${reactionStr}</code></td><td style='padding:4px 6px;border-bottom:1px solid rgba(0,0,0,0.06)'>${formatEnergyTerm(ev.deltaE)}</td><td style='padding:4px 6px;border-bottom:1px solid rgba(0,0,0,0.06)'>${envalStr}</td><td style='padding:4px 6px;border-bottom:1px solid rgba(0,0,0,0.06)'>${formatEnergyTerm(ev.envalEnergy)}</td><td style='padding:4px 6px;border-bottom:1px solid rgba(0,0,0,0.06)'>${formatEnergyTerm(ev.chemicalDelta)}</td><td style='padding:4px 6px;border-bottom:1px solid rgba(0,0,0,0.06)'>${formatEnergyTerm(ev.bondEnergyDelta)}</td><td style='padding:4px 6px;border-bottom:1px solid rgba(0,0,0,0.06)'>${formatEnergyTerm(ev.transmutationEnergyDelta)}</td><td style='padding:4px 6px;border-bottom:1px solid rgba(0,0,0,0.06)'>${formatEnergyTerm(ev.catalyticCost)}</td></tr>`;
        }
        reactionHtml += "</tbody></table>";
    }

    const deathNote = (cell.deathSimTime !== null && cell.deathSimTime !== undefined) ? " (dead)" : "";

    return {
        title: `Cell — lineage ${cell.lineageId}`,
        body: `
            <div><strong>Age:</strong> ${ageSstr}s${deathNote}</div>
            <div><strong>Energy:</strong> ${cell.energy.toFixed(3)} (total stored: ${totalStoredEnergy.toFixed(3)})</div>
            <div><strong>Optimal enval:</strong> ${optimalEnval.toFixed(3)}</div>
            <div><strong>Local enval:</strong> ${localEnval.toFixed(3)} (5×5 average)</div>
            <div><strong>World avg enval:</strong> ${worldAvgEnval.toFixed(3)}</div>
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
        <div>Births: ${rates.totals.births + 1}</div>
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
            return `[t=${time}s] ${reactionStr} | ΔE=${formatEnergyTerm(ev.deltaE)} | enval=${formatEnvalExchange(ev)} | Eenv=${formatEnergyTerm(ev.envalEnergy)} | Echem=${formatEnergyTerm(ev.chemicalDelta)} | Ebond=${formatEnergyTerm(ev.bondEnergyDelta)} | Etrans=${formatEnergyTerm(ev.transmutationEnergyDelta)} | Ecat=${formatEnergyTerm(ev.catalyticCost)}`;
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

        const spawnProb = spawnChanceBasedOnPopulation(livingCells.size);
        if (chance(spawnProb)) world.spawnRandomCell(randomGenome);

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
