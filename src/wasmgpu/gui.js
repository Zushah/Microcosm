import { MICROCOSM_DISPLAY_MODES } from "./render.js";

const DISPLAY_LABELS = Object.freeze({
    enval: "Enval",
    occupancy: "Occupancy",
    mass: "Mass",
    molecules: "Molecules",
    "element-a": "Element A presence",
    "element-b": "Element B presence",
    "element-c": "Element C presence",
    "element-d": "Element D presence",
    "element-e": "Element E presence",
    "element-f": "Element F presence"
});

const ELEMENT_KEYS = Object.freeze(["A", "B", "C", "D", "E", "F"]);
const DETAIL_REFRESH_INTERVAL_MS = 1500;
const LINEAGE_REFRESH_INTERVAL_MS = 2500;
const LINEAGE_LIST_REFRESH_INTERVAL_MS = 5000;
const CELL_MOLECULE_LIMIT = 64;
const CELL_REACTION_LIMIT = 24;
const LINEAGE_LIST_LIMIT = 32;

const displayValue = (value) => {
    if (typeof value === "bigint") return value.toString();
    if (typeof value === "number") {
        if (!Number.isFinite(value)) return String(value);
        if (Number.isInteger(value)) return String(value);
        return value.toFixed(4);
    }
    return String(value ?? "—");
};

const escapeHtml = (value) => String(value ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");

const htmlValue = (value) => ({ html: value });

const renderValue = (value) => { if (value && typeof value === "object" && Object.prototype.hasOwnProperty.call(value, "html")) return value.html; return escapeHtml(displayValue(value)); };

const formatArraySample = (array, count = 8) => { if (!array) return "—"; const size = Math.min(count, array.length), out = []; for (let i = 0; i < size; i++) out.push(displayValue(array[i])); return out.length > 0 ? out.join(", ") : "—"; };

const detailRows = (rows) => rows.map(([label, value]) => `<div class="detailRow"><span>${escapeHtml(label)}</span><strong>${renderValue(value)}</strong></div>`).join("");

const elementCountValue = (counts, index, key) => { if (!counts) return null; if (Array.isArray(counts)) return counts[index] ?? 0; return counts[key] ?? counts[key.toLowerCase()] ?? 0; };

const elementCountsText = (counts) => { if (!counts) return "—"; return ELEMENT_KEYS.map((key, index) => `${key}:${displayValue(elementCountValue(counts, index, key))}`).join("  "); };

const compositionText = (counts) => { if (!counts) return "—"; return ELEMENT_KEYS.map((key, index) => [key, elementCountValue(counts, index, key)]).filter(([, value]) => Number(value || 0) > 0).map(([key, value]) => `${key}${Number(value) > 1 ? displayValue(value) : ""}`).join(" ") || "empty"; };

const statusClass = (status) => status === "edit" ? "modeEdit" : "modeExplore";

const errorText = (error) => error && error.message ? error.message : String(error || "Unknown error");

const jsonPreview = (value) => `<pre class="jsonPreview">${escapeHtml(JSON.stringify(value ?? null, null, 2))}</pre>`;

const payloadMessage = (message, className = "smallNote") => `<div class="${className}">${escapeHtml(message)}</div>`;

const makeTable = (columns, rows, emptyMessage = "No records.") => { if (!rows || rows.length === 0) return payloadMessage(emptyMessage); return ` <table class="dataTable"> <thead><tr>${columns.map((column) => `<th>${escapeHtml(column.label)}</th>`).join("")}</tr></thead> <tbody> ${rows.map((row) => ` <tr> ${columns.map((column) => `<td>${renderValue(column.value(row))}</td>`).join("")} </tr> `).join("")} </tbody> </table> `; };

const buttonForLineage = (lineageId) => `<button class="inlineButton" type="button" data-lineage-id="${escapeHtml(displayValue(lineageId))}">Select</button>`;

export class MicrocosmGUI {
    static bind(root = document) {
        return new MicrocosmGUI({
            status: root.getElementById("status"),
            diagnostics: root.getElementById("diagnostics"),
            renderStats: root.getElementById("renderStats"),
            views: root.getElementById("views"),
            samples: root.getElementById("samples"),
            errors: root.getElementById("errors"),
            pauseButton: root.getElementById("pauseButton"),
            stepButton: root.getElementById("stepButton"),
            resetButton: root.getElementById("resetButton"),
            seedInput: root.getElementById("seedInput"),
            displayMode: root.getElementById("displayMode"),
            interactionMode: root.getElementById("interactionMode"),
            brushWidth: root.getElementById("brushWidth"),
            brushHeight: root.getElementById("brushHeight"),
            brushDelta: root.getElementById("brushDelta"),
            clearSelectionButton: root.getElementById("clearSelectionButton"),
            clearLineageButton: root.getElementById("clearLineageButton"),
            refreshDetailsButton: root.getElementById("refreshDetailsButton"),
            copyInspectionButton: root.getElementById("copyInspectionButton"),
            copyCellDetailButton: root.getElementById("copyCellDetailButton"),
            copyTileButton: root.getElementById("copyTileButton"),
            copyLineageButton: root.getElementById("copyLineageButton"),
            copyMoleculesButton: root.getElementById("copyMoleculesButton"),
            copyReactionsButton: root.getElementById("copyReactionsButton"),
            hoverInspector: root.getElementById("hoverInspector"),
            tileInspector: root.getElementById("tileInspector"),
            cellInspector: root.getElementById("cellInspector"),
            lineageInspector: root.getElementById("lineageInspector"),
            cellDetailInspector: root.getElementById("cellDetailInspector"),
            genomeInspector: root.getElementById("genomeInspector"),
            enzymeInspector: root.getElementById("enzymeInspector"),
            moleculeInspector: root.getElementById("moleculeInspector"),
            reactionInspector: root.getElementById("reactionInspector"),
            lineageDetailInspector: root.getElementById("lineageDetailInspector"),
            lineageList: root.getElementById("lineageList"),
            canvas: root.getElementById("wasmgpuCanvas")
        });
    }

    constructor(elements) {
        this.elements = elements;
        this._lastInspectionPayload = null;
        this._forceDetailRefresh = true;
        this._detail = {
            selectedCellId: null,
            selectedLineageId: null,
            cellDetail: null,
            molecules: null,
            reactions: null,
            lineage: null,
            lineageList: null,
            cellError: null,
            moleculeError: null,
            reactionError: null,
            lineageError: null,
            lineageListError: null,
            lastCellRefreshMs: 0,
            lastLineageRefreshMs: 0,
            lastLineageListRefreshMs: 0
        };
        this.populateDisplayModes();
    }

    get canvas() {
        return this.elements.canvas;
    }

    get seed() {
        return this.elements.seedInput ? this.elements.seedInput.value : "demo";
    }

    set seed(value) {
        if (this.elements.seedInput) this.elements.seedInput.value = value;
    }

    get displayMode() {
        return this.elements.displayMode ? this.elements.displayMode.value : "enval";
    }

    get mode() {
        return this.elements.interactionMode ? this.elements.interactionMode.value : "explore";
    }

    get brushOptions() {
        return {
            width: this.elements.brushWidth ? this.elements.brushWidth.value : 10,
            height: this.elements.brushHeight ? this.elements.brushHeight.value : 10,
            delta: this.elements.brushDelta ? this.elements.brushDelta.value : 0.05
        };
    }

    populateDisplayModes() {
        const select = this.elements.displayMode;
        if (!select) return;
        select.innerHTML = MICROCOSM_DISPLAY_MODES.map((mode) => `<option value="${mode}">${DISPLAY_LABELS[mode] || mode}</option>`).join("");
        select.value = "enval";
    }

    bindControls(handlers) {
        if (this.elements.pauseButton) this.elements.pauseButton.addEventListener("click", () => handlers.pause && handlers.pause());
        if (this.elements.stepButton) this.elements.stepButton.addEventListener("click", () => handlers.step && handlers.step());
        if (this.elements.resetButton) this.elements.resetButton.addEventListener("click", () => handlers.reset && handlers.reset());
        if (this.elements.displayMode) this.elements.displayMode.addEventListener("change", () => handlers.displayMode && handlers.displayMode(this.displayMode));
        if (this.elements.interactionMode) this.elements.interactionMode.addEventListener("change", () => handlers.mode && handlers.mode(this.mode));
        for (const input of [this.elements.brushWidth, this.elements.brushHeight, this.elements.brushDelta]) {
            if (!input) continue;
            input.addEventListener("change", () => handlers.brush && handlers.brush(this.brushOptions));
            input.addEventListener("input", () => handlers.brushPreview && handlers.brushPreview(this.brushOptions));
        }
        if (this.elements.clearSelectionButton) this.elements.clearSelectionButton.addEventListener("click", () => handlers.clearSelection && handlers.clearSelection());
        if (this.elements.clearLineageButton) this.elements.clearLineageButton.addEventListener("click", () => handlers.clearLineage && handlers.clearLineage());
        if (this.elements.refreshDetailsButton) this.elements.refreshDetailsButton.addEventListener("click", () => { this.forceRefreshDetails(); if (handlers.refreshDetails) handlers.refreshDetails(); });
        if (this.elements.copyInspectionButton) this.elements.copyInspectionButton.addEventListener("click", () => this.copyInspection());
        if (this.elements.copyCellDetailButton) this.elements.copyCellDetailButton.addEventListener("click", () => this.copyPayload("cell", this.elements.copyCellDetailButton));
        if (this.elements.copyTileButton) this.elements.copyTileButton.addEventListener("click", () => this.copyPayload("tile", this.elements.copyTileButton));
        if (this.elements.copyLineageButton) this.elements.copyLineageButton.addEventListener("click", () => this.copyPayload("lineage", this.elements.copyLineageButton));
        if (this.elements.copyMoleculesButton) this.elements.copyMoleculesButton.addEventListener("click", () => this.copyPayload("molecules", this.elements.copyMoleculesButton));
        if (this.elements.copyReactionsButton) this.elements.copyReactionsButton.addEventListener("click", () => this.copyPayload("reactions", this.elements.copyReactionsButton));
        if (this.elements.lineageList) {
            this.elements.lineageList.addEventListener("click", (event) => {
                const button = event.target instanceof Element ? event.target.closest("[data-lineage-id]") : null;
                if (!button) return;
                const lineageId = Number(button.getAttribute("data-lineage-id"));
                if (Number.isFinite(lineageId) && handlers.selectLineage) handlers.selectLineage(lineageId);
            });
        }
        if (this.elements.seedInput) {
            this.elements.seedInput.addEventListener("keydown", (event) => {
                if (event.key !== "Enter") return;
                event.preventDefault();
                if (handlers.reset) handlers.reset();
            });
        }
    }

    forceRefreshDetails() {
        this._forceDetailRefresh = true;
    }

    setPaused(paused) {
        if (this.elements.pauseButton) this.elements.pauseButton.textContent = paused ? "Resume" : "Pause";
    }

    setMode(mode) {
        if (this.elements.interactionMode) this.elements.interactionMode.value = mode === "edit" ? "edit" : "explore";
    }

    setBrushOptions(options = {}) {
        if (this.elements.brushWidth && options.width !== undefined) this.elements.brushWidth.value = String(options.width);
        if (this.elements.brushHeight && options.height !== undefined) this.elements.brushHeight.value = String(options.height);
        if (this.elements.brushDelta && options.delta !== undefined) this.elements.brushDelta.value = String(options.delta);
    }

    setStatus(text) {
        if (this.elements.status) this.elements.status.textContent = text;
    }

    setError(error) {
        const message = error && error.stack ? error.stack : String(error || "Unknown error");
        console.error(error);
        if (this.elements.errors) this.elements.errors.textContent = message;
        this.setStatus("Error");
    }

    clearError() {
        if (this.elements.errors) this.elements.errors.textContent = "";
    }

    update(state) {
        const runtime = state.runtime;
        const renderer = state.renderer;
        if (!runtime || !runtime.ready) return;
        const stats = runtime.stats || {};
        const renderStats = renderer ? renderer.diagnostics : {};
        const interaction = state.interaction ? state.interaction.state : null;
        if (interaction) this.setMode(interaction.mode);
        this.updateDetailQueries(runtime, interaction);
        const diagnostics = [
            ["Runtime", runtime.ready ? "ready" : "not ready"],
            ["Version", runtime.version],
            ["Handle", runtime.handle],
            ["Tick", stats.tick_count],
            ["Sim time", stats.sim_time_seconds],
            ["FPS", state.fps],
            ["TPS", state.tps],
            ["World", `${stats.width} × ${stats.height}`],
            ["Live cells", stats.live_cell_count],
            ["Molecules", stats.molecule_count],
            ["Births", stats.births],
            ["Deaths", stats.deaths],
            ["Avg cell energy", stats.average_cell_energy],
            ["Avg enval", stats.average_enval],
            ["Reaction successes", stats.reaction_successes],
            ["Render epoch", runtime.renderEpoch]
        ];
        if (this.elements.diagnostics) this.elements.diagnostics.innerHTML = diagnostics.map(([label, value]) => `
            <div class="datum">
                <span>${escapeHtml(label)}</span>
                <strong>${escapeHtml(displayValue(value))}</strong>
            </div>
        `).join("");
        if (this.elements.renderStats) {
            const rows = [
                ["Strategy", renderStats.strategy || "—"],
                ["Display mode", renderStats.displayMode || "—"],
                ["Tiles rendered", renderStats.tileCountRendered],
                ["Cells rendered", renderStats.cellCountRendered],
                ["Selected lineage", renderStats.selectedLineage],
                ["Selected cell", renderStats.selectedCellId],
                ["Renderer frames", renderStats.frameCount],
                ["Paused", state.paused ? "yes" : "no"]
            ];
            this.elements.renderStats.innerHTML = rows.map(([label, value]) => `
                <div class="datum">
                    <span>${escapeHtml(label)}</span>
                    <strong>${escapeHtml(displayValue(value))}</strong>
                </div>
            `).join("");
        }
        this.updateViewTable(runtime);
        this.updateSamples(runtime);
        this.updateInspection(interaction);
        this.updateRichInspection(interaction);
    }

    updateDetailQueries(runtime, interaction) {
        const now = performance.now();
        const selectedCellId = interaction && interaction.selectedCellId != null ? Number(interaction.selectedCellId) >>> 0 : null;
        const selectedLineageId = interaction && interaction.selectedLineageId != null ? Number(interaction.selectedLineageId) >>> 0 : null;
        const force = this._forceDetailRefresh;
        if (selectedCellId !== this._detail.selectedCellId) {
            this._detail.selectedCellId = selectedCellId;
            this._detail.cellDetail = null;
            this._detail.molecules = null;
            this._detail.reactions = null;
            this._detail.cellError = null;
            this._detail.moleculeError = null;
            this._detail.reactionError = null;
            this._detail.lastCellRefreshMs = 0;
        }
        if (selectedLineageId !== this._detail.selectedLineageId) {
            this._detail.selectedLineageId = selectedLineageId;
            this._detail.lineage = null;
            this._detail.lineageError = null;
            this._detail.lastLineageRefreshMs = 0;
        }
        if (selectedCellId !== null && (force || now - this._detail.lastCellRefreshMs >= DETAIL_REFRESH_INTERVAL_MS)) {
            this.refreshCellDetails(runtime, selectedCellId);
            this._detail.lastCellRefreshMs = now;
        }
        if (selectedLineageId !== null && (force || now - this._detail.lastLineageRefreshMs >= LINEAGE_REFRESH_INTERVAL_MS)) {
            this.refreshLineageDetail(runtime, selectedLineageId);
            this._detail.lastLineageRefreshMs = now;
        }
        if (force || !this._detail.lineageList || now - this._detail.lastLineageListRefreshMs >= LINEAGE_LIST_REFRESH_INTERVAL_MS) {
            this.refreshLineageList(runtime);
            this._detail.lastLineageListRefreshMs = now;
        }
        this._forceDetailRefresh = false;
    }

    refreshCellDetails(runtime, cellId) {
        try { this._detail.cellDetail = runtime.inspectCellDetail(cellId, { moleculeLimit: CELL_MOLECULE_LIMIT, reactionLimit: CELL_REACTION_LIMIT }); this._detail.cellError = null; }
        catch (error) { this._detail.cellDetail = null; this._detail.cellError = errorText(error); }
        try { this._detail.molecules = runtime.inspectCellMolecules(cellId, { limit: CELL_MOLECULE_LIMIT }); this._detail.moleculeError = null; }
        catch (error) { this._detail.molecules = null; this._detail.moleculeError = errorText(error); }
        try { this._detail.reactions = runtime.inspectCellReactions(cellId, { limit: CELL_REACTION_LIMIT }); this._detail.reactionError = null; }
        catch (error) { this._detail.reactions = null; this._detail.reactionError = errorText(error); }
    }

    refreshLineageDetail(runtime, lineageId) {
        try { this._detail.lineage = runtime.inspectLineage(lineageId); this._detail.lineageError = null; }
        catch (error) { this._detail.lineage = null; this._detail.lineageError = errorText(error); }
    }

    refreshLineageList(runtime) {
        try { this._detail.lineageList = runtime.listLineages({ limit: LINEAGE_LIST_LIMIT }); this._detail.lineageListError = null; }
        catch (error) { this._detail.lineageList = null; this._detail.lineageListError = errorText(error); }
    }

    updateInspection(interaction) {
        if (!interaction) return;
        this._lastInspectionPayload = {
            hoverTile: interaction.hoverTileInfo,
            selectedTile: interaction.selectedTileInfo,
            selectedCell: interaction.selectedCellInfo,
            selectedCellDetail: this._detail.cellDetail,
            selectedCellMolecules: this._detail.molecules,
            selectedCellReactions: this._detail.reactions,
            selectedLineage: this._detail.lineage,
            lineageList: this._detail.lineageList,
            selectedLineageId: interaction.selectedLineageId,
            mode: interaction.mode,
            brush: interaction.brush
        };
        if (this.elements.hoverInspector) {
            const tile = interaction.hoverTileInfo;
            this.elements.hoverInspector.innerHTML = tile ? detailRows([
                ["Tile", `${tile.x}, ${tile.y}`],
                ["Enval", tile.enval],
                ["Cell", tile.cell_id ?? "—"],
                ["Molecules", tile.molecule_count],
                ["Mass", tile.mass_count]
            ]) : payloadMessage("Move over the canvas to probe a tile.");
        }
        if (this.elements.tileInspector) {
            const tile = interaction.selectedTileInfo;
            this.elements.tileInspector.innerHTML = tile ? detailRows([
                ["Tile", `${tile.x}, ${tile.y}`],
                ["Tile id", tile.tile_id],
                ["Enval", tile.enval],
                ["Occupied cell", tile.cell_id ?? "—"],
                ["Molecule count", tile.molecule_count],
                ["Mass count", tile.mass_count],
                ["Element counts", elementCountsText(tile.element_counts)],
                ["Element mask", tile.element_mask]
            ]) : payloadMessage("Left-click a tile to select it.");
        }
        if (this.elements.cellInspector) {
            const cell = this.cellDetailPayload()?.cell || interaction.selectedCellInfo;
            this.elements.cellInspector.innerHTML = cell ? detailRows([
                ["Cell id", cell.cell_id],
                ["Tile", `${cell.x}, ${cell.y}`],
                ["Lineage", cell.lineage_id],
                ["Energy", cell.energy],
                ["Age", `${displayValue(cell.age_seconds)}s`],
                ["Optimal enval", cell.optimal_enval],
                ["Local enval avg", cell.local_enval_average],
                ["Enzymes", cell.enzyme_count],
                ["Internal atoms", cell.internal_atom_count],
                ["Attack", cell.combat_attack_total],
                ["Defense", cell.combat_defense_total],
                ["Repro threshold", cell.repro_threshold],
                ["Decay time", cell.decay_time]
            ]) : (this._detail.cellError ? payloadMessage(`Selected cell detail unavailable: ${this._detail.cellError}`, "smallNote errorText") : payloadMessage("No active cell selected."));
        }
        if (this.elements.lineageInspector) {
            this.elements.lineageInspector.innerHTML = detailRows([
                ["Selected lineage", interaction.selectedLineageId ?? "—"],
                ["Mode", htmlValue(`<span class="${statusClass(interaction.mode)}">${escapeHtml(interaction.mode)}</span>`)],
                ["Brush", `${interaction.brush.width} × ${interaction.brush.height}`],
                ["Δ enval", interaction.brush.delta]
            ]);
        }
    }

    updateRichInspection(interaction) {
        const detail = this.cellDetailPayload();
        this.renderCellDetail(detail);
        this.renderGenome(detail && detail.genome);
        this.renderEnzymes(detail && detail.genome && detail.genome.enzymes);
        this.renderMolecules(detail);
        this.renderReactions(detail);
        this.renderLineageDetail(interaction);
        this.renderLineageList();
    }

    cellDetailPayload() {
        return this._detail.cellDetail && this._detail.cellDetail.cell_detail ? this._detail.cellDetail.cell_detail : null;
    }

    moleculePayload(detail) {
        if (this._detail.molecules && this._detail.molecules.internal) return this._detail.molecules.internal;
        return detail && detail.internal ? detail.internal : null;
    }

    reactionPayload(detail) {
        if (this._detail.reactions && this._detail.reactions.recent_reactions) return this._detail.reactions.recent_reactions;
        return detail && detail.recent_reactions ? detail.recent_reactions : null;
    }

    renderCellDetail(detail) {
        if (!this.elements.cellDetailInspector) return;
        if (this._detail.selectedCellId === null) { this.elements.cellDetailInspector.innerHTML = payloadMessage("Select a live cell to load Rust-authoritative detail."); return; }
        if (this._detail.cellError && !detail) { this.elements.cellDetailInspector.innerHTML = payloadMessage(`Selected cell detail unavailable: ${this._detail.cellError}`, "smallNote errorText"); return; }
        if (!detail) { this.elements.cellDetailInspector.innerHTML = payloadMessage("Loading selected cell detail..."); return; }
        const cell = detail.cell || {};
        this.elements.cellDetailInspector.innerHTML = detailRows([
            ["Cell id", cell.cell_id],
            ["State", detail.state],
            ["Tile", `${cell.x}, ${cell.y}`],
            ["Lineage", cell.lineage_id],
            ["Energy", cell.energy],
            ["Age", `${displayValue(cell.age_seconds)}s`],
            ["Optimal enval", cell.optimal_enval],
            ["Local enval average", cell.local_enval_average],
            ["Time without food", detail.time_without_food],
            ["Maintenance / sec", detail.maintenance_cost_per_sec],
            ["Death sim time", detail.death_sim_time ?? "—"],
            ["Enzyme count", cell.enzyme_count],
            ["Internal atoms", cell.internal_atom_count],
            ["Attack total", cell.combat_attack_total],
            ["Defense total", cell.combat_defense_total],
            ["Reproduction threshold", cell.repro_threshold],
            ["Decay time", cell.decay_time]
        ]);
    }

    renderGenome(genome) {
        if (!this.elements.genomeInspector) return;
        if (!genome) { this.elements.genomeInspector.innerHTML = payloadMessage("Genome detail is available after selecting a live cell."); return; }
        this.elements.genomeInspector.innerHTML = detailRows([
            ["Optimal enval", genome.optimal_enval],
            ["Lineage", genome.lineage_id],
            ["Enzyme count", genome.enzyme_count],
            ["Enzyme bounds", `${displayValue(genome.min_cell_enzymes)}-${displayValue(genome.max_cell_enzymes)}`],
            ["Mutation rate", genome.mutation_rate],
            ["Reproduction threshold", genome.repro_threshold],
            ["Initial energy", genome.initial_energy],
            ["Decay time", genome.decay_time],
            ["Default secretion", genome.default_secretion_prob],
            ["Desired reserve", genome.desired_element_reserve],
            ["Enval stress factor", genome.enval_stress_factor],
            ["Enval mutation floor", genome.enval_mutation_floor],
            ["Maintenance / sec", genome.maintenance_cost_per_sec],
            ["Post-divide mortality", genome.post_divide_mortality]
        ]);
    }

    renderEnzymes(enzymes) {
        if (!this.elements.enzymeInspector) return;
        this.elements.enzymeInspector.innerHTML = makeTable([
            { label: "#", value: (enzyme) => enzyme.index },
            { label: "Type", value: (enzyme) => enzyme.enzyme_type },
            { label: "Specificity", value: (enzyme) => Array.isArray(enzyme.specificity_elements) && enzyme.specificity_elements.length > 0 ? enzyme.specificity_elements.join("") : "—" },
            { label: "σ", value: (enzyme) => enzyme.enval_sigma },
            { label: "Throughput", value: (enzyme) => enzyme.enval_throughput },
            { label: "Secretion", value: (enzyme) => enzyme.secretion_prob },
            { label: "Bond", value: (enzyme) => enzyme.is_combat ? `level ${displayValue(enzyme.combat_level)}` : `×${displayValue(enzyme.bond_multiplier)} / cost ${displayValue(enzyme.bond_cost_fraction)} / harvest ${displayValue(enzyme.bond_harvest_fraction)}` }
        ], enzymes || [], "Select a live cell to inspect enzymes.");
    }

    renderMolecules(detail) {
        if (!this.elements.moleculeInspector) return;
        if (this._detail.moleculeError && !this.moleculePayload(detail)) { this.elements.moleculeInspector.innerHTML = payloadMessage(`Molecule detail unavailable: ${this._detail.moleculeError}`, "smallNote errorText"); return; }
        const internal = this.moleculePayload(detail);
        if (!internal) { this.elements.moleculeInspector.innerHTML = payloadMessage("Internal molecule detail is available after selecting a live cell."); return; }
        const shown = internal.molecules ? internal.molecules.length : 0;
        const summary = `
            <div class="detailSubsection">
                ${detailRows([
                    ["Molecule count", internal.molecule_count],
                    ["Returned", `${shown} / ${displayValue(internal.molecule_count)}`],
                    ["Atom count", internal.atom_count],
                    ["Element counts", elementCountsText(internal.element_counts)],
                    ["Limit", internal.limit],
                    ["Truncated", internal.truncated ? "yes" : "no"]
                ])}
                ${internal.truncated ? payloadMessage(`Showing ${shown} of ${displayValue(internal.molecule_count)} internal molecules.`, "smallNote warningText") : ""}
            </div>
        `;
        const table = makeTable([
            { label: "#", value: (molecule) => molecule.list_index },
            { label: "Formula", value: (molecule) => molecule.formula || compositionText(molecule.composition_counts) },
            { label: "Counts", value: (molecule) => compositionText(molecule.composition_counts) },
            { label: "Size", value: (molecule) => molecule.size },
            { label: "Mask", value: (molecule) => molecule.element_mask },
            { label: "Bond", value: (molecule) => molecule.bond_multiplier },
            { label: "Energy", value: (molecule) => molecule.energy },
            { label: "Polarity", value: (molecule) => molecule.polarity },
            { label: "Diffusion", value: (molecule) => `${displayValue(molecule.diffusion_rate)} / ${displayValue(molecule.diffusion_period)}` }
        ], internal.molecules || [], "No internal molecules returned.");
        this.elements.moleculeInspector.innerHTML = summary + table;
    }

    renderReactions(detail) {
        if (!this.elements.reactionInspector) return;
        if (this._detail.reactionError && !this.reactionPayload(detail)) { this.elements.reactionInspector.innerHTML = payloadMessage(`Reaction detail unavailable: ${this._detail.reactionError}`, "smallNote errorText"); return; }
        const reactions = this.reactionPayload(detail);
        if (!reactions) { this.elements.reactionInspector.innerHTML = payloadMessage("Reaction data is available after selecting a live cell."); return; }
        if (reactions.available === false) {
            this.elements.reactionInspector.innerHTML = `
                ${payloadMessage("Recent reaction logs are not recorded by the Rust core yet.", "smallNote warningText")}
                ${detailRows([
                    ["Reason", reactions.reason || "not_recorded"],
                    ["Limit", reactions.limit],
                    ["Truncated", reactions.truncated ? "yes" : "no"]
                ])}
            `;
            return;
        }
        this.elements.reactionInspector.innerHTML = makeTable([
            { label: "Tick", value: (record) => record.tick_count ?? record.tick ?? "—" },
            { label: "Enzyme", value: (record) => record.enzyme_type ?? record.enzyme_index ?? "—" },
            { label: "Status", value: (record) => record.status ?? record.reason ?? "—" },
            { label: "ΔE", value: (record) => record.delta_cell_energy ?? record.delta_energy ?? "—" },
            { label: "Note", value: (record) => record.note ?? "—" }
        ], reactions.reactions || reactions.records || [], "No recent reactions returned.");
    }

    renderLineageDetail(interaction) {
        if (!this.elements.lineageDetailInspector) return;
        if (!interaction || interaction.selectedLineageId == null) { this.elements.lineageDetailInspector.innerHTML = payloadMessage("Right-click a cell or select a lineage from the table."); return; }
        if (this._detail.lineageError && !this._detail.lineage) { this.elements.lineageDetailInspector.innerHTML = payloadMessage(`Lineage detail unavailable: ${this._detail.lineageError}`, "smallNote errorText"); return; }
        const lineage = this._detail.lineage && this._detail.lineage.lineage;
        if (!lineage) { this.elements.lineageDetailInspector.innerHTML = payloadMessage("Loading lineage detail..."); return; }
        this.elements.lineageDetailInspector.innerHTML = detailRows([
            ["Lineage", lineage.lineage_id],
            ["Population", lineage.population],
            ["Births", lineage.births],
            ["Deaths", lineage.deaths],
            ["Extinct", lineage.extinct ? "yes" : "no"],
            ["Live share", lineage.share],
            ["Avg energy", lineage.average_energy],
            ["Avg enzymes", lineage.average_enzyme_count],
            ["Avg attack", lineage.average_attack_total],
            ["Avg defense", lineage.average_defense_total],
            ["Max attack", lineage.max_attack_total],
            ["Max defense", lineage.max_defense_total],
            ["Cells with attackase", lineage.cells_with_attackase],
            ["Cells with defensase", lineage.cells_with_defensase]
        ]);
    }

    renderLineageList() {
        if (!this.elements.lineageList) return;
        if (this._detail.lineageListError && !this._detail.lineageList) { this.elements.lineageList.innerHTML = payloadMessage(`Lineage list unavailable: ${this._detail.lineageListError}`, "smallNote errorText"); return; }
        const list = this._detail.lineageList && this._detail.lineageList.lineage_list;
        if (!list) { this.elements.lineageList.innerHTML = payloadMessage("Lineage list has not loaded yet."); return; }
        const header = `
            <div class="detailSubsection">
                ${detailRows([
                    ["Extant lineages", list.extant_lineage_count],
                    ["Total records", list.total_lineage_records],
                    ["Limit", list.limit],
                    ["Truncated", list.truncated ? "yes" : "no"]
                ])}
            </div>
        `;
        const table = makeTable([
            { label: "Lineage", value: (lineage) => lineage.lineage_id },
            { label: "Population", value: (lineage) => lineage.population },
            { label: "Births", value: (lineage) => lineage.births },
            { label: "Deaths", value: (lineage) => lineage.deaths },
            { label: "Share", value: (lineage) => lineage.share },
            { label: "Avg E", value: (lineage) => lineage.average_energy },
            { label: "Avg enz", value: (lineage) => lineage.average_enzyme_count },
            { label: "Action", value: (lineage) => htmlValue(buttonForLineage(lineage.lineage_id)) }
        ], list.lineages || [], "No extant lineages returned.");
        this.elements.lineageList.innerHTML = header + table;
    }

    updateViewTable(runtime) {
        if (!this.elements.views) return;
        const rows = runtime.viewDiagnostics().map((view) => `
            <tr class="${view.ok ? "" : "isBad"}">
                <td>${escapeHtml(view.key)}</td>
                <td>${escapeHtml(view.dtype)}</td>
                <td>${escapeHtml(displayValue(view.ptr))}</td>
                <td>${escapeHtml(displayValue(view.length))}</td>
                <td>${escapeHtml(displayValue(view.expectedLength))}</td>
                <td>${view.ok ? "ok" : "mismatch"}</td>
            </tr>
        `).join("");
        this.elements.views.innerHTML = `
            <table>
                <thead><tr><th>View</th><th>dtype</th><th>ptr</th><th>length</th><th>expected</th><th>status</th></tr></thead>
                <tbody>${rows}</tbody>
            </table>
        `;
    }

    updateSamples(runtime) {
        if (!this.elements.samples) return;
        this.elements.samples.innerHTML = `
            <div><strong>tileEnval[0..]</strong>: ${escapeHtml(formatArraySample(runtime.views.tileEnval && runtime.views.tileEnval.array()))}</div>
            <div><strong>tileOccupancy[0..]</strong>: ${escapeHtml(formatArraySample(runtime.views.tileOccupancy && runtime.views.tileOccupancy.array()))}</div>
            <div><strong>tileElementMask[0..]</strong>: ${escapeHtml(formatArraySample(runtime.views.tileElementMask && runtime.views.tileElementMask.array()))}</div>
            <div><strong>cellId[0..]</strong>: ${escapeHtml(formatArraySample(runtime.views.cellId && runtime.views.cellId.array()))}</div>
            <div><strong>cellX[0..]</strong>: ${escapeHtml(formatArraySample(runtime.views.cellX && runtime.views.cellX.array()))}</div>
            <div><strong>cellY[0..]</strong>: ${escapeHtml(formatArraySample(runtime.views.cellY && runtime.views.cellY.array()))}</div>
            <div><strong>cellEnergy[0..]</strong>: ${escapeHtml(formatArraySample(runtime.views.cellEnergy && runtime.views.cellEnergy.array()))}</div>
            <div><strong>cellLineage[0..]</strong>: ${escapeHtml(formatArraySample(runtime.views.cellLineage && runtime.views.cellLineage.array()))}</div>
        `;
    }

    payloadFor(kind) {
        switch (kind) {
            case "cell": return this._detail.cellDetail;
            case "tile": return this._lastInspectionPayload && this._lastInspectionPayload.selectedTile;
            case "lineage": return this._detail.lineage;
            case "molecules": return this._detail.molecules;
            case "reactions": return this._detail.reactions;
            default: return this._lastInspectionPayload;
        }
    }

    async copyPayload(kind, button = null) {
        const payload = this.payloadFor(kind);
        if (!payload) { this.setError(new Error(`No ${kind} JSON is loaded yet.`)); return; }
        await this.copyObject(payload, button);
    }

    async copyInspection() {
        if (!this._lastInspectionPayload) return;
        await this.copyObject(this._lastInspectionPayload, this.elements.copyInspectionButton);
    }

    async copyObject(payload, button) {
        const text = JSON.stringify(payload, null, 2);
        try {
            await navigator.clipboard.writeText(text);
            if (button) {
                const prev = button.textContent;
                button.textContent = "Copied";
                setTimeout(() => { if (button.isConnected) button.textContent = prev; }, 900);
            }
        } catch (error) { this.setError(error); }
    }
}
