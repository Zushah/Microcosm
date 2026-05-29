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

const displayValue = (value) => {
    if (typeof value === "bigint") return value.toString();
    if (typeof value === "number") {
        if (!Number.isFinite(value)) return String(value);
        if (Number.isInteger(value)) return String(value);
        return value.toFixed(4);
    }
    return String(value ?? "—");
};

const formatArraySample = (array, count = 8) => {
    if (!array) return "—";
    const size = Math.min(count, array.length);
    const out = [];
    for (let i = 0; i < size; i++) out.push(displayValue(array[i]));
    return out.length > 0 ? out.join(", ") : "—";
};

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
            canvas: root.getElementById("wasmgpuCanvas")
        });
    }

    constructor(elements) {
        this.elements = elements;
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
        if (this.elements.seedInput) {
            this.elements.seedInput.addEventListener("keydown", (event) => {
                if (event.key !== "Enter") return;
                event.preventDefault();
                if (handlers.reset) handlers.reset();
            });
        }
    }

    setPaused(paused) {
        if (this.elements.pauseButton) this.elements.pauseButton.textContent = paused ? "Resume" : "Pause";
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
                <span>${label}</span>
                <strong>${displayValue(value)}</strong>
            </div>
        `).join("");
        if (this.elements.renderStats) {
            const rows = [
                ["Strategy", renderStats.strategy || "—"],
                ["Display mode", renderStats.displayMode || "—"],
                ["Tiles rendered", renderStats.tileCountRendered],
                ["Cells rendered", renderStats.cellCountRendered],
                ["Renderer frames", renderStats.frameCount],
                ["Paused", state.paused ? "yes" : "no"]
            ];
            this.elements.renderStats.innerHTML = rows.map(([label, value]) => `
                <div class="datum">
                    <span>${label}</span>
                    <strong>${displayValue(value)}</strong>
                </div>
            `).join("");
        }
        this.updateViewTable(runtime);
        this.updateSamples(runtime);
    }

    updateViewTable(runtime) {
        if (!this.elements.views) return;
        const rows = runtime.viewDiagnostics().map((view) => `
            <tr class="${view.ok ? "" : "isBad"}">
                <td>${view.key}</td>
                <td>${view.dtype}</td>
                <td>${view.ptr}</td>
                <td>${view.length}</td>
                <td>${view.expectedLength}</td>
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
            <div><strong>tileEnval[0..]</strong>: ${formatArraySample(runtime.views.tileEnval && runtime.views.tileEnval.array())}</div>
            <div><strong>tileOccupancy[0..]</strong>: ${formatArraySample(runtime.views.tileOccupancy && runtime.views.tileOccupancy.array())}</div>
            <div><strong>tileElementMask[0..]</strong>: ${formatArraySample(runtime.views.tileElementMask && runtime.views.tileElementMask.array())}</div>
            <div><strong>cellId[0..]</strong>: ${formatArraySample(runtime.views.cellId && runtime.views.cellId.array())}</div>
            <div><strong>cellX[0..]</strong>: ${formatArraySample(runtime.views.cellX && runtime.views.cellX.array())}</div>
            <div><strong>cellY[0..]</strong>: ${formatArraySample(runtime.views.cellY && runtime.views.cellY.array())}</div>
            <div><strong>cellEnergy[0..]</strong>: ${formatArraySample(runtime.views.cellEnergy && runtime.views.cellEnergy.array())}</div>
            <div><strong>cellLineage[0..]</strong>: ${formatArraySample(runtime.views.cellLineage && runtime.views.cellLineage.array())}</div>
        `;
    }
}
