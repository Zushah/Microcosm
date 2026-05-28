import { MicrocosmRuntime } from "./wasmgpu/runtime.js";

const SEED_QUERY_PARAM = "seed";
const STEP_MS = 10;
const MAX_STEPS_PER_FRAME = 6;
const HUD_UPDATE_INTERVAL_MS = 200;

const ui = {
    status: document.getElementById("status"),
    diagnostics: document.getElementById("diagnostics"),
    views: document.getElementById("views"),
    samples: document.getElementById("samples"),
    errors: document.getElementById("errors"),
    pauseButton: document.getElementById("pauseButton"),
    stepButton: document.getElementById("stepButton"),
    resetButton: document.getElementById("resetButton"),
    seedInput: document.getElementById("seedInput")
};

let runtime = null;
let paused = false;
let lastWallTimeMs = performance.now();
let accumulatorMs = 0;
let lastHudUpdateMs = 0;
let fpsFrames = 0;
let fpsLastSampleMs = performance.now();
let fpsValue = 0;
let tpsTicks = 0;
let tpsLastSampleMs = performance.now();
let tpsValue = 0;

const setStatus = (text) => { if (ui.status) ui.status.textContent = text; };

const setError = (error) => { const message = error && error.stack ? error.stack : String(error || "Unknown error"); console.error(error); if (ui.errors) ui.errors.textContent = message; setStatus("Error"); };

const clearError = () => { if (ui.errors) ui.errors.textContent = ""; };

const displayValue = (value) => {
    if (typeof value === "bigint") return value.toString();
    if (typeof value === "number") {
        if (!Number.isFinite(value)) return String(value);
        if (Number.isInteger(value)) return String(value);
        return value.toFixed(4);
    }
    return String(value ?? "-");
};

const formatArraySample = (array, count = 8) => {
    if (!array) return "—";
    const size = Math.min(count, array.length);
    const out = [];
    for (let i = 0; i < size; i++) out.push(displayValue(array[i]));
    return out.join(", ");
};

const resolveWasmGPU = () => { const candidate = window.WasmGPU && (window.WasmGPU.default || window.WasmGPU); if (!candidate || !candidate.webassembly) throw new Error("WasmGPU was not loaded."); return candidate; };

const normalizeSeedValue = (value) => { const text = `${value ?? ""}`.trim(); return text !== "" ? text : "demo"; };

const initialSeed = () => normalizeSeedValue(new URL(window.location.href).searchParams.get(SEED_QUERY_PARAM));

const seedUrl = (seed) => { const url = new URL(window.location.href); url.searchParams.set(SEED_QUERY_PARAM, seed); return url; };

const buildConfig = () => ({
    seed: normalizeSeedValue(ui.seedInput ? ui.seedInput.value : initialSeed()),
    width: 320,
    height: 240,
    initial_cells: 32,
    predation_enabled: true
});

const updatePauseButton = () => { if (ui.pauseButton) ui.pauseButton.textContent = paused ? "Resume" : "Pause"; };

const renderDiagnostics = () => {
    if (!runtime || !runtime.ready) return;
    const stats = runtime.stats || {};
    const diagnostics = [
        ["Runtime", "ready"],
        ["WebAssembly URL", runtime.bridge.wasmUrl instanceof URL ? runtime.bridge.wasmUrl.href : String(runtime.bridge.wasmUrl)],
        ["Version", runtime.version],
        ["ABI version", runtime.abiVersion],
        ["Handle", runtime.handle],
        ["Tick", stats.tick_count],
        ["Sim time", stats.sim_time_seconds],
        ["FPS", fpsValue],
        ["TPS", tpsValue],
        ["World", `${stats.width} × ${stats.height}`],
        ["Tile count", stats.tile_count],
        ["Occupied tiles", stats.occupied_tile_count],
        ["Live cells", stats.live_cell_count],
        ["Molecules", stats.molecule_count],
        ["Births", stats.births],
        ["Deaths", stats.deaths],
        ["Avg cell energy", stats.average_cell_energy],
        ["Avg enval", stats.average_enval],
        ["Reaction attempts", stats.reaction_attempts],
        ["Reaction successes", stats.reaction_successes],
        ["Render epoch", runtime.renderEpoch]
    ];
    if (ui.diagnostics) {
        ui.diagnostics.innerHTML = diagnostics.map(([label, value]) => `
            <div class="datum">
                <span>${label}</span>
                <strong>${displayValue(value)}</strong>
            </div>
        `).join("");
    }
    if (ui.views) {
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
        ui.views.innerHTML = `
            <table>
                <thead><tr><th>View</th><th>dtype</th><th>ptr</th><th>length</th><th>expected</th><th>status</th></tr></thead>
                <tbody>${rows}</tbody>
            </table>
        `;
    }
    if (ui.samples) ui.samples.innerHTML = `
        <div><strong>tileEnval[0..]</strong>: ${formatArraySample(runtime.views.tileEnval && runtime.views.tileEnval.array())}</div>
        <div><strong>tileOccupancy[0..]</strong>: ${formatArraySample(runtime.views.tileOccupancy && runtime.views.tileOccupancy.array())}</div>
        <div><strong>cellId[0..]</strong>: ${formatArraySample(runtime.views.cellId && runtime.views.cellId.array())}</div>
        <div><strong>cellX[0..]</strong>: ${formatArraySample(runtime.views.cellX && runtime.views.cellX.array())}</div>
        <div><strong>cellY[0..]</strong>: ${formatArraySample(runtime.views.cellY && runtime.views.cellY.array())}</div>
        <div><strong>cellEnergy[0..]</strong>: ${formatArraySample(runtime.views.cellEnergy && runtime.views.cellEnergy.array())}</div>
        <div><strong>cellLineage[0..]</strong>: ${formatArraySample(runtime.views.cellLineage && runtime.views.cellLineage.array())}</div>
    `;
};

const resetRuntime = () => {
    if (!runtime) return;
    clearError();
    const config = buildConfig();
    runtime.reset(config);
    window.history.replaceState(null, "", seedUrl(config.seed).toString());
    renderDiagnostics();
};

const stepOnce = () => {
    if (!runtime) return;
    clearError();
    runtime.step(1);
    tpsTicks++;
    renderDiagnostics();
};

const frame = () => {
    if (!runtime || !runtime.ready) { requestAnimationFrame(frame); return; }
    const nowMs = performance.now();
    let frameElapsedMs = nowMs - lastWallTimeMs;
    lastWallTimeMs = nowMs;
    if (frameElapsedMs > 200) frameElapsedMs = 200;
    if (!paused) {
        accumulatorMs += frameElapsedMs;
        const maxAccumulatedMs = STEP_MS * MAX_STEPS_PER_FRAME;
        if (accumulatorMs > maxAccumulatedMs) accumulatorMs = maxAccumulatedMs;
        const steps = Math.min(MAX_STEPS_PER_FRAME, Math.floor(accumulatorMs / STEP_MS));
        if (steps > 0) {
            try { runtime.step(steps); tpsTicks += steps; accumulatorMs -= steps * STEP_MS; }
            catch (error) { paused = true; updatePauseButton(); setError(error); }
        }
    } else accumulatorMs = 0;
    fpsFrames++;
    if (nowMs - fpsLastSampleMs >= 500) { fpsValue = (fpsFrames * 1000) / (nowMs - fpsLastSampleMs); fpsFrames = 0; fpsLastSampleMs = nowMs; }
    if (nowMs - tpsLastSampleMs >= 500) { tpsValue = (tpsTicks * 1000) / (nowMs - tpsLastSampleMs); tpsTicks = 0; tpsLastSampleMs = nowMs; }
    if (nowMs - lastHudUpdateMs >= HUD_UPDATE_INTERVAL_MS) { renderDiagnostics(); lastHudUpdateMs = nowMs; }
    requestAnimationFrame(frame);
};

const init = async () => {
    try {
        setStatus("Loading WasmGPU and Microcosm WebAssembly...");
        const seed = initialSeed();
        if (ui.seedInput) ui.seedInput.value = seed;
        window.history.replaceState(null, "", seedUrl(seed).toString());
        runtime = await MicrocosmRuntime.create({
            WasmGPU: resolveWasmGPU(),
            wasmUrl: new URL("./rust/microcosm.wasm", import.meta.url),
            config: buildConfig()
        });
        setStatus("Running Rust/WasmGPU Microcosm runtime");
        updatePauseButton();
        renderDiagnostics();
        requestAnimationFrame(frame);
    } catch (error) { setError(error); }
};

if (ui.pauseButton) ui.pauseButton.addEventListener("click", () => { paused = !paused; updatePauseButton(); renderDiagnostics(); });

if (ui.stepButton) ui.stepButton.addEventListener("click", () => { try { paused = true; updatePauseButton(); stepOnce(); } catch (error) { setError(error); } });

if (ui.resetButton) ui.resetButton.addEventListener("click", () => { try { resetRuntime(); } catch (error) { setError(error); } });

if (ui.seedInput) ui.seedInput.addEventListener("keydown", (event) => { if (event.key !== "Enter") return; event.preventDefault(); try { resetRuntime(); } catch (error) { setError(error); } });

window.addEventListener("beforeunload", () => { if (runtime) runtime.destroy(); });

init();
