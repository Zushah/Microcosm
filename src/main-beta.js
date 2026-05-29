import { MicrocosmRuntime } from "./wasmgpu/runtime.js";
import { MicrocosmRenderer } from "./wasmgpu/render.js";
import { MicrocosmGUI } from "./wasmgpu/gui.js";

const SEED_QUERY_PARAM = "seed";
const STEP_MS = 10;
const MAX_STEPS_PER_FRAME = 6;
const HUD_UPDATE_INTERVAL_MS = 200;

const gui = MicrocosmGUI.bind();

let runtime = null;
let renderer = null;
let paused = false;
let animationFrameId = null;
let lastWallTimeMs = performance.now();
let accumulatorMs = 0;
let lastHudUpdateMs = 0;
let fpsFrames = 0;
let fpsLastSampleMs = performance.now();
let fpsValue = 0;
let tpsTicks = 0;
let tpsLastSampleMs = performance.now();
let tpsValue = 0;

const resolveWasmGPU = () => {
    const candidate = window.WasmGPU && (window.WasmGPU.default || window.WasmGPU);
    if (!candidate || !candidate.webassembly || typeof candidate.create !== "function") throw new Error("WasmGPU was not loaded or does not expose the expected API.");
    return candidate;
};

const normalizeSeedValue = (value) => {
    const text = `${value ?? ""}`.trim();
    return text !== "" ? text : "42";
};

const initialSeed = () => normalizeSeedValue(new URL(window.location.href).searchParams.get(SEED_QUERY_PARAM));

const seedUrl = (seed) => {
    const url = new URL(window.location.href);
    url.searchParams.set(SEED_QUERY_PARAM, seed);
    return url;
};

const buildConfig = () => ({
    seed: normalizeSeedValue(gui.seed || initialSeed()),
    width: 320,
    height: 240,
    initial_cells: 32,
    predation_enabled: true
});

const updateGui = () => gui.update({ runtime, renderer, paused, fps: fpsValue, tps: tpsValue });

const renderOnce = () => {
    if (!runtime || !runtime.ready || !renderer) return;
    renderer.updateFromRuntime(runtime);
    renderer.render();
    updateGui();
};

const stepRuntime = (steps) => {
    if (!runtime || !runtime.ready) return;
    runtime.step(steps, { refresh: false });
    runtime.refreshRenderBuffers();
    tpsTicks += steps;
};

const resetRuntime = () => {
    if (!runtime || !renderer) return;
    gui.clearError();
    const config = buildConfig();
    runtime.reset(config);
    window.history.replaceState(null, "", seedUrl(config.seed).toString());
    renderer.updateFromRuntime(runtime);
    renderer.render();
    updateGui();
};

const stepOnce = () => {
    if (!runtime || !renderer) return;
    gui.clearError();
    paused = true;
    renderer.setPausedVisualState(paused);
    gui.setPaused(paused);
    stepRuntime(1);
    renderer.updateFromRuntime(runtime);
    renderer.render();
    updateGui();
};

const setPaused = (value) => {
    paused = !!value;
    if (renderer) renderer.setPausedVisualState(paused);
    gui.setPaused(paused);
    updateGui();
};

const applyDisplayMode = (mode) => {
    if (!renderer) return;
    gui.clearError();
    renderer.setDisplayMode(mode);
    if (runtime && runtime.ready) renderOnce();
};

const frame = () => {
    animationFrameId = requestAnimationFrame(frame);
    if (!runtime || !runtime.ready || !renderer) return;
    const nowMs = performance.now();
    let frameElapsedMs = nowMs - lastWallTimeMs;
    lastWallTimeMs = nowMs;
    if (frameElapsedMs > 200) frameElapsedMs = 200;
    try {
        if (!paused) {
            accumulatorMs += frameElapsedMs;
            const maxAccumulatedMs = STEP_MS * MAX_STEPS_PER_FRAME;
            if (accumulatorMs > maxAccumulatedMs) accumulatorMs = maxAccumulatedMs;
            const steps = Math.min(MAX_STEPS_PER_FRAME, Math.floor(accumulatorMs / STEP_MS));
            if (steps > 0) {
                stepRuntime(steps);
                accumulatorMs -= steps * STEP_MS;
                renderer.updateFromRuntime(runtime);
            }
        } else accumulatorMs = 0;
        renderer.render();
    } catch (error) {
        setPaused(true);
        gui.setError(error);
    }
    fpsFrames++;
    if (nowMs - fpsLastSampleMs >= 500) {
        fpsValue = (fpsFrames * 1000) / (nowMs - fpsLastSampleMs);
        fpsFrames = 0;
        fpsLastSampleMs = nowMs;
    }
    if (nowMs - tpsLastSampleMs >= 500) {
        tpsValue = (tpsTicks * 1000) / (nowMs - tpsLastSampleMs);
        tpsTicks = 0;
        tpsLastSampleMs = nowMs;
    }
    if (nowMs - lastHudUpdateMs >= HUD_UPDATE_INTERVAL_MS) {
        updateGui();
        lastHudUpdateMs = nowMs;
    }
};

const init = async () => {
    try {
        gui.setStatus("Loading WasmGPU renderer and Microcosm WebAssembly...");
        const seed = initialSeed();
        gui.seed = seed;
        window.history.replaceState(null, "", seedUrl(seed).toString());
        const WasmGPU = resolveWasmGPU();
        renderer = await MicrocosmRenderer.create({
            WasmGPU,
            canvas: gui.canvas,
            displayMode: gui.displayMode
        });
        runtime = await MicrocosmRuntime.create({
            WasmGPU,
            wasmUrl: new URL("./rust/microcosm.wasm", import.meta.url),
            config: buildConfig()
        });
        renderer.updateFromRuntime(runtime);
        renderer.render();
        gui.setStatus("Running Rust/WasmGPU Microcosm beta implementation");
        gui.setPaused(paused);
        updateGui();
        lastWallTimeMs = performance.now();
        animationFrameId = requestAnimationFrame(frame);
    } catch (error) {
        if (runtime) { runtime.destroy(); runtime = null; }
        if (renderer) { renderer.destroy(); renderer = null; }
        gui.setError(error);
    }
};

gui.bindControls({
    pause: () => setPaused(!paused),
    step: () => { try { stepOnce(); } catch (error) { gui.setError(error); } },
    reset: () => { try { resetRuntime(); } catch (error) { gui.setError(error); } },
    displayMode: (mode) => { try { applyDisplayMode(mode); } catch (error) { gui.setError(error); } }
});

window.addEventListener("beforeunload", () => {
    if (animationFrameId !== null) cancelAnimationFrame(animationFrameId);
    if (renderer) renderer.destroy();
    if (runtime) runtime.destroy();
});

init();
