import { MicrocosmRuntime } from "./wasmgpu/runtime.js";
import { MicrocosmRenderer } from "./wasmgpu/render.js";
import { MicrocosmGUI } from "./wasmgpu/gui.js";
import { MicrocosmInteraction } from "./wasmgpu/interact.js";

const SEED_QUERY_PARAM = "seed";
const STEP_MS = 10;
const MAX_STEPS_PER_FRAME = 6;
const HUD_UPDATE_INTERVAL_MS = 200;

const gui = MicrocosmGUI.bind();

let runtime = null;
let renderer = null;
let interaction = null;
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
let fallbackCanvasExpanded = false;
let startupInfoMathRendered = false;
const startupInfoUI = { modal: null, closeButton: null, openButton: null };

const resolveWasmGPU = () => { const candidate = window.WasmGPU && (window.WasmGPU.default || window.WasmGPU); if (!candidate || !candidate.webassembly || typeof candidate.create !== "function") throw new Error("WasmGPU was not loaded or does not expose the expected API."); return candidate; };

const normalizeSeedValue = (value) => { const text = `${value ?? ""}`.trim(); return text !== "" ? text : "42"; };

const initialSeed = () => normalizeSeedValue(new URL(window.location.href).searchParams.get(SEED_QUERY_PARAM));

const seedUrl = (seed) => { const url = new URL(window.location.href); url.searchParams.set(SEED_QUERY_PARAM, seed); return url; };

const buildConfig = () => ({ seed: normalizeSeedValue(gui.seed || initialSeed()), width: 320, height: 240, initial_cells: 32, predation_enabled: true });

const updateGui = () => gui.update({ runtime, renderer, interaction, paused, fps: fpsValue, tps: tpsValue });

const canvasFullscreenActive = () => Boolean(gui.visualPanel && document.fullscreenElement === gui.visualPanel);

const canvasExpandedActive = () => canvasFullscreenActive() || fallbackCanvasExpanded;

const refreshCanvasViewport = (options = {}) => {
    if (!renderer) return;
    requestAnimationFrame(() => {
        try {
            if (options.fit !== false) renderer.fitView({ saveState: true });
            else renderer.resize(true);
            renderer.render();
            updateGui();
        } catch (error) { gui.setError(error); }
    });
};

const setFallbackCanvasExpanded = (active) => {
    fallbackCanvasExpanded = !!active;
    if (gui.visualPanel) gui.visualPanel.classList.toggle("isExpanded", fallbackCanvasExpanded);
    document.body.classList.toggle("isCanvasExpanded", fallbackCanvasExpanded);
    gui.setFullscreen(canvasExpandedActive());
    refreshCanvasViewport({ fit: true });
};

const handleFullscreenChange = () => {
    if (fallbackCanvasExpanded && canvasFullscreenActive()) setFallbackCanvasExpanded(false);
    gui.setFullscreen(canvasExpandedActive());
    refreshCanvasViewport({ fit: true });
};

const toggleCanvasFullscreen = async () => {
    if (!gui.visualPanel) return;
    gui.clearError();
    if (canvasFullscreenActive()) { if (document.exitFullscreen) await document.exitFullscreen(); return; }
    if (fallbackCanvasExpanded) { setFallbackCanvasExpanded(false); return; }
    if (gui.visualPanel.requestFullscreen) { try { await gui.visualPanel.requestFullscreen(); return; } catch (error) { setFallbackCanvasExpanded(true); return; } }
    setFallbackCanvasExpanded(true);
};

const refreshAfterGenomeEdit = (result) => {
    gui.setGenomeEditResult(result);
    gui.forceRefreshDetails();
    if (interaction) interaction.refreshSelection();
    if (renderer && runtime && runtime.ready) {
        renderer.updateFromRuntime(runtime);
        renderer.render();
    }
    updateGui();
};

const applySelectedGenomePatch = (patch) => {
    if (!runtime || !runtime.ready || !interaction) throw new Error("Genome patch requires a ready runtime and selected cell.");
    if (interaction.selectedCellId == null) throw new Error("Select a live cell before applying a genome patch.");
    gui.clearError();
    const result = runtime.applyCellGenomePatch(interaction.selectedCellId, patch);
    refreshAfterGenomeEdit(result);
};

const applyGenomeBrushAtSelection = (patch) => {
    if (!runtime || !runtime.ready || !interaction) throw new Error("Genome brush requires a ready runtime and selected tile.");
    const tile = interaction.selectedTile || interaction.hoverTile;
    if (!tile) throw new Error("Select or hover a tile before applying a genome brush from the GUI.");
    gui.clearError();
    const brush = interaction.brush || gui.brushOptions;
    const result = runtime.applyGenomeBrush(tile.x, tile.y, brush.width, brush.height, patch);
    refreshAfterGenomeEdit(result);
};

const updateInteractionVisuals = () => {
    if (!runtime || !runtime.ready || !renderer) return;
    renderer.updateFromRuntime(runtime);
    renderer.render();
    updateGui();
};

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
    gui.forceRefreshDetails();
    gui.clearGenomeEditResult();
    const config = buildConfig();
    runtime.reset(config);
    window.history.replaceState(null, "", seedUrl(config.seed).toString());
    if (interaction) interaction.clearSelection();
    renderer.updateFromRuntime(runtime);
    renderer.fitView({ saveState: true });
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

const renderStartupInfoMath = () => {
    if (!startupInfoUI.modal || startupInfoMathRendered) return;
    if (typeof window.renderMathInElement !== "function") return;
    window.renderMathInElement(startupInfoUI.modal, { delimiters: [{ left: "$$", right: "$$", display: true }, { left: "\\(", right: "\\)", display: false }], throwOnError: false });
    startupInfoMathRendered = true;
};

const openStartupInfoModal = () => {
    if (!startupInfoUI.modal) return;
    startupInfoUI.modal.hidden = false;
    startupInfoUI.modal.setAttribute("aria-hidden", "false");
    renderStartupInfoMath();
};

const closeStartupInfoModal = () => {
    if (!startupInfoUI.modal) return;
    startupInfoUI.modal.hidden = true;
    startupInfoUI.modal.setAttribute("aria-hidden", "true");
};

startupInfoUI.modal = document.getElementById("startupInfoModal");
startupInfoUI.closeButton = document.getElementById("startupInfoClose");
startupInfoUI.openButton = document.getElementById("infoButton");
if (startupInfoUI.modal) {
    startupInfoUI.modal.hidden = true; startupInfoUI.modal.setAttribute("aria-hidden", "true"); openStartupInfoModal();
    if (typeof window.renderMathInElement !== "function") window.addEventListener("load", () => renderStartupInfoMath(), { once: true });
    startupInfoUI.modal.addEventListener("click", (event) => { if (event.target === startupInfoUI.modal) closeStartupInfoModal(); });
}
if (startupInfoUI.closeButton) startupInfoUI.closeButton.addEventListener("click", () => closeStartupInfoModal());
if (startupInfoUI.openButton) startupInfoUI.openButton.addEventListener("click", () => openStartupInfoModal());

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
        renderer.render(frameElapsedMs / 1000);
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
        if (interaction) interaction.refreshSelection();
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
        interaction = MicrocosmInteraction.create({
            runtime,
            renderer,
            canvas: gui.canvas,
            mode: gui.mode,
            brush: gui.brushOptions,
            genomePatch: () => gui.genomeBrushPatchDraft,
            onError: (error) => gui.setError(error),
            onChange: () => updateInteractionVisuals(),
            onMutation: (event) => {
                if (event && event.type === "genome-brush") gui.setGenomeEditResult(event.result);
                if (event && event.type === "genome-brush") gui.forceRefreshDetails();
                updateInteractionVisuals();
            }
        });
        gui.setBrushOptions(interaction.brush);
        renderer.updateFromRuntime(runtime);
        renderer.fitView({ saveState: true });
        renderer.render();
        gui.setStatus("Running Rust/WasmGPU Microcosm beta implementation");
        gui.setPaused(paused);
        updateGui();
        lastWallTimeMs = performance.now();
        animationFrameId = requestAnimationFrame(frame);
    } catch (error) {
        if (interaction) { interaction.destroy(); interaction = null; }
        if (runtime) { runtime.destroy(); runtime = null; }
        if (renderer) { renderer.destroy(); renderer = null; }
        gui.setError(error);
    }
};

gui.bindControls({
    pause: () => setPaused(!paused),
    step: () => { try { stepOnce(); } catch (error) { gui.setError(error); } },
    reset: () => { try { resetRuntime(); } catch (error) { gui.setError(error); } },
    info: () => { try { openStartupInfoModal(); } catch (error) { gui.setError(error); } },
    displayMode: (mode) => { try { applyDisplayMode(mode); } catch (error) { gui.setError(error); } },
    mode: (mode) => { try { if (interaction) interaction.setMode(mode); } catch (error) { gui.setError(error); } },
    brush: (options) => { try { if (interaction) interaction.setBrushOptions(options); } catch (error) { gui.setError(error); } },
    brushPreview: (options) => { try { if (interaction) interaction.setBrushOptions(options); } catch (error) { gui.setError(error); } },
    clearSelection: () => { try { if (interaction) interaction.clearSelection(); } catch (error) { gui.setError(error); } },
    clearLineage: () => { try { if (interaction) interaction.clearLineageSelection(); } catch (error) { gui.setError(error); } },
    refreshDetails: () => { try { updateGui(); } catch (error) { gui.setError(error); } },
    fitView: () => { try { if (renderer) { renderer.fitView({ saveState: true }); renderer.render(); updateGui(); } } catch (error) { gui.setError(error); } },
    fullscreen: () => { toggleCanvasFullscreen().catch((error) => gui.setError(error)); },
    applyGenomePatch: (patch) => { try { applySelectedGenomePatch(patch); } catch (error) { gui.setError(error); } },
    applyGenomeBrush: (patch) => { try { applyGenomeBrushAtSelection(patch); } catch (error) { gui.setError(error); } },
    selectLineage: (lineageId) => { try { if (interaction) interaction.selectLineage(lineageId); } catch (error) { gui.setError(error); } }
});

document.addEventListener("fullscreenchange", handleFullscreenChange);

window.addEventListener("beforeunload", () => {
    document.removeEventListener("fullscreenchange", handleFullscreenChange);
    if (animationFrameId !== null) cancelAnimationFrame(animationFrameId);
    if (interaction) interaction.destroy();
    if (renderer) renderer.destroy();
    if (runtime) runtime.destroy();
});

init();
