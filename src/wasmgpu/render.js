const TILE_DISPLAY_MODES = Object.freeze(["enval", "occupancy", "mass", "molecules", "element-a", "element-b", "element-c", "element-d", "element-e", "element-f"]);

const normalizeWasmGPU = (WasmGPU) => {
    const candidate = WasmGPU && (WasmGPU.default || WasmGPU);
    if (!candidate || typeof candidate.create !== "function") throw new Error("MicrocosmRenderer requires a WasmGPU object with WasmGPU.create(...).");
    return candidate;
};

const sameTile = (a, b) => a === b || Boolean(a && b && a.x === b.x && a.y === b.y);

const sameBrush = (a, b) => a === b || Boolean(a && b && a.x === b.x && a.y === b.y && a.width === b.width && a.height === b.height);

export const MICROCOSM_DISPLAY_MODES = TILE_DISPLAY_MODES;

export class MicrocosmRenderer {
    static async create(options = {}) {
        const WasmGPU = normalizeWasmGPU(options.WasmGPU || globalThis.WasmGPU);
        const canvas = options.canvas || document.getElementById("wasmgpuCanvas");
        if (!(canvas instanceof HTMLCanvasElement)) throw new Error("MicrocosmRenderer requires a canvas element.");
        const wgpu = await WasmGPU.create(canvas, {
            antialias: false,
            powerPreference: "high-performance",
            frustumCulling: false,
            occlusionCulling: false,
            ...(options.descriptor || {})
        });
        if (typeof wgpu.createLatticeSpace !== "function") throw new Error("MicrocosmRenderer requires WasmGPU v0.9.0 or newer with createLatticeSpace(...).");
        const renderer = new MicrocosmRenderer(wgpu, canvas, options);
        renderer.initializeScene();
        return renderer;
    }

    constructor(wgpu, canvas, options = {}) {
        this._wgpu = wgpu;
        this._canvas = canvas;
        this._scene = null;
        this._camera = null;
        this._controls = null;
        this._lattice = null;
        this._latticeDataView = null;
        this._displayMode = options.displayMode || "enval";
        this._selectedLineage = null;
        this._selectedTile = null;
        this._selectedCellId = null;
        this._hoverTile = null;
        this._brushPreview = null;
        this._pausedVisualState = false;
        this._visualStateDirty = true;
        this._width = 0;
        this._height = 0;
        this._tileCount = 0;
        this._cellCount = 0;
        this._lastCanvasWidth = 0;
        this._lastCanvasHeight = 0;
        this._viewInitialized = false;
        this._frameCount = 0;
        this._destroyed = false;
    }

    get wgpu() { return this._wgpu; }
    get scene() { return this._scene; }
    get camera() { return this._camera; }
    get canvas() { return this._canvas; }
    get displayMode() { return this._displayMode; }
    get width() { return this._width; }
    get height() { return this._height; }
    get tileCountRendered() { return this._tileCount; }
    get cellCountRendered() { return this._cellCount; }
    get frameCount() { return this._frameCount; }

    get diagnostics() {
        return {
            strategy: "Wasm-backed composited LatticeSpace tile and cell field",
            displayMode: this._displayMode,
            tileCountRendered: this._tileCount,
            cellCountRendered: this._cellCount,
            frameCount: this._frameCount,
            world: this._width > 0 && this._height > 0 ? `${this._width} × ${this._height}` : "—",
            navigation: this._controls ? "WasmGPU OrbitControls pan/zoom" : "—",
            zoom: this._controls ? this._controls.zoom : 1,
            cameraCenter: this.cameraCenterLabel(),
            selectedLineage: this._selectedLineage,
            selectedCellId: this._selectedCellId,
            selectedTile: this._selectedTile
        };
    }

    initializeScene() {
        this._scene = this._wgpu.createScene([0.965, 0.965, 0.965]);
        this._camera = this._wgpu.createCamera.orthographic({ near: 0.01, far: 5000 });
        this._camera.transform.setPosition(0, 0, 1000);
        this._camera.lookAt(0, 0, 0);
        this._controls = this._wgpu.createControls.orbit(this._camera, this._canvas, {
            enableRotate: false,
            enablePan: true,
            enableZoom: true,
            zoomOnCursor: true,
            enableDamping: false,
            minZoom: 1,
            maxZoom: 75,
            mouseButtons: { pan: 0, zoom: 1, rotate: -1 }
        });
    }

    setDisplayMode(mode) {
        if (!TILE_DISPLAY_MODES.includes(mode)) throw new Error(`Unsupported Microcosm display mode: ${mode}`);
        if (mode === this._displayMode) return;
        this._displayMode = mode;
        this._visualStateDirty = true;
    }

    setSelectedLineage(lineageId) {
        const next = lineageId == null ? null : Number(lineageId) >>> 0;
        if (next === this._selectedLineage) return;
        this._selectedLineage = next;
        this._visualStateDirty = true;
    }

    setSelectedTile(tile) {
        const next = tile ? { x: Number(tile.x) | 0, y: Number(tile.y) | 0 } : null;
        if (sameTile(next, this._selectedTile)) return;
        this._selectedTile = next;
        this._visualStateDirty = true;
    }

    setSelectedCell(cellId) {
        const next = cellId == null ? null : Number(cellId) >>> 0;
        if (next === this._selectedCellId) return;
        this._selectedCellId = next;
        this._visualStateDirty = true;
    }

    setHoverTile(tile) {
        const next = tile ? { x: Number(tile.x) | 0, y: Number(tile.y) | 0 } : null;
        if (sameTile(next, this._hoverTile)) return;
        this._hoverTile = next;
        this._visualStateDirty = true;
    }

    setBrushPreview(preview) {
        const next = preview ? {
            x: Number(preview.x) | 0,
            y: Number(preview.y) | 0,
            width: Math.max(1, Number(preview.width) | 0),
            height: Math.max(1, Number(preview.height) | 0)
        } : null;
        if (sameBrush(next, this._brushPreview)) return;
        this._brushPreview = next;
        this._visualStateDirty = true;
    }

    setInteractionState(state = {}) {
        if (Object.prototype.hasOwnProperty.call(state, "selectedLineageId")) this.setSelectedLineage(state.selectedLineageId);
        if (Object.prototype.hasOwnProperty.call(state, "selectedCellId")) this.setSelectedCell(state.selectedCellId);
        if (Object.prototype.hasOwnProperty.call(state, "selectedTile")) this.setSelectedTile(state.selectedTile);
        if (Object.prototype.hasOwnProperty.call(state, "hoverTile")) this.setHoverTile(state.hoverTile);
        if (Object.prototype.hasOwnProperty.call(state, "brushPreview")) this.setBrushPreview(state.brushPreview);
    }

    visualState() {
        return {
            displayMode: this._displayMode,
            selectedLineageId: this._selectedLineage,
            selectedCellId: this._selectedCellId,
            selectedTile: this._selectedTile,
            hoverTile: this._hoverTile,
            brushPreview: this._brushPreview
        };
    }

    canvasToWorld(clientX, clientY) {
        if (!this._camera || this._width <= 0 || this._height <= 0) return null;
        const rect = this._canvas.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) return null;
        const x = clientX - rect.left;
        const y = clientY - rect.top;
        if (x < 0 || y < 0 || x > rect.width || y > rect.height) return null;
        const u = x / rect.width;
        const v = y / rect.height;
        const cameraPosition = this._camera.position || [0, 0, 0];
        const projectionX = this._camera.left + u * (this._camera.right - this._camera.left);
        const projectionY = this._camera.top - v * (this._camera.top - this._camera.bottom);
        return {
            x: cameraPosition[0] + projectionX,
            y: cameraPosition[1] + projectionY,
            canvasX: x,
            canvasY: y
        };
    }

    worldToTile(worldX, worldY, options = {}) {
        if (this._width <= 0 || this._height <= 0) return null;
        let x = Math.floor(worldX + this._width * 0.5);
        let y = Math.floor(this._height * 0.5 - worldY);
        if (options.wrap) {
            x = ((x % this._width) + this._width) % this._width;
            y = ((y % this._height) + this._height) % this._height;
            return { x, y, index: this.tileIndex(x, y) };
        }
        if (x < 0 || y < 0 || x >= this._width || y >= this._height) return null;
        return { x, y, index: this.tileIndex(x, y) };
    }

    tileFromClient(clientX, clientY, options = {}) {
        const world = this.canvasToWorld(clientX, clientY);
        if (!world) return null;
        const tile = this.worldToTile(world.x, world.y, options);
        return tile ? { ...tile, worldX: world.x, worldY: world.y, canvasX: world.canvasX, canvasY: world.canvasY } : null;
    }

    tileIndex(x, y) {
        if (this._height <= 0) return -1;
        return (Number(x) | 0) * this._height + (Number(y) | 0);
    }

    setPausedVisualState(paused) { this._pausedVisualState = !!paused; }

    invalidateVisualState() { this._visualStateDirty = true; }

    updateFromRuntime(runtime) {
        this.assertLive();
        if (!runtime || !runtime.ready) throw new Error("MicrocosmRenderer.updateFromRuntime requires a ready MicrocosmRuntime.");
        const stats = runtime.stats || {};
        const width = Math.max(1, Number(stats.width || 0) || 1);
        const height = Math.max(1, Number(stats.height || 0) || 1);
        const tileCount = Math.max(0, Number(runtime.tileCount || stats.tile_count || 0) | 0);
        const cellCount = Math.max(0, Number(runtime.cellCount || stats.live_cell_count || 0) | 0);
        const dimensionsChanged = width !== this._width || height !== this._height;
        this._width = width;
        this._height = height;
        this._tileCount = tileCount;
        this._cellCount = cellCount;

        if (this._visualStateDirty) {
            runtime.setRenderVisualState(this.visualState());
            this._visualStateDirty = false;
        }

        const dataView = runtime.views.latticeRgba;
        if (!dataView || dataView.dtype !== "f32" || dataView.length !== tileCount * 4) throw new Error(`Microcosm lattice RGBA view must contain ${tileCount * 4} f32 values.`);
        if (!this._lattice || dimensionsChanged || dataView !== this._latticeDataView) {
            this.createLattice(dataView);
        } else {
            this._lattice.refreshFromWasm({ keepCPUData: false });
        }

        if (dimensionsChanged || !this._viewInitialized) this.fitView({ saveState: true });
    }

    createLattice(dataView) {
        if (this._lattice) {
            this._scene.remove(this._lattice);
            this._lattice.clearWasmSources();
            this._lattice.destroy();
        }
        this._latticeDataView = dataView;
        this._lattice = this._wgpu.createLatticeSpace({
            dimensions: [this._height, this._width],
            componentCount: 4,
            wasmData: dataView,
            origin: [-this._height * 0.5 + 0.5, -this._width * 0.5 + 0.5, 0],
            spacing: [1, 1, 1],
            cellScale: 1,
            colorMode: "rgba",
            colorSpace: "srgb",
            opacity: 1,
            lit: false,
            blendMode: "opaque",
            depthWrite: true,
            depthTest: true,
            keepCPUData: false,
            name: "microcosm.composited-lattice"
        });
        this._lattice.transform.setRotationFromEuler(0, 0, -Math.PI * 0.5);
        this._scene.add(this._lattice);
    }

    measureCanvas() {
        return {
            width: Math.max(1, this._canvas.clientWidth || this._canvas.width || 1),
            height: Math.max(1, this._canvas.clientHeight || this._canvas.height || 1)
        };
    }

    fittedFrustum() {
        const canvas = this.measureCanvas();
        const aspect = canvas.width / canvas.height;
        const worldAspect = this._width / this._height;
        const margin = 3;
        let halfWidth = this._width * 0.5 + margin;
        let halfHeight = this._height * 0.5 + margin;
        if (aspect > worldAspect) halfWidth = halfHeight * aspect;
        else halfHeight = halfWidth / aspect;
        return { canvas, halfWidth, halfHeight };
    }

    applyCameraView(center, zoom = 1, options = {}) {
        if (!this._camera || this._width <= 0 || this._height <= 0) return;
        const fitted = this.fittedFrustum();
        this._lastCanvasWidth = fitted.canvas.width;
        this._lastCanvasHeight = fitted.canvas.height;
        this._camera.left = -fitted.halfWidth;
        this._camera.right = fitted.halfWidth;
        this._camera.top = fitted.halfHeight;
        this._camera.bottom = -fitted.halfHeight;
        const z = Math.max(this._width, this._height, 10) * 2;
        const target = [Number(center[0]) || 0, Number(center[1]) || 0, 0];
        this._camera.transform.setPosition(target[0], target[1], z);
        this._camera.lookAt(target[0], target[1], target[2]);
        if (this._controls) {
            if (typeof this._controls.setTarget === "function") this._controls.setTarget(target);
            else this._controls.target = target;
            this._controls.syncFromCamera();
            this._controls.zoom = Math.max(1, Math.min(75, Number(zoom) || 1));
            this._controls.update(0);
            if (options.saveState) this._controls.saveState();
        }
        this._viewInitialized = true;
    }

    fitView(options = {}) { this.applyCameraView([0, 0, 0], 1, { saveState: options.saveState !== false }); }

    cameraCenter() {
        if (this._controls && Array.isArray(this._controls.target)) return this._controls.target;
        if (this._camera && this._camera.position) return [this._camera.position[0] || 0, this._camera.position[1] || 0, 0];
        return [0, 0, 0];
    }

    cameraCenterLabel() {
        const center = this.cameraCenter();
        return `${center[0].toFixed(2)}, ${center[1].toFixed(2)}`;
    }

    resize(force = false) {
        if (!this._camera || this._width <= 0 || this._height <= 0) return;
        const canvas = this.measureCanvas();
        if (!force && canvas.width === this._lastCanvasWidth && canvas.height === this._lastCanvasHeight) return;
        const center = this.cameraCenter();
        const zoom = this._controls ? this._controls.zoom : 1;
        this.applyCameraView(center, zoom, { saveState: false });
    }

    updateControls(dtSeconds = 0) { if (this._controls) this._controls.update(dtSeconds); }

    render(dtSeconds = 0) {
        this.assertLive();
        this.resize();
        this.updateControls(dtSeconds);
        this._wgpu.render(this._scene, this._camera);
        this._frameCount++;
    }

    destroy() {
        if (this._destroyed) return;
        this._controls?.dispose?.();
        this._controls = null;
        if (this._lattice) {
            this._scene?.remove?.(this._lattice);
            this._lattice.clearWasmSources();
            this._lattice.destroy();
        }
        this._lattice = null;
        this._latticeDataView = null;
        this._wgpu?.destroy?.();
        this._destroyed = true;
    }

    assertLive() { if (this._destroyed) throw new Error("MicrocosmRenderer has been destroyed."); }
}
