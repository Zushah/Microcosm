const EMPTY_CELL_ID = 0xffffffff;
const TILE_NEUTRAL_RGB255 = [246, 246, 246];
const TILE_ELEMENT_ORDER = Object.freeze(["a", "b", "c", "d", "e", "f"]);
const TILE_DISPLAY_MODES = Object.freeze(["enval", "occupancy", "mass", "molecules", "element-a", "element-b", "element-c", "element-d", "element-e", "element-f"]);

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

const hslToRgb255 = (hDeg, s01, l01) => {
    const h = ((hDeg % 360) + 360) % 360;
    const s = clamp(s01, 0, 1);
    const l = clamp(l01, 0, 1);
    const c = (1 - Math.abs(2 * l - 1)) * s;
    const hp = h / 60;
    const x = c * (1 - Math.abs((hp % 2) - 1));
    let r1 = 0;
    let g1 = 0;
    let b1 = 0;
    if (hp >= 0 && hp < 1) { r1 = c; g1 = x; b1 = 0; }
    else if (hp >= 1 && hp < 2) { r1 = x; g1 = c; b1 = 0; }
    else if (hp >= 2 && hp < 3) { r1 = 0; g1 = c; b1 = x; }
    else if (hp >= 3 && hp < 4) { r1 = 0; g1 = x; b1 = c; }
    else if (hp >= 4 && hp < 5) { r1 = x; g1 = 0; b1 = c; }
    else { r1 = c; g1 = 0; b1 = x; }
    const m = l - c / 2;
    const r = Math.round((r1 + m) * 255);
    const g = Math.round((g1 + m) * 255);
    const b = Math.round((b1 + m) * 255);
    return [clamp(r, 0, 255), clamp(g, 0, 255), clamp(b, 0, 255)];
};

const TILE_ELEMENT_RGB255 = Object.freeze({
    a: hslToRgb255(132, 0.46, 0.48),
    b: hslToRgb255(220, 0.62, 0.54),
    c: hslToRgb255(186, 0.62, 0.46),
    d: hslToRgb255(12, 0.66, 0.56),
    e: hslToRgb255(46, 0.68, 0.52),
    f: hslToRgb255(284, 0.44, 0.56)
});

const normalizeWasmGPU = (WasmGPU) => {
    const candidate = WasmGPU && (WasmGPU.default || WasmGPU);
    if (!candidate || typeof candidate.create !== "function") throw new Error("MicrocosmRenderer requires a WasmGPU object with WasmGPU.create(...).");
    return candidate;
};

const createGpuBuffer = (device, byteLength, label) => device.createBuffer({
    label,
    size: Math.max(16, byteLength >>> 0),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
});

const nextCapacity = (required, current = 0) => {
    let capacity = Math.max(1, current || 1);
    while (capacity < required) capacity *= 2;
    return capacity;
};

const setRgb255 = (target, offset, rgb255, alpha = 1) => {
    target[offset + 0] = rgb255[0] / 255;
    target[offset + 1] = rgb255[1] / 255;
    target[offset + 2] = rgb255[2] / 255;
    target[offset + 3] = alpha;
};

const setBlendRgb255 = (target, offset, baseRgb, targetRgb, intensity, alpha = 1) => {
    const t = clamp(intensity, 0, 1);
    target[offset + 0] = (baseRgb[0] + (targetRgb[0] - baseRgb[0]) * t) / 255;
    target[offset + 1] = (baseRgb[1] + (targetRgb[1] - baseRgb[1]) * t) / 255;
    target[offset + 2] = (baseRgb[2] + (targetRgb[2] - baseRgb[2]) * t) / 255;
    target[offset + 3] = alpha;
};

const setEnvalRgb = (target, offset, value) => {
    const v = Number.isFinite(value) ? value : 0;
    const mapped = Math.atan(v * 1.25) / (Math.PI * 0.5);
    const mag = Math.min(1, Math.abs(mapped));
    let r = TILE_NEUTRAL_RGB255[0];
    let g = TILE_NEUTRAL_RGB255[1];
    let b = TILE_NEUTRAL_RGB255[2];
    if (mapped >= 0) {
        r = Math.floor(246 - 10 * mag);
        g = Math.floor(246 - 92 * mag);
        b = Math.floor(246 - 205 * mag);
    } else {
        r = Math.floor(246 - 205 * mag);
        g = Math.floor(246 - 118 * mag);
        b = Math.floor(246 - 10 * mag);
    }
    target[offset + 0] = clamp(r, 0, 255) / 255;
    target[offset + 1] = clamp(g, 0, 255) / 255;
    target[offset + 2] = clamp(b, 0, 255) / 255;
    target[offset + 3] = 1;
};

const lineageRgb01 = (lineageId, cache) => {
    const id = Number(lineageId) >>> 0;
    if (cache.has(id)) return cache.get(id);
    let x = id ^ 0x9e3779b9;
    x = Math.imul(x ^ (x >>> 16), 0x85ebca6b);
    x = Math.imul(x ^ (x >>> 13), 0xc2b2ae35);
    x = (x ^ (x >>> 16)) >>> 0;
    const hue = x % 360;
    const [r255, g255, b255] = hslToRgb255(hue, 0.70, 0.55);
    const out = [r255 / 255, g255 / 255, b255 / 255];
    cache.set(id, out);
    return out;
};

const displayModeElement = (mode) => {
    if (!mode || !mode.startsWith("element-")) return null;
    const element = mode.slice("element-".length).toLowerCase();
    const index = TILE_ELEMENT_ORDER.indexOf(element);
    return index >= 0 ? { element, index, mask: 1 << index } : null;
};

const axisInWrappedSpan = (value, start, span, size) => {
    if (span >= size) return true;
    const normalized = ((start % size) + size) % size;
    const end = normalized + span;
    return end <= size ? value >= normalized && value < end : value >= normalized || value < (end - size);
};

const tileInBrush = (x, y, brush, width, height) => {
    if (!brush || !Number.isInteger(brush.x) || !Number.isInteger(brush.y)) return false;
    const brushWidth = Math.max(1, Math.min(width, Number(brush.width) | 0));
    const brushHeight = Math.max(1, Math.min(height, Number(brush.height) | 0));
    const startX = brush.x - Math.floor(brushWidth * 0.5);
    const startY = brush.y - Math.floor(brushHeight * 0.5);
    return axisInWrappedSpan(x, startX, brushWidth, width) && axisInWrappedSpan(y, startY, brushHeight, height);
};

const updateLayerBounds = (cloud, width, height, z = 0) => {
    const halfWidth = Math.max(0.5, width * 0.5);
    const halfHeight = Math.max(0.5, height * 0.5);
    cloud.boundsMin = [-halfWidth, -halfHeight, z - 0.5];
    cloud.boundsMax = [halfWidth, halfHeight, z + 0.5];
    cloud.boundsCenter = [0, 0, z];
    cloud.boundsRadius = Math.hypot(halfWidth, halfHeight, 1);
};

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
        const renderer = new MicrocosmRenderer(wgpu, canvas, options);
        renderer.initializeScene();
        return renderer;
    }

    constructor(wgpu, canvas, options = {}) {
        this._wgpu = wgpu;
        this._canvas = canvas;
        this._scene = null;
        this._camera = null;
        this._tileCloud = null;
        this._cellCloud = null;
        this._tileLayer = null;
        this._cellLayer = null;
        this._displayMode = options.displayMode || "enval";
        this._selectedLineage = null;
        this._selectedTile = null;
        this._selectedCellId = null;
        this._hoverTile = null;
        this._brushPreview = null;
        this._pausedVisualState = false;
        this._width = 0;
        this._height = 0;
        this._tileCount = 0;
        this._cellCount = 0;
        this._tileCoordinatesDirty = true;
        this._lastCanvasWidth = 0;
        this._lastCanvasHeight = 0;
        this._frameCount = 0;
        this._lineageRgbCache = new Map();
        this._destroyed = false;
    }

    get wgpu() {
        return this._wgpu;
    }

    get scene() {
        return this._scene;
    }

    get camera() {
        return this._camera;
    }

    get canvas() {
        return this._canvas;
    }

    get displayMode() {
        return this._displayMode;
    }

    get width() {
        return this._width;
    }

    get height() {
        return this._height;
    }

    get tileCountRendered() {
        return this._tileCount;
    }

    get cellCountRendered() {
        return this._cellCount;
    }

    get frameCount() {
        return this._frameCount;
    }

    get diagnostics() {
        return {
            strategy: "WasmGPU PointCloud tile field + PointCloud cell overlay",
            displayMode: this._displayMode,
            tileCountRendered: this._tileCount,
            cellCountRendered: this._cellCount,
            frameCount: this._frameCount,
            world: this._width > 0 && this._height > 0 ? `${this._width} × ${this._height}` : "—",
            selectedLineage: this._selectedLineage,
            selectedCellId: this._selectedCellId,
            selectedTile: this._selectedTile
        };
    }

    initializeScene() {
        const device = this._wgpu.gpu.device;
        this._scene = this._wgpu.createScene([0.965, 0.965, 0.965]);
        this._camera = this._wgpu.createCamera.orthographic({ near: 0.01, far: 5000 });
        this._camera.transform.setPosition(0, 0, 1000);
        this._camera.lookAt(0, 0, 0);
        this._tileLayer = this.createPointLayer(device, "microcosm.tiles", 1);
        this._cellLayer = this.createPointLayer(device, "microcosm.cells", 1);
        this._tileCloud = this._wgpu.createPointCloud({
            pointsBuffer: this._tileLayer.pointsBuffer,
            colorsBuffer: this._tileLayer.colorsBuffer,
            pointCount: 1,
            colorMode: "rgba",
            scaleTransform: { mode: "linear", domainMin: 0, domainMax: 1 },
            basePointSize: 2,
            minPointSize: 1,
            maxPointSize: 8,
            sizeAttenuation: 0,
            opacity: 1,
            softness: 0,
            blendMode: "opaque",
            depthWrite: false,
            depthTest: true,
            name: "microcosm.tile-field"
        });
        this._cellCloud = this._wgpu.createPointCloud({
            pointsBuffer: this._cellLayer.pointsBuffer,
            colorsBuffer: this._cellLayer.colorsBuffer,
            pointCount: 1,
            colorMode: "rgba",
            scaleTransform: { mode: "linear", domainMin: 0, domainMax: 100 },
            basePointSize: 4,
            minPointSize: 2,
            maxPointSize: 16,
            sizeAttenuation: 0,
            opacity: 1,
            softness: 0.05,
            blendMode: "transparent",
            depthWrite: false,
            depthTest: true,
            name: "microcosm.cells"
        });
        updateLayerBounds(this._tileCloud, 1, 1, 0);
        updateLayerBounds(this._cellCloud, 1, 1, 1);
        this._scene.add(this._tileCloud);
        this._scene.add(this._cellCloud);
    }

    createPointLayer(device, label, capacity) {
        const pointCapacity = Math.max(1, capacity | 0);
        const byteLength = pointCapacity * 4 * 4;
        return {
            label,
            capacity: pointCapacity,
            count: 1,
            points: new Float32Array(pointCapacity * 4),
            colors: new Float32Array(pointCapacity * 4),
            pointsView: null,
            colorsView: null,
            pointsBuffer: createGpuBuffer(device, byteLength, `${label}.points`),
            colorsBuffer: createGpuBuffer(device, byteLength, `${label}.colors`)
        };
    }

    setDisplayMode(mode) {
        if (!TILE_DISPLAY_MODES.includes(mode)) throw new Error(`Unsupported Microcosm display mode: ${mode}`);
        if (mode === this._displayMode) return;
        this._displayMode = mode;
    }

    setSelectedLineage(lineageId) {
        this._selectedLineage = lineageId == null ? null : Number(lineageId) >>> 0;
    }

    setSelectedTile(tile) {
        this._selectedTile = tile ? { x: Number(tile.x) | 0, y: Number(tile.y) | 0 } : null;
    }

    setSelectedCell(cellId) {
        this._selectedCellId = cellId == null ? null : Number(cellId) >>> 0;
    }

    setHoverTile(tile) {
        this._hoverTile = tile ? { x: Number(tile.x) | 0, y: Number(tile.y) | 0 } : null;
    }

    setBrushPreview(preview) {
        this._brushPreview = preview ? {
            x: Number(preview.x) | 0,
            y: Number(preview.y) | 0,
            width: Math.max(1, Number(preview.width) | 0),
            height: Math.max(1, Number(preview.height) | 0)
        } : null;
    }

    setInteractionState(state = {}) {
        if (Object.prototype.hasOwnProperty.call(state, "selectedLineageId")) this.setSelectedLineage(state.selectedLineageId);
        if (Object.prototype.hasOwnProperty.call(state, "selectedCellId")) this.setSelectedCell(state.selectedCellId);
        if (Object.prototype.hasOwnProperty.call(state, "selectedTile")) this.setSelectedTile(state.selectedTile);
        if (Object.prototype.hasOwnProperty.call(state, "hoverTile")) this.setHoverTile(state.hoverTile);
        if (Object.prototype.hasOwnProperty.call(state, "brushPreview")) this.setBrushPreview(state.brushPreview);
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
        return {
            x: this._camera.left + u * (this._camera.right - this._camera.left),
            y: this._camera.top - v * (this._camera.top - this._camera.bottom),
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

    setPausedVisualState(paused) {
        this._pausedVisualState = !!paused;
    }

    updateFromRuntime(runtime) {
        this.assertLive();
        if (!runtime || !runtime.ready) throw new Error("MicrocosmRenderer.updateFromRuntime requires a ready MicrocosmRuntime.");
        const stats = runtime.stats || {};
        const width = Math.max(1, Number(stats.width || 0) || 1);
        const height = Math.max(1, Number(stats.height || 0) || 1);
        const tileCount = Math.max(0, Number(runtime.tileCount || stats.tile_count || 0) | 0);
        const cellCount = Math.max(0, Number(runtime.cellCount || stats.live_cell_count || 0) | 0);
        if (width !== this._width || height !== this._height || tileCount !== this._tileCount) {
            this._width = width;
            this._height = height;
            this._tileCoordinatesDirty = true;
            this.ensureLayerCapacity(this._tileLayer, this._tileCloud, tileCount);
            this._tileCount = tileCount;
            updateLayerBounds(this._tileCloud, width, height, 0);
            updateLayerBounds(this._cellCloud, width, height, 1);
            this.resize(true);
        }
        this.ensureLayerCapacity(this._cellLayer, this._cellCloud, Math.max(1, cellCount));
        this._cellCount = cellCount;
        this.updateTileLayer(runtime);
        this.updateCellLayer(runtime);
        this.syncPointSizes();
    }

    ensureLayerCapacity(layer, cloud, requiredCount) {
        const renderCount = Math.max(1, requiredCount | 0);
        const device = this._wgpu.gpu.device;
        if (renderCount > layer.capacity) {
            const capacity = nextCapacity(renderCount, layer.capacity);
            const byteLength = capacity * 4 * 4;
            const nextPoints = createGpuBuffer(device, byteLength, `${layer.label}.points`);
            const nextColors = createGpuBuffer(device, byteLength, `${layer.label}.colors`);
            layer.pointsBuffer.destroy();
            layer.colorsBuffer.destroy();
            layer.pointsBuffer = nextPoints;
            layer.colorsBuffer = nextColors;
            layer.capacity = capacity;
            layer.points = new Float32Array(capacity * 4);
            layer.colors = new Float32Array(capacity * 4);
            layer.pointsView = null;
            layer.colorsView = null;
            cloud.setPointsBuffer(layer.pointsBuffer, renderCount);
            cloud.setColorsBuffer(layer.colorsBuffer);
            layer.count = renderCount;
            return;
        }
        if (renderCount !== layer.count) {
            cloud.setPointsBuffer(layer.pointsBuffer, renderCount);
            cloud.setColorsBuffer(layer.colorsBuffer);
            layer.count = renderCount;
            layer.pointsView = null;
            layer.colorsView = null;
        }
    }

    layerPointsView(layer) {
        const length = layer.count * 4;
        if (!layer.pointsView || layer.pointsView.length !== length) layer.pointsView = layer.points.subarray(0, length);
        return layer.pointsView;
    }

    layerColorsView(layer) {
        const length = layer.count * 4;
        if (!layer.colorsView || layer.colorsView.length !== length) layer.colorsView = layer.colors.subarray(0, length);
        return layer.colorsView;
    }

    updateTileLayer(runtime) {
        const count = this._tileCount;
        if (count <= 0) {
            this._tileCloud.visible = false;
            return;
        }
        this._tileCloud.visible = true;
        const points = this.layerPointsView(this._tileLayer);
        const colors = this.layerColorsView(this._tileLayer);
        if (this._tileCoordinatesDirty) {
            for (let i = 0; i < count; i++) {
                const x = Math.floor(i / this._height);
                const y = i % this._height;
                const offset = i * 4;
                points[offset + 0] = x + 0.5 - this._width * 0.5;
                points[offset + 1] = this._height * 0.5 - y - 0.5;
                points[offset + 2] = 0;
                points[offset + 3] = 0;
            }
            this._wgpu.gpu.queue.writeBuffer(this._tileLayer.pointsBuffer, 0, points.buffer, points.byteOffset, points.byteLength);
            this._tileCoordinatesDirty = false;
        }
        this.writeTileColors(runtime, colors, count);
        this._wgpu.gpu.queue.writeBuffer(this._tileLayer.colorsBuffer, 0, colors.buffer, colors.byteOffset, colors.byteLength);
    }

    writeTileColors(runtime, colors, count) {
        const views = runtime.views;
        const mode = this._displayMode;
        const elementMode = displayModeElement(mode);
        const enval = views.tileEnval && views.tileEnval.array();
        const occupancy = views.tileOccupancy && views.tileOccupancy.array();
        const mass = views.tileMass && views.tileMass.array();
        const molecules = views.tileMoleculeCount && views.tileMoleculeCount.array();
        const mask = views.tileElementMask && views.tileElementMask.array();
        const selectedTileIndex = this._selectedTile ? this.tileIndex(this._selectedTile.x, this._selectedTile.y) : -1;
        const hoverTileIndex = this._hoverTile ? this.tileIndex(this._hoverTile.x, this._hoverTile.y) : -1;
        for (let i = 0; i < count; i++) {
            const offset = i * 4;
            const x = Math.floor(i / this._height);
            const y = i % this._height;
            if (mode === "enval") { setEnvalRgb(colors, offset, enval ? enval[i] : 0); }
            else if (mode === "occupancy") { const occupied = occupancy && occupancy[i] !== EMPTY_CELL_ID; setBlendRgb255(colors, offset, TILE_NEUTRAL_RGB255, [46, 57, 72], occupied ? 0.92 : 0.0, 1); }
            else if (mode === "mass") { const intensity = 1 - Math.exp(-Math.max(0, mass ? mass[i] : 0) * 0.08); setBlendRgb255(colors, offset, TILE_NEUTRAL_RGB255, [86, 154, 112], intensity, 1); }
            else if (mode === "molecules") { const intensity = 1 - Math.exp(-Math.max(0, molecules ? molecules[i] : 0) * 0.30); setBlendRgb255(colors, offset, TILE_NEUTRAL_RGB255, [69, 139, 186], intensity, 1); }
            else if (elementMode) { const present = mask && ((mask[i] & elementMode.mask) !== 0); setBlendRgb255(colors, offset, TILE_NEUTRAL_RGB255, TILE_ELEMENT_RGB255[elementMode.element], present ? 1.0 : 0.0, 1); }
            else { setRgb255(colors, offset, TILE_NEUTRAL_RGB255, 1); }
            if (this._brushPreview && tileInBrush(x, y, this._brushPreview, this._width, this._height)) {
                colors[offset + 0] = Math.min(1, colors[offset + 0] * 0.72 + 0.28 * 0.56);
                colors[offset + 1] = Math.min(1, colors[offset + 1] * 0.72 + 0.28 * 0.26);
                colors[offset + 2] = Math.min(1, colors[offset + 2] * 0.72 + 0.28 * 0.82);
            }
            if (i === hoverTileIndex) {
                colors[offset + 0] = Math.min(1, colors[offset + 0] * 0.65 + 0.35);
                colors[offset + 1] = Math.min(1, colors[offset + 1] * 0.65 + 0.35);
                colors[offset + 2] = Math.min(1, colors[offset + 2] * 0.65 + 0.35);
            }
            if (i === selectedTileIndex) {
                colors[offset + 0] = Math.min(1, colors[offset + 0] * 0.35 + 0.65);
                colors[offset + 1] = Math.min(1, colors[offset + 1] * 0.35 + 0.55);
                colors[offset + 2] = Math.min(1, colors[offset + 2] * 0.35 + 0.10);
            }
        }
    }

    updateCellLayer(runtime) {
        const count = this._cellCount;
        const layer = this._cellLayer;
        const points = this.layerPointsView(layer);
        const colors = this.layerColorsView(layer);
        if (count <= 0) {
            this._cellCloud.visible = false;
            points[0] = 0; points[1] = 0; points[2] = 1; points[3] = 0;
            colors[0] = 0; colors[1] = 0; colors[2] = 0; colors[3] = 0;
            this._wgpu.gpu.queue.writeBuffer(layer.pointsBuffer, 0, points.buffer, points.byteOffset, 16);
            this._wgpu.gpu.queue.writeBuffer(layer.colorsBuffer, 0, colors.buffer, colors.byteOffset, 16);
            return;
        }
        this._cellCloud.visible = true;
        const views = runtime.views;
        const cellX = views.cellX && views.cellX.array();
        const cellY = views.cellY && views.cellY.array();
        const cellEnergy = views.cellEnergy && views.cellEnergy.array();
        const cellLineage = views.cellLineage && views.cellLineage.array();
        const cellId = views.cellId && views.cellId.array();
        for (let i = 0; i < count; i++) {
            const offset = i * 4;
            const x = cellX ? cellX[i] : 0;
            const y = cellY ? cellY[i] : 0;
            const lineage = cellLineage ? cellLineage[i] : 0;
            const id = cellId ? cellId[i] : 0;
            const color = lineageRgb01(lineage, this._lineageRgbCache);
            const lineageSelected = this._selectedLineage === null || this._selectedLineage === (Number(lineage) >>> 0);
            const cellSelected = this._selectedCellId !== null && this._selectedCellId === (Number(id) >>> 0);
            points[offset + 0] = x + 0.5 - this._width * 0.5;
            points[offset + 1] = this._height * 0.5 - y - 0.5;
            points[offset + 2] = 1;
            points[offset + 3] = cellEnergy ? cellEnergy[i] : 0;
            if (cellSelected) {
                colors[offset + 0] = 1.0;
                colors[offset + 1] = 0.93;
                colors[offset + 2] = 0.30;
                colors[offset + 3] = 1.0;
            } else {
                colors[offset + 0] = color[0];
                colors[offset + 1] = color[1];
                colors[offset + 2] = color[2];
                colors[offset + 3] = lineageSelected ? 1.0 : 0.20;
            }
        }
        this._wgpu.gpu.queue.writeBuffer(layer.pointsBuffer, 0, points.buffer, points.byteOffset, points.byteLength);
        this._wgpu.gpu.queue.writeBuffer(layer.colorsBuffer, 0, colors.buffer, colors.byteOffset, colors.byteLength);
    }

    syncPointSizes() {
        const width = Math.max(1, this._canvas.clientWidth || this._canvas.width || 1);
        const height = Math.max(1, this._canvas.clientHeight || this._canvas.height || 1);
        const pixelPerTile = Math.max(1, Math.min(width / Math.max(1, this._width), height / Math.max(1, this._height)));
        const tileSize = clamp(pixelPerTile * 1.18, 1.0, 7.0);
        const cellSize = clamp(pixelPerTile * 1.85, 2.0, 18.0);
        this._tileCloud.basePointSize = tileSize;
        this._tileCloud.minPointSize = Math.max(1, tileSize * 0.75);
        this._tileCloud.maxPointSize = Math.max(tileSize, tileSize * 1.35);
        this._cellCloud.basePointSize = cellSize;
        this._cellCloud.minPointSize = Math.max(2, cellSize * 0.75);
        this._cellCloud.maxPointSize = Math.max(cellSize, cellSize * 1.45);
    }

    resize(force = false) {
        if (!this._camera || this._width <= 0 || this._height <= 0) return;
        const canvasWidth = Math.max(1, this._canvas.clientWidth || this._canvas.width || 1);
        const canvasHeight = Math.max(1, this._canvas.clientHeight || this._canvas.height || 1);
        if (!force && canvasWidth === this._lastCanvasWidth && canvasHeight === this._lastCanvasHeight) return;
        this._lastCanvasWidth = canvasWidth;
        this._lastCanvasHeight = canvasHeight;
        const aspect = canvasWidth / canvasHeight;
        const worldAspect = this._width / this._height;
        const margin = 3;
        let halfWidth = this._width * 0.5 + margin;
        let halfHeight = this._height * 0.5 + margin;
        if (aspect > worldAspect) halfWidth = halfHeight * aspect;
        else halfHeight = halfWidth / aspect;
        this._camera.left = -halfWidth;
        this._camera.right = halfWidth;
        this._camera.top = halfHeight;
        this._camera.bottom = -halfHeight;
        this._camera.transform.setPosition(0, 0, Math.max(this._width, this._height, 10) * 2);
        this._camera.lookAt(0, 0, 0);
        this.syncPointSizes();
    }

    render() {
        this.assertLive();
        this.resize();
        this._wgpu.render(this._scene, this._camera);
        this._frameCount++;
    }

    destroy() {
        if (this._destroyed) return;
        this._tileCloud?.destroy?.();
        this._cellCloud?.destroy?.();
        this._tileLayer = null;
        this._cellLayer = null;
        this._wgpu?.destroy?.();
        this._destroyed = true;
    }

    assertLive() {
        if (this._destroyed) throw new Error("MicrocosmRenderer has been destroyed.");
    }
}
