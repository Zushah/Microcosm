const DEFAULT_BRUSH = Object.freeze({ width: 10, height: 10, delta: 0.05 });
const CLICK_DRAG_THRESHOLD_PX = 4;

const normalizeMode = (mode) => mode === "edit" || mode === "edit-genome" ? mode : "explore";

const normalizeBrushSpan = (value, fallback) => { const parsed = Math.round(Number(value)); if (!Number.isFinite(parsed)) return fallback; return Math.max(1, parsed); };

const normalizeBrushDelta = (value, fallback) => { const parsed = Number(value); return Number.isFinite(parsed) ? parsed : fallback; };

const sameTile = (a, b) => Boolean(a && b && a.x === b.x && a.y === b.y);

const isEditableShortcutTarget = (target) => {
    if (!(target instanceof Element)) return false;
    const tag = target.tagName.toLowerCase();
    return tag === "input" || tag === "select" || tag === "textarea" || target.isContentEditable;
};

export class MicrocosmInteraction {
    static create(options = {}) {
        const interaction = new MicrocosmInteraction(options);
        if (options.attach !== false) interaction.attach();
        return interaction;
    }

    constructor(options = {}) {
        if (!options.runtime) throw new Error("MicrocosmInteraction requires a MicrocosmRuntime.");
        if (!options.renderer) throw new Error("MicrocosmInteraction requires a MicrocosmRenderer.");
        this.runtime = options.runtime;
        this.renderer = options.renderer;
        this.canvas = options.canvas || options.renderer.canvas;
        this.onChange = options.onChange || null;
        this.onMutation = options.onMutation || null;
        this.onError = options.onError || null;
        this.genomePatch = options.genomePatch || null;
        this.mode = normalizeMode(options.mode || "explore");
        this.brush = { ...DEFAULT_BRUSH, ...(options.brush || {}) };
        this.hoverTile = null;
        this.hoverTileInfo = null;
        this.selectedTile = null;
        this.selectedTileInfo = null;
        this.selectedCellId = null;
        this.selectedCellInfo = null;
        this.selectedLineageId = null;
        this.isPainting = false;
        this.lastPaintKey = null;
        this.pendingLeftClick = null;
        this.attached = false;
        this._listeners = [];
        this.syncRendererState();
    }

    get state() {
        return {
            mode: this.mode,
            brush: { ...this.brush },
            hoverTile: this.hoverTile ? { ...this.hoverTile } : null,
            hoverTileInfo: this.hoverTileInfo,
            selectedTile: this.selectedTile ? { ...this.selectedTile } : null,
            selectedTileInfo: this.selectedTileInfo,
            selectedCellId: this.selectedCellId,
            selectedCellInfo: this.selectedCellInfo,
            selectedLineageId: this.selectedLineageId,
            isPainting: this.isPainting
        };
    }

    attach() {
        if (this.attached) return;
        this.addListener(this.canvas, "pointermove", (event) => this.handlePointerMove(event));
        this.addListener(this.canvas, "pointerdown", (event) => this.handlePointerDown(event));
        this.addListener(window, "pointerup", (event) => this.handlePointerUp(event));
        this.addListener(this.canvas, "pointerleave", () => this.handlePointerLeave());
        this.addListener(this.canvas, "contextmenu", (event) => this.handleContextMenu(event));
        this.addListener(window, "keydown", (event) => this.handleKeyDown(event));
        this.attached = true;
    }

    detach() {
        for (const entry of this._listeners) entry.target.removeEventListener(entry.type, entry.listener, entry.options);
        this._listeners = [];
        this.attached = false;
        this.isPainting = false;
        this.lastPaintKey = null;
        this.pendingLeftClick = null;
    }

    destroy() {
        this.detach();
        this.renderer.setInteractionState({ hoverTile: null, brushPreview: null });
    }

    addListener(target, type, listener, options) {
        target.addEventListener(type, listener, options);
        this._listeners.push({ target, type, listener, options });
    }

    setMode(mode) {
        this.mode = normalizeMode(mode);
        this.isPainting = false;
        this.lastPaintKey = null;
        this.pendingLeftClick = null;
        this.syncRendererState();
        this.notifyChange();
    }

    setBrushOptions(options = {}) {
        this.brush = {
            width: normalizeBrushSpan(options.width ?? this.brush.width, this.brush.width),
            height: normalizeBrushSpan(options.height ?? this.brush.height, this.brush.height),
            delta: normalizeBrushDelta(options.delta ?? this.brush.delta, this.brush.delta)
        };
        this.syncRendererState();
        this.notifyChange();
    }

    clearSelection() {
        this.selectedTile = null;
        this.selectedTileInfo = null;
        this.selectedCellId = null;
        this.selectedCellInfo = null;
        this.selectedLineageId = null;
        this.pendingLeftClick = null;
        this.syncRendererState();
        this.notifyChange();
    }

    clearLineageSelection() {
        this.selectedLineageId = null;
        this.syncRendererState();
        this.notifyChange();
    }

    selectLineage(lineageId) {
        const parsed = Number(lineageId);
        this.selectedLineageId = Number.isFinite(parsed) ? (parsed >>> 0) : null;
        this.syncRendererState();
        this.notifyChange();
    }

    refreshSelection() {
        if (!this.runtime || !this.runtime.ready) return;
        try {
            if (this.selectedTile) {
                this.selectedTileInfo = this.runtime.inspectTile(this.selectedTile.x, this.selectedTile.y);
                if (this.selectedTileInfo && this.selectedTileInfo.cell_id != null) {
                    this.selectedCellId = Number(this.selectedTileInfo.cell_id) >>> 0;
                    this.selectedCellInfo = this.runtime.inspectCell(this.selectedCellId);
                } else { this.selectedCellId = null; this.selectedCellInfo = null; }
            }
            if (this.hoverTile) this.hoverTileInfo = this.runtime.inspectTile(this.hoverTile.x, this.hoverTile.y);
        } catch (error) { if (this.selectedCellId !== null) { this.selectedCellId = null; this.selectedCellInfo = null; } }
        this.syncRendererState();
    }

    tileFromEvent(event) {
        return this.renderer.tileFromClient(event.clientX, event.clientY);
    }

    inspectTile(tile) {
        if (!tile) return null;
        return this.runtime.inspectTile(tile.x, tile.y);
    }

    inspectCell(cellId) {
        if (cellId == null) return null;
        return this.runtime.inspectCell(cellId);
    }

    handlePointerMove(event) {
        this.updatePendingLeftClick(event);
        const tile = this.tileFromEvent(event);
        const refreshHoverInfo = !this.pendingLeftClick || !this.pendingLeftClick.moved;
        if (!sameTile(tile, this.hoverTile)) {
            this.hoverTile = tile ? { x: tile.x, y: tile.y } : null;
            this.hoverTileInfo = this.hoverTile && refreshHoverInfo ? this.safeInspectTile(this.hoverTile) : null;
            this.syncRendererState();
            this.notifyChange();
        } else if (this.hoverTile && refreshHoverInfo && !this.hoverTileInfo) { this.hoverTileInfo = this.safeInspectTile(this.hoverTile); this.notifyChange(); }
        if ((this.mode === "edit" || this.mode === "edit-genome") && this.isPainting) this.applyBrush(tile, event);
    }

    handlePointerDown(event) {
        const tile = this.tileFromEvent(event);
        if (event.button === 0) {
            this.pendingLeftClick = this.mode === "explore" ? {
                pointerId: event.pointerId,
                startClientX: event.clientX,
                startClientY: event.clientY,
                startTile: tile ? { x: tile.x, y: tile.y } : null,
                moved: false
            } : null;
            return;
        }
        if ((this.mode === "edit" || this.mode === "edit-genome") && event.button === 2) {
            event.preventDefault();
            this.pendingLeftClick = null;
            this.isPainting = true;
            this.lastPaintKey = null;
            try { this.canvas.setPointerCapture(event.pointerId); } catch { /* ignore */ }
            this.applyBrush(tile, event);
            return;
        }
        if (event.button === 2) {
            event.preventDefault();
            this.pendingLeftClick = null;
            this.selectLineageFromTile(tile);
        }
    }

    handlePointerUp(event) {
        if (event.button === 0) { this.finishPendingLeftClick(event); return; }
        if (event.button === 2) {
            this.isPainting = false;
            this.lastPaintKey = null;
            try { this.canvas.releasePointerCapture(event.pointerId); } catch { /* ignore */ }
            this.syncRendererState();
            this.notifyChange();
        }
    }

    handlePointerLeave() {
        this.hoverTile = null;
        this.hoverTileInfo = null;
        if (!this.isPainting) this.renderer.setInteractionState({ hoverTile: null, brushPreview: null });
        this.notifyChange();
    }

    handleContextMenu(event) {
        event.preventDefault();
    }

    handleKeyDown(event) {
        if (isEditableShortcutTarget(event.target)) return;
        if (event.key === "Escape") { event.preventDefault(); this.clearSelection(); return; }
        if (event.key.toLowerCase() === "e") { event.preventDefault(); this.setMode(this.mode === "explore" ? "edit" : "explore"); }
    }

    updatePendingLeftClick(event) {
        const pending = this.pendingLeftClick;
        if (!pending || event.pointerId !== pending.pointerId) return;
        const dx = event.clientX - pending.startClientX;
        const dy = event.clientY - pending.startClientY;
        if ((dx * dx + dy * dy) > CLICK_DRAG_THRESHOLD_PX * CLICK_DRAG_THRESHOLD_PX) pending.moved = true;
    }

    finishPendingLeftClick(event) {
        const pending = this.pendingLeftClick;
        if (!pending || event.pointerId !== pending.pointerId) return;
        this.updatePendingLeftClick(event);
        this.pendingLeftClick = null;
        if (pending.moved || this.mode !== "explore") return;
        const tile = this.tileFromEvent(event) || pending.startTile;
        this.selectTile(tile);
    }

    selectTile(tile) {
        if (!tile) return;
        this.selectedTile = { x: tile.x, y: tile.y };
        this.selectedTileInfo = this.safeInspectTile(tile);
        const cellId = this.selectedTileInfo && this.selectedTileInfo.cell_id != null ? Number(this.selectedTileInfo.cell_id) >>> 0 : null;
        this.selectedCellId = cellId;
        this.selectedCellInfo = cellId === null ? null : this.safeInspectCell(cellId);
        this.syncRendererState();
        this.notifyChange();
    }

    selectLineageFromTile(tile) {
        if (!tile) { this.selectedLineageId = null; this.syncRendererState(); this.notifyChange(); return; }
        this.selectTile(tile);
        if (this.selectedCellInfo && this.selectedCellInfo.lineage_id != null) this.selectedLineageId = Number(this.selectedCellInfo.lineage_id) >>> 0;
        else this.selectedLineageId = null;
        this.syncRendererState();
        this.notifyChange();
    }

    applyBrush(tile, event) {
        if (!tile) return;
        const key = `${this.mode}:${tile.x},${tile.y}`;
        if (key === this.lastPaintKey) return;
        this.lastPaintKey = key;
        event.preventDefault();
        let mutation = null;
        try {
            if (this.mode === "edit-genome") {
                const patch = typeof this.genomePatch === "function" ? this.genomePatch() : null;
                if (!patch) throw new Error("Genome brush requires a genome patch draft.");
                const result = this.runtime.applyGenomeBrush(tile.x, tile.y, this.brush.width, this.brush.height, patch);
                mutation = { type: "genome-brush", tile: { x: tile.x, y: tile.y }, brush: { ...this.brush }, result };
            } else {
                this.runtime.applyEnvalBrush(tile.x, tile.y, this.brush.width, this.brush.height, this.brush.delta);
                mutation = { type: "enval-brush", tile: { x: tile.x, y: tile.y }, brush: { ...this.brush } };
            }
        } catch (error) {
            this.lastPaintKey = null;
            if (this.onError) this.onError(error);
            else throw error;
            return;
        }
        this.hoverTile = { x: tile.x, y: tile.y };
        this.hoverTileInfo = this.safeInspectTile(this.hoverTile);
        this.syncRendererState();
        if (this.onMutation) this.onMutation(mutation);
        this.notifyChange();
    }

    safeInspectTile(tile) {
        try { return this.inspectTile(tile); }
        catch (error) { return { kind: "tile-error", x: tile.x, y: tile.y, error: error.message }; }
    }

    safeInspectCell(cellId) {
        try { return this.inspectCell(cellId); }
        catch (error) { return { kind: "cell-error", cell_id: cellId, error: error.message }; }
    }

    syncRendererState() {
        const brushPreview = (this.mode === "edit" || this.mode === "edit-genome") && this.hoverTile ? {
            x: this.hoverTile.x, y: this.hoverTile.y,
            width: this.brush.width, height: this.brush.height
        } : null;
        this.renderer.setInteractionState({
            selectedLineageId: this.selectedLineageId,
            selectedCellId: this.selectedCellId,
            selectedTile: this.selectedTile,
            hoverTile: this.hoverTile,
            brushPreview
        });
    }

    notifyChange() {
        if (this.onChange) this.onChange(this.state);
    }
}
