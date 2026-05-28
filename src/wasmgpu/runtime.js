import { createMicrocosmBridge, REQUIRED_MICROCOSM_EXPORTS } from "./bridge.js";

const STATUS = Object.freeze({
    OK: 0,
    INVALID_HANDLE: 1,
    NULL_POINTER: 2,
    CONFIG_ERROR: 3,
    WORLD_ERROR: 4,
    LOCK_ERROR: 5,
    ALLOC_ERROR: 6
});

const STATUS_LABELS = Object.freeze({
    [STATUS.OK]: "ok",
    [STATUS.INVALID_HANDLE]: "invalid handle",
    [STATUS.NULL_POINTER]: "null pointer",
    [STATUS.CONFIG_ERROR]: "config error",
    [STATUS.WORLD_ERROR]: "world error",
    [STATUS.LOCK_ERROR]: "runtime lock error",
    [STATUS.ALLOC_ERROR]: "allocation error"
});

const TILE_VIEW_DEFINITIONS = Object.freeze([
    { key: "tileEnval", ptr: "microcosm_tile_enval_ptr", length: "microcosm_tile_count", dtype: "f32", name: "microcosm.tile_enval" },
    { key: "tileOccupancy", ptr: "microcosm_tile_occupancy_ptr", length: "microcosm_tile_count", dtype: "u32", name: "microcosm.tile_occupancy" },
    { key: "tileMass", ptr: "microcosm_tile_mass_ptr", length: "microcosm_tile_count", dtype: "u32", name: "microcosm.tile_mass" },
    { key: "tileMoleculeCount", ptr: "microcosm_tile_molecule_count_ptr", length: "microcosm_tile_count", dtype: "u32", name: "microcosm.tile_molecule_count" },
    { key: "tileElementMask", ptr: "microcosm_tile_element_mask_ptr", length: "microcosm_tile_count", dtype: "u32", name: "microcosm.tile_element_mask" }
]);

const CELL_VIEW_DEFINITIONS = Object.freeze([
    { key: "cellId", ptr: "microcosm_cell_id_ptr", length: "microcosm_cell_count", dtype: "u32", name: "microcosm.cell_id" },
    { key: "cellX", ptr: "microcosm_cell_x_ptr", length: "microcosm_cell_count", dtype: "u32", name: "microcosm.cell_x" },
    { key: "cellY", ptr: "microcosm_cell_y_ptr", length: "microcosm_cell_count", dtype: "u32", name: "microcosm.cell_y" },
    { key: "cellEnergy", ptr: "microcosm_cell_energy_ptr", length: "microcosm_cell_count", dtype: "f32", name: "microcosm.cell_energy" },
    { key: "cellLineage", ptr: "microcosm_cell_lineage_ptr", length: "microcosm_cell_count", dtype: "u32", name: "microcosm.cell_lineage" },
    { key: "cellFlags", ptr: "microcosm_cell_flags_ptr", length: "microcosm_cell_count", dtype: "u32", name: "microcosm.cell_flags" },
    { key: "cellEnzymeCount", ptr: "microcosm_cell_enzyme_count_ptr", length: "microcosm_cell_count", dtype: "u32", name: "microcosm.cell_enzyme_count" },
    { key: "cellAgeSeconds", ptr: "microcosm_cell_age_seconds_ptr", length: "microcosm_cell_count", dtype: "f32", name: "microcosm.cell_age_seconds" },
    { key: "cellAttack", ptr: "microcosm_cell_attack_ptr", length: "microcosm_cell_count", dtype: "u32", name: "microcosm.cell_attack" },
    { key: "cellDefense", ptr: "microcosm_cell_defense_ptr", length: "microcosm_cell_count", dtype: "u32", name: "microcosm.cell_defense" }
]);

const VIEW_DEFINITIONS = Object.freeze([...TILE_VIEW_DEFINITIONS, ...CELL_VIEW_DEFINITIONS]);

const STATS_FIELD_TYPES = Object.freeze([
    ["tick_count", "u64"],
    ["sim_time_seconds", "f64"],
    ["width", "u32"],
    ["height", "u32"],
    ["tile_count", "u32"],
    ["occupied_tile_count", "u32"],
    ["empty_tile_count", "u32"],
    ["occupancy_fraction", "f64"],
    ["molecule_count", "u32"],
    ["tile_molecule_count", "u32"],
    ["cell_molecule_count", "u32"],
    ["free_molecule_record_count", "u32"],
    ["active_molecule_record_count", "u32"],
    ["molecule_arena_len", "u32"],
    ["molecule_arena_high_water_mark", "u32"],
    ["molecule_slots_reused", "u64"],
    ["molecule_slots_newly_allocated", "u64"],
    ["total_atom_count", "u64"],
    ["tile_atom_count", "u64"],
    ["cell_atom_count", "u64"],
    ["live_cell_count", "u32"],
    ["cell_record_count", "u32"],
    ["dead_cell_count", "u32"],
    ["births", "u64"],
    ["deaths", "u64"],
    ["predation_events", "u64"],
    ["cells_consumed", "u64"],
    ["predator_energy_gained", "f64"],
    ["predation_enzyme_transfers", "u64"],
    ["predation_enzyme_replacements", "u64"],
    ["lineage_count", "u32"],
    ["total_lineage_records", "u32"],
    ["extinct_lineage_count", "u32"],
    ["dominant_lineage_id", "u64"],
    ["dominant_lineage_population", "u64"],
    ["dominant_lineage_share", "f64"],
    ["average_cell_energy", "f64"],
    ["min_cell_energy", "f64"],
    ["max_cell_energy", "f64"],
    ["total_cell_energy", "f64"],
    ["average_enzyme_count", "f64"],
    ["min_enzyme_count", "u32"],
    ["max_enzyme_count", "u32"],
    ["cells_at_enzyme_cap", "u32"],
    ["fraction_cells_at_enzyme_cap", "f64"],
    ["enzyme_anabolase_count", "u64"],
    ["enzyme_catabolase_count", "u64"],
    ["enzyme_transmutase_count", "u64"],
    ["enzyme_defensase_count", "u64"],
    ["enzyme_attackase_count", "u64"],
    ["average_attack_total", "f64"],
    ["max_attack_total", "u32"],
    ["average_defense_total", "f64"],
    ["max_defense_total", "u32"],
    ["average_enval", "f32"],
    ["min_enval", "f32"],
    ["max_enval", "f32"],
    ["enval_std_dev", "f32"],
    ["enval_p05", "f32"],
    ["enval_p50", "f32"],
    ["enval_p95", "f32"],
    ["reaction_attempts", "u64"],
    ["reaction_gates_passed", "u64"],
    ["reaction_successes", "u64"],
    ["molecule_uptakes", "u64"],
    ["molecule_outputs", "u64"],
    ["divisions", "u64"],
    ["cell_steps", "u64"],
    ["enzyme_entries_seen", "u64"],
    ["metabolic_enzyme_attempts", "u64"],
    ["render_epoch", "u32"]
]);

const TYPE_LAYOUT = Object.freeze({
    u32: { size: 4, align: 4, read: (view, offset) => view.getUint32(offset, true) },
    f32: { size: 4, align: 4, read: (view, offset) => view.getFloat32(offset, true) },
    u64: { size: 8, align: 8, read: (view, offset) => normalizeU64(view.getBigUint64(offset, true)) },
    f64: { size: 8, align: 8, read: (view, offset) => view.getFloat64(offset, true) }
});

const MAX_SAFE_BIGINT = BigInt(Number.MAX_SAFE_INTEGER);

const normalizeU64 = (value) => value <= MAX_SAFE_BIGINT ? Number(value) : value;

const alignOffset = (offset, align) => (offset % align) === 0 ? offset : offset + align - (offset % align);

const buildStatsLayout = () => {
    let offset = 0;
    let maxAlign = 1;
    const fields = [];
    for (const [name, type] of STATS_FIELD_TYPES) {
        const layout = TYPE_LAYOUT[type];
        maxAlign = Math.max(maxAlign, layout.align);
        offset = alignOffset(offset, layout.align);
        fields.push({ name, type, offset, read: layout.read });
        offset += layout.size;
    }
    const byteLength = alignOffset(offset, maxAlign);
    return Object.freeze({ fields: Object.freeze(fields), byteLength });
};

const WASM_STATS_LAYOUT = buildStatsLayout();

const camelStatName = (name) => name.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase());

const readUintStatus = (value) => Number(value) >>> 0;

const assertPositiveInteger = (value, label) => { const number = Number(value); if (!Number.isInteger(number) || number <= 0) throw new Error(`${label} must be a positive integer.`); return number; };

const isPlainObject = (value) => typeof value === "object" && value !== null && !Array.isArray(value);

export class MicrocosmRuntime {
    static async create(options = {}) {
        const bridge = options.bridge || await createMicrocosmBridge(options);
        const runtime = new MicrocosmRuntime(bridge, options);
        runtime.validateAbi();
        if (options.createWorld !== false) runtime.createWorld(options.config || {});
        return runtime;
    }

    constructor(bridge, options = {}) {
        this.bridge = bridge;
        this.module = bridge.module;
        this.memory = bridge.memory;
        this._handle = 0;
        this._ready = false;
        this._destroyed = false;
        this._views = Object.create(null);
        this._stats = null;
        this._version = "";
        this._abiVersion = "";
        this._renderEpoch = 0;
        this._textEncoder = new TextEncoder();
        this._validateStatsSize = options.validateStatsSize !== false;
        this._functions = this._bindFunctions();
    }

    get ready() {
        return this._ready && !this._destroyed && this._handle !== 0;
    }

    get handle() {
        return this._handle;
    }

    get version() {
        return this._version;
    }

    get abiVersion() {
        return this._abiVersion;
    }

    get stats() {
        return this._stats;
    }

    get renderEpoch() {
        return this._renderEpoch;
    }

    get tileCount() {
        return this.ready ? Number(this._functions.tileCount(this._handle)) : 0;
    }

    get cellCount() {
        return this.ready ? Number(this._functions.cellCount(this._handle)) : 0;
    }

    get views() {
        return this._views;
    }

    validateAbi() {
        REQUIRED_MICROCOSM_EXPORTS.forEach((name) => this.bridge.getExport(name));
        this._version = this.readExportedString("microcosm_version_ptr", "microcosm_version_len", "microcosm.version");
        this._abiVersion = this.readExportedString("microcosm_abi_version_ptr", "microcosm_abi_version_len", "microcosm.abi_version");
        if (this._validateStatsSize) if (Number(this._functions.statsSize()) !== WASM_STATS_LAYOUT.byteLength) throw new Error(`Microcosm WasmStats layout mismatch: Rust reports ${Number(this._functions.statsSize())} bytes, JavaScript expects ${WASM_STATS_LAYOUT.byteLength} bytes.`);
    }

    createWorld(config = {}) {
        this.assertNotDestroyed();
        if (this._handle !== 0) this.destroy();
        this._destroyed = false;
        const handle = this.withConfigBytes(config, (ptr, len) => Number(this._functions.create(ptr, len)) >>> 0);
        if (handle === 0) throw new Error(`microcosm_create failed: ${this.readLastError() || "no Rust error was provided"}`);
        this._handle = handle;
        this._ready = true;
        this.createViews();
        this.refreshRenderBuffers();
        return this;
    }

    step(ticks = 1) {
        this.assertReady();
        const count = Math.max(0, Number(ticks) | 0);
        if (count > 0) this.assertStatus(readUintStatus(this._functions.step(this._handle, count)), "microcosm_step");
        this.refreshRenderBuffers();
        return this._stats;
    }

    refreshRenderBuffers() {
        this.assertReady();
        const status = readUintStatus(this._functions.refreshRenderBuffers(this._handle));
        this.assertStatus(status, "microcosm_refresh_render_buffers");
        this.refreshViews();
        this._renderEpoch = Number(this._functions.renderEpoch(this._handle)) >>> 0;
        this._stats = this.decodeStats();
        return this._stats;
    }

    refreshViews() {
        this.assertReady();
        for (const view of Object.values(this._views)) view.refresh();
        return this._views;
    }

    reset(config = {}) {
        this.assertNotDestroyed();
        if (!this.ready) return this.createWorld(config);
        const status = this.withConfigBytes(config, (ptr, len) => readUintStatus(this._functions.reset(this._handle, ptr, len)));
        this.assertStatus(status, "microcosm_reset");
        this.refreshRenderBuffers();
        return this;
    }

    destroy() {
        if (this._handle !== 0) this.assertStatus(readUintStatus(this._functions.destroy(this._handle)), "microcosm_destroy");
        this._handle = 0;
        this._ready = false;
        this._destroyed = true;
        this._views = Object.create(null);
        this._stats = null;
        this._renderEpoch = 0;
    }

    readLastError() {
        try { if (Number(this._functions.lastErrorLen()) <= 0) return ""; return this.readExportedString("microcosm_last_error_ptr", "microcosm_last_error_len", "microcosm.last_error"); }
        catch (error) { return `failed to read Rust last error: ${error.message}`; }
    }

    viewDiagnostics() {
        const expectedTileCount = this.tileCount, expectedCellCount = this.cellCount;
        return VIEW_DEFINITIONS.map((definition) => ({
                key: definition.key,
                name: definition.name,
                dtype: this._views[definition.key] ? this._views[definition.key].dtype : definition.dtype,
                ptr: this._views[definition.key] ? this._views[definition.key].ptr : 0,
                length: this._views[definition.key] ? this._views[definition.key].length : 0,
                byteLength: this._views[definition.key] ? this._views[definition.key].byteLength : 0,
                expectedLength: definition.length === "microcosm_tile_count" ? expectedTileCount : expectedCellCount,
                ok: Boolean(this._views[definition.key]) && this._views[definition.key].length === (definition.length === "microcosm_tile_count" ? expectedTileCount : expectedCellCount) && this._views[definition.key].dtype === definition.dtype
            })
        );
    }

    _bindFunctions() {
        const get = (name) => this.bridge.getFunction(name);
        return {
            versionPtr: get("microcosm_version_ptr"),
            versionLen: get("microcosm_version_len"),
            abiVersionPtr: get("microcosm_abi_version_ptr"),
            abiVersionLen: get("microcosm_abi_version_len"),
            statsSize: get("microcosm_stats_size"),
            lastErrorPtr: get("microcosm_last_error_ptr"),
            lastErrorLen: get("microcosm_last_error_len"),
            alloc: get("microcosm_alloc"),
            free: get("microcosm_free"),
            create: get("microcosm_create"),
            destroy: get("microcosm_destroy"),
            reset: get("microcosm_reset"),
            step: get("microcosm_step"),
            refreshRenderBuffers: get("microcosm_refresh_render_buffers"),
            statsPtr: get("microcosm_stats_ptr"),
            tileCount: get("microcosm_tile_count"),
            cellCount: get("microcosm_cell_count"),
            renderEpoch: get("microcosm_render_epoch")
        };
    }

    createViews() {
        this.assertReady();
        const views = Object.create(null);
        for (const definition of VIEW_DEFINITIONS) {
            views[definition.key] = this.bridge.view({
                ptr: { export: definition.ptr, args: [this._handle] },
                length: { export: definition.length, args: [this._handle] },
                dtype: definition.dtype,
                name: definition.name
            });
        }
        this._views = views;
        return views;
    }

    decodeStats() {
        this.assertReady();
        const statsPtr = Number(this._functions.statsPtr(this._handle));
        if (statsPtr === 0) throw new Error("microcosm_stats_ptr returned a null pointer.");
        const statsSize = Number(this._functions.statsSize());
        const dataView = this.bridge.dataView({ ptr: statsPtr, byteLength: statsSize, name: "microcosm.stats" });
        const stats = {};
        for (const field of WASM_STATS_LAYOUT.fields) {
            const value = field.read(dataView, field.offset);
            stats[field.name] = value;
            stats[camelStatName(field.name)] = value;
        }
        return stats;
    }

    readExportedString(ptrExport, lenExport, name) {
        return this.bridge.readUtf8({ export: ptrExport }, { export: lenExport }, { name });
    }

    withConfigBytes(config, callback) {
        if (!isPlainObject(config) || Object.keys(config).length === 0) return callback(0, 0);
        const bytes = this._textEncoder.encode(JSON.stringify(config));
        if (bytes.length === 0) return callback(0, 0);
        const ptr = Number(this._functions.alloc(bytes.length, 1));
        if (ptr === 0) throw new Error(`microcosm_alloc failed for ${bytes.length} config byte(s).`);
        try { new Uint8Array(this.memory.buffer, ptr, bytes.length).set(bytes); return callback(ptr, bytes.length); }
        finally { this._functions.free(ptr, bytes.length, 1); }
    }

    assertStatus(status, label) {
        if (status === STATUS.OK) return;
        const statusLabel = STATUS_LABELS[status] || "unknown status";
        const lastError = this.readLastError();
        throw new Error(`${label} failed with status ${status} (${statusLabel})${lastError ? `: ${lastError}` : "."}`);
    }

    assertReady() {
        this.assertNotDestroyed();
        assertPositiveInteger(this._handle, "MicrocosmRuntime handle");
        if (!this._ready) throw new Error("MicrocosmRuntime is not ready.");
    }

    assertNotDestroyed() {
        if (this._destroyed) throw new Error("MicrocosmRuntime has been destroyed.");
    }
}

export const MicrocosmStatus = STATUS;
export const MicrocosmViewDefinitions = VIEW_DEFINITIONS;
export const MicrocosmStatsLayout = WASM_STATS_LAYOUT;
