const DEFAULT_BRIDGE_NAME = "microcosm";
const DEFAULT_MEMORY_EXPORT = "memory";

export const REQUIRED_MICROCOSM_EXPORTS = Object.freeze([
    DEFAULT_MEMORY_EXPORT,
    "microcosm_version_ptr",
    "microcosm_version_len",
    "microcosm_abi_version_ptr",
    "microcosm_abi_version_len",
    "microcosm_stats_size",
    "microcosm_last_error_ptr",
    "microcosm_last_error_len",
    "microcosm_alloc",
    "microcosm_free",
    "microcosm_create",
    "microcosm_destroy",
    "microcosm_reset",
    "microcosm_step",
    "microcosm_refresh_render_buffers",
    "microcosm_stats_ptr",
    "microcosm_tile_count",
    "microcosm_cell_count",
    "microcosm_render_epoch",
    "microcosm_query_result_ptr",
    "microcosm_query_result_len",
    "microcosm_inspect_tile",
    "microcosm_inspect_cell",
    "microcosm_set_tile_enval",
    "microcosm_adjust_tile_enval",
    "microcosm_brush_enval_rect",
    "microcosm_tile_enval_ptr",
    "microcosm_tile_occupancy_ptr",
    "microcosm_tile_mass_ptr",
    "microcosm_tile_molecule_count_ptr",
    "microcosm_tile_element_mask_ptr",
    "microcosm_cell_id_ptr",
    "microcosm_cell_x_ptr",
    "microcosm_cell_y_ptr",
    "microcosm_cell_energy_ptr",
    "microcosm_cell_lineage_ptr",
    "microcosm_cell_flags_ptr",
    "microcosm_cell_enzyme_count_ptr",
    "microcosm_cell_age_seconds_ptr",
    "microcosm_cell_attack_ptr",
    "microcosm_cell_defense_ptr"
]);

const isObject = (value) => typeof value === "object" && value !== null;

const normalizeWasmGPU = (WasmGPU) => {
    const candidate = WasmGPU && (WasmGPU.default || WasmGPU);
    if (!candidate || !candidate.webassembly) throw new Error("Microcosm bridge requires a WasmGPU object with a WebAssembly interop accessor.");
    return candidate;
};

const describeUrl = (url) => url instanceof URL ? url.href : String(url);

export const normalizeInstantiateResult = (result) => {
    if (result instanceof WebAssembly.Instance) return { instance: result, module: null, exports: result.exports };
    if (isObject(result) && result.instance instanceof WebAssembly.Instance) return { instance: result.instance, module: result.module || null, exports: result.instance.exports };
    if (isObject(result) && isObject(result.exports)) return { instance: result, module: null, exports: result.exports };
    throw new Error("Microcosm WebAssembly instantiation returned an unsupported result shape.");
};

export const loadMicrocosmWasm = async (wasmUrl, imports = {}) => {
    const response = await fetch(wasmUrl);
    if (!response.ok) throw new Error(`Failed to fetch Microcosm WebAssembly from ${describeUrl(wasmUrl)} (${response.status} ${response.statusText}).`);
    const contentType = response.headers.get("content-type") || "";
    if (WebAssembly.instantiateStreaming && contentType.includes("application/wasm")) {
        try { return await WebAssembly.instantiateStreaming(response, imports); }
        catch (error) { throw new Error(`Failed to instantiate Microcosm WebAssembly stream: ${error.message}`); }
    }
    const bytes = await response.arrayBuffer();
    try { return await WebAssembly.instantiate(bytes, imports); }
    catch (error) { throw new Error(`Failed to instantiate Microcosm WebAssembly bytes: ${error.message}`); }
};

export const assertMicrocosmExports = (exportsObject, requiredExports = REQUIRED_MICROCOSM_EXPORTS) => {
    const missing = [];
    for (const name of requiredExports) if (!Object.prototype.hasOwnProperty.call(exportsObject, name)) missing.push(name);
    if (missing.length > 0) throw new Error(`Microcosm WebAssembly is missing required export(s): ${missing.join(", ")}.`);
};

export const readExportedUtf8 = (module, ptrExport, lenExport, options = {}) => module.readUtf8(
    { export: ptrExport, args: options.args || [] },
    { export: lenExport, args: options.args || [] },
    { name: options.name || `${ptrExport}/${lenExport}`, fatal: options.fatal ?? false }
);

export const createExportView = (module, descriptor) => module.view({
    ptr: { export: descriptor.ptr, args: descriptor.args || [] },
    length: { export: descriptor.length, args: descriptor.args || [] },
    dtype: descriptor.dtype,
    name: descriptor.name
});

export const createMicrocosmBridge = async (options = {}) => {
    const WasmGPU = normalizeWasmGPU(options.WasmGPU || globalThis.WasmGPU);
    const wasmUrl = options.wasmUrl || new URL("../rust/microcosm.wasm", import.meta.url);
    const imports = options.imports || {};
    const name = options.name || DEFAULT_BRIDGE_NAME;
    const instantiateResult = options.instance ? options.instance : await loadMicrocosmWasm(wasmUrl, imports);
    const normalized = normalizeInstantiateResult(instantiateResult);
    const exportsObject = normalized.exports;
    assertMicrocosmExports(exportsObject, options.requiredExports || REQUIRED_MICROCOSM_EXPORTS);
    const memorySource = Object.prototype.hasOwnProperty.call(exportsObject, DEFAULT_MEMORY_EXPORT) ? DEFAULT_MEMORY_EXPORT : undefined;
    const module = normalized.instance instanceof WebAssembly.Instance
        ? WasmGPU.webassembly.fromInstance(normalized.instance, { memory: memorySource, name })
        : WasmGPU.webassembly.fromExports(exportsObject, { memory: memorySource, name });
    const memory = module.memory(memorySource);
    const getExport = (exportName) => module.getExport(exportName);
    const getFunction = (exportName) => module.getFunction(exportName);
    const hasExport = (exportName) => Object.prototype.hasOwnProperty.call(exportsObject, exportName);
    return {
        module,
        exports: exportsObject,
        instance: normalized.instance,
        memory,
        wasmUrl,
        getExport,
        getFunction,
        hasExport,
        readUtf8: (ptrDescriptor, lengthDescriptor, readOptions = {}) => module.readUtf8(ptrDescriptor, lengthDescriptor, readOptions),
        readExportedUtf8: (ptrExport, lenExport, readOptions = {}) => readExportedUtf8(module, ptrExport, lenExport, readOptions),
        view: (descriptor) => module.view(descriptor),
        createExportView: (descriptor) => createExportView(module, descriptor),
        dataView: (descriptor) => module.dataView(descriptor)
    };
};
