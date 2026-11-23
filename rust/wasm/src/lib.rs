use std::alloc::{Layout, alloc, dealloc};
use std::sync::{Mutex, OnceLock};

use microcosmcore::{
    CellBodyConfig, CellId, CellMotionConfig, Config, ElementFieldConfig, EnvalLandscapeConfig,
    EnvalRechargeConfig, GenomePatch, LineageId, Position, RenderBrushPreview, RenderBuffers,
    RenderDisplayMode, RenderVisualState, SurfaceConfig, VERSION, World, WorldStats,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

pub const ABI_VERSION: &str = VERSION;
pub const STATUS_OK: u32 = 0;
pub const STATUS_INVALID_HANDLE: u32 = 1;
pub const STATUS_NULL_POINTER: u32 = 2;
pub const STATUS_CONFIG_ERROR: u32 = 3;
pub const STATUS_WORLD_ERROR: u32 = 4;
pub const STATUS_LOCK_ERROR: u32 = 5;
pub const STATUS_ALLOC_ERROR: u32 = 6;

const DEFAULT_CELL_FLUX_LIMIT: usize = 32;
const DEFAULT_LINEAGE_LIMIT: usize = 64;
const MAX_CELL_FLUX_LIMIT: usize = 128;
const MAX_LINEAGE_LIMIT: usize = 512;
const VISUAL_HAS_SELECTED_LINEAGE: u32 = 1 << 0;
const VISUAL_HAS_SELECTED_CELL: u32 = 1 << 1;
const VISUAL_HAS_SELECTED_TILE: u32 = 1 << 2;
const VISUAL_HAS_HOVER_TILE: u32 = 1 << 3;
const VISUAL_HAS_BRUSH_PREVIEW: u32 = 1 << 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WasmStats {
    pub tick_count: u64,
    pub sim_time_seconds: f64,
    pub width: u32,
    pub height: u32,
    pub tile_count: u32,
    pub occupied_tile_count: u32,
    pub empty_tile_count: u32,
    pub occupancy_fraction: f64,
    pub extracellular_a: f64,
    pub extracellular_b: f64,
    pub extracellular_c: f64,
    pub extracellular_d: f64,
    pub extracellular_e: f64,
    pub extracellular_f: f64,
    pub intracellular_a: f64,
    pub intracellular_b: f64,
    pub intracellular_c: f64,
    pub intracellular_d: f64,
    pub intracellular_e: f64,
    pub intracellular_f: f64,
    pub system_a: f64,
    pub system_b: f64,
    pub system_c: f64,
    pub system_d: f64,
    pub system_e: f64,
    pub system_f: f64,
    pub total_element_amount: f64,
    pub live_cell_count: u32,
    pub births: u64,
    pub deaths: u64,
    pub fissions: u64,
    pub energy_deaths: u64,
    pub lyses: u64,
    pub radius_median: f32,
    pub radius_p95: f32,
    pub radius_max: f32,
    pub mean_strain: f64,
    pub lineage_count: u32,
    pub total_lineage_records: u32,
    pub extinct_lineage_count: u32,
    pub dominant_lineage_id: u64,
    pub dominant_lineage_population: u64,
    pub dominant_lineage_share: f64,
    pub average_cell_energy: f64,
    pub min_cell_energy: f64,
    pub max_cell_energy: f64,
    pub total_cell_energy: f64,
    pub average_enzyme_count: f64,
    pub min_enzyme_count: u32,
    pub max_enzyme_count: u32,
    pub cells_at_enzyme_cap: u32,
    pub fraction_cells_at_enzyme_cap: f64,
    pub total_enzyme_count: u64,
    pub average_enval: f32,
    pub min_enval: f32,
    pub max_enval: f32,
    pub enval_std_dev: f32,
    pub enval_p05: f32,
    pub enval_p50: f32,
    pub enval_p95: f32,
    pub reaction_attempts: u64,
    pub reaction_successes: u64,
    pub executed_metabolic_flux: f64,
    pub uptake_flux: f64,
    pub leak_flux: f64,
    pub secretion_flux: f64,
    pub divisions: u64,
    pub element_field_diffusion_tiles: u64,
    pub element_uptake_events: u64,
    pub cell_steps: u64,
    pub enzyme_entries_seen: u64,
    pub metabolic_enzyme_attempts: u64,
    pub spatial_candidate_checks: u64,
    pub overlap_candidates: u64,
    pub overlap_corrections: u64,
    pub energy_founder: f64,
    pub energy_injected: f64,
    pub energy_extracted: f64,
    pub energy_chemical_harvest: f64,
    pub energy_chemical_cost: f64,
    pub energy_enval_harvest: f64,
    pub energy_pump_cost: f64,
    pub energy_assembly_cost: f64,
    pub energy_maintenance: f64,
    pub energy_death_loss: f64,
    pub energy_ledger_residual: f64,
    pub renewable_coverage: f64,
    pub enval_source_inflow: f64,
    pub enval_cell_uptake: f64,
    pub enval_cell_emission: f64,
    pub enval_recharge: f64,
    pub enval_edits: f64,
    pub recharge_amount: f64,
    pub recharge_energy: f64,
    pub mean_exposure: f64,
    pub max_exposure: f32,
    pub exposed_cell_count: u32,
    pub contact_pairs: u64,
    pub cross_lineage_contact_pairs: u64,
    pub field_surface_extent: f64,
    pub cell_surface_extent: f64,
    pub boundary_digested: f64,
    pub cross_lineage_boundary_digested: f64,
    pub lyses_after_digestion: u64,
    pub mean_enval_optimum_spread: f64,
    pub mean_niche_mismatch: f64,
    pub mixed_polarity_cells: u32,
    pub render_epoch: u32,
}

impl From<&WorldStats> for WasmStats {
    fn from(stats: &WorldStats) -> Self {
        Self {
            tick_count: stats.tick_count,
            sim_time_seconds: stats.sim_time_seconds,
            width: clamp_usize_to_u32(stats.width),
            height: clamp_usize_to_u32(stats.height),
            tile_count: clamp_usize_to_u32(stats.tile_count),
            occupied_tile_count: clamp_usize_to_u32(stats.occupied_tile_count),
            empty_tile_count: clamp_usize_to_u32(stats.empty_tile_count),
            occupancy_fraction: stats.occupancy_fraction,
            extracellular_a: stats.extracellular_element_amounts[0],
            extracellular_b: stats.extracellular_element_amounts[1],
            extracellular_c: stats.extracellular_element_amounts[2],
            extracellular_d: stats.extracellular_element_amounts[3],
            extracellular_e: stats.extracellular_element_amounts[4],
            extracellular_f: stats.extracellular_element_amounts[5],
            intracellular_a: stats.intracellular_element_amounts[0],
            intracellular_b: stats.intracellular_element_amounts[1],
            intracellular_c: stats.intracellular_element_amounts[2],
            intracellular_d: stats.intracellular_element_amounts[3],
            intracellular_e: stats.intracellular_element_amounts[4],
            intracellular_f: stats.intracellular_element_amounts[5],
            system_a: stats.system_element_amounts[0],
            system_b: stats.system_element_amounts[1],
            system_c: stats.system_element_amounts[2],
            system_d: stats.system_element_amounts[3],
            system_e: stats.system_element_amounts[4],
            system_f: stats.system_element_amounts[5],
            total_element_amount: stats.total_element_amount,
            live_cell_count: clamp_usize_to_u32(stats.live_cell_count),
            births: stats.births,
            deaths: stats.deaths,
            fissions: stats.fissions,
            energy_deaths: stats.energy_deaths,
            lyses: stats.lyses,
            radius_median: stats.radius_median,
            radius_p95: stats.radius_p95,
            radius_max: stats.radius_max,
            mean_strain: stats.mean_strain,
            lineage_count: clamp_usize_to_u32(stats.lineage_count),
            total_lineage_records: clamp_usize_to_u32(stats.total_lineage_records),
            extinct_lineage_count: clamp_usize_to_u32(stats.extinct_lineage_count),
            dominant_lineage_id: stats.dominant_lineage_id,
            dominant_lineage_population: stats.dominant_lineage_population,
            dominant_lineage_share: stats.dominant_lineage_share,
            average_cell_energy: stats.average_cell_energy,
            min_cell_energy: stats.min_cell_energy,
            max_cell_energy: stats.max_cell_energy,
            total_cell_energy: stats.total_cell_energy,
            average_enzyme_count: stats.average_enzyme_count,
            min_enzyme_count: clamp_usize_to_u32(stats.min_enzyme_count),
            max_enzyme_count: clamp_usize_to_u32(stats.max_enzyme_count),
            cells_at_enzyme_cap: clamp_usize_to_u32(stats.cells_at_enzyme_cap),
            fraction_cells_at_enzyme_cap: stats.fraction_cells_at_enzyme_cap,
            total_enzyme_count: stats.total_enzyme_count,
            average_enval: stats.average_enval,
            min_enval: stats.min_enval,
            max_enval: stats.max_enval,
            enval_std_dev: stats.enval_std_dev,
            enval_p05: stats.enval_p05,
            enval_p50: stats.enval_p50,
            enval_p95: stats.enval_p95,
            reaction_attempts: stats.reaction_counters.total_attempts(),
            reaction_successes: stats.reaction_counters.total_successes(),
            executed_metabolic_flux: stats.reaction_counters.executed_metabolic_flux,
            uptake_flux: stats.reaction_counters.uptake_flux,
            leak_flux: stats.reaction_counters.leak_flux,
            secretion_flux: stats.reaction_counters.secretion_flux,
            divisions: stats.reaction_counters.divisions,
            element_field_diffusion_tiles: stats.operation_counters.element_field_diffusion_tiles,
            element_uptake_events: stats.operation_counters.element_uptake_events,
            cell_steps: stats.operation_counters.cell_steps,
            enzyme_entries_seen: stats.operation_counters.enzyme_entries_seen,
            metabolic_enzyme_attempts: stats.operation_counters.metabolic_enzyme_attempts,
            spatial_candidate_checks: stats.operation_counters.spatial_candidate_checks,
            overlap_candidates: stats.operation_counters.overlap_candidates,
            overlap_corrections: stats.operation_counters.overlap_corrections,
            energy_founder: stats.energy_ledger.founder_energy,
            energy_injected: stats.energy_ledger.injected_energy,
            energy_extracted: stats.energy_ledger.extracted_energy,
            energy_chemical_harvest: stats.energy_ledger.chemical_harvest,
            energy_chemical_cost: stats.energy_ledger.chemical_cost,
            energy_enval_harvest: stats.energy_ledger.enval_harvest,
            energy_pump_cost: stats.energy_ledger.pump_cost,
            energy_assembly_cost: stats.energy_ledger.assembly_cost,
            energy_maintenance: stats.energy_ledger.maintenance,
            energy_death_loss: stats.energy_ledger.death_loss,
            energy_ledger_residual: stats.energy_ledger_residual,
            renewable_coverage: stats.renewable_coverage,
            enval_source_inflow: stats.enval_ledger.source_inflow,
            enval_cell_uptake: stats.enval_ledger.cell_uptake,
            enval_cell_emission: stats.enval_ledger.cell_emission,
            enval_recharge: stats.enval_ledger.recharge,
            enval_edits: stats.enval_ledger.edits,
            recharge_amount: stats.enval_ledger.recharge_amount,
            recharge_energy: stats.enval_ledger.recharge_energy,
            mean_exposure: stats.mean_exposure,
            max_exposure: stats.max_exposure,
            exposed_cell_count: clamp_usize_to_u32(stats.exposed_cell_count),
            contact_pairs: stats.contact_pairs,
            cross_lineage_contact_pairs: stats.cross_lineage_contact_pairs,
            field_surface_extent: stats.interaction.field_surface_extent,
            cell_surface_extent: stats.interaction.cell_surface_extent,
            boundary_digested: stats.interaction.boundary_digested,
            cross_lineage_boundary_digested: stats.interaction.cross_lineage_boundary_digested,
            lyses_after_digestion: stats.interaction.lyses_after_digestion,
            mean_enval_optimum_spread: stats.mean_enval_optimum_spread,
            mean_niche_mismatch: stats.mean_niche_mismatch,
            mixed_polarity_cells: clamp_usize_to_u32(stats.mixed_polarity_cells),
            render_epoch: (stats.tick_count & u64::from(u32::MAX)) as u32,
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
struct WasmInitConfig {
    cell_body: Option<CellBodyConfig>,
    founder_energy: Option<f64>,
    seed: Option<String>,
    width: Option<usize>,
    height: Option<usize>,
    initial_cells: Option<usize>,
    dt_seconds: Option<f64>,
    enval_diffusivity_per_second: Option<f32>,
    element_fields: Option<ElementFieldConfig>,
    membrane_permeability: Option<f32>,
    catalyst_upkeep_per_sec: Option<f64>,
    enval_landscape: Option<EnvalLandscapeConfig>,
    enval_recharge: Option<EnvalRechargeConfig>,
    cell_motion: Option<CellMotionConfig>,
    surface: Option<SurfaceConfig>,
}

impl WasmInitConfig {
    fn into_core_config(self) -> Config {
        let mut config = Config::default();
        if let Some(cell_body) = self.cell_body {
            config.cell_body = cell_body;
        }
        if let Some(energy) = self.founder_energy {
            config.founder_energy = energy;
        }
        if let Some(seed) = self.seed {
            config.seed = seed;
        }
        if let Some(width) = self.width {
            config.width = width;
        }
        if let Some(height) = self.height {
            config.height = height;
        }
        if let Some(initial_cells) = self.initial_cells {
            config.initial_founder_count = initial_cells;
        }
        if let Some(dt_seconds) = self.dt_seconds {
            config.dt_seconds = dt_seconds;
        }
        if let Some(diffusivity) = self.enval_diffusivity_per_second {
            config.enval_diffusivity_per_second = diffusivity;
        }
        if let Some(element_fields) = self.element_fields {
            config.element_fields = element_fields;
        }
        if let Some(permeability) = self.membrane_permeability {
            config.membrane_permeability = permeability;
        }
        if let Some(upkeep) = self.catalyst_upkeep_per_sec {
            config.catalyst_upkeep_per_sec = upkeep;
        }
        if let Some(landscape) = self.enval_landscape {
            config.enval_landscape = landscape;
        }
        if let Some(recharge) = self.enval_recharge {
            config.enval_recharge = recharge;
        }
        if let Some(cell_motion) = self.cell_motion {
            config.cell_motion = cell_motion;
        }
        if let Some(surface) = self.surface {
            config.surface = surface;
        }
        config
    }
}

struct WasmInstance {
    world: World,
    stats: WasmStats,
    render_buffers: RenderBuffers,
    render_visual_state: RenderVisualState,
}

impl WasmInstance {
    fn new(mut world: World) -> Self {
        world.rebuild_derived_caches();
        let stats = WasmStats::from(&world.stats());
        let mut render_buffers = RenderBuffers::default();
        let render_visual_state = RenderVisualState::default();
        world.write_render_buffers_with_visual(&mut render_buffers, &render_visual_state);
        Self {
            world,
            stats,
            render_buffers,
            render_visual_state,
        }
    }

    fn refresh_stats(&mut self) {
        self.stats = WasmStats::from(&self.world.stats());
        self.stats.render_epoch = self.render_buffers.render_epoch;
    }

    fn refresh_render_buffers(&mut self) {
        self.world
            .write_render_buffers_with_visual(&mut self.render_buffers, &self.render_visual_state);
    }

    fn refresh_lattice_rgba(&mut self) {
        self.render_buffers
            .refresh_lattice_rgba(&self.render_visual_state);
    }
}

#[derive(Default)]
struct WasmRuntime {
    instances: Vec<Option<Box<WasmInstance>>>,
    last_error: Vec<u8>,
    query_result: Vec<u8>,
}

impl WasmRuntime {
    fn insert_instance(&mut self, instance: WasmInstance) -> u32 {
        for (index, slot) in self.instances.iter_mut().enumerate() {
            if slot.is_none() {
                *slot = Some(Box::new(instance));
                return (index + 1) as u32;
            }
        }
        self.instances.push(Some(Box::new(instance)));
        self.instances.len() as u32
    }

    fn instance(&self, handle: u32) -> Option<&WasmInstance> {
        let index = handle.checked_sub(1)? as usize;
        self.instances.get(index)?.as_deref()
    }

    fn instance_mut(&mut self, handle: u32) -> Option<&mut WasmInstance> {
        let index = handle.checked_sub(1)? as usize;
        self.instances.get_mut(index)?.as_deref_mut()
    }
}

static RUNTIME: OnceLock<Mutex<WasmRuntime>> = OnceLock::new();

fn runtime() -> &'static Mutex<WasmRuntime> {
    RUNTIME.get_or_init(|| Mutex::new(WasmRuntime::default()))
}

fn lock_runtime() -> Result<std::sync::MutexGuard<'static, WasmRuntime>, u32> {
    runtime().lock().map_err(|_| STATUS_LOCK_ERROR)
}

fn parse_config_from_bytes(ptr: *const u8, len: usize) -> Result<Config, String> {
    if len == 0 {
        return Ok(Config::default());
    }
    if ptr.is_null() {
        return Err("nonzero config length with null pointer".to_owned());
    }
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    let init: WasmInitConfig = serde_json::from_slice(bytes).map_err(|err| err.to_string())?;
    Ok(init.into_core_config())
}

fn parse_genome_patch_from_bytes(ptr: *const u8, len: usize) -> Result<GenomePatch, String> {
    if len == 0 {
        return Err("genome patch payload is empty".to_owned());
    }
    if ptr.is_null() {
        return Err("nonzero genome patch length with null pointer".to_owned());
    }
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    serde_json::from_slice(bytes).map_err(|err| err.to_string())
}

fn create_instance(config: Config) -> Result<WasmInstance, String> {
    config.validate().map_err(|err| err.to_string())?;
    let initial_cells = config.initial_founder_count;
    let mut world = World::new(config).map_err(|err| err.to_string())?;
    if initial_cells > 0 {
        world
            .spawn_founder_cells(initial_cells)
            .map_err(|err| err.to_string())?;
    }
    Ok(WasmInstance::new(world))
}

fn set_last_error(status: u32, message: impl Into<String>) -> u32 {
    match lock_runtime() {
        Ok(mut runtime) => {
            runtime.last_error = message.into().into_bytes();
            status
        }
        Err(err) => err,
    }
}

fn set_runtime_error(runtime: &mut WasmRuntime, status: u32, message: impl Into<String>) -> u32 {
    runtime.last_error = message.into().into_bytes();
    runtime.query_result.clear();
    status
}

fn set_query_result(runtime: &mut WasmRuntime, value: serde_json::Value) -> u32 {
    match serde_json::to_vec(&value) {
        Ok(bytes) => {
            runtime.query_result = bytes;
            runtime.last_error.clear();
            STATUS_OK
        }
        Err(err) => set_runtime_error(runtime, STATUS_WORLD_ERROR, err.to_string()),
    }
}

fn set_serialized_query_result<T: Serialize>(runtime: &mut WasmRuntime, value: T) -> u32 {
    match serde_json::to_vec(&value) {
        Ok(bytes) => {
            runtime.query_result = bytes;
            runtime.last_error.clear();
            STATUS_OK
        }
        Err(err) => set_runtime_error(runtime, STATUS_WORLD_ERROR, err.to_string()),
    }
}

fn clamp_query_limit(value: u32, default_value: usize, max_value: usize) -> usize {
    if value == 0 {
        default_value
    } else {
        (value as usize).min(max_value)
    }
}

fn clamp_usize_to_u32(value: usize) -> u32 {
    value.min(u32::MAX as usize) as u32
}

#[unsafe(no_mangle)]
pub extern "C" fn microcosm_version_ptr() -> *const u8 {
    VERSION.as_ptr()
}

#[unsafe(no_mangle)]
pub extern "C" fn microcosm_version_len() -> usize {
    VERSION.len()
}

#[unsafe(no_mangle)]
pub extern "C" fn microcosm_abi_version_ptr() -> *const u8 {
    ABI_VERSION.as_ptr()
}

#[unsafe(no_mangle)]
pub extern "C" fn microcosm_abi_version_len() -> usize {
    ABI_VERSION.len()
}

#[unsafe(no_mangle)]
pub extern "C" fn microcosm_stats_size() -> usize {
    std::mem::size_of::<WasmStats>()
}

#[unsafe(no_mangle)]
pub extern "C" fn microcosm_last_error_ptr() -> *const u8 {
    match lock_runtime() {
        Ok(runtime) => runtime.last_error.as_ptr(),
        Err(_) => std::ptr::null(),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn microcosm_last_error_len() -> usize {
    match lock_runtime() {
        Ok(runtime) => runtime.last_error.len(),
        Err(_) => 0,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn microcosm_query_result_ptr() -> *const u8 {
    match lock_runtime() {
        Ok(runtime) => runtime.query_result.as_ptr(),
        Err(_) => std::ptr::null(),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn microcosm_query_result_len() -> usize {
    match lock_runtime() {
        Ok(runtime) => runtime.query_result.len(),
        Err(_) => 0,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn microcosm_alloc(len: usize, align: usize) -> *mut u8 {
    if len == 0 || align == 0 || !align.is_power_of_two() {
        return std::ptr::null_mut();
    }
    let Ok(layout) = Layout::from_size_align(len, align) else {
        return std::ptr::null_mut();
    };
    unsafe { alloc(layout) }
}

/// # Safety
/// `ptr` must be null or a pointer returned by `microcosm_alloc` with the same
/// `len` and `align`, and it must not be used or freed again afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn microcosm_free(ptr: *mut u8, len: usize, align: usize) {
    if ptr.is_null() || len == 0 || align == 0 || !align.is_power_of_two() {
        return;
    }
    if let Ok(layout) = Layout::from_size_align(len, align) {
        unsafe {
            dealloc(ptr, layout);
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn microcosm_create(config_ptr: *const u8, config_len: usize) -> u32 {
    let config = match parse_config_from_bytes(config_ptr, config_len) {
        Ok(config) => config,
        Err(err) => {
            let _ = set_last_error(STATUS_CONFIG_ERROR, err);
            return 0;
        }
    };
    let instance = match create_instance(config) {
        Ok(instance) => instance,
        Err(err) => {
            let _ = set_last_error(STATUS_WORLD_ERROR, err);
            return 0;
        }
    };
    match lock_runtime() {
        Ok(mut runtime) => {
            runtime.last_error.clear();
            runtime.insert_instance(instance)
        }
        Err(_) => 0,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn microcosm_destroy(handle: u32) -> u32 {
    match lock_runtime() {
        Ok(mut runtime) => {
            let Some(index) = handle.checked_sub(1).map(|value| value as usize) else {
                return STATUS_INVALID_HANDLE;
            };
            let Some(slot) = runtime.instances.get_mut(index) else {
                return STATUS_INVALID_HANDLE;
            };
            *slot = None;
            STATUS_OK
        }
        Err(err) => err,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn microcosm_reset(handle: u32, config_ptr: *const u8, config_len: usize) -> u32 {
    let config = match parse_config_from_bytes(config_ptr, config_len) {
        Ok(config) => config,
        Err(err) => return set_last_error(STATUS_CONFIG_ERROR, err),
    };
    let instance = match create_instance(config) {
        Ok(instance) => instance,
        Err(err) => return set_last_error(STATUS_WORLD_ERROR, err),
    };
    match lock_runtime() {
        Ok(mut runtime) => {
            let Some(slot) = handle
                .checked_sub(1)
                .and_then(|index| runtime.instances.get_mut(index as usize))
            else {
                return STATUS_INVALID_HANDLE;
            };
            if slot.is_none() {
                return STATUS_INVALID_HANDLE;
            }
            *slot = Some(Box::new(instance));
            runtime.last_error.clear();
            STATUS_OK
        }
        Err(err) => err,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn microcosm_step_no_stats(handle: u32, ticks: u32) -> u32 {
    match lock_runtime() {
        Ok(mut runtime) => {
            let Some(instance) = runtime.instance_mut(handle) else {
                return STATUS_INVALID_HANDLE;
            };
            instance.world.step_many(ticks);
            runtime.last_error.clear();
            STATUS_OK
        }
        Err(err) => err,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn microcosm_refresh_render_buffers(handle: u32) -> u32 {
    match lock_runtime() {
        Ok(mut runtime) => {
            let Some(instance) = runtime.instance_mut(handle) else {
                return STATUS_INVALID_HANDLE;
            };
            instance.refresh_render_buffers();
            runtime.last_error.clear();
            STATUS_OK
        }
        Err(err) => err,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn microcosm_refresh_stats(handle: u32) -> u32 {
    match lock_runtime() {
        Ok(mut runtime) => {
            let Some(instance) = runtime.instance_mut(handle) else {
                return STATUS_INVALID_HANDLE;
            };
            instance.refresh_stats();
            runtime.last_error.clear();
            STATUS_OK
        }
        Err(err) => err,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn microcosm_set_render_visual_state(
    handle: u32,
    display_mode: u32,
    flags: u32,
    selected_lineage: u32,
    selected_cell: u32,
    selected_tile_x: u32,
    selected_tile_y: u32,
    hover_tile_x: u32,
    hover_tile_y: u32,
    brush_x: u32,
    brush_y: u32,
    brush_width: u32,
    brush_height: u32,
) -> u32 {
    let Some(display_mode) = RenderDisplayMode::from_u32(display_mode) else {
        return set_last_error(
            STATUS_CONFIG_ERROR,
            format!("unsupported render display mode {display_mode}"),
        );
    };
    match lock_runtime() {
        Ok(mut runtime) => {
            let Some(instance) = runtime.instance_mut(handle) else {
                return STATUS_INVALID_HANDLE;
            };
            instance.render_visual_state = RenderVisualState {
                display_mode,
                selected_lineage: (flags & VISUAL_HAS_SELECTED_LINEAGE != 0)
                    .then_some(selected_lineage),
                selected_cell: (flags & VISUAL_HAS_SELECTED_CELL != 0).then_some(selected_cell),
                selected_tile: (flags & VISUAL_HAS_SELECTED_TILE != 0)
                    .then_some((selected_tile_x, selected_tile_y)),
                hover_tile: (flags & VISUAL_HAS_HOVER_TILE != 0)
                    .then_some((hover_tile_x, hover_tile_y)),
                brush_preview: (flags & VISUAL_HAS_BRUSH_PREVIEW != 0).then_some(
                    RenderBrushPreview {
                        x: brush_x,
                        y: brush_y,
                        width: brush_width.max(1),
                        height: brush_height.max(1),
                    },
                ),
            };
            instance.refresh_lattice_rgba();
            runtime.last_error.clear();
            STATUS_OK
        }
        Err(err) => err,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn microcosm_stats_ptr(handle: u32) -> *const WasmStats {
    match lock_runtime() {
        Ok(runtime) => runtime
            .instance(handle)
            .map(|instance| &instance.stats as *const WasmStats)
            .unwrap_or(std::ptr::null()),
        Err(_) => std::ptr::null(),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn microcosm_tile_count(handle: u32) -> u32 {
    match lock_runtime() {
        Ok(runtime) => runtime
            .instance(handle)
            .map(|instance| clamp_usize_to_u32(instance.render_buffers.tile_count()))
            .unwrap_or(0),
        Err(_) => 0,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn microcosm_cell_count(handle: u32) -> u32 {
    match lock_runtime() {
        Ok(runtime) => runtime
            .instance(handle)
            .map(|instance| clamp_usize_to_u32(instance.render_buffers.cell_count()))
            .unwrap_or(0),
        Err(_) => 0,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn microcosm_render_epoch(handle: u32) -> u32 {
    match lock_runtime() {
        Ok(runtime) => runtime
            .instance(handle)
            .map(|instance| instance.render_buffers.render_epoch)
            .unwrap_or(0),
        Err(_) => 0,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn microcosm_inspect_tile(handle: u32, x: u32, y: u32) -> u32 {
    match lock_runtime() {
        Ok(mut runtime) => {
            let Some(instance) = runtime.instance(handle) else {
                return STATUS_INVALID_HANDLE;
            };
            let Some(info) = instance.world.inspect_tile_xy(x as usize, y as usize) else {
                return set_runtime_error(
                    &mut runtime,
                    STATUS_WORLD_ERROR,
                    format!("no tile at ({x}, {y})"),
                );
            };
            set_query_result(
                &mut runtime,
                json!({
                    "kind": "tile",
                    "tile_id": info.tile_id.index(),
                    "x": info.x,
                    "y": info.y,
                    "enval": info.enval,
                    "enval_landscape_target": info.enval_landscape_target,
                    "cell_center_count": info.cell_center_count,
                    "element_concentrations": {
                        "A": info.element_concentrations[0],
                        "B": info.element_concentrations[1],
                        "C": info.element_concentrations[2],
                        "D": info.element_concentrations[3],
                        "E": info.element_concentrations[4],
                        "F": info.element_concentrations[5]
                    },
                    "total_element_concentration": info.total_element_concentration,
                    "mass_density": info.mass_density
                }),
            )
        }
        Err(err) => err,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn microcosm_inspect_cell(handle: u32, cell_id: u32) -> u32 {
    match lock_runtime() {
        Ok(mut runtime) => {
            let Some(instance) = runtime.instance(handle) else {
                return STATUS_INVALID_HANDLE;
            };
            let Some(info) = instance.world.inspect_cell(CellId(cell_id as usize)) else {
                return set_runtime_error(
                    &mut runtime,
                    STATUS_WORLD_ERROR,
                    format!("no active cell with id {cell_id}"),
                );
            };
            set_query_result(
                &mut runtime,
                json!({
                    "kind": "cell",
                    "cell_id": info.cell_id.index(),
                    "x": info.x,
                    "y": info.y,
                    "radius": info.radius,
                    "energy": info.energy,
                    "lineage_id": info.lineage_id.raw(),
                    "enzyme_count": info.enzyme_count,
                    "total_internal_elements": info.total_internal_elements,
                    "age_seconds": info.age_seconds,
                    "mean_enval_optimum": info.mean_enval_optimum,
                    "enval_optimum_spread": info.enval_optimum_spread,
                    "local_enval_average": info.local_enval_average,
                    "body_total": info.body_total,
                    "boundary_total": info.boundary_total,
                    "area": info.area,
                    "strain": info.strain,
                    "membrane_polarity": info.membrane_polarity,
                    "field_share": info.field_share,
                    "contacts": info.contacts
                }),
            )
        }
        Err(err) => err,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn microcosm_pick_cell(handle: u32, x: f32, y: f32) -> u32 {
    match lock_runtime() {
        Ok(mut runtime) => {
            let Some(instance) = runtime.instance(handle) else {
                return STATUS_INVALID_HANDLE;
            };
            if !x.is_finite() || !y.is_finite() {
                return set_runtime_error(
                    &mut runtime,
                    STATUS_CONFIG_ERROR,
                    format!("cell pick position must be finite, got ({x}, {y})"),
                );
            }
            let cell_id = instance
                .world
                .pick_cell(Position::new(x, y))
                .map(CellId::index);
            set_query_result(
                &mut runtime,
                json!({
                    "schema": "microcosm.cell_pick.v1",
                    "kind": "cell_pick",
                    "x": x,
                    "y": y,
                    "cell_id": cell_id
                }),
            )
        }
        Err(err) => err,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn microcosm_inspect_cell_detail(handle: u32, cell_id: u32, flux_limit: u32) -> u32 {
    match lock_runtime() {
        Ok(mut runtime) => {
            let Some(instance) = runtime.instance_mut(handle) else {
                return STATUS_INVALID_HANDLE;
            };
            let flux_limit =
                clamp_query_limit(flux_limit, DEFAULT_CELL_FLUX_LIMIT, MAX_CELL_FLUX_LIMIT);
            let Some(info) = instance
                .world
                .inspect_cell_detail(CellId(cell_id as usize), flux_limit)
            else {
                return set_runtime_error(
                    &mut runtime,
                    STATUS_WORLD_ERROR,
                    format!("no active cell with id {cell_id}"),
                );
            };
            set_serialized_query_result(
                &mut runtime,
                json!({
                    "schema": "microcosm.cell_detail.v1",
                    "kind": "cell_detail",
                    "cell_detail": info
                }),
            )
        }
        Err(err) => err,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn microcosm_inspect_cell_elements(handle: u32, cell_id: u32) -> u32 {
    match lock_runtime() {
        Ok(mut runtime) => {
            let Some(instance) = runtime.instance(handle) else {
                return STATUS_INVALID_HANDLE;
            };
            let Some(info) = instance.world.inspect_cell(CellId(cell_id as usize)) else {
                return set_runtime_error(
                    &mut runtime,
                    STATUS_WORLD_ERROR,
                    format!("no active cell with id {cell_id}"),
                );
            };
            let internal_elements = instance
                .world
                .cell(CellId(cell_id as usize))
                .map(|cell| *cell.internal_elements.as_array())
                .unwrap_or_default();
            set_serialized_query_result(
                &mut runtime,
                json!({
                    "schema": "microcosm.cell_elements.v1",
                    "kind": "cell_elements",
                    "cell_id": cell_id,
                    "internal_elements": internal_elements,
                    "total_internal_elements": info.total_internal_elements
                }),
            )
        }
        Err(err) => err,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn microcosm_inspect_cell_fluxes(handle: u32, cell_id: u32, flux_limit: u32) -> u32 {
    match lock_runtime() {
        Ok(mut runtime) => {
            let Some(instance) = runtime.instance_mut(handle) else {
                return STATUS_INVALID_HANDLE;
            };
            let flux_limit =
                clamp_query_limit(flux_limit, DEFAULT_CELL_FLUX_LIMIT, MAX_CELL_FLUX_LIMIT);
            let Some(info) = instance
                .world
                .inspect_cell_fluxes(CellId(cell_id as usize), flux_limit)
            else {
                return set_runtime_error(
                    &mut runtime,
                    STATUS_WORLD_ERROR,
                    format!("no active cell with id {cell_id}"),
                );
            };
            set_serialized_query_result(
                &mut runtime,
                json!({
                    "schema": "microcosm.cell_fluxes.v1",
                    "kind": "cell_fluxes",
                    "cell_id": cell_id,
                    "recent_fluxes": info
                }),
            )
        }
        Err(err) => err,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn microcosm_inspect_lineage(handle: u32, lineage_id: u32) -> u32 {
    match lock_runtime() {
        Ok(mut runtime) => {
            let Some(instance) = runtime.instance(handle) else {
                return STATUS_INVALID_HANDLE;
            };
            let Some(info) = instance
                .world
                .inspect_lineage(LineageId(u64::from(lineage_id)))
            else {
                return set_runtime_error(
                    &mut runtime,
                    STATUS_WORLD_ERROR,
                    format!("no lineage with id {lineage_id}"),
                );
            };
            set_serialized_query_result(
                &mut runtime,
                json!({
                    "schema": "microcosm.lineage.v1",
                    "kind": "lineage",
                    "lineage": info
                }),
            )
        }
        Err(err) => err,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn microcosm_list_lineages(handle: u32, limit: u32) -> u32 {
    match lock_runtime() {
        Ok(mut runtime) => {
            let Some(instance) = runtime.instance(handle) else {
                return STATUS_INVALID_HANDLE;
            };
            let limit = clamp_query_limit(limit, DEFAULT_LINEAGE_LIMIT, MAX_LINEAGE_LIMIT);
            let info = instance.world.list_lineages(limit);
            set_serialized_query_result(
                &mut runtime,
                json!({
                    "schema": "microcosm.lineage_list.v1",
                    "kind": "lineage_list",
                    "lineage_list": info
                }),
            )
        }
        Err(err) => err,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn microcosm_apply_cell_genome_patch(
    handle: u32,
    cell_id: u32,
    patch_ptr: *const u8,
    patch_len: usize,
) -> u32 {
    let patch = match parse_genome_patch_from_bytes(patch_ptr, patch_len) {
        Ok(patch) => patch,
        Err(err) => return set_last_error(STATUS_CONFIG_ERROR, err),
    };
    match lock_runtime() {
        Ok(mut runtime) => {
            let result = {
                let Some(instance) = runtime.instance_mut(handle) else {
                    return STATUS_INVALID_HANDLE;
                };
                let result = instance
                    .world
                    .apply_cell_genome_patch(CellId(cell_id as usize), &patch)
                    .map_err(|err| err.to_string());
                if result.is_ok() {
                    instance.refresh_render_buffers();
                    instance.refresh_stats();
                }
                result
            };
            match result {
                Ok(info) => set_serialized_query_result(
                    &mut runtime,
                    json!({
                        "schema": "microcosm.genome_edit_result.v1",
                        "kind": "genome_edit_result",
                        "result": info
                    }),
                ),
                Err(err) => set_runtime_error(&mut runtime, STATUS_WORLD_ERROR, err),
            }
        }
        Err(err) => err,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn microcosm_apply_genome_brush(
    handle: u32,
    center_x: u32,
    center_y: u32,
    brush_width: u32,
    brush_height: u32,
    patch_ptr: *const u8,
    patch_len: usize,
) -> u32 {
    let patch = match parse_genome_patch_from_bytes(patch_ptr, patch_len) {
        Ok(patch) => patch,
        Err(err) => return set_last_error(STATUS_CONFIG_ERROR, err),
    };
    match lock_runtime() {
        Ok(mut runtime) => {
            let result = {
                let Some(instance) = runtime.instance_mut(handle) else {
                    return STATUS_INVALID_HANDLE;
                };
                let result = instance
                    .world
                    .apply_genome_brush(
                        center_x as usize,
                        center_y as usize,
                        brush_width as usize,
                        brush_height as usize,
                        &patch,
                    )
                    .map_err(|err| err.to_string());
                if result.is_ok() {
                    instance.refresh_render_buffers();
                    instance.refresh_stats();
                }
                result
            };
            match result {
                Ok(info) => set_serialized_query_result(
                    &mut runtime,
                    json!({
                        "schema": "microcosm.genome_edit_result.v1",
                        "kind": "genome_edit_result",
                        "result": info
                    }),
                ),
                Err(err) => set_runtime_error(&mut runtime, STATUS_WORLD_ERROR, err),
            }
        }
        Err(err) => err,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn microcosm_set_tile_enval(handle: u32, x: u32, y: u32, value: f32) -> u32 {
    match lock_runtime() {
        Ok(mut runtime) => {
            let Some(instance) = runtime.instance_mut(handle) else {
                return STATUS_INVALID_HANDLE;
            };
            let result: Result<(), String> = (|| {
                let tile_id = instance
                    .world
                    .tile_id(x as usize, y as usize)
                    .ok_or_else(|| format!("no tile at ({x}, {y})"))?;
                instance
                    .world
                    .set_tile_enval(tile_id, value)
                    .map_err(|err| err.to_string())?;
                instance.refresh_stats();
                Ok(())
            })();
            match result {
                Ok(()) => {
                    runtime.last_error.clear();
                    STATUS_OK
                }
                Err(err) => set_runtime_error(&mut runtime, STATUS_WORLD_ERROR, err),
            }
        }
        Err(err) => err,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn microcosm_adjust_tile_enval(handle: u32, x: u32, y: u32, delta: f32) -> u32 {
    match lock_runtime() {
        Ok(mut runtime) => {
            let Some(instance) = runtime.instance_mut(handle) else {
                return STATUS_INVALID_HANDLE;
            };
            let result: Result<(), String> = (|| {
                let tile_id = instance
                    .world
                    .tile_id(x as usize, y as usize)
                    .ok_or_else(|| format!("no tile at ({x}, {y})"))?;
                instance
                    .world
                    .adjust_tile_enval(tile_id, delta)
                    .map_err(|err| err.to_string())?;
                instance.refresh_stats();
                Ok(())
            })();
            match result {
                Ok(()) => {
                    runtime.last_error.clear();
                    STATUS_OK
                }
                Err(err) => set_runtime_error(&mut runtime, STATUS_WORLD_ERROR, err),
            }
        }
        Err(err) => err,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn microcosm_brush_enval_rect(
    handle: u32,
    center_x: u32,
    center_y: u32,
    brush_width: u32,
    brush_height: u32,
    delta: f32,
) -> u32 {
    match lock_runtime() {
        Ok(mut runtime) => {
            let Some(instance) = runtime.instance_mut(handle) else {
                return STATUS_INVALID_HANDLE;
            };
            let result: Result<(), String> = (|| {
                if !delta.is_finite() {
                    return Err(format!("non-finite enval brush delta {delta}"));
                }
                let width = (brush_width as usize).min(instance.world.width()).max(1);
                let height = (brush_height as usize).min(instance.world.height()).max(1);
                let start_x = center_x as isize - (width as isize / 2);
                let start_y = center_y as isize - (height as isize / 2);
                for dx in 0..width {
                    for dy in 0..height {
                        let tile_id = instance
                            .world
                            .wrapped_tile_id(start_x + dx as isize, start_y + dy as isize);
                        instance
                            .world
                            .adjust_tile_enval(tile_id, delta)
                            .map_err(|err| err.to_string())?;
                    }
                }
                instance.refresh_stats();
                Ok(())
            })();
            match result {
                Ok(()) => {
                    runtime.last_error.clear();
                    STATUS_OK
                }
                Err(err) => set_runtime_error(&mut runtime, STATUS_WORLD_ERROR, err),
            }
        }
        Err(err) => err,
    }
}

macro_rules! ptr_fn {
    ($name:ident, $field:ident, $ty:ty) => {
        #[unsafe(no_mangle)]
        pub extern "C" fn $name(handle: u32) -> *const $ty {
            match lock_runtime() {
                Ok(runtime) => runtime
                    .instance(handle)
                    .map(|instance| instance.render_buffers.$field.as_ptr())
                    .unwrap_or(std::ptr::null()),
                Err(_) => std::ptr::null(),
            }
        }
    };
}

ptr_fn!(microcosm_tile_enval_ptr, tile_enval, f32);
ptr_fn!(microcosm_tile_cell_count_ptr, tile_cell_count, u32);
ptr_fn!(microcosm_tile_mass_density_ptr, tile_mass_density, f32);
ptr_fn!(microcosm_tile_total_elements_ptr, tile_total_elements, f32);
ptr_fn!(
    microcosm_tile_element_concentrations_ptr,
    tile_element_concentrations,
    f32
);

#[unsafe(no_mangle)]
pub extern "C" fn microcosm_tile_element_concentrations_len(handle: u32) -> u32 {
    match lock_runtime() {
        Ok(runtime) => runtime
            .instance(handle)
            .map(|instance| {
                clamp_usize_to_u32(instance.render_buffers.tile_element_concentrations.len())
            })
            .unwrap_or(0),
        Err(_) => 0,
    }
}
ptr_fn!(microcosm_lattice_rgba_ptr, lattice_rgba, f32);

#[unsafe(no_mangle)]
pub extern "C" fn microcosm_lattice_rgba_len(handle: u32) -> u32 {
    match lock_runtime() {
        Ok(runtime) => runtime
            .instance(handle)
            .map(|instance| clamp_usize_to_u32(instance.render_buffers.lattice_rgba.len()))
            .unwrap_or(0),
        Err(_) => 0,
    }
}

ptr_fn!(microcosm_cell_id_ptr, cell_id, u32);
ptr_fn!(microcosm_cell_point_data_ptr, cell_point_data, f32);
ptr_fn!(microcosm_cell_rotation_data_ptr, cell_rotation_data, f32);
ptr_fn!(microcosm_cell_scale_data_ptr, cell_scale_data, f32);
ptr_fn!(microcosm_cell_rgba_ptr, cell_rgba, f32);
ptr_fn!(microcosm_cell_radius_ptr, cell_radius, f32);

#[unsafe(no_mangle)]
pub extern "C" fn microcosm_cell_point_data_len(handle: u32) -> u32 {
    match lock_runtime() {
        Ok(runtime) => runtime
            .instance(handle)
            .map(|instance| clamp_usize_to_u32(instance.render_buffers.cell_point_data.len()))
            .unwrap_or(0),
        Err(_) => 0,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn microcosm_cell_rotation_data_len(handle: u32) -> u32 {
    match lock_runtime() {
        Ok(runtime) => runtime
            .instance(handle)
            .map(|instance| clamp_usize_to_u32(instance.render_buffers.cell_rotation_data.len()))
            .unwrap_or(0),
        Err(_) => 0,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn microcosm_cell_scale_data_len(handle: u32) -> u32 {
    match lock_runtime() {
        Ok(runtime) => runtime
            .instance(handle)
            .map(|instance| clamp_usize_to_u32(instance.render_buffers.cell_scale_data.len()))
            .unwrap_or(0),
        Err(_) => 0,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn microcosm_cell_rgba_len(handle: u32) -> u32 {
    match lock_runtime() {
        Ok(runtime) => runtime
            .instance(handle)
            .map(|instance| clamp_usize_to_u32(instance.render_buffers.cell_rgba.len()))
            .unwrap_or(0),
        Err(_) => 0,
    }
}
ptr_fn!(microcosm_cell_energy_ptr, cell_energy, f32);
ptr_fn!(microcosm_cell_lineage_ptr, cell_lineage, u32);
ptr_fn!(microcosm_cell_flags_ptr, cell_flags, u32);
ptr_fn!(microcosm_cell_enzyme_count_ptr, cell_enzyme_count, u32);
ptr_fn!(microcosm_cell_age_seconds_ptr, cell_age_seconds, f32);
