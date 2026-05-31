use std::alloc::{alloc, dealloc, Layout};
use std::sync::{Mutex, OnceLock};

use microcosmcore::{
    CellId, Config, MoleculeSeedingConfig, RenderBuffers, World, WorldStats, VERSION,
};
use serde::Deserialize;
use serde_json::json;

pub const ABI_VERSION: &str = VERSION;
pub const STATUS_OK: u32 = 0;
pub const STATUS_INVALID_HANDLE: u32 = 1;
pub const STATUS_NULL_POINTER: u32 = 2;
pub const STATUS_CONFIG_ERROR: u32 = 3;
pub const STATUS_WORLD_ERROR: u32 = 4;
pub const STATUS_LOCK_ERROR: u32 = 5;
pub const STATUS_ALLOC_ERROR: u32 = 6;

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
    pub molecule_count: u32,
    pub tile_molecule_count: u32,
    pub cell_molecule_count: u32,
    pub free_molecule_record_count: u32,
    pub active_molecule_record_count: u32,
    pub molecule_arena_len: u32,
    pub molecule_arena_high_water_mark: u32,
    pub molecule_slots_reused: u64,
    pub molecule_slots_newly_allocated: u64,
    pub total_atom_count: u64,
    pub tile_atom_count: u64,
    pub cell_atom_count: u64,
    pub live_cell_count: u32,
    pub cell_record_count: u32,
    pub dead_cell_count: u32,
    pub births: u64,
    pub deaths: u64,
    pub predation_events: u64,
    pub cells_consumed: u64,
    pub predator_energy_gained: f64,
    pub predation_enzyme_transfers: u64,
    pub predation_enzyme_replacements: u64,
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
    pub enzyme_anabolase_count: u64,
    pub enzyme_catabolase_count: u64,
    pub enzyme_transmutase_count: u64,
    pub enzyme_defensase_count: u64,
    pub enzyme_attackase_count: u64,
    pub average_attack_total: f64,
    pub max_attack_total: u32,
    pub average_defense_total: f64,
    pub max_defense_total: u32,
    pub average_enval: f32,
    pub min_enval: f32,
    pub max_enval: f32,
    pub enval_std_dev: f32,
    pub enval_p05: f32,
    pub enval_p50: f32,
    pub enval_p95: f32,
    pub reaction_attempts: u64,
    pub reaction_gates_passed: u64,
    pub reaction_successes: u64,
    pub molecule_uptakes: u64,
    pub molecule_outputs: u64,
    pub divisions: u64,
    pub cell_steps: u64,
    pub enzyme_entries_seen: u64,
    pub metabolic_enzyme_attempts: u64,
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
            molecule_count: clamp_usize_to_u32(stats.molecule_count),
            tile_molecule_count: clamp_usize_to_u32(stats.tile_molecule_count),
            cell_molecule_count: clamp_usize_to_u32(stats.cell_molecule_count),
            free_molecule_record_count: clamp_usize_to_u32(stats.free_molecule_record_count),
            active_molecule_record_count: clamp_usize_to_u32(stats.active_molecule_record_count),
            molecule_arena_len: clamp_usize_to_u32(stats.molecule_arena_len),
            molecule_arena_high_water_mark: clamp_usize_to_u32(
                stats.molecule_arena_high_water_mark,
            ),
            molecule_slots_reused: stats.molecule_slots_reused,
            molecule_slots_newly_allocated: stats.molecule_slots_newly_allocated,
            total_atom_count: stats.total_atom_count,
            tile_atom_count: stats.tile_atom_count,
            cell_atom_count: stats.cell_atom_count,
            live_cell_count: clamp_usize_to_u32(stats.live_cell_count),
            cell_record_count: clamp_usize_to_u32(stats.cell_record_count),
            dead_cell_count: clamp_usize_to_u32(stats.dead_cell_count),
            births: stats.births,
            deaths: stats.deaths,
            predation_events: stats.predation_events,
            cells_consumed: stats.cells_consumed,
            predator_energy_gained: stats.predator_energy_gained,
            predation_enzyme_transfers: stats.predation_enzyme_transfers,
            predation_enzyme_replacements: stats.predation_enzyme_replacements,
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
            enzyme_anabolase_count: stats.enzyme_type_totals.anabolase,
            enzyme_catabolase_count: stats.enzyme_type_totals.catabolase,
            enzyme_transmutase_count: stats.enzyme_type_totals.transmutase,
            enzyme_defensase_count: stats.enzyme_type_totals.defensase,
            enzyme_attackase_count: stats.enzyme_type_totals.attackase,
            average_attack_total: stats.average_attack_total,
            max_attack_total: stats.max_attack_total,
            average_defense_total: stats.average_defense_total,
            max_defense_total: stats.max_defense_total,
            average_enval: stats.average_enval,
            min_enval: stats.min_enval,
            max_enval: stats.max_enval,
            enval_std_dev: stats.enval_std_dev,
            enval_p05: stats.enval_p05,
            enval_p50: stats.enval_p50,
            enval_p95: stats.enval_p95,
            reaction_attempts: stats.reaction_counters.total_attempts(),
            reaction_gates_passed: stats.reaction_counters.gates_passed_by_type.total(),
            reaction_successes: stats.reaction_counters.total_successes(),
            molecule_uptakes: stats.reaction_counters.molecule_uptakes,
            molecule_outputs: stats.reaction_counters.molecule_outputs,
            divisions: stats.reaction_counters.divisions,
            cell_steps: stats.operation_counters.cell_steps,
            enzyme_entries_seen: stats.operation_counters.enzyme_entries_seen,
            metabolic_enzyme_attempts: stats.operation_counters.metabolic_enzyme_attempts,
            render_epoch: (stats.tick_count & u64::from(u32::MAX)) as u32,
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
struct WasmInitConfig {
    seed: Option<String>,
    width: Option<usize>,
    height: Option<usize>,
    initial_cells: Option<usize>,
    dt_seconds: Option<f64>,
    molecule_diffusion_wheel_size: Option<usize>,
    enval_diffusion_alpha: Option<f32>,
    molecule_seeding: Option<MoleculeSeedingConfig>,
    predation_enabled: Option<bool>,
}

impl WasmInitConfig {
    fn into_core_config(self) -> Config {
        let mut config = Config::default();
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
        if let Some(size) = self.molecule_diffusion_wheel_size {
            config.molecule_diffusion_wheel_size = size;
        }
        if let Some(alpha) = self.enval_diffusion_alpha {
            config.enval_diffusion_alpha = alpha;
        }
        if let Some(seeding) = self.molecule_seeding {
            config.molecule_seeding = seeding;
        }
        if let Some(enabled) = self.predation_enabled {
            config.predation_enabled = enabled;
        }
        config
    }
}

struct WasmInstance {
    world: World,
    stats: WasmStats,
    render_buffers: RenderBuffers,
}

impl WasmInstance {
    fn new(mut world: World) -> Self {
        let stats = WasmStats::from(&world.stats());
        let render_buffers = world.build_render_buffers();
        world.rebuild_derived_caches();
        Self {
            world,
            stats,
            render_buffers,
        }
    }

    fn refresh_stats(&mut self) {
        self.stats = WasmStats::from(&self.world.stats());
        self.stats.render_epoch = self.render_buffers.render_epoch;
    }

    fn refresh_render_buffers(&mut self) {
        self.render_buffers = self.world.build_render_buffers();
        self.refresh_stats();
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

fn clamp_usize_to_u32(value: usize) -> u32 {
    value.min(u32::MAX as usize) as u32
}

#[no_mangle]
pub extern "C" fn microcosm_version_ptr() -> *const u8 {
    VERSION.as_ptr()
}

#[no_mangle]
pub extern "C" fn microcosm_version_len() -> usize {
    VERSION.len()
}

#[no_mangle]
pub extern "C" fn microcosm_abi_version_ptr() -> *const u8 {
    ABI_VERSION.as_ptr()
}

#[no_mangle]
pub extern "C" fn microcosm_abi_version_len() -> usize {
    ABI_VERSION.len()
}

#[no_mangle]
pub extern "C" fn microcosm_stats_size() -> usize {
    std::mem::size_of::<WasmStats>()
}

#[no_mangle]
pub extern "C" fn microcosm_last_error_ptr() -> *const u8 {
    match lock_runtime() {
        Ok(runtime) => runtime.last_error.as_ptr(),
        Err(_) => std::ptr::null(),
    }
}

#[no_mangle]
pub extern "C" fn microcosm_last_error_len() -> usize {
    match lock_runtime() {
        Ok(runtime) => runtime.last_error.len(),
        Err(_) => 0,
    }
}

#[no_mangle]
pub extern "C" fn microcosm_query_result_ptr() -> *const u8 {
    match lock_runtime() {
        Ok(runtime) => runtime.query_result.as_ptr(),
        Err(_) => std::ptr::null(),
    }
}

#[no_mangle]
pub extern "C" fn microcosm_query_result_len() -> usize {
    match lock_runtime() {
        Ok(runtime) => runtime.query_result.len(),
        Err(_) => 0,
    }
}

#[no_mangle]
pub extern "C" fn microcosm_alloc(len: usize, align: usize) -> *mut u8 {
    if len == 0 || align == 0 || !align.is_power_of_two() {
        return std::ptr::null_mut();
    }
    let Ok(layout) = Layout::from_size_align(len, align) else {
        return std::ptr::null_mut();
    };
    unsafe { alloc(layout) }
}

#[no_mangle]
pub extern "C" fn microcosm_free(ptr: *mut u8, len: usize, align: usize) {
    if ptr.is_null() || len == 0 || align == 0 || !align.is_power_of_two() {
        return;
    }
    if let Ok(layout) = Layout::from_size_align(len, align) {
        unsafe {
            dealloc(ptr, layout);
        }
    }
}

#[no_mangle]
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

#[no_mangle]
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

#[no_mangle]
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

#[no_mangle]
pub extern "C" fn microcosm_step(handle: u32, ticks: u32) -> u32 {
    match lock_runtime() {
        Ok(mut runtime) => {
            let Some(instance) = runtime.instance_mut(handle) else {
                return STATUS_INVALID_HANDLE;
            };
            instance.world.step_many(ticks);
            instance.refresh_stats();
            runtime.last_error.clear();
            STATUS_OK
        }
        Err(err) => err,
    }
}

#[no_mangle]
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

#[no_mangle]
pub extern "C" fn microcosm_stats_ptr(handle: u32) -> *const WasmStats {
    match lock_runtime() {
        Ok(runtime) => runtime
            .instance(handle)
            .map(|instance| &instance.stats as *const WasmStats)
            .unwrap_or(std::ptr::null()),
        Err(_) => std::ptr::null(),
    }
}

#[no_mangle]
pub extern "C" fn microcosm_tile_count(handle: u32) -> u32 {
    match lock_runtime() {
        Ok(runtime) => runtime
            .instance(handle)
            .map(|instance| clamp_usize_to_u32(instance.render_buffers.tile_count()))
            .unwrap_or(0),
        Err(_) => 0,
    }
}

#[no_mangle]
pub extern "C" fn microcosm_cell_count(handle: u32) -> u32 {
    match lock_runtime() {
        Ok(runtime) => runtime
            .instance(handle)
            .map(|instance| clamp_usize_to_u32(instance.render_buffers.cell_count()))
            .unwrap_or(0),
        Err(_) => 0,
    }
}

#[no_mangle]
pub extern "C" fn microcosm_render_epoch(handle: u32) -> u32 {
    match lock_runtime() {
        Ok(runtime) => runtime
            .instance(handle)
            .map(|instance| instance.render_buffers.render_epoch)
            .unwrap_or(0),
        Err(_) => 0,
    }
}

#[no_mangle]
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
                    "cell_id": info.cell.map(|cell_id| cell_id.index()),
                    "occupied": info.cell.is_some(),
                    "molecule_count": info.molecule_count,
                    "mass_count": info.mass_count,
                    "element_counts": {
                        "A": info.element_counts[0],
                        "B": info.element_counts[1],
                        "C": info.element_counts[2],
                        "D": info.element_counts[3],
                        "E": info.element_counts[4],
                        "F": info.element_counts[5]
                    },
                    "element_mask": info.element_mask
                }),
            )
        }
        Err(err) => err,
    }
}

#[no_mangle]
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
                    "tile_id": info.tile_id.index(),
                    "x": info.x,
                    "y": info.y,
                    "energy": info.energy,
                    "lineage_id": info.lineage_id.raw(),
                    "enzyme_count": info.enzyme_count,
                    "internal_atom_count": info.internal_atom_count,
                    "combat_attack_total": info.combat_attack_total,
                    "combat_defense_total": info.combat_defense_total,
                    "age_seconds": info.age_seconds,
                    "optimal_enval": info.optimal_enval,
                    "local_enval_average": info.local_enval_average,
                    "repro_threshold": info.repro_threshold,
                    "decay_time": info.decay_time
                }),
            )
        }
        Err(err) => err,
    }
}

#[no_mangle]
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

#[no_mangle]
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

#[no_mangle]
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
        #[no_mangle]
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
ptr_fn!(microcosm_tile_occupancy_ptr, tile_occupancy, u32);
ptr_fn!(microcosm_tile_mass_ptr, tile_mass, u32);
ptr_fn!(microcosm_tile_molecule_count_ptr, tile_molecule_count, u32);
ptr_fn!(microcosm_tile_element_mask_ptr, tile_element_mask, u32);
ptr_fn!(microcosm_cell_id_ptr, cell_id, u32);
ptr_fn!(microcosm_cell_x_ptr, cell_x, u32);
ptr_fn!(microcosm_cell_y_ptr, cell_y, u32);
ptr_fn!(microcosm_cell_energy_ptr, cell_energy, f32);
ptr_fn!(microcosm_cell_lineage_ptr, cell_lineage, u32);
ptr_fn!(microcosm_cell_flags_ptr, cell_flags, u32);
ptr_fn!(microcosm_cell_enzyme_count_ptr, cell_enzyme_count, u32);
ptr_fn!(microcosm_cell_age_seconds_ptr, cell_age_seconds, f32);
ptr_fn!(microcosm_cell_attack_ptr, cell_attack, u32);
ptr_fn!(microcosm_cell_defense_ptr, cell_defense, u32);

#[cfg(test)]
mod tests {
    use super::*;

    fn exported_string(ptr: *const u8, len: usize) -> String {
        assert!(!ptr.is_null());
        let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
        std::str::from_utf8(bytes).unwrap().to_owned()
    }

    #[test]
    fn empty_config_uses_default_world() {
        let config = parse_config_from_bytes(std::ptr::null(), 0).unwrap();
        assert_eq!(config.width, Config::default().width);
    }

    #[test]
    fn json_config_overrides_core_fields() {
        let json =
            br#"{"seed":"wasm-test","width":8,"height":6,"initial_cells":3,"predation_enabled":false}"#;
        let config = parse_config_from_bytes(json.as_ptr(), json.len()).unwrap();
        assert_eq!(config.seed, "wasm-test");
        assert_eq!(config.width, 8);
        assert_eq!(config.height, 6);
        assert_eq!(config.initial_founder_count, 3);
        assert!(!config.predation_enabled);
    }

    #[test]
    fn handle_lifecycle_and_render_buffers_work_natively() {
        let json = br#"{"seed":"wasm-handle","width":8,"height":6,"initial_cells":4}"#;
        let handle = microcosm_create(json.as_ptr(), json.len());
        assert!(handle > 0);
        assert_eq!(microcosm_tile_count(handle), 48);
        assert_eq!(microcosm_cell_count(handle), 4);
        assert!(!microcosm_stats_ptr(handle).is_null());
        assert!(!microcosm_tile_enval_ptr(handle).is_null());
        assert_eq!(microcosm_step(handle, 5), STATUS_OK);
        assert_eq!(microcosm_refresh_render_buffers(handle), STATUS_OK);
        assert!(microcosm_render_epoch(handle) >= 5);
        assert_eq!(microcosm_destroy(handle), STATUS_OK);
        assert_eq!(microcosm_step(handle, 1), STATUS_INVALID_HANDLE);
    }

    #[test]
    fn handle_table_insertions_do_not_move_existing_stats_storage() {
        let json = br#"{"seed":"wasm-stable-pointers","width":4,"height":4,"initial_cells":1}"#;
        let first = microcosm_create(json.as_ptr(), json.len());
        assert!(first > 0);
        let first_stats = microcosm_stats_ptr(first) as usize;
        assert_ne!(first_stats, 0);

        let mut handles = Vec::new();
        for index in 0..32 {
            let json = format!(
                r#"{{"seed":"wasm-stable-pointers-{index}","width":4,"height":4,"initial_cells":1}}"#
            );
            let handle = microcosm_create(json.as_ptr(), json.len());
            assert!(handle > 0);
            handles.push(handle);
        }

        assert_eq!(microcosm_stats_ptr(first) as usize, first_stats);
        for handle in handles {
            assert_eq!(microcosm_destroy(handle), STATUS_OK);
        }
        assert_eq!(microcosm_destroy(first), STATUS_OK);
    }

    #[test]
    fn stats_size_and_versions_are_exposed() {
        assert_eq!(
            exported_string(microcosm_version_ptr(), microcosm_version_len()),
            "0.17.2"
        );
        assert_eq!(
            exported_string(microcosm_abi_version_ptr(), microcosm_abi_version_len()),
            "0.17.2"
        );
        assert_eq!(microcosm_stats_size(), std::mem::size_of::<WasmStats>());
    }

    #[test]
    fn inspection_and_enval_edit_exports_work() {
        let json = br#"{"seed":"wasm-inspect","width":8,"height":6,"initial_cells":4}"#;
        let handle = microcosm_create(json.as_ptr(), json.len());
        assert!(handle > 0);
        assert_eq!(microcosm_inspect_tile(handle, 0, 0), STATUS_OK);
        assert!(microcosm_query_result_len() > 0);
        assert!(!microcosm_query_result_ptr().is_null());
        let before = unsafe { *microcosm_stats_ptr(handle) }.average_enval;
        assert_eq!(microcosm_adjust_tile_enval(handle, 0, 0, 1.0), STATUS_OK);
        assert_eq!(microcosm_refresh_render_buffers(handle), STATUS_OK);
        let after = unsafe { *microcosm_stats_ptr(handle) }.average_enval;
        assert!(after > before);
        assert_eq!(microcosm_brush_enval_rect(handle, 0, 0, 2, 2, -0.25), STATUS_OK);
        assert_eq!(microcosm_inspect_cell(handle, 0), STATUS_OK);
        assert!(microcosm_query_result_len() > 0);
        assert_eq!(microcosm_destroy(handle), STATUS_OK);
    }

    #[test]
    fn expanded_wasm_stats_are_populated() {
        let json = br#"{"seed":"wasm-expanded-stats","width":8,"height":6,"initial_cells":4}"#;
        let handle = microcosm_create(json.as_ptr(), json.len());
        assert!(handle > 0);
        let stats_ptr = microcosm_stats_ptr(handle);
        assert!(!stats_ptr.is_null());
        let stats = unsafe { *stats_ptr };
        assert_eq!(stats.tile_count, 48);
        assert_eq!(stats.live_cell_count, 4);
        assert_eq!(stats.occupied_tile_count, 4);
        assert!(stats.molecule_count >= 48);
        assert_eq!(stats.active_molecule_record_count, stats.molecule_count);
        assert!(stats.molecule_arena_len >= stats.active_molecule_record_count);
        assert!(stats.molecule_arena_high_water_mark >= stats.molecule_arena_len);
        assert!(stats.enzyme_anabolase_count >= 4);
        assert!(stats.average_cell_energy.is_finite());
        assert!(stats.enval_std_dev.is_finite());

        assert_eq!(microcosm_step(handle, 3), STATUS_OK);
        let stepped = unsafe { *microcosm_stats_ptr(handle) };
        assert!(stepped.cell_steps >= 4);
        assert!(stepped.metabolic_enzyme_attempts > 0);
        assert_eq!(microcosm_destroy(handle), STATUS_OK);
    }
}
