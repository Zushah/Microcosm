use std::sync::{Mutex, MutexGuard};

use microcosmwasm::*;
use serde_json::Value;

static LOCK: Mutex<()> = Mutex::new(());

fn lock() -> MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|error| error.into_inner())
}

const VISUAL_HAS_SELECTED_CELL: u32 = 1 << 1;

fn exported_string(ptr: *const u8, len: usize) -> String {
    assert!(!ptr.is_null());
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    std::str::from_utf8(bytes).unwrap().to_owned()
}

fn query_json() -> Value {
    serde_json::from_str(&exported_string(
        microcosm_query_result_ptr(),
        microcosm_query_result_len(),
    ))
    .unwrap()
}

fn last_error() -> String {
    exported_string(microcosm_last_error_ptr(), microcosm_last_error_len())
}

fn create(json: &str) -> u32 {
    let handle = microcosm_create(json.as_ptr(), json.len());
    assert!(handle > 0, "{json}: {}", last_error());
    handle
}

fn stats(handle: u32) -> WasmStats {
    unsafe { *microcosm_stats_ptr(handle) }
}

fn tile_enval(handle: u32, x: u32, y: u32) -> f32 {
    assert_eq!(microcosm_inspect_tile(handle, x, y), STATUS_OK);
    query_json()["enval"].as_f64().unwrap() as f32
}

fn cell_detail(handle: u32, cell: u32) -> Value {
    assert_eq!(microcosm_inspect_cell_detail(handle, cell, 4), STATUS_OK);
    query_json()
}

#[test]
fn versions_and_the_stats_size_are_exposed() {
    let _guard = lock();
    assert_eq!(
        exported_string(microcosm_version_ptr(), microcosm_version_len()),
        microcosmcore::VERSION
    );
    assert_eq!(
        exported_string(microcosm_abi_version_ptr(), microcosm_abi_version_len()),
        microcosmcore::VERSION
    );
    assert_eq!(ABI_VERSION, microcosmcore::VERSION);
    assert_eq!(microcosm_stats_size(), std::mem::size_of::<WasmStats>());
}

#[test]
fn the_javascript_stats_layout_matches_wasm_stats() {
    let _guard = lock();
    macro_rules! rust_fields {
        ($($name:ident: $ty:ident),* $(,)?) => {
            vec![$((stringify!($name), stringify!($ty), std::mem::offset_of!(WasmStats, $name))),*]
        };
    }
    let rust = rust_fields!(
        tick_count: u64, sim_time_seconds: f64, width: u32, height: u32, tile_count: u32,
        occupied_tile_count: u32, empty_tile_count: u32, occupancy_fraction: f64,
        extracellular_a: f64, extracellular_b: f64, extracellular_c: f64,
        extracellular_d: f64, extracellular_e: f64, extracellular_f: f64,
        intracellular_a: f64, intracellular_b: f64, intracellular_c: f64,
        intracellular_d: f64, intracellular_e: f64, intracellular_f: f64,
        system_a: f64, system_b: f64, system_c: f64, system_d: f64, system_e: f64,
        system_f: f64, total_element_amount: f64, live_cell_count: u32, births: u64,
        deaths: u64, fissions: u64, energy_deaths: u64, lyses: u64, radius_median: f32, radius_p95: f32, radius_max: f32, mean_strain: f64, lineage_count: u32, total_lineage_records: u32,
        extinct_lineage_count: u32, dominant_lineage_id: u64,
        dominant_lineage_population: u64, dominant_lineage_share: f64,
        average_cell_energy: f64, min_cell_energy: f64, max_cell_energy: f64,
        total_cell_energy: f64, average_enzyme_count: f64, min_enzyme_count: u32,
        max_enzyme_count: u32, cells_at_enzyme_cap: u32, fraction_cells_at_enzyme_cap: f64,
        total_enzyme_count: u64, average_enval: f32,
        min_enval: f32, max_enval: f32, enval_std_dev: f32, enval_p05: f32, enval_p50: f32,
        enval_p95: f32, reaction_attempts: u64, reaction_successes: u64,
        executed_metabolic_flux: f64, uptake_flux: f64, leak_flux: f64,
        secretion_flux: f64, divisions: u64, element_field_diffusion_tiles: u64,
        element_uptake_events: u64, cell_steps: u64, enzyme_entries_seen: u64,
        metabolic_enzyme_attempts: u64, spatial_candidate_checks: u64,
        overlap_candidates: u64, overlap_corrections: u64, energy_founder: f64,
        energy_injected: f64, energy_extracted: f64, energy_chemical_harvest: f64,
        energy_chemical_cost: f64, energy_enval_harvest: f64, energy_pump_cost: f64, energy_assembly_cost: f64,
        energy_maintenance: f64, energy_death_loss: f64,
        energy_ledger_residual: f64, renewable_coverage: f64, enval_source_inflow: f64,
        enval_cell_uptake: f64, enval_cell_emission: f64, enval_recharge: f64,
        enval_edits: f64, recharge_amount: f64, recharge_energy: f64, mean_exposure: f64,
        max_exposure: f32, exposed_cell_count: u32, contact_pairs: u64,
        cross_lineage_contact_pairs: u64, field_surface_extent: f64,
        cell_surface_extent: f64, boundary_digested: f64,
        cross_lineage_boundary_digested: f64, lyses_after_digestion: u64,
        mean_enval_optimum_spread: f64, mean_niche_mismatch: f64, mixed_polarity_cells: u32,
        render_epoch: u32,
    );

    let source = include_str!("../../wasmgpu/runtime.js");
    let start = source.find("const STATS_FIELD_TYPES").unwrap();
    let end = start + source[start..].find("]);").unwrap();
    let mut offset = 0_usize;
    let mut max_align = 1_usize;
    let mut javascript = Vec::new();
    for line in source[start..end].lines() {
        let line = line.trim();
        let Some(entry) = line.strip_prefix("[\"") else {
            continue;
        };
        let (name, rest) = entry.split_once('"').unwrap();
        let ty = rest.trim_start_matches(", \"").split('"').next().unwrap();
        let size = match ty {
            "u32" | "f32" => 4,
            "u64" | "f64" => 8,
            other => panic!("unknown stats type {other}"),
        };
        max_align = max_align.max(size);
        offset = offset.div_ceil(size) * size;
        javascript.push((name.to_owned(), ty.to_owned(), offset));
        offset += size;
    }
    assert_eq!(javascript.len(), rust.len(), "field count differs");
    for ((js_name, js_type, js_offset), (name, ty, rust_offset)) in javascript.iter().zip(&rust) {
        assert_eq!(js_name, name);
        assert_eq!(js_type, ty, "{name}");
        assert_eq!(*js_offset, *rust_offset, "{name}");
    }
    assert_eq!(
        offset.div_ceil(max_align) * max_align,
        std::mem::size_of::<WasmStats>()
    );
}

#[test]
fn handles_are_created_reset_and_destroyed() {
    let _guard = lock();
    let handle = create(r#"{"seed":"wasm-handle","width":8,"height":6,"initial_cells":4}"#);
    assert_eq!(microcosm_tile_count(handle), 48);
    assert_eq!(microcosm_cell_count(handle), 4);
    assert_eq!(stats(handle).live_cell_count, 4);
    assert!(!microcosm_tile_enval_ptr(handle).is_null());
    assert_eq!(microcosm_lattice_rgba_len(handle), 48 * 4);
    assert_eq!(microcosm_cell_point_data_len(handle), 4 * 4);
    assert_eq!(microcosm_cell_rgba_len(handle), 4 * 4);
    assert_eq!(
        microcosm_set_render_visual_state(
            handle,
            0,
            VISUAL_HAS_SELECTED_CELL,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
        ),
        STATUS_OK
    );
    assert_eq!(
        microcosm_set_render_visual_state(handle, 99, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1),
        STATUS_CONFIG_ERROR
    );
    assert_eq!(microcosm_step_no_stats(handle, 5), STATUS_OK);
    assert_eq!(microcosm_refresh_render_buffers(handle), STATUS_OK);
    assert_eq!(microcosm_render_epoch(handle), 5);

    let default = microcosm_create(std::ptr::null(), 0);
    assert!(default > 0, "{}", last_error());
    assert_eq!(microcosm_tile_count(default), 76_800);
    assert_eq!(microcosm_destroy(default), STATUS_OK);

    let config = r#"{"seed":"wasm-handle-reset","width":5,"height":4,"initial_cells":2}"#;
    assert_eq!(
        microcosm_reset(handle, config.as_ptr(), config.len()),
        STATUS_OK
    );
    assert_eq!(microcosm_tile_count(handle), 20);
    assert_eq!(microcosm_cell_count(handle), 2);
    assert_eq!(stats(handle).tick_count, 0);

    assert_eq!(microcosm_destroy(handle), STATUS_OK);
    for dead in [handle, 0, u32::MAX] {
        assert_eq!(microcosm_step_no_stats(dead, 1), STATUS_INVALID_HANDLE);
        assert_eq!(microcosm_refresh_stats(dead), STATUS_INVALID_HANDLE);
        assert_eq!(
            microcosm_refresh_render_buffers(dead),
            STATUS_INVALID_HANDLE
        );
        assert_eq!(microcosm_inspect_tile(dead, 0, 0), STATUS_INVALID_HANDLE);
        assert_eq!(
            microcosm_reset(dead, config.as_ptr(), config.len()),
            STATUS_INVALID_HANDLE
        );
        assert!(microcosm_stats_ptr(dead).is_null());
        assert_eq!(microcosm_tile_count(dead), 0);
    }
}

#[test]
fn bad_configs_are_reported() {
    let _guard = lock();
    for bad in [r#"{"width":"#, r#"{"width":0}"#, r#"{"dt_seconds":-1.0}"#] {
        assert_eq!(microcosm_create(bad.as_ptr(), bad.len()), 0, "{bad}");
        assert!(!last_error().is_empty(), "{bad}");
    }
    let handle = create(r#"{"seed":"wasm-bad","width":8,"height":6,"initial_cells":2}"#);
    let malformed = r#"{"width":"#;
    assert_eq!(
        microcosm_reset(handle, malformed.as_ptr(), malformed.len()),
        STATUS_CONFIG_ERROR
    );
    assert!(!last_error().is_empty());
    let impossible = r#"{"width":0}"#;
    assert_eq!(
        microcosm_reset(handle, impossible.as_ptr(), impossible.len()),
        STATUS_WORLD_ERROR
    );
    assert!(!last_error().is_empty());
    assert_eq!(microcosm_tile_count(handle), 48);
    assert_eq!(microcosm_destroy(handle), STATUS_OK);
}

#[test]
fn stats_change_only_on_an_explicit_refresh() {
    let _guard = lock();
    let handle = create(r#"{"seed":"wasm-no-stats-step","width":8,"height":6,"initial_cells":4}"#);
    let before = stats(handle).tick_count;

    assert_eq!(microcosm_step_no_stats(handle, 3), STATUS_OK);
    assert_eq!(stats(handle).tick_count, before);

    assert_eq!(microcosm_refresh_render_buffers(handle), STATUS_OK);
    assert!(microcosm_render_epoch(handle) >= 3);
    assert_eq!(stats(handle).tick_count, before);

    assert_eq!(microcosm_refresh_stats(handle), STATUS_OK);
    let refreshed = stats(handle);
    assert_eq!(refreshed.tick_count, before + 3);
    assert_eq!(refreshed.render_epoch, microcosm_render_epoch(handle));
    assert_eq!(refreshed.tile_count, 48);
    assert!(refreshed.live_cell_count > 0);
    assert!(refreshed.total_element_amount > refreshed.system_a);
    assert!(refreshed.energy_founder > 0.0);
    assert!(refreshed.energy_maintenance > 0.0);
    assert!(refreshed.cell_steps > 0);
    assert!(refreshed.enval_source_inflow != 0.0);
    assert!(refreshed.energy_ledger_residual.abs() < 1.0e-9);
    assert_eq!(microcosm_destroy(handle), STATUS_OK);
}

#[test]
fn stats_storage_stays_put_when_handles_are_added() {
    let _guard = lock();
    let first = create(r#"{"seed":"wasm-stable-pointers","width":4,"height":4,"initial_cells":1}"#);
    let first_stats = microcosm_stats_ptr(first) as usize;
    assert_ne!(first_stats, 0);

    let mut handles = Vec::new();
    for index in 0..32 {
        handles.push(create(&format!(
            r#"{{"seed":"wasm-stable-pointers-{index}","width":4,"height":4,"initial_cells":1}}"#
        )));
    }

    assert_eq!(microcosm_stats_ptr(first) as usize, first_stats);
    for handle in handles {
        assert_eq!(microcosm_destroy(handle), STATUS_OK);
    }
    assert_eq!(microcosm_destroy(first), STATUS_OK);
}

#[test]
fn inspection_exports_return_their_schemas() {
    let _guard = lock();
    let handle = create(r#"{"seed":"wasm-inspect","width":8,"height":6,"initial_cells":4}"#);
    assert_eq!(microcosm_step_no_stats(handle, 5), STATUS_OK);

    assert_eq!(microcosm_inspect_tile(handle, 2, 3), STATUS_OK);
    let tile = query_json();
    assert_eq!(tile["kind"], "tile");
    assert_eq!((tile["x"].as_u64(), tile["y"].as_u64()), (Some(2), Some(3)));
    assert!(tile["enval_landscape_target"].as_f64().is_some());

    assert_eq!(microcosm_inspect_cell(handle, 0), STATUS_OK);
    let cell = query_json();
    assert_eq!(cell["kind"], "cell");
    assert_eq!(cell["cell_id"], 0);
    assert!(cell["radius"].as_f64().unwrap() > 0.0);

    let (x, y) = (cell["x"].as_f64().unwrap(), cell["y"].as_f64().unwrap());
    assert_eq!(microcosm_pick_cell(handle, x as f32, y as f32), STATUS_OK);
    let pick = query_json();
    assert_eq!(pick["schema"], "microcosm.cell_pick.v1");
    assert_eq!(pick["cell_id"], 0);

    let detail = cell_detail(handle, 0);
    assert_eq!(detail["schema"], "microcosm.cell_detail.v1");
    assert_eq!(detail["cell_detail"]["cell"]["cell_id"], 0);
    assert!(
        !detail["cell_detail"]["genome"]["enzymes"]
            .as_array()
            .unwrap()
            .is_empty()
    );

    assert_eq!(microcosm_inspect_cell_elements(handle, 0), STATUS_OK);
    let elements = query_json();
    assert_eq!(elements["schema"], "microcosm.cell_elements.v1");
    assert_eq!(elements["internal_elements"].as_array().unwrap().len(), 6);

    assert_eq!(microcosm_step_no_stats(handle, 5), STATUS_OK);
    assert_eq!(microcosm_inspect_cell_fluxes(handle, 0, 4), STATUS_OK);
    let fluxes = query_json();
    assert_eq!(fluxes["schema"], "microcosm.cell_fluxes.v1");
    assert_eq!(fluxes["recent_fluxes"]["limit"], 4);
    assert!(fluxes["recent_fluxes"]["fluxes"].as_array().is_some());

    let lineage = cell["lineage_id"].as_u64().unwrap() as u32;
    assert_eq!(microcosm_inspect_lineage(handle, lineage), STATUS_OK);
    let lineage_detail = query_json();
    assert_eq!(lineage_detail["schema"], "microcosm.lineage.v1");
    assert_eq!(lineage_detail["lineage"]["lineage_id"], u64::from(lineage));

    assert_eq!(microcosm_list_lineages(handle, 2), STATUS_OK);
    let lineages = query_json();
    assert_eq!(lineages["schema"], "microcosm.lineage_list.v1");
    assert!(
        lineages["lineage_list"]["lineages"]
            .as_array()
            .unwrap()
            .len()
            <= 2
    );

    assert_eq!(microcosm_inspect_tile(handle, 8, 0), STATUS_WORLD_ERROR);
    assert_eq!(microcosm_inspect_tile(handle, 0, 6), STATUS_WORLD_ERROR);
    assert_eq!(microcosm_inspect_cell(handle, u32::MAX), STATUS_WORLD_ERROR);
    assert_eq!(
        microcosm_inspect_cell_detail(handle, 99, 2),
        STATUS_WORLD_ERROR
    );
    assert_eq!(
        microcosm_inspect_cell_elements(handle, 99),
        STATUS_WORLD_ERROR
    );
    assert_eq!(
        microcosm_inspect_cell_fluxes(handle, 99, 2),
        STATUS_WORLD_ERROR
    );
    let unknown_lineage = if lineage == u32::MAX { 0 } else { lineage + 1 };
    assert_eq!(
        microcosm_inspect_lineage(handle, unknown_lineage),
        STATUS_WORLD_ERROR
    );
    assert_eq!(
        microcosm_pick_cell(handle, f32::NAN, 0.0),
        STATUS_CONFIG_ERROR
    );
    assert!(!last_error().is_empty());
    assert_eq!(microcosm_destroy(handle), STATUS_OK);
}

#[test]
fn genome_patch_exports_apply_atomically() {
    let _guard = lock();
    let handle = create(r#"{"seed":"wasm-genome-patch","width":8,"height":6,"initial_cells":4}"#);

    let patch = br#"{"schema":"microcosm.genome_patch.v6","enzymes":[{"op":"update","index":0,"fields":{"rate":0.625}}]}"#;
    assert_eq!(
        microcosm_apply_cell_genome_patch(handle, 0, patch.as_ptr(), patch.len()),
        STATUS_OK
    );
    let result = query_json();
    assert_eq!(result["schema"], "microcosm.genome_edit_result.v1");
    assert_eq!(result["result"]["target"], "cell");
    assert_eq!(result["result"]["patched_cell_count"], 1);
    assert_eq!(result["result"]["changed_fields"][0], "enzyme[0].rate");
    assert_eq!(
        cell_detail(handle, 0)["cell_detail"]["genome"]["enzymes"][0]["rate"],
        0.625
    );

    let brush = br#"{"schema":"microcosm.genome_patch.v6","enzymes":[{"op":"update","index":0,"fields":{"rate":0.375}}]}"#;
    assert_eq!(
        microcosm_apply_genome_brush(handle, 4, 3, 8, 6, brush.as_ptr(), brush.len()),
        STATUS_OK
    );
    let result = query_json();
    assert_eq!(result["schema"], "microcosm.genome_edit_result.v1");
    assert_eq!(result["result"]["target"], "brush");
    assert_eq!(result["result"]["visited_tile_count"], 48);
    assert_eq!(result["result"]["patched_cell_count"], 4);
    for cell in 0..4 {
        assert_eq!(
            cell_detail(handle, cell)["cell_detail"]["genome"]["enzymes"][0]["rate"],
            0.375
        );
    }

    let genome = cell_detail(handle, 0)["cell_detail"]["genome"].clone();
    let invalid = br#"{"schema":"microcosm.genome_patch.v6","enzymes":[{"op":"update","index":0,"fields":{"rate":2.0}},{"op":"update","index":1,"fields":{"exposure":1.5}}]}"#;
    assert_eq!(
        microcosm_apply_cell_genome_patch(handle, 0, invalid.as_ptr(), invalid.len()),
        STATUS_WORLD_ERROR
    );
    assert_eq!(
        microcosm_apply_genome_brush(handle, 4, 3, 8, 6, invalid.as_ptr(), invalid.len()),
        STATUS_WORLD_ERROR
    );
    let stale = br#"{"schema":"microcosm.genome_patch.v5","enzymes":[{"op":"update","index":0,"fields":{"rate":2.0}}]}"#;
    assert_eq!(
        microcosm_apply_cell_genome_patch(handle, 0, stale.as_ptr(), stale.len()),
        STATUS_WORLD_ERROR
    );
    let unparsable = br#"{"enzymes":[{"op":"update","index":0,"fields":{"level":7}}]}"#;
    assert_eq!(
        microcosm_apply_cell_genome_patch(handle, 0, unparsable.as_ptr(), unparsable.len()),
        STATUS_CONFIG_ERROR
    );
    assert!(!last_error().is_empty());
    assert_eq!(cell_detail(handle, 0)["cell_detail"]["genome"], genome);
    assert_eq!(microcosm_destroy(handle), STATUS_OK);
}

#[test]
fn enval_edit_exports_change_the_field() {
    let _guard = lock();
    let handle = create(r#"{"seed":"wasm-enval-edit","width":8,"height":6,"initial_cells":0}"#);

    let old = tile_enval(handle, 2, 3);
    assert_eq!(microcosm_set_tile_enval(handle, 2, 3, 0.75), STATUS_OK);
    assert_eq!(tile_enval(handle, 2, 3), 0.75);
    assert_eq!(microcosm_adjust_tile_enval(handle, 2, 3, -0.25), STATUS_OK);
    assert_eq!(tile_enval(handle, 2, 3), 0.5);
    assert_eq!(stats(handle).enval_edits, (0.75 - f64::from(old)) - 0.25);

    let before = (0..8)
        .flat_map(|x| (0..6).map(move |y| (x, y)))
        .map(|(x, y)| ((x, y), tile_enval(handle, x, y)))
        .collect::<Vec<_>>();
    assert_eq!(
        microcosm_brush_enval_rect(handle, 0, 0, 3, 3, 0.125),
        STATUS_OK
    );
    for ((x, y), old) in &before {
        let inside = [7, 0, 1].contains(x) && [5, 0, 1].contains(y);
        let expected = if inside { old + 0.125 } else { *old };
        assert_eq!(tile_enval(handle, *x, *y), expected, "tile ({x}, {y})");
    }

    let edits = stats(handle).enval_edits;
    for status in [
        microcosm_set_tile_enval(handle, 2, 3, f32::NAN),
        microcosm_set_tile_enval(handle, 2, 3, f32::INFINITY),
        microcosm_brush_enval_rect(handle, 2, 3, 2, 2, f32::NAN),
        microcosm_set_tile_enval(handle, 8, 0, 1.0),
        microcosm_adjust_tile_enval(handle, 0, 6, 1.0),
    ] {
        assert_eq!(status, STATUS_WORLD_ERROR);
        assert!(!last_error().is_empty());
    }
    assert_eq!(tile_enval(handle, 2, 3), 0.5);
    assert_eq!(stats(handle).enval_edits, edits);
    assert_eq!(microcosm_destroy(handle), STATUS_OK);
}
