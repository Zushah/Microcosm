pub const EMPTY_CELL_ID: u32 = u32::MAX;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RenderBuffers {
    pub width: u32,
    pub height: u32,
    pub tick_count: u64,
    pub sim_time_seconds: f64,
    pub render_epoch: u32,

    pub tile_enval: Vec<f32>,
    pub tile_occupancy: Vec<u32>,
    pub tile_mass: Vec<u32>,
    pub tile_molecule_count: Vec<u32>,
    pub tile_element_mask: Vec<u32>,

    pub cell_id: Vec<u32>,
    pub cell_x: Vec<u32>,
    pub cell_y: Vec<u32>,
    pub cell_energy: Vec<f32>,
    pub cell_lineage: Vec<u32>,
    pub cell_flags: Vec<u32>,
    pub cell_enzyme_count: Vec<u32>,
    pub cell_age_seconds: Vec<f32>,
    pub cell_attack: Vec<u32>,
    pub cell_defense: Vec<u32>,
}

impl RenderBuffers {
    pub fn tile_count(&self) -> usize {
        self.tile_enval.len()
    }

    pub fn cell_count(&self) -> usize {
        self.cell_id.len()
    }

    pub fn clear(&mut self) {
        self.width = 0;
        self.height = 0;
        self.tick_count = 0;
        self.sim_time_seconds = 0.0;
        self.render_epoch = 0;
        self.tile_enval.clear();
        self.tile_occupancy.clear();
        self.tile_mass.clear();
        self.tile_molecule_count.clear();
        self.tile_element_mask.clear();
        self.cell_id.clear();
        self.cell_x.clear();
        self.cell_y.clear();
        self.cell_energy.clear();
        self.cell_lineage.clear();
        self.cell_flags.clear();
        self.cell_enzyme_count.clear();
        self.cell_age_seconds.clear();
        self.cell_attack.clear();
        self.cell_defense.clear();
    }
}
