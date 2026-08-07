pub const EMPTY_CELL_ID: u32 = u32::MAX;

const TILE_NEUTRAL_RGB255: [f64; 3] = [246.0, 246.0, 246.0];
const TILE_ELEMENT_RGB255: [[f64; 3]; 6] = [
    [66.0, 179.0, 89.0],
    [65.0, 113.0, 210.0],
    [45.0, 175.0, 190.0],
    [217.0, 98.0, 69.0],
    [216.0, 177.0, 49.0],
    [166.0, 93.0, 192.0],
];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum RenderDisplayMode {
    #[default]
    Enval = 0,
    Occupancy = 1,
    Mass = 2,
    Molecules = 3,
    ElementA = 4,
    ElementB = 5,
    ElementC = 6,
    ElementD = 7,
    ElementE = 8,
    ElementF = 9,
}

impl RenderDisplayMode {
    pub fn from_u32(value: u32) -> Option<Self> {
        Some(match value {
            0 => Self::Enval,
            1 => Self::Occupancy,
            2 => Self::Mass,
            3 => Self::Molecules,
            4 => Self::ElementA,
            5 => Self::ElementB,
            6 => Self::ElementC,
            7 => Self::ElementD,
            8 => Self::ElementE,
            9 => Self::ElementF,
            _ => return None,
        })
    }

    fn element_index(self) -> Option<usize> {
        match self {
            Self::ElementA => Some(0),
            Self::ElementB => Some(1),
            Self::ElementC => Some(2),
            Self::ElementD => Some(3),
            Self::ElementE => Some(4),
            Self::ElementF => Some(5),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RenderBrushPreview {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RenderVisualState {
    pub display_mode: RenderDisplayMode,
    pub selected_lineage: Option<u32>,
    pub selected_cell: Option<u32>,
    pub selected_tile: Option<(u32, u32)>,
    pub hover_tile: Option<(u32, u32)>,
    pub brush_preview: Option<RenderBrushPreview>,
}

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

    pub lattice_rgba: Vec<f32>,

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
        self.lattice_rgba.clear();
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

    pub(crate) fn clear_for_refresh(&mut self) {
        let lattice_rgba = std::mem::take(&mut self.lattice_rgba);
        self.clear();
        self.lattice_rgba = lattice_rgba;
    }

    pub fn refresh_lattice_rgba(&mut self, visual: &RenderVisualState) {
        let tile_count = self.tile_count();
        self.lattice_rgba.resize(tile_count.saturating_mul(4), 0.0);
        let width = self.width as usize;
        let height = self.height as usize;

        debug_assert_eq!(tile_count, width.saturating_mul(height));
        for x in 0..width {
            for y in 0..height {
                let index = x * height + y;
                if index >= tile_count {
                    break;
                }
                let offset = index * 4;
                let mut color = self.tile_color(index, visual.display_mode);

                if visual
                    .brush_preview
                    .is_some_and(|brush| tile_in_brush(x, y, brush, width, height))
                {
                    color[0] = (color[0] * 0.72 + 0.28 * 0.56).min(1.0);
                    color[1] = (color[1] * 0.72 + 0.28 * 0.26).min(1.0);
                    color[2] = (color[2] * 0.72 + 0.28 * 0.82).min(1.0);
                }
                if visual.hover_tile == Some((x as u32, y as u32)) {
                    color[0] = (color[0] * 0.65 + 0.35).min(1.0);
                    color[1] = (color[1] * 0.65 + 0.35).min(1.0);
                    color[2] = (color[2] * 0.65 + 0.35).min(1.0);
                }
                if visual.selected_tile == Some((x as u32, y as u32)) {
                    color[0] = (color[0] * 0.35 + 0.65).min(1.0);
                    color[1] = (color[1] * 0.35 + 0.55).min(1.0);
                    color[2] = (color[2] * 0.35 + 0.10).min(1.0);
                }
                self.lattice_rgba[offset..offset + 4].copy_from_slice(&color);
            }
        }

        debug_assert_eq!(self.cell_x.len(), self.cell_count());
        debug_assert_eq!(self.cell_y.len(), self.cell_count());
        debug_assert_eq!(self.cell_lineage.len(), self.cell_count());
        for cell_index in 0..self.cell_count() {
            let x = self.cell_x[cell_index] as usize;
            let y = self.cell_y[cell_index] as usize;
            if x >= width || y >= height {
                continue;
            }
            let tile_index = x * height + y;
            if tile_index >= tile_count {
                continue;
            }
            let id = self.cell_id[cell_index];
            let lineage = self.cell_lineage[cell_index];
            let cell_color = if visual.selected_cell == Some(id) {
                [1.0, 0.93, 0.30]
            } else {
                lineage_rgb01(lineage)
            };
            let alpha = if visual.selected_cell == Some(id)
                || visual.selected_lineage.is_none()
                || visual.selected_lineage == Some(lineage)
            {
                1.0
            } else {
                0.20
            };
            let offset = tile_index * 4;
            if alpha >= 1.0 {
                self.lattice_rgba[offset..offset + 3].copy_from_slice(&cell_color);
            } else {
                for (component, cell_component) in cell_color.iter().copied().enumerate() {
                    let tile_linear = srgb_to_linear(self.lattice_rgba[offset + component]);
                    let cell_linear = srgb_to_linear(cell_component);
                    self.lattice_rgba[offset + component] =
                        linear_to_srgb(tile_linear + (cell_linear - tile_linear) * alpha);
                }
            }
            self.lattice_rgba[offset + 3] = 1.0;
        }
    }

    fn tile_color(&self, index: usize, mode: RenderDisplayMode) -> [f32; 4] {
        let rgb = match mode {
            RenderDisplayMode::Enval => enval_rgb01(self.tile_enval[index]),
            RenderDisplayMode::Occupancy => blend_rgb01(
                [46.0, 57.0, 72.0],
                if self.tile_occupancy[index] != EMPTY_CELL_ID {
                    0.92
                } else {
                    0.0
                },
            ),
            RenderDisplayMode::Mass => blend_rgb01(
                [86.0, 154.0, 112.0],
                1.0 - (-(self.tile_mass[index] as f64) * 0.08).exp(),
            ),
            RenderDisplayMode::Molecules => blend_rgb01(
                [69.0, 139.0, 186.0],
                1.0 - (-(self.tile_molecule_count[index] as f64) * 0.30).exp(),
            ),
            element_mode => {
                let element = element_mode.element_index().unwrap_or(0);
                let present = self.tile_element_mask[index] & (1 << element) != 0;
                blend_rgb01(
                    TILE_ELEMENT_RGB255[element],
                    if present { 1.0 } else { 0.0 },
                )
            }
        };
        [rgb[0], rgb[1], rgb[2], 1.0]
    }
}

fn blend_rgb01(target: [f64; 3], intensity: f64) -> [f32; 3] {
    let t = intensity.clamp(0.0, 1.0);
    [
        ((TILE_NEUTRAL_RGB255[0] + (target[0] - TILE_NEUTRAL_RGB255[0]) * t) / 255.0) as f32,
        ((TILE_NEUTRAL_RGB255[1] + (target[1] - TILE_NEUTRAL_RGB255[1]) * t) / 255.0) as f32,
        ((TILE_NEUTRAL_RGB255[2] + (target[2] - TILE_NEUTRAL_RGB255[2]) * t) / 255.0) as f32,
    ]
}

fn enval_rgb01(value: f32) -> [f32; 3] {
    let value = if value.is_finite() { value as f64 } else { 0.0 };
    let mapped = (value * 1.25).atan() / (std::f64::consts::PI * 0.5);
    let magnitude = mapped.abs().min(1.0);
    let rgb = if mapped >= 0.0 {
        [
            (246.0 - 10.0 * magnitude).floor(),
            (246.0 - 92.0 * magnitude).floor(),
            (246.0 - 205.0 * magnitude).floor(),
        ]
    } else {
        [
            (246.0 - 205.0 * magnitude).floor(),
            (246.0 - 118.0 * magnitude).floor(),
            (246.0 - 10.0 * magnitude).floor(),
        ]
    };
    [
        (rgb[0] / 255.0) as f32,
        (rgb[1] / 255.0) as f32,
        (rgb[2] / 255.0) as f32,
    ]
}

fn lineage_rgb01(lineage_id: u32) -> [f32; 3] {
    let mut value = lineage_id ^ 0x9e37_79b9;
    value = (value ^ (value >> 16)).wrapping_mul(0x85eb_ca6b);
    value = (value ^ (value >> 13)).wrapping_mul(0xc2b2_ae35);
    value ^= value >> 16;
    hsl_to_rgb01((value % 360) as f64, 0.70, 0.55)
}

fn hsl_to_rgb01(hue_degrees: f64, saturation: f64, lightness: f64) -> [f32; 3] {
    let hue = hue_degrees.rem_euclid(360.0);
    let saturation = saturation.clamp(0.0, 1.0);
    let lightness = lightness.clamp(0.0, 1.0);
    let chroma = (1.0 - (2.0 * lightness - 1.0).abs()) * saturation;
    let hue_prime = hue / 60.0;
    let x = chroma * (1.0 - (hue_prime.rem_euclid(2.0) - 1.0).abs());
    let (r1, g1, b1) = match hue_prime as u32 {
        0 => (chroma, x, 0.0),
        1 => (x, chroma, 0.0),
        2 => (0.0, chroma, x),
        3 => (0.0, x, chroma),
        4 => (x, 0.0, chroma),
        _ => (chroma, 0.0, x),
    };
    let m = lightness - chroma * 0.5;
    [
        ((r1 + m) * 255.0).round().clamp(0.0, 255.0) as f32 / 255.0,
        ((g1 + m) * 255.0).round().clamp(0.0, 255.0) as f32 / 255.0,
        ((b1 + m) * 255.0).round().clamp(0.0, 255.0) as f32 / 255.0,
    ]
}

fn tile_in_brush(
    x: usize,
    y: usize,
    brush: RenderBrushPreview,
    width: usize,
    height: usize,
) -> bool {
    if width == 0 || height == 0 {
        return false;
    }
    let brush_width = (brush.width as usize).clamp(1, width);
    let brush_height = (brush.height as usize).clamp(1, height);
    let start_x = brush.x as i64 - (brush_width / 2) as i64;
    let start_y = brush.y as i64 - (brush_height / 2) as i64;
    axis_in_wrapped_span(x, start_x, brush_width, width)
        && axis_in_wrapped_span(y, start_y, brush_height, height)
}

fn axis_in_wrapped_span(value: usize, start: i64, span: usize, size: usize) -> bool {
    if span >= size {
        return true;
    }
    let normalized = start.rem_euclid(size as i64) as usize;
    let end = normalized + span;
    if end <= size {
        value >= normalized && value < end
    } else {
        value >= normalized || value < end - size
    }
}

fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(value: f32) -> f32 {
    if value <= 0.0031308 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lattice_rgba_retains_native_y_fastest_order_and_composites_cells() {
        let mut buffers = RenderBuffers {
            width: 2,
            height: 3,
            tile_enval: vec![0.0; 6],
            tile_occupancy: vec![EMPTY_CELL_ID; 6],
            tile_mass: vec![0; 6],
            tile_molecule_count: vec![0; 6],
            tile_element_mask: vec![0; 6],
            cell_id: vec![17],
            cell_x: vec![1],
            cell_y: vec![2],
            cell_lineage: vec![9],
            ..RenderBuffers::default()
        };
        buffers.refresh_lattice_rgba(&RenderVisualState {
            selected_cell: Some(17),
            ..RenderVisualState::default()
        });

        assert_eq!(buffers.lattice_rgba.len(), 24);
        let selected_offset = 5 * 4;
        assert_eq!(
            &buffers.lattice_rgba[selected_offset..selected_offset + 4],
            &[1.0, 0.93, 0.30, 1.0]
        );
        assert_ne!(&buffers.lattice_rgba[0..4], &[1.0, 0.93, 0.30, 1.0]);
    }

    #[test]
    fn brush_preview_wraps_across_native_grid_edges() {
        let brush = RenderBrushPreview {
            x: 0,
            y: 0,
            width: 3,
            height: 3,
        };
        assert!(tile_in_brush(3, 2, brush, 4, 3));
        assert!(tile_in_brush(0, 0, brush, 4, 3));
        assert!(!tile_in_brush(2, 0, brush, 4, 3));
    }
}
