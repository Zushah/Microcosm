use super::{
    LOCAL_ENVAL_RADIUS, LOCAL_ENVAL_WINDOW_AREA, LOCAL_ENVAL_WINDOW_DIAMETER, TileId, World,
    WorldError,
};
use crate::chem::{ELEMENT_COUNT, ELEMENT_ORDER, ElementAmounts};
use crate::environment;
use crate::spatial::{BilinearStencil, Position};

#[derive(Clone, Copy, Debug)]
pub(super) struct TileStencil {
    pub(super) tiles: [usize; 4],
    pub(super) weights: [f32; 4],
}

impl World {
    pub fn tile_enval(&self, tile_id: TileId) -> Option<f32> {
        self.enval.get(tile_id.index()).copied()
    }

    pub fn tile_element_amounts(&self, tile_id: TileId) -> Option<ElementAmounts> {
        self.element_fields.get(tile_id.index()).copied()
    }

    #[cfg(test)]
    pub(crate) fn set_tile_element_amounts(
        &mut self,
        tile_id: TileId,
        amounts: ElementAmounts,
    ) -> Result<(), WorldError> {
        if tile_id.index() >= self.tile_count {
            return Err(WorldError::InvalidTile(tile_id));
        }
        amounts.validate_nonnegative("tile_element_amounts")?;
        self.element_fields[tile_id.index()] = amounts;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn element_field_totals(&self) -> [f64; ELEMENT_COUNT] {
        let mut totals = [0.0_f64; ELEMENT_COUNT];
        for amounts in &self.element_fields {
            for element in ELEMENT_ORDER {
                totals[element.index()] += f64::from(amounts[element]);
            }
        }
        totals
    }

    pub fn set_tile_enval(&mut self, tile_id: TileId, value: f32) -> Result<(), WorldError> {
        if tile_id.index() >= self.tile_count {
            return Err(WorldError::InvalidTile(tile_id));
        }
        if !value.is_finite() {
            return Err(WorldError::NonFiniteEnvalInput(value));
        }
        let old = self.enval[tile_id.index()];
        self.enval[tile_id.index()] = value;
        let delta = f64::from(value) - f64::from(old);
        self.enval_sum += delta;
        self.avg_enval = (self.enval_sum / self.tile_count as f64) as f32;
        self.enval_ledger.edits += delta;
        Ok(())
    }

    pub fn adjust_tile_enval(&mut self, tile_id: TileId, delta: f32) -> Result<(), WorldError> {
        let applied = self.adjust_tile_enval_unrecorded(tile_id, delta)?;
        self.enval_ledger.edits += applied;
        Ok(())
    }

    pub(super) fn adjust_tile_enval_unrecorded(
        &mut self,
        tile_id: TileId,
        delta: f32,
    ) -> Result<f64, WorldError> {
        let applied = self.apply_tile_enval_delta(tile_id, delta)?;
        self.avg_enval = (self.enval_sum / self.tile_count as f64) as f32;
        Ok(applied)
    }

    pub(super) fn apply_tile_enval_delta(
        &mut self,
        tile_id: TileId,
        delta: f32,
    ) -> Result<f64, WorldError> {
        if tile_id.index() >= self.tile_count {
            return Err(WorldError::InvalidTile(tile_id));
        }
        if !delta.is_finite() || delta == 0.0 {
            return Ok(0.0);
        }
        let old = self.enval[tile_id.index()];
        let value = old + delta;
        if !value.is_finite() {
            return Err(WorldError::NonFiniteEnvalInput(value));
        }
        self.enval[tile_id.index()] = value;
        let applied = f64::from(value) - f64::from(old);
        self.enval_sum += applied;
        Ok(applied)
    }

    #[cfg(test)]
    pub(crate) fn set_all_enval(&mut self, value: f32) -> Result<(), WorldError> {
        if !value.is_finite() {
            return Err(WorldError::NonFiniteEnvalInput(value));
        }
        let before = self
            .enval
            .iter()
            .map(|value| f64::from(*value))
            .sum::<f64>();
        self.enval.fill(value);
        self.enval_sum = f64::from(value) * self.tile_count as f64;
        self.avg_enval = (self.enval_sum / self.tile_count as f64) as f32;
        self.enval_ledger.edits += self.enval_sum - before;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn sample_enval_at(&self, position: Position) -> Option<f32> {
        let stencil = BilinearStencil::new(position, self.width, self.height)?;
        let mut value = 0.0_f64;
        for sample in stencil.samples {
            value +=
                f64::from(self.enval[self.index_xy(sample.x, sample.y)]) * f64::from(sample.weight);
        }
        Some(value as f32)
    }

    pub fn sample_element_fields_at(&self, position: Position) -> Option<ElementAmounts> {
        let stencil = BilinearStencil::new(position, self.width, self.height)?;
        let mut result = ElementAmounts::ZERO;
        for element in ELEMENT_ORDER {
            let mut value = 0.0_f64;
            for sample in stencil.samples {
                value += f64::from(self.element_fields[self.index_xy(sample.x, sample.y)][element])
                    * f64::from(sample.weight);
            }
            result[element] = value as f32;
        }
        Some(result)
    }

    pub(super) fn tile_stencil(&self, position: Position) -> Option<TileStencil> {
        let position = position.wrapped(self.width as f32, self.height as f32)?;
        Some(self.canonical_tile_stencil(position))
    }

    pub(super) fn canonical_tile_stencil(&self, position: Position) -> TileStencil {
        debug_assert_eq!(
            position.wrapped(self.width as f32, self.height as f32),
            Some(position)
        );
        let (x0, y0) = (position.x as usize, position.y as usize);
        let x1 = if x0 + 1 == self.width { 0 } else { x0 + 1 };
        let y1 = if y0 + 1 == self.height { 0 } else { y0 + 1 };
        let fx = position.x - x0 as f32;
        let fy = position.y - y0 as f32;
        let one_minus_fx = 1.0 - fx;
        let one_minus_fy = 1.0 - fy;
        TileStencil {
            tiles: [
                self.index_xy(x0, y0),
                self.index_xy(x1, y0),
                self.index_xy(x0, y1),
                self.index_xy(x1, y1),
            ],
            weights: [
                one_minus_fx * one_minus_fy,
                fx * one_minus_fy,
                one_minus_fx * fy,
                fx * fy,
            ],
        }
    }

    pub(super) fn deposit_elements_at(
        &mut self,
        position: Position,
        amounts: ElementAmounts,
    ) -> Result<(), WorldError> {
        let stencil = self
            .tile_stencil(position)
            .ok_or(WorldError::InvalidCellPosition(position))?;
        self.deposit_elements_with(&stencil, amounts);
        Ok(())
    }

    pub(super) fn deposit_elements_with(&mut self, stencil: &TileStencil, amounts: ElementAmounts) {
        for (index, amount) in amounts.as_array().iter().enumerate() {
            if *amount == 0.0 {
                continue;
            }
            for (tile, weight) in stencil.tiles.into_iter().zip(stencil.weights) {
                self.element_fields[tile].as_mut_array()[index] += *amount * weight;
            }
        }
    }

    #[cfg(test)]
    pub(super) fn adjust_enval_at(
        &mut self,
        position: Position,
        delta: f32,
    ) -> Result<f64, WorldError> {
        if !delta.is_finite() || delta == 0.0 {
            return Ok(0.0);
        }
        let stencil = self
            .tile_stencil(position)
            .ok_or(WorldError::NonFiniteEnvalInput(delta))?;
        let applied = self.adjust_enval_with(&stencil, delta);
        self.refresh_average_enval();
        applied
    }

    pub(super) fn refresh_average_enval(&mut self) {
        self.avg_enval = (self.enval_sum / self.tile_count as f64) as f32;
    }

    pub(super) fn adjust_enval_with(
        &mut self,
        stencil: &TileStencil,
        delta: f32,
    ) -> Result<f64, WorldError> {
        if !delta.is_finite() || delta == 0.0 {
            return Ok(0.0);
        }
        let mut applied = 0.0;
        let mut result = Ok(());
        for (tile, weight) in stencil.tiles.into_iter().zip(stencil.weights) {
            let tile_delta = delta * weight;
            if tile_delta == 0.0 {
                continue;
            }
            let old = self.enval[tile];
            let value = old + tile_delta;
            if !value.is_finite() {
                result = Err(WorldError::NonFiniteEnvalInput(value));
                break;
            }
            self.enval[tile] = value;
            applied += f64::from(value) - f64::from(old);
        }
        self.enval_sum += applied;
        result.map(|()| applied)
    }

    pub(super) fn enval_phase(&mut self) {
        self.diffuse_enval_field();
        let kappa = self.landscape_kappa();
        let outcome = environment::recharge_and_relax(
            &mut self.enval,
            &mut self.element_fields,
            &self.enval_landscape,
            f64::from(self.config.enval_recharge.rate_per_second),
            self.config.dt_seconds,
            kappa,
        );
        self.enval_ledger.recharge += outcome.recharge.enval_delta;
        self.enval_ledger.recharge_amount += outcome.recharge.amount;
        self.enval_ledger.recharge_energy += outcome.recharge.energy;
        self.enval_ledger.source_inflow += outcome.relaxation;
        self.enval_sum = outcome.enval_sum;
        self.avg_enval = (outcome.enval_sum / self.tile_count as f64) as f32;
    }

    pub(super) fn landscape_kappa(&self) -> f64 {
        let rate = f64::from(self.config.enval_landscape.relaxation_per_second);
        if rate <= 0.0 {
            return 0.0;
        }
        1.0 - (-rate * self.config.dt_seconds).exp()
    }

    #[cfg(test)]
    pub(super) fn apply_recharge(&mut self) -> bool {
        let recharge_rate = f64::from(self.config.enval_recharge.rate_per_second);
        if recharge_rate <= 0.0 {
            return false;
        }
        let outcome = environment::recharge(
            &mut self.enval,
            &mut self.element_fields,
            recharge_rate,
            self.config.dt_seconds,
        );
        self.enval_ledger.recharge += outcome.enval_delta;
        self.enval_ledger.recharge_amount += outcome.amount;
        self.enval_ledger.recharge_energy += outcome.energy;
        true
    }

    #[cfg(test)]
    pub(super) fn relax_enval_landscape(&mut self) -> bool {
        let kappa = self.landscape_kappa();
        if kappa <= 0.0 {
            return false;
        }
        self.enval_ledger.source_inflow +=
            environment::relax_enval(&mut self.enval, &self.enval_landscape, kappa);
        true
    }

    #[cfg(test)]
    pub(super) fn refresh_enval_sum(&mut self) {
        let sum = self
            .enval
            .iter()
            .map(|value| f64::from(*value))
            .sum::<f64>();
        self.enval_sum = sum;
        self.avg_enval = (sum / self.tile_count as f64) as f32;
    }

    #[cfg(test)]
    pub(crate) fn diffuse_enval(&mut self) {
        self.diffuse_enval_field();
        self.refresh_enval_sum();
    }

    pub(super) fn diffuse_enval_field(&mut self) {
        let mix = (f64::from(self.config.enval_diffusivity_per_second) * self.config.dt_seconds
            / 9.0) as f32;
        let (width, height) = (self.width, self.height);
        let rows = &mut self.enval_row_sums;
        rows.resize(self.tile_count, 0.0);
        for x in 0..width {
            let left = if x == 0 { width - 1 } else { x - 1 } * height;
            let right = if x + 1 == width { 0 } else { x + 1 } * height;
            let here = x * height;
            for (sum, ((l, c), r)) in rows[here..here + height].iter_mut().zip(
                self.enval[left..left + height]
                    .iter()
                    .zip(&self.enval[here..here + height])
                    .zip(&self.enval[right..right + height]),
            ) {
                *sum = l + c + r;
            }
        }
        let mix_tile = |center: f32, up: f32, row: f32, down: f32| {
            let value = center + mix * (up + row + down - 9.0 * center);
            if value.is_finite() { value } else { 0.0 }
        };
        for x in 0..width {
            let here = x * height;
            let rows = &rows[here..here + height];
            let centers = &self.enval[here..here + height];
            let next = &mut self.enval_next[here..here + height];
            if height >= 3 {
                for (y, value) in next[1..height - 1].iter_mut().enumerate() {
                    *value = mix_tile(centers[y + 1], rows[y], rows[y + 1], rows[y + 2]);
                }
            }
            for y in [0, height - 1] {
                let up = if y == 0 { height - 1 } else { y - 1 };
                let down = if y + 1 == height { 0 } else { y + 1 };
                next[y] = mix_tile(centers[y], rows[up], rows[y], rows[down]);
            }
        }
        std::mem::swap(&mut self.enval, &mut self.enval_next);
    }

    pub(super) fn diffuse_element_fields(&mut self) {
        let (width, height) = (self.width, self.height);
        let mut mix = [0.0_f32; ELEMENT_COUNT];
        for element in ELEMENT_ORDER {
            mix[element.index()] = (f64::from(self.config.element_fields.diffusivities[element])
                * self.config.dt_seconds
                / 9.0) as f32;
        }
        let rows = &mut self.element_row_sums;
        rows.resize(self.tile_count, [0.0; ELEMENT_COUNT]);
        for x in 0..width {
            let left = if x == 0 { width - 1 } else { x - 1 } * height;
            let right = if x + 1 == width { 0 } else { x + 1 } * height;
            let here = x * height;
            for (sum, ((l, c), r)) in rows[here..here + height].iter_mut().zip(
                self.element_fields[left..left + height]
                    .iter()
                    .zip(&self.element_fields[here..here + height])
                    .zip(&self.element_fields[right..right + height]),
            ) {
                let (l, c, r) = (l.as_array(), c.as_array(), r.as_array());
                *sum = std::array::from_fn(|element| l[element] + c[element] + r[element]);
            }
        }
        let mix_tile = |center: &ElementAmounts,
                        up: &[f32; ELEMENT_COUNT],
                        row: &[f32; ELEMENT_COUNT],
                        down: &[f32; ELEMENT_COUNT]| {
            let center = center.as_array();
            let mut next = [0.0_f32; ELEMENT_COUNT];
            for element in 0..ELEMENT_COUNT {
                next[element] = center[element]
                    + mix[element]
                        * (up[element] + row[element] + down[element] - 9.0 * center[element]);
            }
            ElementAmounts::new(next)
        };
        for x in 0..width {
            let here = x * height;
            let rows = &rows[here..here + height];
            let centers = &self.element_fields[here..here + height];
            let next = &mut self.element_fields_next[here..here + height];
            let mut up = rows[height - 1];
            let mut row = rows[0];
            for ((value, center), down) in next.iter_mut().zip(centers).zip(&rows[1..]) {
                *value = mix_tile(center, &up, &row, down);
                up = row;
                row = *down;
            }
            next[height - 1] = mix_tile(&centers[height - 1], &up, &row, &rows[0]);
        }

        self.operation_counters.element_field_diffusion_tiles = self
            .operation_counters
            .element_field_diffusion_tiles
            .saturating_add(self.tile_count as u64);

        std::mem::swap(&mut self.element_fields, &mut self.element_fields_next);
    }

    pub(super) fn refresh_enval_smoothed(&mut self) {
        smooth_enval(
            &self.enval,
            self.width,
            self.height,
            &mut self.enval_smoothing_scratch,
            &mut self.enval_smoothed,
        );
        self.enval_smoothed_current = true;
    }

    pub(super) fn fresh_smoothed_enval(&self) -> Vec<f32> {
        let mut smoothed = Vec::new();
        smooth_enval(
            &self.enval,
            self.width,
            self.height,
            &mut Vec::new(),
            &mut smoothed,
        );
        smoothed
    }

    pub fn local_enval_at(&self, position: Position) -> Option<f32> {
        let stencil = self.tile_stencil(position)?;
        Some(sample_smoothed_enval(
            &self.fresh_smoothed_enval(),
            &stencil,
        ))
    }
}

pub(super) fn smooth_enval(
    enval: &[f32],
    width: usize,
    height: usize,
    scratch: &mut Vec<f64>,
    smoothed: &mut Vec<f32>,
) {
    let tile_count = enval.len();
    let radius = LOCAL_ENVAL_RADIUS;
    let padded_height = height + 2 * radius;
    scratch.resize(tile_count + padded_height, 0.0);
    smoothed.resize(tile_count, 0.0);
    let (column_sums, padded) = scratch.split_at_mut(tile_count);
    for x in 0..width {
        let column = &enval[x * height..(x + 1) * height];
        for (value, source) in padded[radius..radius + height].iter_mut().zip(column) {
            *value = f64::from(*source);
        }
        for offset in 0..radius {
            padded[offset] = f64::from(column[(offset + height * radius - radius) % height]);
            padded[radius + height + offset] = f64::from(column[offset % height]);
        }
        let sums = &mut column_sums[x * height..(x + 1) * height];
        for (y, sum) in sums.iter_mut().enumerate() {
            *sum = padded[y] + padded[y + 1] + padded[y + 2] + padded[y + 3] + padded[y + 4];
        }
    }
    let scale = 1.0 / LOCAL_ENVAL_WINDOW_AREA as f64;
    for x in 0..width {
        let columns: [usize; LOCAL_ENVAL_WINDOW_DIAMETER] =
            std::array::from_fn(|offset| (x + width * radius + offset - radius) % width * height);
        let row = &mut smoothed[x * height..(x + 1) * height];
        for (y, value) in row.iter_mut().enumerate() {
            let sum = column_sums[columns[0] + y]
                + column_sums[columns[1] + y]
                + column_sums[columns[2] + y]
                + column_sums[columns[3] + y]
                + column_sums[columns[4] + y];
            *value = (sum * scale) as f32;
        }
    }
}

#[inline]
pub(super) fn sample_smoothed_enval(smoothed: &[f32], stencil: &TileStencil) -> f32 {
    let mut value = 0.0_f64;
    for (tile, weight) in stencil.tiles.into_iter().zip(stencil.weights) {
        value += f64::from(smoothed[tile]) * f64::from(weight);
    }
    value as f32
}
