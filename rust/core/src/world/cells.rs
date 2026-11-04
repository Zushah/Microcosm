use super::fields::{TileStencil, sample_smoothed_enval};
use super::{LYSIS_ATTRIBUTION_TICKS, MAX_TRANSPORT_FRACTION, MIN_DAUGHTER_BODY, World};
use crate::bio::{self, EnzymeTerms, FluxCell, FluxExtent, mask_elements};
use crate::cell::{Cell, CellId, FluxRecord, ReactionSite};
use crate::chem::{ELEMENT_COUNT, ELEMENT_ORDER, ELEMENT_POLARITIES, ElementAmounts};
use crate::genome::Enzyme;
use crate::interaction::Contact;
use crate::spatial::Position;

#[derive(Clone, Copy, Debug)]
pub(super) struct CellReactionContext {
    pub(super) cell_id: CellId,
    pub(super) slot: usize,
    pub(super) position: Position,
    pub(super) stencil: TileStencil,
    pub(super) flux_cell: FluxCell,
    pub(super) watched: bool,
}

#[derive(Clone, Copy)]
pub(super) struct ReactingEnzyme<'a> {
    pub(super) index: usize,
    pub(super) enzyme: &'a Enzyme,
    pub(super) terms: &'a EnzymeTerms,
}

impl World {
    pub(super) fn catalyst_upkeep_per_sec(&self, cell: &Cell) -> f64 {
        self.config.catalyst_upkeep_per_sec * cell.genome.enzymes.len() as f64
    }

    pub(super) fn step_cells(&mut self) {
        self.cell_phase_scratch.clear();
        self.cell_phase_scratch.extend(self.cells.ids());
        if self.cell_phase_scratch.is_empty() {
            return;
        }
        self.refresh_enval_smoothed();
        for phase_index in 0..self.cell_phase_scratch.len() {
            let cell_id = self.cell_phase_scratch[phase_index];
            let Some(slot) = self.cells.slot(cell_id) else {
                continue;
            };
            self.cell_phase_row = Some(phase_index);
            self.step_cell(cell_id, slot);
        }
        self.cell_phase_row = None;
        self.enval_smoothed_current = false;
        self.refresh_average_enval();
    }

    pub(super) fn step_cell(&mut self, cell_id: CellId, slot: usize) {
        let dt_seconds = self.config.dt_seconds;
        let cell_body = self.config.cell_body;
        let cell = self.cells.at_slot_mut(slot);
        let position = cell.position;
        let radius = cell.radius;
        let enzymes = std::mem::take(&mut cell.genome.enzymes);
        let stencil = self.canonical_tile_stencil(position);
        debug_assert!(self.enval_smoothed_current);
        let local_enval = sample_smoothed_enval(&self.enval_smoothed, &stencil);
        let context = CellReactionContext {
            cell_id,
            slot,
            position,
            stencil,
            flux_cell: FluxCell::new(
                local_enval,
                std::f32::consts::PI * radius * radius,
                cell_body,
            ),
            watched: self.watched_cell_id == Some(cell_id),
        };
        let mut terms = std::mem::take(&mut self.enzyme_terms);
        terms.clear();
        terms.extend(
            enzymes
                .iter()
                .map(|enzyme| EnzymeTerms::new(enzyme, &context.flux_cell, dt_seconds)),
        );

        self.operation_counters.cell_steps += 1;
        self.operation_counters.local_enval_average_calls += 1;
        self.operation_counters.enzyme_entries_seen += enzymes.len() as u64;
        for (catalyst_index, enzyme) in enzymes.iter().enumerate() {
            self.count_reaction_attempt();
            let cell = self.cells.at_slot_mut(slot);
            let energy_before = cell.energy;
            let Some(flux) = bio::compute_extent(
                &terms[catalyst_index],
                enzyme,
                &cell.internal_elements,
                energy_before,
                1.0 - f64::from(enzyme.exposure),
                context.flux_cell.cell_area,
                &context.flux_cell,
            ) else {
                self.reaction_counters.no_substrate += 1;
                continue;
            };
            let internal = cell.internal_elements.as_mut_array();
            let coefficients = enzyme.reactants.as_array();
            for index in mask_elements(terms[catalyst_index].reactant_mask) {
                let before = internal[index];
                let consumed = flux.consumed(coefficients[index]);
                debug_assert!(consumed <= before + 1.0e-5);
                internal[index] = if consumed >= before {
                    0.0
                } else {
                    before - consumed
                };
            }
            self.apply_reaction_outcome(
                &context,
                ReactingEnzyme {
                    index: catalyst_index,
                    enzyme,
                    terms: &terms[catalyst_index],
                },
                ReactionSite::Internal,
                &flux,
                energy_before,
            );
        }

        let secreted_mask = std::mem::take(&mut self.pending_secretion_mask);
        if secreted_mask != 0 {
            let secretion = std::mem::replace(&mut self.pending_secretion, ElementAmounts::ZERO);
            for index in mask_elements(secreted_mask) {
                let amount = secretion.as_array()[index];
                for (tile, weight) in stencil.tiles.into_iter().zip(stencil.weights) {
                    self.element_fields[tile].as_mut_array()[index] += amount * weight;
                }
            }
        }
        self.step_surface_catalysis(&context, &enzymes, &terms);
        self.enzyme_terms = terms;
        self.apply_enval_exchange(&context);

        let cell = self.cells.at_slot_mut(slot);
        let area = cell.area(cell_body.body_density);
        cell.radius = (area / std::f64::consts::PI).sqrt() as f32;
        let radius = cell.radius;
        let maintenance_loss = (cell.maintenance_cost_per_area * area
            + self.config.catalyst_upkeep_per_sec * enzymes.len() as f64)
            * dt_seconds;
        cell.genome.enzymes = enzymes;
        self.max_cell_radius = self.max_cell_radius.max(radius);
        if maintenance_loss > 0.0 {
            self.energy_ledger.maintenance += maintenance_loss.min(cell.energy);
            cell.energy -= maintenance_loss;
        }
        if cell.energy <= 0.0 {
            cell.energy = 0.0;
            self.energy_death_count += 1;
            self.kill_cell_and_release(cell_id);
            return;
        }

        let (inward, outward) = self.transport_elements_with(slot, &stencil);
        self.reaction_counters.uptake_flux += inward;
        self.reaction_counters.leak_flux += outward;

        let cell = self.cells.at_slot(slot);
        let strain = cell.strain(cell_body.boundary_density);
        if strain < cell_body.lysis_strain {
            self.lysis_count += 1;
            if cell
                .last_digested_tick
                .is_some_and(|tick| self.tick_count.saturating_sub(tick) <= LYSIS_ATTRIBUTION_TICKS)
            {
                self.interaction_counters.lyses_after_digestion += 1;
            }
            self.kill_cell_and_release(cell_id);
        } else if strain >= cell_body.fission_strain {
            self.fission_cell(cell_id);
        }
    }

    #[inline]
    pub(super) fn count_reaction_attempt(&mut self) {
        self.operation_counters.metabolic_enzyme_attempts += 1;
        self.reaction_counters.attempts += 1;
    }

    #[inline(always)]
    pub(super) fn apply_reaction_outcome(
        &mut self,
        context: &CellReactionContext,
        reacting: ReactingEnzyme,
        site: ReactionSite,
        flux: &FluxExtent,
        energy_before: f64,
    ) {
        let ReactingEnzyme {
            index: catalyst_index,
            enzyme,
            terms,
        } = reacting;
        let cell = self.cells.at_slot_mut(context.slot);
        let coefficients = enzyme.products.as_array();
        let mut secreted_total = 0.0_f64;
        for index in mask_elements(terms.product_mask) {
            let produced = f64::from(coefficients[index]) * flux.executed_extent;
            cell.internal_elements.as_mut_array()[index] +=
                (produced * flux.retained_fraction) as f32;
            cell.body.as_mut_array()[index] += (produced * flux.body_fraction) as f32;
            cell.boundary.as_mut_array()[index] += (produced * flux.boundary_fraction) as f32;
            let secreted = (produced * flux.secretion_fraction) as f32;
            if secreted != 0.0 {
                secreted_total += f64::from(secreted);
                if site == ReactionSite::Internal {
                    self.pending_secretion.as_mut_array()[index] += secreted;
                    self.pending_secretion_mask |= 1 << index;
                } else {
                    for (tile, weight) in context
                        .stencil
                        .tiles
                        .into_iter()
                        .zip(context.stencil.weights)
                    {
                        self.element_fields[tile].as_mut_array()[index] += secreted * weight;
                    }
                }
            }
        }
        let energy_after = energy_before + flux.energy_delta;
        cell.energy = if energy_after < 0.0 && energy_after > -1.0e-9 {
            0.0
        } else {
            energy_after
        };
        debug_assert!(cell.energy >= 0.0);
        let energy_after = cell.energy;
        if flux.chemical_energy_delta >= 0.0 {
            self.energy_ledger.chemical_harvest += flux.chemical_energy_delta;
        } else {
            self.energy_ledger.chemical_cost -= flux.chemical_energy_delta;
        }
        self.energy_ledger.enval_harvest += flux.enval_energy;
        self.energy_ledger.pump_cost += flux.pump_cost;
        self.energy_ledger.assembly_cost += flux.assembly_cost;
        self.pending_enval_uptake += f64::from(flux.enval_input);
        self.pending_enval_emission += f64::from(flux.enval_output);

        self.operation_counters.reactions_succeeded += 1;
        self.reaction_counters.successes += 1;
        self.reaction_counters.energy_delta += flux.energy_delta;
        self.reaction_counters.enval_input += f64::from(flux.enval_input);
        self.reaction_counters.enval_output += f64::from(flux.enval_output);
        self.reaction_counters.executed_metabolic_flux += f64::from(flux.executed_extent_f32());
        self.reaction_counters.secretion_flux += secreted_total;

        if context.watched {
            let outcome = flux.outcome(enzyme);
            self.flux_log.push_flux_record(FluxRecord {
                tick_count: self.tick_count,
                sim_time_seconds: self.sim_time_seconds,
                cell_id: context.cell_id.index(),
                x: context.position.x,
                y: context.position.y,
                catalyst_index,
                site,
                reactants: *enzyme.reactants.as_array(),
                products: *enzyme.products.as_array(),
                requested_extent: outcome.requested_extent,
                executed_extent: outcome.executed_extent,
                element_deltas: outcome.element_deltas,
                secreted_elements: *outcome.secreted_products.as_array(),
                energy_before,
                energy_after,
                delta_cell_energy: outcome.energy_delta,
                raw_chemical_energy: outcome.raw_chemical_energy,
                enval_energy: outcome.enval_energy,
                enval_input: outcome.enval_input,
                enval_output: outcome.enval_output,
                local_enval: context.flux_cell.local_enval,
                enval_optimum: enzyme.enval_optimum,
            });
        }
    }

    pub(super) fn step_surface_catalysis(
        &mut self,
        context: &CellReactionContext,
        enzymes: &[Enzyme],
        terms: &[EnzymeTerms],
    ) {
        let mut exposed_mask = 0_u32;
        for (index, enzyme) in enzymes.iter().enumerate() {
            exposed_mask |= u32::from(enzyme.exposure > 0.0) << index;
        }
        if exposed_mask == 0 {
            return;
        }
        let radius = self.cells.at_slot(context.slot).radius;
        self.operation_counters.surface_contact_queries += 1;
        let mut contacts = std::mem::take(&mut self.contact_scratch);
        let field_share =
            self.collect_contacts(context.cell_id, context.position, radius, &mut contacts);
        while exposed_mask != 0 {
            let catalyst_index = exposed_mask.trailing_zeros() as usize;
            exposed_mask &= exposed_mask - 1;
            let reacting = ReactingEnzyme {
                index: catalyst_index,
                enzyme: &enzymes[catalyst_index],
                terms: &terms[catalyst_index],
            };
            let exposure = f64::from(reacting.enzyme.exposure);
            if field_share > 0.0 {
                self.surface_reaction_on_field(context, reacting, exposure * field_share);
            }
            for contact in &contacts {
                if contact.share > 0.0 {
                    self.surface_reaction_on_cell(
                        context,
                        reacting,
                        exposure * contact.share,
                        contact,
                    );
                }
            }
        }
        self.contact_scratch = contacts;
    }

    pub(super) fn surface_reaction_on_field(
        &mut self,
        context: &CellReactionContext,
        reacting: ReactingEnzyme,
        capacity_fraction: f64,
    ) {
        let ReactingEnzyme { enzyme, terms, .. } = reacting;
        let stencil = &context.stencil;
        let tile_indices = stencil.tiles;
        let reactants = enzyme.reactants.as_array();
        let mut weighted = [[0.0_f64; 4]; ELEMENT_COUNT];
        let mut pool = ElementAmounts::ZERO;
        for index in mask_elements(terms.reactant_mask) {
            let mut total = 0.0_f64;
            for (corner, weight) in stencil.weights.iter().enumerate() {
                let amount = f64::from(*weight)
                    * f64::from(self.element_fields[tile_indices[corner]].as_array()[index]);
                weighted[index][corner] = amount;
                total += amount;
            }
            pool.as_mut_array()[index] = total as f32;
        }

        self.count_reaction_attempt();
        let energy_before = self.cells.at_slot(context.slot).energy;
        let Some(flux) = bio::compute_extent(
            terms,
            enzyme,
            &pool,
            energy_before,
            capacity_fraction,
            1.0,
            &context.flux_cell,
        ) else {
            self.reaction_counters.no_substrate += 1;
            return;
        };

        for index in mask_elements(terms.reactant_mask) {
            let consumed = f64::from(flux.consumed(reactants[index]));
            let shares = weighted[index];
            let total = shares.iter().sum::<f64>();
            if consumed <= 0.0 || total <= 0.0 {
                continue;
            }
            let last = (0..4)
                .rev()
                .find(|corner| shares[*corner] > 0.0)
                .expect("a positive pool has a contributing tile");
            let mut remaining = consumed;
            for corner in 0..4 {
                if shares[corner] <= 0.0 {
                    continue;
                }
                let wanted = if corner == last {
                    remaining
                } else {
                    consumed * shares[corner] / total
                };
                let tile = &mut self.element_fields[tile_indices[corner]].as_mut_array()[index];
                let wanted = wanted as f32;
                let amount = if wanted < *tile { wanted } else { *tile };
                let amount = if amount > 0.0 { amount } else { 0.0 };
                *tile -= amount;
                remaining -= f64::from(amount);
            }
        }
        self.interaction_counters.field_surface_extent += f64::from(flux.executed_extent_f32());
        self.apply_reaction_outcome(context, reacting, ReactionSite::Field, &flux, energy_before);
    }

    pub(super) fn surface_reaction_on_cell(
        &mut self,
        context: &CellReactionContext,
        reacting: ReactingEnzyme,
        capacity_fraction: f64,
        contact: &Contact,
    ) {
        let ReactingEnzyme { enzyme, terms, .. } = reacting;
        let target = contact.cell_id;
        let target_slot = contact.slot;
        debug_assert_ne!(target, context.cell_id);
        debug_assert_eq!(self.cells.slot(target), Some(target_slot));
        let energy_before = self.cells.at_slot(context.slot).energy;
        let target_cell = self.cells.at_slot_mut(target_slot);
        let perimeter = std::f32::consts::TAU * target_cell.radius;

        self.reaction_counters.attempts += 1;
        self.operation_counters.metabolic_enzyme_attempts += 1;
        let Some(flux) = bio::compute_extent(
            terms,
            enzyme,
            &target_cell.boundary,
            energy_before,
            capacity_fraction,
            f64::from(perimeter),
            &context.flux_cell,
        ) else {
            self.reaction_counters.no_substrate += 1;
            return;
        };

        let boundary = target_cell.boundary.as_mut_array();
        let mut digested = 0.0_f64;
        let coefficients = enzyme.reactants.as_array();
        for index in mask_elements(terms.reactant_mask) {
            let before = boundary[index];
            let consumed = flux.consumed(coefficients[index]);
            debug_assert!(consumed <= before + 1.0e-5);
            boundary[index] = if consumed >= before {
                0.0
            } else {
                before - consumed
            };
            digested += f64::from(consumed.min(before));
        }
        target_cell.last_digested_tick = Some(self.tick_count);
        let target_lineage = target_cell.lineage_id;
        self.interaction_counters.cell_surface_extent += f64::from(flux.executed_extent_f32());
        self.interaction_counters.boundary_digested += digested;
        if target_lineage != self.cells.at_slot(context.slot).lineage_id {
            self.interaction_counters.cross_lineage_boundary_digested += digested;
        }
        self.apply_reaction_outcome(
            context,
            reacting,
            ReactionSite::Cell(target),
            &flux,
            energy_before,
        );
    }

    #[cfg(test)]
    pub(super) fn transport_elements(&mut self, slot: usize, position: Position) -> (f64, f64) {
        let Some(stencil) = self.tile_stencil(position) else {
            return (0.0, 0.0);
        };
        self.transport_elements_with(slot, &stencil)
    }

    pub(super) fn transport_elements_with(
        &mut self,
        slot: usize,
        stencil: &TileStencil,
    ) -> (f64, f64) {
        let cell = self.cells.at_slot_mut(slot);
        let radius = f64::from(cell.radius);
        let area = std::f64::consts::PI * radius * radius;
        let perimeter = 2.0 * std::f64::consts::PI * radius;
        let base_fraction =
            f64::from(self.config.membrane_permeability) * perimeter * self.config.dt_seconds
                / (2.0 * area);
        let membrane_polarity = cell.membrane_polarity();
        let sigma = f64::from(self.config.cell_body.permeability_sigma);
        let width_term = 2.0 * sigma * sigma;
        let tile_indices = stencil.tiles;
        let weights = stencil.weights.map(f64::from);
        let internal = cell.internal_elements.as_mut_array();

        let mut inward = 0.0_f64;
        let mut outward = 0.0_f64;
        for index in 0..ELEMENT_COUNT {
            let distance = ELEMENT_POLARITIES[index] - membrane_polarity;
            let permeability_factor = 0.25 + 1.5 * (-distance * distance / width_term).exp();
            let fraction = (base_fraction * permeability_factor).min(MAX_TRANSPORT_FRACTION);
            let mut weighted_sources = [0.0_f64; 4];
            for corner in 0..4 {
                weighted_sources[corner] = weights[corner]
                    * f64::from(self.element_fields[tile_indices[corner]].as_array()[index]);
            }
            let outside = weighted_sources.iter().sum::<f64>();
            let held = internal[index];
            let inside = f64::from(held) / area;
            let delta = fraction * (outside - inside) * area;
            if delta > 0.0 && outside > 0.0 {
                let wanted = weighted_sources.map(|source| (delta * source / outside) as f32);
                let mut moved = 0.0_f32;
                for corner in 0..4 {
                    if weighted_sources[corner] <= 0.0 {
                        continue;
                    }
                    let tile = &mut self.element_fields[tile_indices[corner]].as_mut_array()[index];
                    let amount = if wanted[corner] < *tile {
                        wanted[corner]
                    } else {
                        *tile
                    };
                    *tile -= amount;
                    moved += amount;
                }
                internal[index] = held + moved;
                inward += f64::from(moved);
            } else if delta < 0.0 {
                let amount = ((-delta) as f32).min(held);
                if amount <= 0.0 {
                    continue;
                }
                internal[index] = held - amount;
                for (tile, weight) in tile_indices.into_iter().zip(stencil.weights) {
                    self.element_fields[tile].as_mut_array()[index] += amount * weight;
                }
                outward += f64::from(amount);
            }
        }
        if inward > 0.0 || outward > 0.0 {
            self.operation_counters.element_uptake_events += 1;
        }
        (inward, outward)
    }

    pub(super) fn apply_enval_exchange(&mut self, context: &CellReactionContext) {
        let uptake = std::mem::take(&mut self.pending_enval_uptake) as f32;
        let emission = std::mem::take(&mut self.pending_enval_emission) as f32;
        if uptake != 0.0
            && let Ok(applied) = self.adjust_enval_with(&context.stencil, -uptake)
        {
            self.enval_ledger.cell_uptake += applied;
        }
        if emission != 0.0
            && let Ok(applied) = self.adjust_enval_with(&context.stencil, emission)
        {
            self.enval_ledger.cell_emission += applied;
        }
    }

    pub(super) fn fission_cell(&mut self, cell_id: CellId) {
        let parent = &self.cells[cell_id];
        if parent.strain(self.config.cell_body.boundary_density)
            < self.config.cell_body.fission_strain
            || parent.body.total() * 0.5 < MIN_DAUGHTER_BODY
        {
            return;
        }
        let fraction = 0.5 + (self.rng.next_f64() - 0.5) * 0.1;
        if parent.body.total() * fraction.min(1.0 - fraction) < MIN_DAUGHTER_BODY {
            return;
        }
        let angle = self.rng.range(0.0, std::f32::consts::TAU);
        let axis = Position::new(angle.cos(), angle.sin());
        let old_position = parent.position;
        let lineage = parent.lineage_id;
        let genome = parent.genome.mutate(&mut self.rng);
        let child_energy = parent.energy * fraction;
        let mut body = ElementAmounts::ZERO;
        let mut boundary = ElementAmounts::ZERO;
        let mut free = ElementAmounts::ZERO;
        let parent = &mut self.cells[cell_id];
        for element in ELEMENT_ORDER {
            body[element] = (f64::from(parent.body[element]) * fraction) as f32;
            boundary[element] = (f64::from(parent.boundary[element]) * fraction) as f32;
            free[element] = (f64::from(parent.internal_elements[element]) * fraction) as f32;
            parent.body[element] -= body[element];
            parent.boundary[element] -= boundary[element];
            parent.internal_elements[element] -= free[element];
        }
        parent.energy -= child_energy;
        parent.refresh_radius(self.config.cell_body.body_density);
        let parent_radius = parent.radius;
        let mut child = Cell::new(
            genome,
            old_position,
            self.sim_time_seconds,
            child_energy,
            body,
            boundary,
            self.config.cell_body.body_density,
        );
        child.internal_elements = free;
        let parent_position = Position::new(
            old_position.x - axis.x * child.radius,
            old_position.y - axis.y * child.radius,
        )
        .wrapped(self.width as f32, self.height as f32)
        .expect("finite fission position");
        child.position = Position::new(
            old_position.x + axis.x * parent_radius,
            old_position.y + axis.y * parent_radius,
        )
        .wrapped(self.width as f32, self.height as f32)
        .expect("finite fission position");
        self.update_cell_position(cell_id, parent_position);
        let child_position = child.position;
        let child_id = self.cells.insert(child);
        assert!(self.spatial_index.insert(child_id, child_position));
        self.increment_tile_cell_count(child_position);
        self.record_lineage_birth(lineage);
        self.birth_count += 1;
        self.operation_counters.cell_divisions += 1;
        self.reaction_counters.divisions += 1;
    }

    pub(super) fn kill_cell_and_release(&mut self, cell_id: CellId) {
        let Some(cell) = self.cells.get(cell_id) else {
            return;
        };
        let position = cell.position;
        let mut released_elements = cell.internal_elements;
        for element in ELEMENT_ORDER {
            released_elements[element] += cell.body[element] + cell.boundary[element];
        }
        self.deposit_elements_at(position, released_elements)
            .expect("active cell position must support conservative death release");
        let cell = self.remove_live_cell(cell_id);
        self.energy_ledger.death_loss += cell.energy;
        self.record_lineage_death(cell.lineage_id);
        self.death_count = self.death_count.saturating_add(1);
        self.operation_counters.cell_deaths = self.operation_counters.cell_deaths.saturating_add(1);
    }

    pub(super) fn remove_live_cell(&mut self, cell_id: CellId) -> Cell {
        let cell = self
            .cells
            .remove(cell_id)
            .expect("removed cell id must refer to a live cell");
        assert!(self.spatial_index.remove(cell_id, cell.position));
        self.decrement_tile_cell_count(cell.position);
        cell
    }
}
