import { ENZYME_CLASSES, attemptReaction } from "./bio.js";
import { ALL_ELEMENT_MASK, ELEMENT_MASKS, ELEMENT_ORDER, compositionToElementMask, maskToString, normalizeSpecificityMask, refreshMoleculeDerivedState } from "./chem.js";
import { chance, random, randomGaussian, randomInt } from "./rng.js";
import { cloneEnzyme, isCombatEnzymeType, mutateCombatLevel, normalizeCombatLevel } from "./eco.js";

const INTERNAL_ELEMENT_INDEX = { A: 0, B: 1, C: 2, D: 3, E: 4, F: 5 };

export class Cell {
    constructor(genome) {
        this.genome = genome;
        this.energy = genome.initialEnergy || 0;
        this.molecules = [];
        this.timeWithoutFood = 0;
        this.state = "active";
        this.lineageId = genome.lineageId || randomInt(1e9);
        this.birthSimTime = this.birthSimTime ?? (window.SIM_TIME || 0);
        this.deathSimTime = null;
        this._highlight = false;
        this.reactionLog = [];
        this.reactionLogMax = 40;
        this.maintenanceCostPerSec = genome.maintenanceCostPerSec ?? 0.05;
        this._internalElementCounts = new Int32Array(6);
        this._internalAtomCount = 0;
        this._internalCompositionDirty = false;
        this._refreshCombatTotals();
    }

    step(env, tile) {
        if (this.state !== "active") return;

        let positiveEnergyGain = 0;

        const tileMolecules = tile.molecules;
        const internalMolecules = this.molecules;
        const world = tile.__world;

        if (world && tile) {
            env.enval = tile.enval;
            env.localEnval = world.getLocalEnvalAverage(tile.__x, tile.__y, 2);
            env.avgEnval = world.avgEnval;
        }

        for (const enzyme of this.genome.enzymes) {
            const localEnvalBefore = env.localEnval ?? tile.enval ?? 0;
            const result = attemptReaction(enzyme, tileMolecules, env, this, tile, internalMolecules);
            if (!result) continue;

            if (result.elementDelta && typeof window !== "undefined" && typeof window.__recordElementDelta === "function") {
                window.__recordElementDelta(result.elementDelta);
            }

            const logConsumed = snapshotMolecules(result.consumed || []);
            const logProduced = snapshotMolecule(result.produced || null);
            const logByproducts = snapshotMolecules(result.byproducts || []);
            const normalizedSpecificity = normalizeSpecificityMask(enzyme.specificityMask, ALL_ELEMENT_MASK);

            const eDelta = result.energyDelta || 0;
            if (eDelta !== 0) {
                this.energy += eDelta;
                if (eDelta > 0) positiveEnergyGain += eDelta;
            }

            this.consumeSubstrates(result.consumed, tile);

            if (result.produced) {
                if (this.shouldSecrete(result.produced, enzyme)) {
                    if (world) world.addProductAround(tile.__x, tile.__y, result.produced);
                    else tileMolecules.push(result.produced);
                } else {
                    internalMolecules.push(result.produced);
                    this._markInternalCompositionDirty();
                }
            }

            if (result.byproducts && result.byproducts.length > 0) {
                if (world) for (const bp of result.byproducts) world.addProductAround(tile.__x, tile.__y, bp);
                else for (const bp of result.byproducts) tileMolecules.push(bp);
            }

            const envalInput = result.envalInput || 0;
            const envalOutput = result.envalOutput || 0;
            let envalChanged = false;
            if (envalInput !== 0) {
                if (world && typeof world.adjustTileEnval === "function") world.adjustTileEnval(tile, -envalInput);
                else tile.enval -= envalInput;
                envalChanged = true;
            }
            if (envalOutput !== 0) {
                if (world && typeof world.addEnvalAround === "function") world.addEnvalAround(tile.__x, tile.__y, envalOutput);
                else tile.enval += envalOutput;
                envalChanged = true;
            }

            if (envalChanged && world && tile) {
                env.enval = tile.enval;
                env.localEnval = world.getLocalEnvalAverage(tile.__x, tile.__y, 2);
                env.avgEnval = world.avgEnval;
            }

            const ageAtEvent = (typeof this.getAgeSecSim === "function") ? this.getAgeSecSim() : 0;

            this._pushReactionLog({
                enzymeType: enzyme.type || "unknown",
                specificityMask: normalizedSpecificity,
                specificity: maskToString(normalizedSpecificity),
                consumed: logConsumed,
                produced: logProduced,
                byproducts: logByproducts,
                deltaE: Number((eDelta || 0).toFixed(3)),
                chemicalDelta: Number((result.chemicalDelta || 0).toFixed(3)),
                envalEnergy: Number((result.envalEnergy || 0).toFixed(3)),
                bondEnergyDelta: Number((result.bondEnergyDelta || 0).toFixed(3)),
                bondEnergyRaw: Number((result.bondEnergyRaw || 0).toFixed(3)),
                transmutationEnergyDelta: Number((result.transmutationEnergyDelta || 0).toFixed(3)),
                transmutationEnergyRaw: Number((result.transmutationEnergyRaw || 0).toFixed(3)),
                catalyticCost: Number((result.catalyticCost || 0).toFixed(3)),
                localEnval: Number(localEnvalBefore.toFixed(4)),
                envalInput: Number((envalInput || 0).toFixed(4)),
                envalOutput: Number((envalOutput || 0).toFixed(4)),
                deltaEnval: Number((result.deltaEnval || 0).toFixed(4)),
                substrateAtomEnergy: Number((result.substrateAtomEnergy || 0).toFixed(3)),
                productAtomEnergy: Number((result.productAtomEnergy || 0).toFixed(3)),
                substrateBondEnergy: Number((result.substrateBondEnergy || 0).toFixed(3)),
                productBondEnergy: Number((result.productBondEnergy || 0).toFixed(3)),
                rawDelta: Number((result.rawDelta || 0).toFixed(3)),
                transmutedFrom: result.transmutedFrom || null,
                transmutedTo: result.transmutedTo || null,
                transmutedDirection: result.transmutedDirection || null,
                ageAtEventSec: Number(ageAtEvent.toFixed(3)),
                simTime: window.SIM_TIME || 0
            });
        }

        if (positiveEnergyGain > 1e-6) {
            this.timeWithoutFood = Math.max(0, this.timeWithoutFood - positiveEnergyGain * 0.2);
        } else {
            this.timeWithoutFood += env.dt;
        }

        const loss = this.maintenanceCostPerSec * env.dt;
        if (loss > 0) {
            this.energy -= loss;
            if (!(this.energy > 0)) {
                this.energy = 0;
                this._dieAndRelease(tile);
                return;
            }
        }

        const internalCount = this.totalInternalAtoms();
        if (internalCount < (this.genome.desiredElementReserve * 2) && tileMolecules.length > 0) {
            const idx = randomInt(tileMolecules.length);
            let take = tileMolecules[idx];
            if (take) {
                if (world && typeof world._removeMoleculeFromTile === "function") take = world._removeMoleculeFromTile(tile, idx) || take;
                else tileMolecules.splice(idx, 1);
                internalMolecules.push(take);
                this._markInternalCompositionDirty();
            }
        }

        const localEnval = env.localEnval ?? tile.enval ?? 0;
        const optimalEnval = (typeof this.genome.optimalEnval === "number") ? this.genome.optimalEnval : 0;
        const dist = Math.abs(localEnval - optimalEnval);
        const envalStressFactor = this.genome.envalStressFactor ?? 0.02;
        const stressIncrement = Math.pow(dist, 1.6) * envalStressFactor * Math.max(1, this.genome.enzymes.length || 1);

        this.timeWithoutFood += stressIncrement;

        if (this.timeWithoutFood > this.genome.decayTime) {
            this._dieAndRelease(tile);
            return;
        }

        if (this.energy >= this.genome.reproThreshold) {
            this.divide(tile);
        }
    }

    _dieAndRelease(tile) {
        if (tile && tile.__world && typeof tile.__world._addMoleculeToTile === "function") {
            const world = tile.__world;
            for (const m of this.molecules) world._addMoleculeToTile(tile, m);
        } else if (tile && tile.molecules) for (const m of this.molecules) tile.molecules.push(m);
        this.molecules = [];
        this._resetInternalCompositionCache();
        this.deathSimTime = window.SIM_TIME || 0;
        this.state = "dead";
        if (typeof window !== "undefined" && typeof window.__recordLineageDeath === "function") {
            window.__recordLineageDeath(this.lineageId);
        }
        if (typeof window !== "undefined" && typeof window.__recordCellDeath === "function") {
            window.__recordCellDeath(this);
        }
    }

    _pushReactionLog(entry) {
        this.reactionLog.unshift(entry);
        if (this.reactionLog.length > this.reactionLogMax) this.reactionLog.length = this.reactionLogMax;
    }

    addEnzymeToGenome(enzyme) {
        if (!enzyme || !enzyme.type || this.state === "dead") return null;
        if (!this.genome) this.genome = {};
        if (!Array.isArray(this.genome.enzymes)) this.genome.enzymes = [];
        const previousEnzymes = this.genome.enzymes.map((entry) => cloneEnzyme(entry));
        const nextEnzyme = cloneEnzyme(enzyme);
        applyEnzymeClassDefaults(nextEnzyme);
        this.genome.enzymes.push(nextEnzyme);
        this._refreshCombatTotals();
        if (typeof window !== "undefined" && typeof window.__recordCellGenomeChange === "function") window.__recordCellGenomeChange(this, previousEnzymes);
        return nextEnzyme;
    }

    totalInternalAtoms() {
        this._ensureInternalCompositionCache();
        return this._internalAtomCount;
    }

    shouldSecrete(product, enzyme) {
        const reserve = this.genome.desiredElementReserve ?? 2;
        for (const el in product.composition) {
            const have = this.countInternalElement(el);
            if (have < reserve) return false;
        }
        const prob = enzyme.secretionProb ?? this.genome.defaultSecretionProb ?? 0.15;
        return chance(prob);
    }

    countInternalElement(el) {
        this._ensureInternalCompositionCache();
        const idx = INTERNAL_ELEMENT_INDEX[el];
        if (idx === undefined) return 0;
        return this._internalElementCounts[idx] || 0;
    }

    consumeSubstrates(substrates, tile) {
        if (!substrates || substrates.length === 0) return;
        const tilePool = tile && tile.molecules ? tile.molecules : null;
        const cellPool = this.molecules;
        const world = tile && tile.__world;
        let changedInternalMolecules = false;
        const subtractFromPool = (pool, subComp) => {
            if (!pool) return false;
            for (let i = 0; i < pool.length; i++) {
                const m = pool[i];
                if (!m || !m.composition) continue;
                let ok = true;
                for (const el in subComp) if ((m.composition[el] || 0) < subComp[el]) { ok = false; break; }
                if (!ok) continue;
                const changedTileMolecule = pool === tilePool && world && typeof world._applyTileCompositionDelta === "function";
                if (changedTileMolecule) world._applyTileCompositionDelta(tile, subComp, -1);
                for (const el in subComp) {
                    m.composition[el] -= subComp[el];
                    if (m.composition[el] <= 0) delete m.composition[el];
                }
                if (pool === cellPool) changedInternalMolecules = true;
                if (Object.keys(m.composition).length === 0) {
                    if (pool === tilePool && world && typeof world._removeMoleculeFromTile === "function" && m.__tile === tile) world._removeMoleculeFromTile(tile, i);
                    else pool.splice(i, 1);
                } else {
                    refreshMoleculeDerivedState(m);
                    if (changedTileMolecule && typeof world._scheduleMoleculeDiffusion === "function" && m.__tile === tile) world._scheduleMoleculeDiffusion(m);
                }
                return true;
            }
            return false;
        };
        for (let si = 0; si < substrates.length; si++) {
            const sub = substrates[si];
            if (!sub || !sub.composition) continue;
            if (tilePool) {
                if (world && typeof world._removeMoleculeFromTile === "function" && sub.__tile === tile) {
                    world._removeMoleculeFromTile(tile, sub.__tileIndex);
                    continue;
                }
                const idx = tilePool.indexOf(sub);
                if (idx !== -1) {
                    if (world && typeof world._removeMoleculeFromTile === "function") world._removeMoleculeFromTile(tile, idx);
                    else tilePool.splice(idx, 1);
                    continue;
                }
            }
            const idx2 = cellPool.indexOf(sub);
            if (idx2 !== -1) {
                cellPool.splice(idx2, 1);
                changedInternalMolecules = true;
                continue;
            }
            if (!subtractFromPool(tilePool, sub.composition)) subtractFromPool(cellPool, sub.composition);
        }
        if (changedInternalMolecules) this._markInternalCompositionDirty();
    }

    divide(tile) {
        const world = this._worldRef;
        const worldAvgEnval = world ? world.avgEnval : 0;
        const childGenome = mutateGenome(this.genome, worldAvgEnval);

        const childEnergy = this.energy * (0.5 + (random() - 0.5) * 0.1);
        this.energy = Math.max(0, this.energy - childEnergy);

        const childMolecules = [];
        for (let i = this.molecules.length - 1; i >= 0; i--) {
            if (chance(0.5)) {
                childMolecules.push(this.molecules[i]);
                this.molecules.splice(i, 1);
            }
        }
        if (childMolecules.length > 0) this._markInternalCompositionDirty();

        const child = new Cell(childGenome);
        child.energy = childEnergy;
        child.molecules = childMolecules;
        if (childMolecules.length > 0) child._markInternalCompositionDirty();
        child.lineageId = this.lineageId;
        child.birthSimTime = window.SIM_TIME || 0;

        let placed = false;

        if (world && tile && typeof tile.__x === "number" && typeof tile.__y === "number") {
            const radius = 2;
            const candidates = [];

            for (let dx = -radius; dx <= radius; dx++) {
                for (let dy = -radius; dy <= radius; dy++) {
                    if (dx === 0 && dy === 0) continue;
                    const px = (tile.__x + dx + world.width) % world.width;
                    const py = (tile.__y + dy + world.height) % world.height;
                    if (world.grid[px][py].cells.length === 0) {
                        candidates.push([px, py]);
                    }
                }
            }

            if (candidates.length === 0) {
                this.energy += childEnergy;
                for (const m of childMolecules) this.molecules.push(m);
                return;
            }

            const [px, py] = candidates[randomInt(candidates.length)];
            world.grid[px][py].cells.push(child);
            child._worldRef = world;
            child._tileX = px;
            child._tileY = py;
            placed = true;
        } else {
            if (tile && tile.cells && tile.cells.length === 0) {
                tile.cells.push(child);
                placed = true;
            } else if (tile && tile.cells) {
                this.energy += childEnergy;
                for (const m of childMolecules) this.molecules.push(m);
                return;
            }
        }

        if (placed && typeof window !== "undefined" && typeof window.__recordLineageBirth === "function") {
            window.__recordLineageBirth(child.lineageId);
        }

        if (placed && typeof window !== "undefined" && typeof window.__recordCellBirth === "function") {
            window.__recordCellBirth(child);
        }

        if (chance(this.genome.postDivideMortality ?? 0.0)) this._dieAndRelease(tile);
    }

    absorbConsumedCell(victim, combatOutcome = null) {
        if (!victim || victim === this) return;

        const previousEnzymes = (this.genome && Array.isArray(this.genome.enzymes))
            ? this.genome.enzymes.map((enzyme) => cloneEnzyme(enzyme))
            : [];

        if (!this.genome) this.genome = { enzymes: [] };
        if (!Array.isArray(this.genome.enzymes)) this.genome.enzymes = [];

        const victimEnzymes = (victim.genome && Array.isArray(victim.genome.enzymes))
            ? victim.genome.enzymes
            : [];
        for (let i = 0; i < victimEnzymes.length; i++) {
            const absorbed = cloneEnzyme(victimEnzymes[i]);
            applyEnzymeClassDefaults(absorbed);
            this.genome.enzymes.push(absorbed);
        }
        this._refreshCombatTotals();

        if (typeof window !== "undefined" && typeof window.__recordCellGenomeChange === "function") {
            window.__recordCellGenomeChange(this, previousEnzymes);
        }

        const absorbedEnergy = Math.max(0, victim.energy || 0);
        if (absorbedEnergy > 0) {
            this.energy += absorbedEnergy;
            this.timeWithoutFood = Math.max(0, this.timeWithoutFood - absorbedEnergy * 0.2);
        }

        if (Array.isArray(victim.molecules) && victim.molecules.length > 0) {
            for (let i = 0; i < victim.molecules.length; i++) this.molecules.push(victim.molecules[i]);
            this._markInternalCompositionDirty();
        }

        victim.energy = 0;
        victim.molecules = [];
        victim._resetInternalCompositionCache();
        if (victim.genome && Array.isArray(victim.genome.enzymes)) victim.genome.enzymes = [];

        if (combatOutcome) {
            const ageAtEvent = this.getAgeSecSim ? this.getAgeSecSim() : 0;
            this._pushReactionLog({
                enzymeType: "attackase",
                substrates: [`lineage:${victim.lineageId}`],
                product: "phagocytosis",
                byproducts: [],
                deltaE: Number(absorbedEnergy.toFixed(3)),
                chemicalDelta: 0,
                envalEnergy: 0,
                bondEnergyDelta: 0,
                bondEnergyRaw: 0,
                transmutationEnergyDelta: 0,
                transmutationEnergyRaw: 0,
                catalyticCost: 0,
                localEnval: 0,
                envalInput: 0,
                envalOutput: 0,
                deltaEnval: 0,
                substrateAtomEnergy: 0,
                productAtomEnergy: 0,
                substrateBondEnergy: 0,
                productBondEnergy: 0,
                rawDelta: 0,
                attackLevel: combatOutcome.winnerAttack || 0,
                defenseLevel: combatOutcome.loserDefense || 0,
                attackMargin: combatOutcome.winnerMargin || 0,
                ageAtEventSec: Number(ageAtEvent.toFixed(3)),
                simTime: window.SIM_TIME || 0
            });
        }
    }

    _dieByConsumption(tile) {
        this.energy = 0;
        this.molecules = [];
        this._resetInternalCompositionCache();
        this.deathSimTime = window.SIM_TIME || 0;
        this.state = "dead";
        if (typeof window !== "undefined" && typeof window.__recordLineageDeath === "function") {
            window.__recordLineageDeath(this.lineageId);
        }
        if (typeof window !== "undefined" && typeof window.__recordCellDeath === "function") {
            window.__recordCellDeath(this);
        }
    }

    getDominantElement() {
        const counts = {};
        for (const m of this.molecules) for (const el in m.composition) counts[el] = (counts[el] || 0) + m.composition[el];
        let dominant = null;
        let best = 0;
        for (const k in counts) if (counts[k] > best) { best = counts[k]; dominant = k; }
        return dominant;
    }

    getAgeSecSim() {
        const b = this.birthSimTime || 0;
        const d = this.deathSimTime != null ? this.deathSimTime : (window.SIM_TIME || 0);
        return (d - b);
    }

    _refreshCombatTotals() {
        const enzymes = (this.genome && Array.isArray(this.genome.enzymes))
            ? this.genome.enzymes
            : [];
        let attack = 0;
        let defense = 0;
        for (let i = 0; i < enzymes.length; i++) {
            const enzyme = enzymes[i];
            if (!enzyme) continue;
            if (enzyme.type === "attackase") attack += normalizeCombatLevel(enzyme.level);
            else if (enzyme.type === "defensase") defense += normalizeCombatLevel(enzyme.level);
        }
        this.combatAttackTotal = attack;
        this.combatDefenseTotal = defense;
    }

    _markInternalCompositionDirty() {
        this._internalCompositionDirty = true;
    }

    _resetInternalCompositionCache() {
        this._internalElementCounts[0] = 0;
        this._internalElementCounts[1] = 0;
        this._internalElementCounts[2] = 0;
        this._internalElementCounts[3] = 0;
        this._internalElementCounts[4] = 0;
        this._internalElementCounts[5] = 0;
        this._internalAtomCount = 0;
        this._internalCompositionDirty = false;
    }

    _ensureInternalCompositionCache() {
        if (!this._internalCompositionDirty) return;
        const counts = this._internalElementCounts;
        counts[0] = 0;
        counts[1] = 0;
        counts[2] = 0;
        counts[3] = 0;
        counts[4] = 0;
        counts[5] = 0;
        let totalAtoms = 0;
        for (let i = 0; i < this.molecules.length; i++) {
            const molecule = this.molecules[i];
            const comp = molecule && molecule.composition ? molecule.composition : null;
            if (!comp) continue;
            for (const el in comp) {
                const idx = INTERNAL_ELEMENT_INDEX[el];
                if (idx === undefined) continue;
                const count = comp[el] || 0;
                if (count <= 0) continue;
                counts[idx] += count;
                totalAtoms += count;
            }
        }
        this._internalAtomCount = totalAtoms;
        this._internalCompositionDirty = false;
    }

}

const mutateGenome = (genome, worldAvgEnval = 0) => {
    const g = JSON.parse(JSON.stringify(genome));
    const mut = g.mutationRate ?? 0.05;
    const avgEnval = Number.isFinite(worldAvgEnval) ? worldAvgEnval : 0;

    if (typeof g.optimalEnval !== "number") {
        g.optimalEnval = avgEnval;
    }

    if (Array.isArray(g.enzymes)) {
        for (let i = 0; i < g.enzymes.length; i++) applyEnzymeClassDefaults(g.enzymes[i]);
    }

    if (chance(mut)) {
        g.reproThreshold = Math.max(0.1, g.reproThreshold * (1 + (random() - 0.5) * 0.2));
    }
    if (chance(mut)) {
        g.decayTime = Math.max(50, Math.round(g.decayTime * (1 + (random() - 0.5) * 0.2)));
    }
    if (chance(mut)) {
        g.defaultSecretionProb = clamp01((g.defaultSecretionProb ?? 0.15) + (random() - 0.5) * 0.2);
    }
    if (chance(mut * 0.5)) {
        g.envalStressFactor = Math.max(
            0.001,
            (g.envalStressFactor ?? 0.02) * (1 + (random() - 0.5) * 0.3)
        );
    }

    g.optimalEnval = mutateOptimalEnval(g.optimalEnval, avgEnval, g.envalMutationFloor ?? 0.03);

    for (let i = 0; i < g.enzymes.length; i++) {
        const en = g.enzymes[i];
        applyEnzymeClassDefaults(en);

        if (chance(mut)) {
            if (!isCombatEnzymeType(en.type) && chance(mut)) {
                en.specificityMask = mutateSpecificityMask(en.specificityMask);
            }
            if (chance(mut * 0.2)) {
                en.type = pickRandom(EVOLVABLE_ENZYME_TYPES);
                if (isCombatEnzymeType(en.type)) delete en.level;
                applyEnzymeClassDefaults(en);
            }
            if (isCombatEnzymeType(en.type)) {
                en.level = mutateCombatLevel(en.level);
            } else {
                if (chance(mut)) {
                    en.envalSigma = Math.max(
                        0.02,
                        (en.envalSigma ?? 0.18) * (1 + (random() - 0.5) * 0.35)
                    );
                }
                if (chance(mut)) {
                    en.secretionProb = clamp01((en.secretionProb ?? 0.5) + (random() - 0.5) * 0.2);
                }
                if (chance(mut)) {
                    const cls = ENZYME_CLASSES[en.type] || {};
                    en.envalThroughput = Math.max(
                        0.01,
                        (en.envalThroughput ?? cls.envalThroughput ?? 0.10) * (1 + (random() - 0.5) * 0.4)
                    );
                }
                if (en.type === "anabolase" && chance(mut)) {
                    const cls = ENZYME_CLASSES[en.type] || {};
                    en.bondMultiplier = Math.max(
                        1.01,
                        (en.bondMultiplier ?? cls.bondMultiplier ?? 1.15) * (1 + (random() - 0.5) * 0.18)
                    );
                }
                if (en.type === "transmutase" && chance(mut)) {
                    const cls = ENZYME_CLASSES[en.type] || {};
                    en.downhillHarvestFraction = clamp01(
                        (en.downhillHarvestFraction ?? cls.downhillHarvestFraction ?? 0.18) + (random() - 0.5) * 0.08
                    );
                }
            }
        }

        applyEnzymeClassDefaults(en);
    }

    if (chance(mut * 0.3) && g.enzymes.length > 0) {
        const idx = randomInt(g.enzymes.length);
        g.enzymes.splice(idx, 1);
    } else if (chance(mut * 0.3) && g.enzymes.length < 6) {
        g.enzymes.push(makeRandomEnzyme(pickRandom(EVOLVABLE_ENZYME_TYPES)));
    }

    g.reproThreshold = Math.max(0.01, g.reproThreshold);
    g.decayTime = Math.max(50, g.decayTime);

    return g;
};

const clamp01 = (v) => {
    return Math.max(0, Math.min(1, v));
};

const snapshotMolecule = (molecule) => {
    if (!molecule || !molecule.composition) return null;
    return {
        composition: Object.assign({}, molecule.composition),
        bondMultiplier: molecule.bondMultiplier || 1.0
    };
};

const snapshotMolecules = (molecules) => {
    if (!molecules || molecules.length === 0) return [];
    const out = [];
    for (let i = 0; i < molecules.length; i++) {
        const snap = snapshotMolecule(molecules[i]);
        if (snap) out.push(snap);
    }
    return out;
};

export const applyEnzymeClassDefaults = (enzyme) => {
    if (!enzyme || !enzyme.type) return enzyme;

    if (isCombatEnzymeType(enzyme.type)) {
        if (!Number.isFinite(enzyme.level)) enzyme.level = normalizeCombatLevel(randomGaussian(100, 10), 100);
        enzyme.level = normalizeCombatLevel(enzyme.level);
        delete enzyme.specificityMask;
        delete enzyme.envalSigma;
        delete enzyme.envalThroughput;
        delete enzyme.secretionProb;
        delete enzyme.bondMultiplier;
        delete enzyme.bondCostFraction;
        delete enzyme.bondHarvestFraction;
        delete enzyme.downhillHarvestFraction;
        delete enzyme.transmuteProb;
        delete enzyme.transmuteHarvestFraction;
        delete enzyme.harvestFraction;
        delete enzyme.transportRate;
        delete enzyme.activeCostPerAtom;
        delete enzyme.affinity;
        delete enzyme._affinityMask;
        return enzyme;
    }

    const cls = ENZYME_CLASSES[enzyme.type] || {};

    if (typeof enzyme.envalSigma !== "number") enzyme.envalSigma = 0.16 + random() * 0.08;
    if (typeof enzyme.envalThroughput !== "number") enzyme.envalThroughput = cls.envalThroughput ?? 0.10;
    if (typeof enzyme.secretionProb !== "number") enzyme.secretionProb = 0.15;

    if (typeof enzyme.specificityMask !== "number") {
        if (enzyme.affinity) enzyme.specificityMask = compositionToElementMask(enzyme.affinity);
        else enzyme.specificityMask = defaultSpecificityMaskForType(enzyme.type);
    }
    enzyme.specificityMask = normalizeSpecificityMask(
        enzyme.specificityMask,
        defaultSpecificityMaskForType(enzyme.type)
    );

    if (enzyme.type === "anabolase") {
        if (typeof enzyme.bondMultiplier !== "number") enzyme.bondMultiplier = cls.bondMultiplier ?? 1.15;
        if (typeof enzyme.bondCostFraction !== "number") enzyme.bondCostFraction = cls.bondCostFraction ?? 0.90;
        delete enzyme.bondHarvestFraction;
        delete enzyme.downhillHarvestFraction;
    } else if (enzyme.type === "catabolase") {
        enzyme.bondHarvestFraction = Math.max(1, cls.bondHarvestFraction ?? 1.0);
        delete enzyme.bondMultiplier;
        delete enzyme.bondCostFraction;
        delete enzyme.downhillHarvestFraction;
    } else if (enzyme.type === "transmutase") {
        if (typeof enzyme.downhillHarvestFraction !== "number") {
            enzyme.downhillHarvestFraction = cls.downhillHarvestFraction ?? 0.18;
        }
        delete enzyme.bondMultiplier;
        delete enzyme.bondCostFraction;
        delete enzyme.bondHarvestFraction;
    } else {
        delete enzyme.bondMultiplier;
        delete enzyme.bondCostFraction;
        delete enzyme.bondHarvestFraction;
        delete enzyme.downhillHarvestFraction;
    }

    delete enzyme.level;
    delete enzyme.transmuteProb;
    delete enzyme.transmuteHarvestFraction;
    delete enzyme.harvestFraction;
    delete enzyme.transportRate;
    delete enzyme.activeCostPerAtom;
    delete enzyme.affinity;
    delete enzyme._affinityMask;
    return enzyme;
};

const maskFromLetters = (letters) => {
    let mask = 0;
    for (let i = 0; i < letters.length; i++) {
        mask |= ELEMENT_MASKS[letters[i]] || 0;
    }
    return normalizeSpecificityMask(mask, ALL_ELEMENT_MASK);
};

export const defaultSpecificityMaskForType = (type) => {
    if (type === "anabolase") return maskFromLetters("ABC");
    if (type === "catabolase") return maskFromLetters("ABC");
    if (type === "transmutase") return ALL_ELEMENT_MASK;
    return ALL_ELEMENT_MASK;
};

const EVOLVABLE_ENZYME_TYPES = Object.freeze([...Object.keys(ENZYME_CLASSES), "attackase", "defensase"]);

const makeRandomEnzyme = (type) => {
    if (isCombatEnzymeType(type)) return { type, level: normalizeCombatLevel(randomGaussian(100, 10), 100) };
    const cls = ENZYME_CLASSES[type] || {};
    const enzyme = {
        type,
        specificityMask: makeRandomSpecificityMask(),
        envalSigma: 0.16 + random() * 0.08,
        envalThroughput: cls.envalThroughput ?? 0.10,
        secretionProb: random()
    };
    applyEnzymeClassDefaults(enzyme);
    return enzyme;
};

const pickRandom = (arr) => arr[randomInt(arr.length)];

const makeRandomSpecificityMask = () => {
    const shuffled = ELEMENT_ORDER.slice();
    for (let i = shuffled.length - 1; i > 0; i--) {
        const j = randomInt(i + 1);
        const tmp = shuffled[i];
        shuffled[i] = shuffled[j];
        shuffled[j] = tmp;
    }
    const targetCount = 1 + randomInt(3);
    let mask = 0;
    for (let i = 0; i < targetCount; i++) mask |= ELEMENT_MASKS[shuffled[i]] || 0;
    return normalizeSpecificityMask(mask, ALL_ELEMENT_MASK);
};

const mutateSpecificityMask = (mask) => {
    let next = normalizeSpecificityMask(mask, ALL_ELEMENT_MASK);
    const present = [];
    const absent = [];

    for (let i = 0; i < ELEMENT_ORDER.length; i++) {
        const el = ELEMENT_ORDER[i];
        const bit = ELEMENT_MASKS[el];
        if ((next & bit) !== 0) present.push(bit);
        else absent.push(bit);
    }

    if (present.length <= 1 && absent.length === 0) return next;
    if (present.length <= 1) return next | pickRandom(absent);
    if (absent.length === 0) return next & ~pickRandom(present);

    if (chance(0.5)) next |= pickRandom(absent);
    else next &= ~pickRandom(present);

    return normalizeSpecificityMask(next, ALL_ELEMENT_MASK);
};

const mutateOptimalEnval = (parentEnval, worldAvgEnval, floor = 0.03) => {
    const midpoint = (parentEnval + worldAvgEnval) / 2;
    let step = Math.abs(parentEnval - midpoint);
    let towardSign = Math.sign(worldAvgEnval - parentEnval);

    if (step < floor) {
        step = floor;
        if (towardSign === 0) towardSign = chance(0.5) ? -1 : 1;
    }

    const r = random();
    if (r < 0.5) return parentEnval;
    if (r < 0.75) return parentEnval + towardSign * step;
    return parentEnval - towardSign * step;
};
