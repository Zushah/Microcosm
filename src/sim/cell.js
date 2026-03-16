import { ENZYME_CLASSES, attemptReaction } from "./bio.js";
import { ALL_ELEMENT_MASK, ELEMENT_MASKS, ELEMENT_ORDER, compositionToElementMask, createMolecule, maskToString, normalizeSpecificityMask } from "./chem.js";

export class Cell {
    constructor(genome) {
        this.genome = genome;
        this.energy = genome.initialEnergy || 0;
        this.molecules = [];
        this.timeWithoutFood = 0;
        this.state = "active";
        this.lineageId = genome.lineageId || Math.floor(Math.random() * 1e9);
        this.birthSimTime = this.birthSimTime ?? (window.SIM_TIME || 0);
        this.deathSimTime = null;
        this._highlight = false;
        this.reactionLog = [];
        this.reactionLogMax = 40;
        this.maintenanceCostPerSec = genome.maintenanceCostPerSec ?? 0.05;
    }

    step(env, tile) {
        if (this.state !== "active") return;

        let positiveEnergyGain = 0;

        const tileMolecules = tile.molecules;
        const internalMolecules = this.molecules;
        const world = tile.__world;

        const refreshEnv = () => {
            if (!world || !tile) return;
            env.enval = tile.enval;
            env.tileEnval = tile.enval;
            env.localEnval = world.getLocalEnvalAverage(tile.__x, tile.__y, 2);
            env.avgEnval = world.avgEnval;
        };

        refreshEnv();

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
                } else internalMolecules.push(result.produced);
            }

            if (result.byproducts && result.byproducts.length > 0) {
                if (world) for (const bp of result.byproducts) world.addProductAround(tile.__x, tile.__y, bp);
                else for (const bp of result.byproducts) tileMolecules.push(bp);
            }

            const envalInput = result.envalInput || 0;
            const envalOutput = result.envalOutput || 0;
            if (envalInput !== 0) {
                if (world && typeof world.adjustTileEnval === "function") world.adjustTileEnval(tile, -envalInput);
                else tile.enval -= envalInput;
            }
            if (envalOutput !== 0) {
                if (world && typeof world.addEnvalAround === "function") world.addEnvalAround(tile.__x, tile.__y, envalOutput);
                else tile.enval += envalOutput;
            }

            refreshEnv();

            const ageAtEvent = (typeof this.getAgeSecSim === "function") ? this.getAgeSecSim() : 0;

            this._pushReactionLog({
                enzymeType: enzyme.type || "unknown",
                specificityMask: normalizeSpecificityMask(enzyme.specificityMask, ALL_ELEMENT_MASK),
                specificity: maskToString(normalizeSpecificityMask(enzyme.specificityMask, ALL_ELEMENT_MASK)),
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
            const idx = Math.floor(Math.random() * tileMolecules.length);
            let take = tileMolecules[idx];
            if (take) {
                if (world && typeof world._removeMoleculeFromTile === "function") take = world._removeMoleculeFromTile(tile, idx) || take;
                else tileMolecules.splice(idx, 1);
                internalMolecules.push(take);
            }
        }

        refreshEnv();

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

    totalInternalAtoms() {
        let s = 0;
        for (const m of this.molecules) s += m.size || 0;
        return s;
    }

    shouldSecrete(product, enzyme) {
        const reserve = this.genome.desiredElementReserve ?? 2;
        for (const el in product.composition) {
            const have = this.countInternalElement(el);
            if (have < reserve) return false;
        }
        const prob = enzyme.secretionProb ?? this.genome.defaultSecretionProb ?? 0.15;
        return Math.random() < prob;
    }

    countInternalElement(el) {
        const arr = [];
        for (let i = 0; i < this.molecules.length; i++) arr.push(this.molecules[i].composition[el] || 0);
        return Chalkboard.stat.sum(arr);
    }

    consumeSubstrates(substrates, tile) {
        if (!substrates || substrates.length === 0) return;
        const tilePool = tile && tile.molecules ? tile.molecules : null;
        const cellPool = this.molecules;
        const world = tile && tile.__world;
        const subtractFromPool = (pool, subComp) => {
            if (!pool) return false;
            for (let i = 0; i < pool.length; i++) {
                const m = pool[i];
                if (!m || !m.composition) continue;
                let ok = true;
                for (const el in subComp) if ((m.composition[el] || 0) < subComp[el]) { ok = false; break; }
                if (!ok) continue;
                for (const el in subComp) {
                    m.composition[el] -= subComp[el];
                    if (m.composition[el] <= 0) delete m.composition[el];
                }
                if (Object.keys(m.composition).length === 0) {
                    if (pool === tilePool && world && typeof world._removeMoleculeFromTile === "function" && m.__tile === tile) world._removeMoleculeFromTile(tile, i);
                    else pool.splice(i, 1);
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
                continue;
            }
            if (!subtractFromPool(tilePool, sub.composition)) subtractFromPool(cellPool, sub.composition);
        }
    }

    divide(tile) {
        const world = this._worldRef;
        const worldAvgEnval = world ? world.avgEnval : 0;
        const childGenome = mutateGenome(this.genome, worldAvgEnval);

        const childEnergy = this.energy * (0.5 + (Math.random() - 0.5) * 0.1);
        this.energy = Math.max(0, this.energy - childEnergy);

        const childMolecules = [];
        for (let i = this.molecules.length - 1; i >= 0; i--) {
            if (Math.random() < 0.5) {
                childMolecules.push(this.molecules[i]);
                this.molecules.splice(i, 1);
            }
        }

        const child = new Cell(childGenome);
        child.energy = childEnergy;
        child.molecules = childMolecules;
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

            const [px, py] = candidates[Math.floor(Math.random() * candidates.length)];
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

        if (Math.random() < (this.genome.postDivideMortality ?? 0.0)) this._dieAndRelease(tile);
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

    _worldWidth() { return this._worldRef ? this._worldRef.width : 200; }
    _worldHeight() { return this._worldRef ? this._worldRef.height : 200; }
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

    if (Math.random() < mut) {
        g.reproThreshold = Math.max(0.1, g.reproThreshold * (1 + (Math.random() - 0.5) * 0.2));
    }
    if (Math.random() < mut) {
        g.decayTime = Math.max(50, Math.round(g.decayTime * (1 + (Math.random() - 0.5) * 0.2)));
    }
    if (Math.random() < mut) {
        g.defaultSecretionProb = clamp01((g.defaultSecretionProb ?? 0.15) + (Math.random() - 0.5) * 0.2);
    }
    if (Math.random() < mut * 0.5) {
        g.envalStressFactor = Math.max(
            0.001,
            (g.envalStressFactor ?? 0.02) * (1 + (Math.random() - 0.5) * 0.3)
        );
    }

    g.optimalEnval = mutateOptimalEnval(g.optimalEnval, avgEnval, g.envalMutationFloor ?? 0.03);

    for (let i = 0; i < g.enzymes.length; i++) {
        const en = g.enzymes[i];
        applyEnzymeClassDefaults(en);

        if (Math.random() < mut) {
            if (Math.random() < mut) {
                en.specificityMask = mutateSpecificityMask(en.specificityMask);
            }
            if (Math.random() < mut * 0.2) {
                const classes = Object.keys(ENZYME_CLASSES);
                en.type = classes[Math.floor(Math.random() * classes.length)];
                applyEnzymeClassDefaults(en);
            }
            if (Math.random() < mut) {
                en.envalSigma = Math.max(
                    0.02,
                    (en.envalSigma ?? 0.18) * (1 + (Math.random() - 0.5) * 0.35)
                );
            }
            if (Math.random() < mut) {
                en.secretionProb = clamp01((en.secretionProb ?? 0.5) + (Math.random() - 0.5) * 0.2);
            }
            if (Math.random() < mut) {
                const cls = ENZYME_CLASSES[en.type] || {};
                en.envalThroughput = Math.max(
                    0.01,
                    (en.envalThroughput ?? cls.envalThroughput ?? 0.10) * (1 + (Math.random() - 0.5) * 0.4)
                );
            }
            if (en.type === "anabolase" && Math.random() < mut) {
                const cls = ENZYME_CLASSES[en.type] || {};
                en.bondMultiplier = Math.max(
                    1.01,
                    (en.bondMultiplier ?? cls.bondMultiplier ?? 1.15) * (1 + (Math.random() - 0.5) * 0.18)
                );
            }
            if (en.type === "transmutase" && Math.random() < mut) {
                const cls = ENZYME_CLASSES[en.type] || {};
                en.downhillHarvestFraction = clamp01(
                    (en.downhillHarvestFraction ?? cls.downhillHarvestFraction ?? 0.18) + (Math.random() - 0.5) * 0.08
                );
            }
        }

        applyEnzymeClassDefaults(en);
    }

    if (Math.random() < mut * 0.3 && g.enzymes.length > 0) {
        const idx = Math.floor(Math.random() * g.enzymes.length);
        g.enzymes.splice(idx, 1);
    } else if (Math.random() < mut * 0.3 && g.enzymes.length < 6) {
        const classes = Object.keys(ENZYME_CLASSES);
        const type = classes[Math.floor(Math.random() * classes.length)];
        const cls = ENZYME_CLASSES[type] || {};
        const enzyme = {
            type,
            specificityMask: makeRandomSpecificityMask(),
            envalSigma: 0.16 + Math.random() * 0.08,
            envalThroughput: cls.envalThroughput ?? 0.10,
            secretionProb: Math.random()
        };
        applyEnzymeClassDefaults(enzyme);
        g.enzymes.push(enzyme);
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
    return createMolecule(Object.assign({}, molecule.composition), molecule.bondMultiplier || 1.0);
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

const applyEnzymeClassDefaults = (enzyme) => {
    if (!enzyme || !enzyme.type) return enzyme;
    const cls = ENZYME_CLASSES[enzyme.type] || {};

    if (typeof enzyme.envalSigma !== "number") enzyme.envalSigma = 0.16 + Math.random() * 0.08;
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
        delete enzyme.transmuteProb;
        delete enzyme.bondHarvestFraction;
        delete enzyme.transmuteHarvestFraction;
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

const defaultSpecificityMaskForType = (type) => {
    if (type === "anabolase") return maskFromLetters("ABC");
    if (type === "catabolase") return maskFromLetters("ABC");
    if (type === "transmutase") return ALL_ELEMENT_MASK;
    return ALL_ELEMENT_MASK;
};

const pickRandom = (arr) => arr[(Math.random() * arr.length) | 0];

const makeRandomSpecificityMask = () => {
    const shuffled = ELEMENT_ORDER.slice();
    for (let i = shuffled.length - 1; i > 0; i--) {
        const j = (Math.random() * (i + 1)) | 0;
        const tmp = shuffled[i];
        shuffled[i] = shuffled[j];
        shuffled[j] = tmp;
    }
    const targetCount = 1 + ((Math.random() * 3) | 0);
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

    if (Math.random() < 0.5) next |= pickRandom(absent);
    else next &= ~pickRandom(present);

    return normalizeSpecificityMask(next, ALL_ELEMENT_MASK);
};

const mutateOptimalEnval = (parentEnval, worldAvgEnval, floor = 0.03) => {
    const midpoint = (parentEnval + worldAvgEnval) / 2;
    let step = Math.abs(parentEnval - midpoint);
    let towardSign = Math.sign(worldAvgEnval - parentEnval);

    if (step < floor) {
        step = floor;
        if (towardSign === 0) towardSign = Math.random() < 0.5 ? -1 : 1;
    }

    const r = Math.random();
    if (r < 0.5) return parentEnval;
    if (r < 0.75) return parentEnval + towardSign * step;
    return parentEnval - towardSign * step;
};
