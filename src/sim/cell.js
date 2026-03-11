import { ENZYME_CLASSES, attemptReaction } from "./bio.js";
import { createMolecule } from "./chem.js";

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

        let gainedEnergy = 0;

        const tileMolecules = tile.molecules;
        const internalMolecules = this.molecules;
        const world = tile.__world;

        for (const enzyme of this.genome.enzymes) {
            const result = attemptReaction(enzyme, tileMolecules, env, this, tile, internalMolecules);
            if (!result) continue;

            if (result.elementDelta && typeof window !== "undefined" && typeof window.__recordElementDelta === "function") {
                window.__recordElementDelta(result.elementDelta);
            }

            const eDelta = result.energyDelta || 0;
            gainedEnergy += eDelta;

            const tBefore = tile.temperature;

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

            const heat = result.heatDelta || 0;
            if (heat !== 0 && world) {
                tile.temperature = Math.max(0, tile.temperature + heat);
                const hx = heat * 0.18;
                const x = tile.__x;
                const y = tile.__y;
                const xL = world._xPrev[x];
                const xR = world._xNext[x];
                const yU = world._yPrev[y];
                const yD = world._yNext[y];
                const grid = world.grid;
                const colL = grid[xL];
                const col = grid[x];
                const colR = grid[xR];
                colL[yU].temperature = Math.max(0, colL[yU].temperature + hx);
                colL[y].temperature = Math.max(0, colL[y].temperature + hx);
                colL[yD].temperature = Math.max(0, colL[yD].temperature + hx);
                col[yU].temperature = Math.max(0, col[yU].temperature + hx);
                col[yD].temperature = Math.max(0, col[yD].temperature + hx);
                colR[yU].temperature = Math.max(0, colR[yU].temperature + hx);
                colR[y].temperature = Math.max(0, colR[y].temperature + hx);
                colR[yD].temperature = Math.max(0, colR[yD].temperature + hx);
            }

            const tAfter = tile.temperature;
            const deltaT = tAfter - tBefore;
            const ageAtEvent = (typeof this.getAgeSecSim === "function") ? this.getAgeSecSim() : 0;

            this._pushReactionLog({
                enzymeType: enzyme.type || "unknown",
                affinity: enzyme.affinity || null,
                consumed: result.consumed || [],
                produced: result.produced || null,
                byproducts: result.byproducts || [],
                deltaE: Number((result.energyDelta || 0).toFixed(3)),
                deltaT: Number(deltaT.toFixed(4)),
                substrateAtomEnergy: Number((result.substrateAtomEnergy || 0).toFixed(3)),
                productAtomEnergy: Number((result.productAtomEnergy || 0).toFixed(3)),
                rawDelta: Number((result.rawDelta || 0).toFixed(3)),
                ageAtEventSec: Number(ageAtEvent.toFixed(3)),
                simTime: window.SIM_TIME || 0
            });
        }

        if (gainedEnergy > 1e-6) {
            this.energy += gainedEnergy;
            this.timeWithoutFood = Math.max(0, this.timeWithoutFood - gainedEnergy * 0.2);
        } else {
            this.timeWithoutFood += env.dt;
        }

        this.energy = Math.max(0, this.energy - this.maintenanceCostPerSec * env.dt);

        const internalCount = this.totalInternalAtoms();
        if (internalCount < (this.genome.desiredElementReserve * 2) && tileMolecules.length > 0) {
            const idx = Math.floor(Math.random() * tileMolecules.length);
            const take = tileMolecules[idx];
            if (take) {
                tileMolecules.splice(idx, 1);
                internalMolecules.push(take);
            }
        }

        const T = tile.temperature;
        let stressIncrement = 0;

        const tOptCell = (this.genome.optimalTemp != null)
            ? this.genome.optimalTemp
            : (this.genome.enzymes.length
                ? this.genome.enzymes.reduce((s, en) => s + (en.tOpt ?? 0.5), 0) / this.genome.enzymes.length
                : 0.5);

        const dist = Math.abs(T - tOptCell);
        const tempStressFactor = this.genome.tempStressFactor ?? 0.02;

        stressIncrement += Math.pow(dist, 1.6) * tempStressFactor * (this.genome.enzymes.length || 1);
        this.timeWithoutFood += stressIncrement;

        if (this.timeWithoutFood > this.genome.decayTime) {
            this._dieAndRelease(tile);
            return;
        }

        if (this.energy >= this.genome.reproThreshold) {
            this.divide(tile);
        }

        for (const en of this.genome.enzymes) {
            const tOpt = en.tOpt ?? 0.5;
            const dist = Math.abs(T - tOpt);
            stressIncrement += Math.pow(dist, 1.6) * (this.genome.tempStressFactor ?? 0.02);
        }
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
        if (tile && tile.__world) {
            for (const m of this.molecules) tile.molecules.push(m);
        }
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
        const subtractFromPool = (pool, subComp) => {
            if (!pool) return false;
            for (let i = 0; i < pool.length; i++) {
                const m = pool[i];
                if (!m || !m.composition) continue;
                let ok = true;
                for (const el in subComp) {
                    if ((m.composition[el] || 0) < subComp[el]) { ok = false; break; }
                }
                if (!ok) continue;
                for (const el in subComp) {
                    m.composition[el] -= subComp[el];
                    if (m.composition[el] <= 0) delete m.composition[el];
                }
                if (Object.keys(m.composition).length === 0) {
                    pool.splice(i, 1);
                }
                return true;
            }
            return false;
        };
        for (let si = 0; si < substrates.length; si++) {
            const sub = substrates[si];
            if (!sub || !sub.composition) continue;
            if (tilePool) {
                const idx = tilePool.indexOf(sub);
                if (idx !== -1) {
                    tilePool.splice(idx, 1);
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
        const childGenome = mutateGenome(this.genome);

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

        const world = this._worldRef;

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

const mutateGenome = (genome) => {
    const g = JSON.parse(JSON.stringify(genome));
    const mut = g.mutationRate ?? 0.05;

    if (typeof g.optimalTemp !== "number") {
        if (g.enzymes && g.enzymes.length > 0 && typeof g.enzymes[0].tOpt === "number") {
            g.optimalTemp = g.enzymes[0].tOpt;
        } else {
            g.optimalTemp = 0.5;
        }
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
        g.tempStressFactor = Math.max(
            0.001,
            (g.tempStressFactor ?? 0.02) * (1 + (Math.random() - 0.5) * 0.3)
        );
    }

    g.optimalTemp = biasedPerturb01(g.optimalTemp, 0.10);

    for (let i = 0; i < g.enzymes.length; i++) {
        if (Math.random() < mut) {
            const en = g.enzymes[i];
            if (!en.affinity) en.affinity = {};
            if (Math.random() < mut) {
                const possible = ["A", "B", "C", "D", "E", "X"];
                const k = possible[Math.floor(Math.random() * possible.length)];
                en.affinity[k] = (en.affinity[k] || 0) + Math.random();
            }
            if (Math.random() < mut * 0.2) {
                const classes = Object.keys(ENZYME_CLASSES);
                en.type = classes[Math.floor(Math.random() * classes.length)];
            }
            if (Math.random() < mut) {
                en.pHOpt = clamp01((en.pHOpt ?? 0.5) + (Math.random() - 0.5) * 0.2);
            }
            if (Math.random() < mut) {
                en.secretionProb = clamp01((en.secretionProb ?? 0.5) + (Math.random() - 0.5) * 0.2);
            }
        }
    }

    if (Math.random() < mut * 0.3 && g.enzymes.length > 0) {
        const idx = Math.floor(Math.random() * g.enzymes.length);
        g.enzymes.splice(idx, 1);
    } else if (Math.random() < mut * 0.3 && g.enzymes.length < 6) {
        const classes = Object.keys(ENZYME_CLASSES);
        const type = classes[Math.floor(Math.random() * classes.length)];
        g.enzymes.push({
            type,
            affinity: { A: Math.random(), B: Math.random(), C: Math.random(), D: Math.random() },
            tOpt: g.optimalTemp,
            pHOpt: Math.random(),
            secretionProb: Math.random()
        });
    }

    for (const en of g.enzymes) {
        en.tOpt = g.optimalTemp;
    }

    g.reproThreshold = Math.max(0.01, g.reproThreshold);
    g.decayTime = Math.max(50, g.decayTime);

    return g;
};

const clamp01 = (v) => {
    return Math.max(0, Math.min(1, v));
};

const biasedPerturb01 = (value, maxDelta) => {
    const r = Math.random();
    if (r < 0.5) return clamp01(value);
    const delta = maxDelta * Math.random();
    let v = value + (r < 0.75 ? -delta : delta);
    return clamp01(v);
};

const compositionToString = (comp) => {
    if (!comp) return "—";
    return Object.entries(comp).map(([k, v]) => `${k}${v}`).join("");
};
