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

        for (const enzyme of this.genome.enzymes) {
            const localPool = tile.molecules.concat(this.molecules);

            const result = attemptReaction(enzyme, localPool, env, this, tile);
            if (!result) continue;

            const eDelta = result.energyDelta || 0;
            gainedEnergy += eDelta;

            const tBefore = tile.temperature;

            this.consumeSubstrates(result.consumed, tile);

            const keepPrimary = result.produced && !this.shouldSecrete(result.produced, enzyme);
            if (keepPrimary && result.produced) {
                this.molecules.push(result.produced);
            } else if (result.produced) {
                if (tile.__world) tile.__world.addProductAround(tile.__x, tile.__y, result.produced);
                else tile.molecules.push(result.produced);
            }

            if (result.byproducts && result.byproducts.length > 0 && tile.__world) {
                for (const bp of result.byproducts) {
                    const cloneBP = createMolecule(Object.assign({}, bp.composition || {}), bp.bondMultiplier || 1.0);
                    tile.__world.addProductAround(tile.__x, tile.__y, cloneBP);
                }
            } else if (result.byproducts && result.byproducts.length > 0) {
                for (const bp of result.byproducts) {
                    const cloneBP = createMolecule(Object.assign({}, bp.composition || {}), bp.bondMultiplier || 1.0);
                    tile.molecules.push(cloneBP);
                }
            }

            const heat = result.heatDelta || 0;
            if (Math.abs(heat) > 1e-12 && tile.__world) {
                tile.temperature = Math.max(0, tile.temperature + heat);
                const neighbors = tile.__world.mooreNeighbors(tile.__x, tile.__y);
                for (const [nx, ny] of neighbors) {
                    tile.__world.grid[nx][ny].temperature = Math.max(0, tile.__world.grid[nx][ny].temperature + heat * 0.18);
                }
            }

            const tAfter = tile.temperature;
            const deltaT = tAfter - tBefore;

            const nowSim = window.SIM_TIME || 0;
            const ageAtEventSec = Number((nowSim - (this.birthSimTime || 0)).toFixed(4));
            this._pushReactionLog({
                timeSim: nowSim,
                ageAtEventSec,
                substrates: (result.consumed || []).map(m => compositionToString(m.composition)),
                product: result.produced ? compositionToString(result.produced.composition) : "—",
                byproducts: (result.byproducts || []).map(bp => compositionToString(bp.composition)),
                deltaE: Number((result.energyDelta || 0).toFixed(6)),
                deltaT: Number(deltaT.toFixed(6)),
                enzymeType: enzyme.type,
                substrateAtomEnergy: Number((result.substrateAtomEnergy || 0).toFixed(6)),
                productAtomEnergy: Number((result.productAtomEnergy || 0).toFixed(6)),
                rawDelta: Number((result.rawDelta || 0).toFixed(6))
            });
        }

        const internalCount = this.totalInternalAtoms();
        if (internalCount < (this.genome.desiredElementReserve * 2) && tile.molecules.length > 0) {
            const idx = Math.floor(Math.random() * tile.molecules.length);
            const taken = tile.molecules.splice(idx, 1)[0];
            if (taken) this.molecules.push(taken);
        }

        if (gainedEnergy > 0) {
            this.energy += gainedEnergy;
            this.timeWithoutFood = 0;
        } else {
            this.timeWithoutFood += env.dt;
        }

        const maintRate = this.genome.maintenanceCostPerSec ?? this.maintenanceCostPerSec ?? 0.05;
        const maintLoss = maintRate * env.dt;
        if (maintLoss > 0) {
            this.energy -= maintLoss;
            if (this.energy <= 0) {
                this.energy = 0;
                this._dieAndRelease(tile);
                return;
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
        let s = 0;
        for (const m of this.molecules) s += m.composition[el] || 0;
        return s;
    }

    consumeSubstrates(substrates, tile) {
        if (!substrates || substrates.length === 0) return;

        function sameComp(a, b) {
            const ka = Object.keys(a || {}).filter(k => (a[k]||0) > 0).sort();
            const kb = Object.keys(b || {}).filter(k => (b[k]||0) > 0).sort();
            if (ka.length !== kb.length) return false;
            for (let i = 0; i < ka.length; i++) {
                if (ka[i] !== kb[i]) return false;
                if ((a[ka[i]] || 0) !== (b[kb[i]] || 0)) return false;
            }
            return true;
        }

        for (const sub of substrates) {
            if (!sub || !sub.composition) continue;
            let removed = false;

            for (let i = 0; i < tile.molecules.length && !removed; i++) {
                const m = tile.molecules[i];
                if (sameComp(m.composition, sub.composition)) {
                    tile.molecules.splice(i, 1);
                    removed = true;
                }
            }

            for (let i = 0; i < this.molecules.length && !removed; i++) {
                const m = this.molecules[i];
                if (sameComp(m.composition, sub.composition)) {
                    this.molecules.splice(i, 1);
                    removed = true;
                }
            }
            if (!removed) {
                for (let i = 0; i < tile.molecules.length && !removed; i++) {
                    const m = tile.molecules[i];
                    if (!m || !m.composition) continue;
                    let ok = true;
                    for (const el in sub.composition) {
                        if ((m.composition[el] || 0) < sub.composition[el]) { ok = false; break; }
                    }
                    if (!ok) continue;
                    for (const el in sub.composition) {
                        m.composition[el] -= sub.composition[el];
                        if (m.composition[el] <= 0) delete m.composition[el];
                    }
                    if (Object.keys(m.composition).length === 0) {
                        tile.molecules.splice(i, 1);
                        i--;
                    }
                    removed = true;
                }
            }

            if (!removed) {
                for (let i = 0; i < this.molecules.length && !removed; i++) {
                    const m = this.molecules[i];
                    if (!m || !m.composition) continue;
                    let ok = true;
                    for (const el in sub.composition) {
                        if ((m.composition[el] || 0) < sub.composition[el]) { ok = false; break; }
                    }
                    if (!ok) continue;
                    for (const el in sub.composition) {
                        m.composition[el] -= sub.composition[el];
                        if (m.composition[el] <= 0) delete m.composition[el];
                    }
                    if (Object.keys(m.composition).length === 0) {
                        this.molecules.splice(i, 1);
                        i--;
                    }
                    removed = true;
                }
            }
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
        } else {
            if (tile && tile.cells && tile.cells.length === 0) {
                tile.cells.push(child);
            } else if (tile && tile.cells) {
                this.energy += childEnergy;
                for (const m of childMolecules) this.molecules.push(m);
                return;
            }
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

function mutateGenome(genome) {
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
}

function clamp01(v) {
    return Math.max(0, Math.min(1, v));
}

function biasedPerturb01(value, maxDelta) {
    const r = Math.random();
    if (r < 0.5) return clamp01(value);
    const delta = maxDelta * Math.random();
    let v = value + (r < 0.75 ? -delta : delta);
    return clamp01(v);
}

function compositionToString(comp) {
    if (!comp) return "—";
    return Object.entries(comp).map(([k, v]) => `${k}${v}`).join("");
}
