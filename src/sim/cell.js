import { attemptReaction } from "./bio.js";

export class Cell {
    constructor(genome) {
        this.genome = genome;
        this.energy = genome.initialEnergy || 0;
        this.molecules = [];
        this.timeWithoutFood = 0;
        this.state = "active";
        this.lineageId = genome.lineageId || Math.floor(Math.random() * 1e9);
        this.birthTime = Date.now();
        this._highlight = false;
    }

    step(env, tile) {
        if (this.state !== "active") return;

        let gainedEnergy = 0;

        for (const enzyme of this.genome.enzymes) {
            const result = attemptReaction(
                enzyme,
                tile.molecules.concat(this.molecules),
                env
            );

            if (!result) continue;

            const eDelta = result.energyDelta || 0;
            gainedEnergy += eDelta;

            this.consumeSubstrates(result.consumed, tile);

            const keepPrimary = !this.shouldSecrete(result.produced, enzyme);
            if (keepPrimary) {
                this.molecules.push(result.produced);
            } else {
                if (tile.__world) {
                    tile.__world.addProductAround(tile.__x, tile.__y, result.produced);
                } else {
                    tile.molecules.push(result.produced);
                }
            }

            if (result.byproducts && result.byproducts.length > 0 && tile.__world) {
                for (const bp of result.byproducts) {
                    tile.__world.addProductAround(tile.__x, tile.__y, bp);
                }
            }

            const heat = result.heatDelta || 0;
            if (Math.abs(heat) > 1e-8 && tile.__world) {
                tile.temperature = Math.max(0, tile.temperature + heat);
                const neighbors = tile.__world.mooreNeighbors(tile.__x, tile.__y);
                for (const [nx, ny] of neighbors) {
                    tile.__world.grid[nx][ny].temperature = Math.max(
                        0,
                        tile.__world.grid[nx][ny].temperature + heat * 0.2
                    );
                }
            }
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

        if (this.timeWithoutFood > this.genome.decayTime) {
            this.state = "dead";
            return;
        }

        if (this.energy >= this.genome.reproThreshold) {
            this.divide(tile);
        }
    }

    totalInternalAtoms() {
        let s = 0;
        for (const m of this.molecules) {
            s += m.size || 0;
        }
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
        substrates.forEach(m => {
            let idx = tile.molecules.indexOf(m);
            if (idx >= 0) {
                tile.molecules.splice(idx, 1);
                return;
            }
            idx = this.molecules.indexOf(m);
            if (idx >= 0) {
                this.molecules.splice(idx, 1);
                return;
            }
        });
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

        const maxOffset = 3;
        let placed = false;
        for (let attempts = 0; attempts < 8 && !placed; attempts++) {
            const dx = Math.floor(Math.random() * (maxOffset * 2 + 1)) - maxOffset;
            const dy = Math.floor(Math.random() * (maxOffset * 2 + 1)) - maxOffset;
            const px = (tile.__x + dx + this._worldWidth()) % this._worldWidth();
            const py = (tile.__y + dy + this._worldHeight()) % this._worldHeight();

            tile.__world.grid[px][py].cells.push(child);
            placed = true;
        }
        if (!placed) tile.cells.push(child);

        if (Math.random() < (this.genome.postDivideMortality ?? 0.0)) {
            this.state = "dead";
        }
    }

    getDominantElement() {
        const counts = {};
        for (const m of this.molecules) {
            for (const el in m.composition) counts[el] = (counts[el] || 0) + m.composition[el];
        }
        let dominant = null;
        let best = 0;
        for (const k in counts) {
            if (counts[k] > best) { best = counts[k]; dominant = k; }
        }
        return dominant;
    }

    getAgeMs() { return Date.now() - this.birthTime; }
    _worldWidth() { return this._worldRef ? this._worldRef.width : 200; }
    _worldHeight() { return this._worldRef ? this._worldRef.height : 200; }
}

import { ENZYME_CLASSES } from "./bio.js";
function mutateGenome(genome) {
    const g = JSON.parse(JSON.stringify(genome));
    const mut = g.mutationRate ?? 0.05;
    if (Math.random() < mut) g.reproThreshold = Math.max(0.1, g.reproThreshold * (1 + (Math.random() - 0.5) * 0.2));
    if (Math.random() < mut) g.decayTime = Math.max(100, Math.round(g.decayTime * (1 + (Math.random() - 0.5) * 0.2)));
    if (Math.random() < mut) g.defaultSecretionProb = clamp01(g.defaultSecretionProb + (Math.random() - 0.5) * 0.2);

    for (let i = 0; i < g.enzymes.length; i++) {
        if (Math.random() < mut) {
            const en = g.enzymes[i];
            if (!en.affinity) en.affinity = {};
            if (Math.random() < 0.5) {
                const keys = Object.keys(en.affinity);
                if (keys.length > 0 && Math.random() < 0.7) {
                    const k = keys[Math.floor(Math.random() * keys.length)];
                    en.affinity[k] = Math.max(0, en.affinity[k] + (Math.random() - 0.5) * 0.4);
                } else {
                    const possible = ["A", "B", "C", "D", "E", "X"];
                    const k = possible[Math.floor(Math.random() * possible.length)];
                    en.affinity[k] = (en.affinity[k] || 0) + Math.random();
                }
            }
            if (Math.random() < mut * 0.2) {
                const classes = Object.keys(ENZYME_CLASSES);
                en.type = classes[Math.floor(Math.random() * classes.length)];
            }
            if (Math.random() < mut) en.tOpt = clamp01((en.tOpt ?? 0.5) + (Math.random() - 0.5) * 0.2);
            if (Math.random() < mut) en.pHOpt = clamp01((en.pHOpt ?? 0.5) + (Math.random() - 0.5) * 0.2);
            if (Math.random() < mut) en.secretionProb = clamp01((en.secretionProb ?? 0.5) + (Math.random() - 0.5) * 0.2);
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
            affinity: { B: Math.random(), C: Math.random() },
            tOpt: Math.random(),
            pHOpt: Math.random(),
            secretionProb: Math.random()
        });
    }

    g.reproThreshold = Math.max(0.01, g.reproThreshold);
    g.decayTime = Math.max(50, g.decayTime);

    return g;
}
function clamp01(v) { return Math.max(0, Math.min(1, v)); }
