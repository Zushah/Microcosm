import { createMolecule, ELEMENT_ORDER } from "./chem.js";
import { Cell } from "./cell.js";

const DIFFUSION_WHEEL_SIZE = 4096;
const DIFFUSION_WHEEL_MASK = DIFFUSION_WHEEL_SIZE - 1;
const TILE_ELEMENT_STRIDE = ELEMENT_ORDER.length;
const ELEMENT_INDEX = { A: 0, B: 1, C: 2, D: 3, E: 4, F: 5 };

export class World {
    constructor(width, height) {
        this.width = width;
        this.height = height;
        this.dt = 10;

        this._diffusionTick = 0;
        this._diffusionWheel = new Array(DIFFUSION_WHEEL_SIZE);
        for (let i = 0; i < DIFFUSION_WHEEL_SIZE; i++) this._diffusionWheel[i] = [];

        this._tileCount = this.width * this.height;
        this._tilesFlat = new Array(this._tileCount);
        this._tileElementCounts = new Int32Array(this._tileCount * TILE_ELEMENT_STRIDE);
        this._tileMassCounts = new Int32Array(this._tileCount);
        this._tileDiffusionRotor = new Uint8Array(this._tileCount);
        for (let ti = 0; ti < this._tileCount; ti++) {
            let hash = (ti ^ 0x9e3779b9) >>> 0;
            hash = Math.imul(hash ^ (hash >>> 16), 0x85ebca6b) >>> 0;
            hash = Math.imul(hash ^ (hash >>> 13), 0xc2b2ae35) >>> 0;
            this._tileDiffusionRotor[ti] = (hash ^ (hash >>> 16)) & 3;
        }

        this.baseEnval = Chalkboard.numb.random(-1, 1);

        this.grid = [];
        for (let x = 0; x < width; x++) {
            this.grid[x] = [];
            for (let y = 0; y < height; y++) {
                const tile = this.createTile();
                tile.__x = x;
                tile.__y = y;
                tile.__world = this;
                const idx = x * height + y;
                tile.__i = idx;
                this._tilesFlat[idx] = tile;
                this.grid[x][y] = tile;
            }
        }

        this._xPrev = new Int32Array(this.width);
        this._xNext = new Int32Array(this.width);
        for (let x = 0; x < this.width; x++) {
            this._xPrev[x] = (x === 0) ? (this.width - 1) : (x - 1);
            this._xNext[x] = (x === this.width - 1) ? 0 : (x + 1);
        }

        this._yPrev = new Int32Array(this.height);
        this._yNext = new Int32Array(this.height);
        for (let y = 0; y < this.height; y++) {
            this._yPrev[y] = (y === 0) ? (this.height - 1) : (y - 1);
            this._yNext[y] = (y === this.height - 1) ? 0 : (y + 1);
        }

        this._eCur = new Float32Array(this._tileCount);
        this._eNext = new Float32Array(this._tileCount);

        this.envalSum = this.baseEnval * this._tileCount;
        this.avgEnval = this._tileCount > 0 ? (this.envalSum / this._tileCount) : (this.baseEnval ?? 0);

        for (let ti = 0; ti < this._tileCount; ti++) {
            const tile = this._tilesFlat[ti];
            const mols = tile.molecules;
            for (let mi = 0; mi < mols.length; mi++) {
                const m = mols[mi];
                this._applyTileCompositionDelta(tile, m.composition, 1);
                m.__tile = tile;
                m.__tileIndex = mi;
                m.__diffusionPhase = -1;
                m.__lastDiffusionDir = -1;
                m.__diffusionWheelIndex = -1;
                m.__diffusionWheelPos = -1;
                this._scheduleMoleculeDiffusion(m);
            }
        }
    }

    createTile() {
        return {
            molecules: this.seedMolecules(),
            cells: [],
            enval: this.baseEnval
        };
    }

    seedMolecules() {
        const arr = [];
        arr.push(createMolecule({ A: 1 }));
        if (Math.random() < 0.6) arr.push(createMolecule({ B: 1 }));
        if (Math.random() < 0.45) arr.push(createMolecule({ C: 1 }));
        if (Math.random() < 0.12) arr.push(createMolecule({ D: 1 }));
        if (Math.random() < 0.08) arr.push(createMolecule({ E: 1 }));
        if (Math.random() < 0.05) arr.push(createMolecule({ F: 1 }));
        if (Math.random() < 0.05) arr.push(createMolecule({ B: 1, C: 1 }));
        return arr;
    }

    spawnRandomCell(genomeFactory) {
        const maxAttempts = 50;
        let x = -1;
        let y = -1;

        for (let attempts = 0; attempts < maxAttempts; attempts++) {
            const rx = Math.floor(Math.random() * this.width);
            const ry = Math.floor(Math.random() * this.height);
            if (this.grid[rx][ry].cells.length === 0) {
                x = rx;
                y = ry;
                break;
            }
        }

        if (x < 0 || y < 0) return;

        const cell = new Cell(genomeFactory());
        const tile = this.grid[x][y];
        tile.cells.push(cell);
        cell.birthSimTime = window.SIM_TIME ?? 0;
        cell._worldRef = this;
        cell._tileX = x;
        cell._tileY = y;

        if (typeof window !== "undefined" && typeof window.__recordCellBirth === "function") {
            window.__recordCellBirth(cell);
        }
    }

    step() {
        this.diffuseMolecules();
        const dtSec = this.dt / 1000;
        const grid = this.grid;
        const env = this._sharedEnv || (this._sharedEnv = { enval: 0, localEnval: 0, avgEnval: 0, dt: dtSec });
        env.dt = dtSec;
        env.avgEnval = this.avgEnval;
        for (let x = 0; x < this.width; x++) {
            const col = grid[x];
            for (let y = 0; y < this.height; y++) {
                const tile = col[y];
                const cells = tile.cells;
                if (!cells || cells.length === 0) continue;
                env.enval = tile.enval;
                env.localEnval = this.getLocalEnvalAverage(x, y, 2);
                env.avgEnval = this.avgEnval;
                for (let i = 0; i < cells.length; i++) {
                    const cell = cells[i];
                    if (cell._worldRef !== this) cell._worldRef = this;
                    cell.step(env, tile);
                }
                let write = 0;
                for (let i = 0; i < cells.length; i++) {
                    const c = cells[i];
                    if (c.state !== "dead") {
                        if (write !== i) cells[write] = c;
                        write++;
                    }
                }
                if (write !== cells.length) cells.length = write;
            }
        }
        this.diffuseEnval();
    }

    _unscheduleMoleculeDiffusion(molecule) {
        const wi = molecule.__diffusionWheelIndex;
        if (typeof wi !== "number" || wi < 0) return;
        const bucket = this._diffusionWheel[wi];
        const pos = molecule.__diffusionWheelPos;
        if (!bucket || bucket.length === 0) {
            molecule.__diffusionWheelIndex = -1;
            molecule.__diffusionWheelPos = -1;
            return;
        }
        const lastPos = bucket.length - 1;
        if (pos >= 0 && pos <= lastPos) {
            if (pos !== lastPos) {
                const swap = bucket[lastPos];
                bucket[pos] = swap;
                swap.__diffusionWheelPos = pos;
            }
            bucket.length = lastPos;
        }
        molecule.__diffusionWheelIndex = -1;
        molecule.__diffusionWheelPos = -1;
    }

    _scheduleMoleculeDiffusion(molecule) {
        if (!molecule) return;
        this._unscheduleMoleculeDiffusion(molecule);
        const wait = molecule.diffusionPeriod || (molecule.diffusionRate > 0 ? Math.max(1, Math.round(1 / molecule.diffusionRate)) : 0);
        if (!(wait > 0)) {
            molecule.__diffusionTick = 0;
            return;
        }

        let phase = molecule.__diffusionPhase;
        if (!Number.isInteger(phase) || phase < 0) {
            const tileIndex = molecule.__tile ? molecule.__tile.__i : 0;
            const tileOffset = Number.isInteger(molecule.__tileIndex) && molecule.__tileIndex > 0 ? molecule.__tileIndex : 0;
            phase = (tileIndex + tileOffset + ((molecule.elementMask || 0) * 7) + ((molecule.size || 0) * 13)) % wait;
        } else {
            phase %= wait;
        }
        molecule.__diffusionPhase = phase;

        let offset = wait - ((this._diffusionTick + phase) % wait);
        if (offset <= 0 || offset > wait) offset = wait;

        const nextTick = (this._diffusionTick + offset) >>> 0;
        this._queueMoleculeDiffusionAt(molecule, nextTick);
    }

    _queueMoleculeDiffusionAt(molecule, nextTick) {
        if (!molecule) return;
        molecule.__diffusionTick = nextTick;
        const wi = nextTick & DIFFUSION_WHEEL_MASK;
        const bucket = this._diffusionWheel[wi];
        molecule.__diffusionWheelIndex = wi;
        molecule.__diffusionWheelPos = bucket.length;
        bucket.push(molecule);
    }

    _applyTileCompositionDelta(tile, composition, sign) {
        if (!tile || !composition || !Number.isFinite(sign) || sign === 0) return;
        const tileIndex = tile.__i;
        if (!Number.isInteger(tileIndex) || tileIndex < 0) return;
        const base = tileIndex * TILE_ELEMENT_STRIDE;
        let totalDelta = 0;
        for (const el in composition) {
            const elementIndex = ELEMENT_INDEX[el];
            if (elementIndex === undefined) continue;
            const count = composition[el] || 0;
            if (count === 0) continue;
            totalDelta += count;
            this._tileElementCounts[base + elementIndex] += sign * count;
        }
        if (totalDelta !== 0) this._tileMassCounts[tileIndex] += sign * totalDelta;
    }

    getTileElementCount(tile, element) {
        if (!tile) return 0;
        const elementIndex = ELEMENT_INDEX[element];
        if (elementIndex === undefined) return 0;
        const tileIndex = tile.__i;
        if (!Number.isInteger(tileIndex) || tileIndex < 0) return 0;
        return this._tileElementCounts[(tileIndex * TILE_ELEMENT_STRIDE) + elementIndex] || 0;
    }

    _getTileCompositionDensity(tile, composition) {
        if (!tile || !composition) return 0;
        const tileIndex = tile.__i;
        if (!Number.isInteger(tileIndex) || tileIndex < 0) return 0;
        const base = tileIndex * TILE_ELEMENT_STRIDE;
        let density = 0;
        for (const el in composition) {
            const elementIndex = ELEMENT_INDEX[el];
            if (elementIndex === undefined) continue;
            const count = composition[el] || 0;
            if (count <= 0) continue;
            density += count * (this._tileElementCounts[base + elementIndex] || 0);
        }
        return density;
    }

    _addMoleculeToTile(tile, molecule) {
        if (!tile || !molecule) return;
        const mols = tile.molecules;
        const idx = mols.length;
        mols.push(molecule);
        this._applyTileCompositionDelta(tile, molecule.composition, 1);
        molecule.__tile = tile;
        molecule.__tileIndex = idx;
        molecule.__diffusionPhase = -1;
        molecule.__lastDiffusionDir = -1;
        molecule.__diffusionWheelIndex = -1;
        molecule.__diffusionWheelPos = -1;
        this._scheduleMoleculeDiffusion(molecule);
    }

    _removeMoleculeFromTile(tile, index) {
        const mols = tile && tile.molecules;
        if (!mols) return null;
        const lastIdx = mols.length - 1;
        if (index < 0 || index > lastIdx) return null;
        const removed = mols[index];
        if (index !== lastIdx) {
            const swap = mols[lastIdx];
            mols[index] = swap;
            swap.__tileIndex = index;
        }
        mols.length = lastIdx;
        if (removed) {
            this._applyTileCompositionDelta(tile, removed.composition, -1);
            removed.__tile = null;
            removed.__tileIndex = -1;
            this._unscheduleMoleculeDiffusion(removed);
        }
        return removed;
    }

    diffuseMolecules() {
        const wheel = this._diffusionWheel;
        if (!wheel) return;
        const tick = ((this._diffusionTick + 1) >>> 0);
        this._diffusionTick = tick;

        const bucket = wheel[tick & DIFFUSION_WHEEL_MASK];
        if (!bucket || bucket.length === 0) return;

        const grid = this.grid;
        const xPrev = this._xPrev;
        const xNext = this._xNext;
        const yPrev = this._yPrev;
        const yNext = this._yNext;

        let i = 0;
        while (i < bucket.length) {
            const m = bucket[i];
            if (!m || !m.__tile) {
                const lastPos = bucket.length - 1;
                if (i !== lastPos) {
                    const swap = bucket[lastPos];
                    bucket[i] = swap;
                    swap.__diffusionWheelPos = i;
                }
                bucket.length = lastPos;
                if (m) {
                    m.__diffusionWheelIndex = -1;
                    m.__diffusionWheelPos = -1;
                }
                continue;
            }

            if (m.__diffusionTick !== tick) {
                i++;
                continue;
            }

            const lastPos = bucket.length - 1;
            if (i !== lastPos) {
                const swap = bucket[lastPos];
                bucket[i] = swap;
                swap.__diffusionWheelPos = i;
            }
            bucket.length = lastPos;
            m.__diffusionWheelIndex = -1;
            m.__diffusionWheelPos = -1;

            const tile = m.__tile;
            const x = tile.__x;
            const y = tile.__y;

            const composition = m.composition;
            const srcDensity = this._getTileCompositionDensity(tile, composition);
            const rightTile = grid[xNext[x]][y];
            const leftTile = grid[xPrev[x]][y];
            const downTile = grid[x][yNext[y]];
            const upTile = grid[x][yPrev[y]];
            const rightDensity = this._getTileCompositionDensity(rightTile, composition);
            const leftDensity = this._getTileCompositionDensity(leftTile, composition);
            const downDensity = this._getTileCompositionDensity(downTile, composition);
            const upDensity = this._getTileCompositionDensity(upTile, composition);
            let minDensity = srcDensity;
            let tieMask = 0;

            if (rightDensity < minDensity) {
                minDensity = rightDensity;
                tieMask = 1;
            } else if (rightDensity === minDensity && rightDensity < srcDensity) tieMask |= 1;
            if (leftDensity < minDensity) {
                minDensity = leftDensity;
                tieMask = 2;
            } else if (leftDensity === minDensity && leftDensity < srcDensity) tieMask |= 2;
            if (downDensity < minDensity) {
                minDensity = downDensity;
                tieMask = 4;
            } else if (downDensity === minDensity && downDensity < srcDensity) tieMask |= 4;
            if (upDensity < minDensity) {
                minDensity = upDensity;
                tieMask = 8;
            } else if (upDensity === minDensity && upDensity < srcDensity) tieMask |= 8;
            if (tieMask === 0) {
                const wait = m.diffusionPeriod || (m.diffusionRate > 0 ? Math.max(1, Math.round(1 / m.diffusionRate)) : 0);
                if (wait > 0) this._queueMoleculeDiffusionAt(m, (tick + wait) >>> 0);
                continue;
            }

            const lastDir = Number.isInteger(m.__lastDiffusionDir) ? (m.__lastDiffusionDir & 3) : -1;
            if (lastDir >= 0) {
                const reverseMask = 1 << (lastDir ^ 1);
                const filteredMask = tieMask & ~reverseMask;
                if (filteredMask !== 0) tieMask = filteredMask;
            }

            let chosenDir = -1;
            if ((tieMask & (tieMask - 1)) === 0) {
                if (tieMask === 1) chosenDir = 0;
                else if (tieMask === 2) chosenDir = 1;
                else if (tieMask === 4) chosenDir = 2;
                else chosenDir = 3;
            } else {
                const rotorIndex = tile.__i;
                const rotorStart = this._tileDiffusionRotor[rotorIndex] & 3;
                for (let step = 0; step < 4; step++) {
                    const dir = (rotorStart + step) & 3;
                    if ((tieMask & (1 << dir)) !== 0) {
                        chosenDir = dir;
                        this._tileDiffusionRotor[rotorIndex] = (dir + 1) & 3;
                        break;
                    }
                }
            }

            let destTile = null;
            if (chosenDir === 0) destTile = rightTile;
            else if (chosenDir === 1) destTile = leftTile;
            else if (chosenDir === 2) destTile = downTile;
            else if (chosenDir === 3) destTile = upTile;

            if (!destTile) {
                const wait = m.diffusionPeriod || (m.diffusionRate > 0 ? Math.max(1, Math.round(1 / m.diffusionRate)) : 0);
                if (wait > 0) this._queueMoleculeDiffusionAt(m, (tick + wait) >>> 0);
                continue;
            }

            const srcMols = tile.molecules;
            const srcIdx = m.__tileIndex;
            const srcLast = srcMols.length - 1;
            if (srcIdx >= 0 && srcIdx <= srcLast) {
                if (srcIdx !== srcLast) {
                    const swapMol = srcMols[srcLast];
                    srcMols[srcIdx] = swapMol;
                    swapMol.__tileIndex = srcIdx;
                }
                srcMols.length = srcLast;
            }

            const destMols = destTile.molecules;
            this._applyTileCompositionDelta(tile, m.composition, -1);
            this._applyTileCompositionDelta(destTile, m.composition, 1);
            m.__tile = destTile;
            m.__tileIndex = destMols.length;
            m.__lastDiffusionDir = chosenDir;
            destMols.push(m);

            const wait = m.diffusionPeriod || (m.diffusionRate > 0 ? Math.max(1, Math.round(1 / m.diffusionRate)) : 0);
            if (wait > 0) this._queueMoleculeDiffusionAt(m, (tick + wait) >>> 0);
        }
    }

    diffuseEnval() {
        const w = this.width;
        const h = this.height;
        const tileCount = this._tileCount;
        const tiles = this._tilesFlat;

        const eCur = this._eCur;
        const eNext = this._eNext;

        for (let i = 0; i < tileCount; i++) {
            const tile = tiles[i];
            eCur[i] = tile.enval;
        }

        const xPrev = this._xPrev;
        const xNext = this._xNext;
        const yPrev = this._yPrev;
        const yNext = this._yNext;

        const alpha = 0.18;
        const oneMinusAlpha = 1 - alpha;
        const inv9 = 1 / 9;

        for (let x = 0; x < w; x++) {
            const xL = xPrev[x];
            const xR = xNext[x];
            const row = x * h;
            const rowL = xL * h;
            const rowR = xR * h;

            for (let y = 0; y < h; y++) {
                const yU = yPrev[y];
                const yD = yNext[y];

                const iC = row + y;
                const iLU = rowL + yU;
                const iL = rowL + y;
                const iLD = rowL + yD;
                const iU = row + yU;
                const iD = row + yD;
                const iRU = rowR + yU;
                const iR = rowR + y;
                const iRD = rowR + yD;

                const eSum = eCur[iC] + eCur[iLU] + eCur[iL] + eCur[iLD] + eCur[iU] + eCur[iD] + eCur[iRU] + eCur[iR] + eCur[iRD];
                const eAvg = eSum * inv9;

                eNext[iC] = alpha * eAvg + oneMinusAlpha * eCur[iC];
            }
        }

        let sumE = 0;
        for (let i = 0; i < tileCount; i++) {
            const tile = tiles[i];
            let e = eNext[i];
            if (!Number.isFinite(e)) e = 0;
            tile.enval = e;
            sumE += e;
        }
        this.envalSum = sumE;
        this.avgEnval = tileCount > 0 ? (sumE / tileCount) : (this.baseEnval ?? 0);
    }

    adjustTileEnval(tile, delta) {
        if (!tile) return;
        if (!Number.isFinite(delta) || delta === 0) return;
        tile.enval += delta;
        this.envalSum += delta;
        this.avgEnval = this._tileCount > 0 ? (this.envalSum / this._tileCount) : (this.baseEnval ?? 0);
    }


    applyEnvalBrush(centerX, centerY, brushWidth, brushHeight, intensity) {
        if (!Number.isFinite(intensity) || intensity === 0) return 0;

        const width = Math.max(1, Math.min(this.width, Math.round(Number(brushWidth) || 1)));
        const height = Math.max(1, Math.min(this.height, Math.round(Number(brushHeight) || 1)));
        const startX = centerX - Math.floor(width * 0.5);
        const startY = centerY - Math.floor(height * 0.5);

        const xIndices = [];
        const yIndices = [];

        if (width >= this.width) for (let x = 0; x < this.width; x++) xIndices.push(x);
        else for (let dx = 0; dx < width; dx++) xIndices.push(((startX + dx) % this.width + this.width) % this.width);
        if (height >= this.height) for (let y = 0; y < this.height; y++) yIndices.push(y);
        else for (let dy = 0; dy < height; dy++) yIndices.push(((startY + dy) % this.height + this.height) % this.height);

        for (let xi = 0; xi < xIndices.length; xi++) {
            const x = xIndices[xi];
            const col = this.grid[x];
            for (let yi = 0; yi < yIndices.length; yi++) col[yIndices[yi]].enval += intensity;
        }

        const affectedTileCount = xIndices.length * yIndices.length;
        this.envalSum += intensity * affectedTileCount;
        this.avgEnval = this._tileCount > 0 ? (this.envalSum / this._tileCount) : (this.baseEnval ?? 0);
        return affectedTileCount;
    }

    getLocalEnvalAverage(centerX, centerY, radius = 2) {
        let sum = 0;
        let count = 0;
        for (let dx = -radius; dx <= radius; dx++) {
            const x = (centerX + dx + this.width) % this.width;
            const col = this.grid[x];
            for (let dy = -radius; dy <= radius; dy++) {
                const y = (centerY + dy + this.height) % this.height;
                sum += col[y].enval;
                count++;
            }
        }
        return count > 0 ? (sum / count) : 0;
    }

    addProductAround(centerX, centerY, product) {
        const ring = this.mooreNeighbors(centerX, centerY).concat([[centerX, centerY]]);
        const [nx, ny] = ring[Math.floor(Math.random() * ring.length)];
        const cloneComp = Object.assign({}, product.composition || {});
        const clone = createMolecule(cloneComp, product.bondMultiplier || 1.0);
        const tile = this.grid[nx][ny];
        this._addMoleculeToTile(tile, clone);
    }

    addEnvalAround(centerX, centerY, envalDelta) {
        if (!Number.isFinite(envalDelta) || envalDelta === 0) return;
        const ring = this.mooreNeighbors(centerX, centerY).concat([[centerX, centerY]]);
        const [nx, ny] = ring[Math.floor(Math.random() * ring.length)];
        this.adjustTileEnval(this.grid[nx][ny], envalDelta);
    }

    mooreNeighbors(x, y) {
        const out = [];
        for (let dx = -1; dx <= 1; dx++) {
            for (let dy = -1; dy <= 1; dy++) {
                if (dx === 0 && dy === 0) continue;
                out.push([(x + dx + this.width) % this.width, (y + dy + this.height) % this.height]);
            }
        }
        return out;
    }
}
