import { createMolecule, ELEMENT_ORDER } from "./chem.js";
import { Cell } from "./cell.js";
import { chance, randomInt, randomRange } from "./rng.js";
import { resolvePredationOutcome } from "./eco.js";

const DIFFUSION_WHEEL_SIZE = 4096;
const DIFFUSION_WHEEL_MASK = DIFFUSION_WHEEL_SIZE - 1;
const TILE_ELEMENT_STRIDE = ELEMENT_ORDER.length;
const ELEMENT_INDEX = { A: 0, B: 1, C: 2, D: 3, E: 4, F: 5 };
const LOCAL_ENVAL_RADIUS = 2;
const LOCAL_ENVAL_WINDOW_AREA = (LOCAL_ENVAL_RADIUS * 2 + 1) ** 2;
const LOCAL_ENVAL_INV_WINDOW_AREA = 1 / LOCAL_ENVAL_WINDOW_AREA;
const MOORE_WITH_CENTER_DX = [-1, -1, -1, 0, 0, 1, 1, 1, 0];
const MOORE_WITH_CENTER_DY = [-1, 0, 1, -1, 1, -1, 0, 1, 0];

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
        this._tileLeft = new Int32Array(this._tileCount);
        this._tileRight = new Int32Array(this._tileCount);
        this._tileUp = new Int32Array(this._tileCount);
        this._tileDown = new Int32Array(this._tileCount);
        this._tileUpLeft = new Int32Array(this._tileCount);
        this._tileUpRight = new Int32Array(this._tileCount);
        this._tileDownLeft = new Int32Array(this._tileCount);
        this._tileDownRight = new Int32Array(this._tileCount);
        this._occupiedTileIndices = new Int32Array(this._tileCount);
        this._occupiedTileCount = 0;
        for (let ti = 0; ti < this._tileCount; ti++) {
            let hash = (ti ^ 0x9e3779b9) >>> 0;
            hash = Math.imul(hash ^ (hash >>> 16), 0x85ebca6b) >>> 0;
            hash = Math.imul(hash ^ (hash >>> 13), 0xc2b2ae35) >>> 0;
            this._tileDiffusionRotor[ti] = (hash ^ (hash >>> 16)) & 3;
        }

        this.baseEnval = randomRange(-1, 1);

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
        const h = this.height;

        for (let ti = 0; ti < this._tileCount; ti++) {
            const tile = this._tilesFlat[ti];
            this._eCur[ti] = tile.enval;
            this._eNext[ti] = tile.enval;
            const x = tile.__x;
            const y = tile.__y;
            const xL = this._xPrev[x];
            const xR = this._xNext[x];
            const yU = this._yPrev[y];
            const yD = this._yNext[y];
            const idx = ti;

            const idxL = xL * h + y;
            const idxR = xR * h + y;
            const idxU = x * h + yU;
            const idxD = x * h + yD;
            this._tileLeft[idx] = idxL;
            this._tileRight[idx] = idxR;
            this._tileUp[idx] = idxU;
            this._tileDown[idx] = idxD;
            this._tileUpLeft[idx] = xL * h + yU;
            this._tileUpRight[idx] = xR * h + yU;
            this._tileDownLeft[idx] = xL * h + yD;
            this._tileDownRight[idx] = xR * h + yD;

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
        if (chance(0.6)) arr.push(createMolecule({ B: 1 }));
        if (chance(0.45)) arr.push(createMolecule({ C: 1 }));
        if (chance(0.12)) arr.push(createMolecule({ D: 1 }));
        if (chance(0.08)) arr.push(createMolecule({ E: 1 }));
        if (chance(0.05)) arr.push(createMolecule({ F: 1 }));
        if (chance(0.05)) arr.push(createMolecule({ B: 1, C: 1 }));
        return arr;
    }

    spawnRandomCell(genomeFactory) {
        const maxAttempts = 50;
        let x = -1;
        let y = -1;

        for (let attempts = 0; attempts < maxAttempts; attempts++) {
            const rx = randomInt(this.width);
            const ry = randomInt(this.height);
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
        const tiles = this._tilesFlat;
        const tileCount = this._tileCount;
        const occupiedTileIndices = this._occupiedTileIndices;
        let occupiedTileCount = 0;
        const env = this._sharedEnv || (this._sharedEnv = { enval: 0, localEnval: 0, avgEnval: 0, dt: dtSec });
        env.dt = dtSec;
        env.avgEnval = this.avgEnval;
        for (let tileIndex = 0; tileIndex < tileCount; tileIndex++) {
            const tile = tiles[tileIndex];
            const cells = tile.cells;
            if (!cells || cells.length === 0) continue;
            env.enval = tile.enval;
            env.avgEnval = this.avgEnval;
            let sawDeadCell = false;
            for (let i = 0; i < cells.length; i++) {
                const cell = cells[i];
                if (!cell) {
                    sawDeadCell = true;
                    continue;
                }
                if (cell._worldRef !== this) cell._worldRef = this;
                cell.step(env, tile);
                if (cell.state === "dead") sawDeadCell = true;
            }
            if (sawDeadCell) this._compactDeadCells(tile);
            if (tile.cells.length > 0) {
                occupiedTileIndices[occupiedTileCount] = tileIndex;
                occupiedTileCount++;
            }
        }
        this._occupiedTileCount = occupiedTileCount;
        this._resolvePredation();
        this.diffuseEnval();
    }

    _compactDeadCells(tile) {
        const cells = tile && tile.cells;
        if (!cells || cells.length === 0) return;
        let write = 0;
        for (let i = 0; i < cells.length; i++) {
            const cell = cells[i];
            if (cell && cell.state !== "dead") {
                if (write !== i) cells[write] = cell;
                write++;
            }
        }
        if (write !== cells.length) cells.length = write;
    }

    _resolvePredation() {
        const tiles = this._tilesFlat;
        const occupied = this._occupiedTileIndices;
        const occupiedCount = this._occupiedTileCount;
        const tileRight = this._tileRight;
        const tileDown = this._tileDown;
        const shouldCheckRight = this.width > 1;
        const shouldCheckDown = this.height > 1;
        const singleHorizontalPair = this.width === 2;
        const singleVerticalPair = this.height === 2;

        for (let i = 0; i < occupiedCount; i++) {
            const tileIndex = occupied[i];
            const tile = tiles[tileIndex];
            if (!tile || !tile.cells || tile.cells.length === 0) continue;
            const x = tile.__x;
            const y = tile.__y;

            if (shouldCheckRight && (!singleHorizontalPair || x === 0)) {
                const rightTile = tiles[tileRight[tileIndex]];
                if (rightTile && rightTile !== tile && rightTile.cells && rightTile.cells.length > 0) {
                    this._resolvePredationBetweenTiles(tile, rightTile);
                }
            }

            if (shouldCheckDown && (!singleVerticalPair || y === 0)) {
                const downTile = tiles[tileDown[tileIndex]];
                if (downTile && downTile !== tile && downTile.cells && downTile.cells.length > 0) {
                    this._resolvePredationBetweenTiles(tile, downTile);
                }
            }
        }
    }

    _resolvePredationBetweenTiles(tileA, tileB) {
        const cellsA = tileA && tileA.cells;
        const cellsB = tileB && tileB.cells;
        if (!cellsA || !cellsB || cellsA.length === 0 || cellsB.length === 0) return;
        let compactA = false;
        let compactB = false;
        let hasAliveA = false;
        let hasAliveB = false;
        let maxAttackA = 0;
        let maxAttackB = 0;
        let minDefenseA = Infinity;
        let minDefenseB = Infinity;

        for (let i = 0; i < cellsA.length; i++) {
            const cell = cellsA[i];
            if (!cell || cell.state === "dead") {
                compactA = true;
                continue;
            }
            const attack = cell.combatAttackTotal;
            const defense = cell.combatDefenseTotal;
            if (!Number.isFinite(attack) || !Number.isFinite(defense)) {
                return this._resolvePredationBetweenTilesFallback(tileA, tileB);
            }
            hasAliveA = true;
            if (attack > maxAttackA) maxAttackA = attack;
            if (defense < minDefenseA) minDefenseA = defense;
        }

        for (let j = 0; j < cellsB.length; j++) {
            const cell = cellsB[j];
            if (!cell || cell.state === "dead") {
                compactB = true;
                continue;
            }
            const attack = cell.combatAttackTotal;
            const defense = cell.combatDefenseTotal;
            if (!Number.isFinite(attack) || !Number.isFinite(defense)) {
                return this._resolvePredationBetweenTilesFallback(tileA, tileB);
            }
            hasAliveB = true;
            if (attack > maxAttackB) maxAttackB = attack;
            if (defense < minDefenseB) minDefenseB = defense;
        }

        if (!hasAliveA || !hasAliveB) {
            if (compactA) this._compactDeadCells(tileA);
            if (compactB) this._compactDeadCells(tileB);
            return;
        }

        if (maxAttackA <= minDefenseB && maxAttackB <= minDefenseA) {
            if (compactA) this._compactDeadCells(tileA);
            if (compactB) this._compactDeadCells(tileB);
            return;
        }

        for (let i = 0; i < cellsA.length; i++) {
            const cellA = cellsA[i];
            if (!cellA || cellA.state === "dead") {
                compactA = true;
                continue;
            }
            const aAttack = cellA.combatAttackTotal;
            const aDefense = cellA.combatDefenseTotal;
            if (aAttack <= minDefenseB && maxAttackB <= aDefense) continue;

            for (let j = 0; j < cellsB.length; j++) {
                const cellB = cellsB[j];
                if (!cellB || cellB.state === "dead") {
                    compactB = true;
                    continue;
                }
                if (cellA.lineageId === cellB.lineageId) continue;

                const bAttack = cellB.combatAttackTotal;
                const bDefense = cellB.combatDefenseTotal;
                const aCanKill = aAttack > 0 && aAttack > bDefense;
                const bCanKill = bAttack > 0 && bAttack > aDefense;
                if (!aCanKill && !bCanKill) continue;

                let outcome = null;
                if (aCanKill && !bCanKill) {
                    outcome = {
                        winner: cellA,
                        loser: cellB,
                        winnerAttack: aAttack,
                        winnerDefense: aDefense,
                        loserAttack: bAttack,
                        loserDefense: bDefense,
                        winnerMargin: aAttack - bDefense,
                        loserMargin: bAttack - aDefense
                    };
                } else if (bCanKill && !aCanKill) {
                    outcome = {
                        winner: cellB,
                        loser: cellA,
                        winnerAttack: bAttack,
                        winnerDefense: bDefense,
                        loserAttack: aAttack,
                        loserDefense: aDefense,
                        winnerMargin: bAttack - aDefense,
                        loserMargin: aAttack - bDefense
                    };
                } else {
                    const aMargin = aAttack - bDefense;
                    const bMargin = bAttack - aDefense;
                    if (aMargin === bMargin) continue;
                    if (aMargin > bMargin) {
                        outcome = {
                            winner: cellA,
                            loser: cellB,
                            winnerAttack: aAttack,
                            winnerDefense: aDefense,
                            loserAttack: bAttack,
                            loserDefense: bDefense,
                            winnerMargin: aMargin,
                            loserMargin: bMargin
                        };
                    } else {
                        outcome = {
                            winner: cellB,
                            loser: cellA,
                            winnerAttack: bAttack,
                            winnerDefense: bDefense,
                            loserAttack: aAttack,
                            loserDefense: aDefense,
                            winnerMargin: bMargin,
                            loserMargin: aMargin
                        };
                    }
                }

                const loserTile = outcome.loser === cellA ? tileA : tileB;
                this._executePredationOutcome(outcome, loserTile);
                if (loserTile === tileA) compactA = true;
                else compactB = true;

                if (cellA.state === "dead") break;
            }
        }

        if (compactA) this._compactDeadCells(tileA);
        if (compactB) this._compactDeadCells(tileB);
    }

    _resolvePredationBetweenTilesFallback(tileA, tileB) {
        const cellsA = tileA && tileA.cells;
        const cellsB = tileB && tileB.cells;
        if (!cellsA || !cellsB || cellsA.length === 0 || cellsB.length === 0) return;

        for (let i = 0; i < cellsA.length; i++) {
            const cellA = cellsA[i];
            if (!cellA || cellA.state === "dead") continue;

            for (let j = 0; j < cellsB.length; j++) {
                const cellB = cellsB[j];
                if (!cellB || cellB.state === "dead") continue;

                const outcome = resolvePredationOutcome(cellA, cellB);
                if (!outcome) continue;

                const loserTile = outcome.loser === cellA ? tileA : tileB;
                this._executePredationOutcome(outcome, loserTile);
                if (cellA.state === "dead") break;
            }
        }

        this._compactDeadCells(tileA);
        this._compactDeadCells(tileB);
    }

    _executePredationOutcome(outcome, loserTile) {
        if (!outcome || !outcome.winner || !outcome.loser) return;
        if (outcome.winner.state === "dead" || outcome.loser.state === "dead") return;
        outcome.winner.absorbConsumedCell(outcome.loser, outcome);
        outcome.loser._dieByConsumption(loserTile);
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
        const wait = this._getMoleculeDiffusionWait(molecule);
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

    _getMoleculeDiffusionWait(molecule) {
        if (!molecule) return 0;
        if (molecule.diffusionPeriod) return molecule.diffusionPeriod;
        if (!(molecule.diffusionRate > 0)) return 0;
        return Math.max(1, Math.round(1 / molecule.diffusionRate));
    }

    _applyTileCompositionDeltaByIndex(tileIndex, composition, sign) {
        if (!composition || !Number.isFinite(sign) || sign === 0) return;
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

    _applyTileCompositionDelta(tile, composition, sign) {
        if (!tile || !composition || !Number.isFinite(sign) || sign === 0) return;
        const tileIndex = tile.__i;
        this._applyTileCompositionDeltaByIndex(tileIndex, composition, sign);
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
        return this._getTileCompositionDensityByIndex(tileIndex, composition);
    }

    _getTileCompositionDensityByIndex(tileIndex, composition) {
        if (!composition) return 0;
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

        const tiles = this._tilesFlat;
        const tileLeft = this._tileLeft;
        const tileRight = this._tileRight;
        const tileUp = this._tileUp;
        const tileDown = this._tileDown;

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
            const srcTileIndex = tile.__i;
            const rightIndex = tileRight[srcTileIndex];
            const leftIndex = tileLeft[srcTileIndex];
            const downIndex = tileDown[srcTileIndex];
            const upIndex = tileUp[srcTileIndex];

            const composition = m.composition;
            const srcDensity = this._getTileCompositionDensityByIndex(srcTileIndex, composition);
            const rightDensity = this._getTileCompositionDensityByIndex(rightIndex, composition);
            const leftDensity = this._getTileCompositionDensityByIndex(leftIndex, composition);
            const downDensity = this._getTileCompositionDensityByIndex(downIndex, composition);
            const upDensity = this._getTileCompositionDensityByIndex(upIndex, composition);
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
                const wait = this._getMoleculeDiffusionWait(m);
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
                const rotorStart = this._tileDiffusionRotor[srcTileIndex] & 3;
                for (let step = 0; step < 4; step++) {
                    const dir = (rotorStart + step) & 3;
                    if ((tieMask & (1 << dir)) !== 0) {
                        chosenDir = dir;
                        this._tileDiffusionRotor[srcTileIndex] = (dir + 1) & 3;
                        break;
                    }
                }
            }

            let destTileIndex = -1;
            if (chosenDir === 0) destTileIndex = rightIndex;
            else if (chosenDir === 1) destTileIndex = leftIndex;
            else if (chosenDir === 2) destTileIndex = downIndex;
            else if (chosenDir === 3) destTileIndex = upIndex;

            if (destTileIndex < 0) {
                const wait = this._getMoleculeDiffusionWait(m);
                if (wait > 0) this._queueMoleculeDiffusionAt(m, (tick + wait) >>> 0);
                continue;
            }

            const destTile = tiles[destTileIndex];
            if (!destTile) {
                const wait = this._getMoleculeDiffusionWait(m);
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
            this._applyTileCompositionDeltaByIndex(srcTileIndex, m.composition, -1);
            this._applyTileCompositionDeltaByIndex(destTileIndex, m.composition, 1);
            m.__tile = destTile;
            m.__tileIndex = destMols.length;
            m.__lastDiffusionDir = chosenDir;
            destMols.push(m);

            const wait = this._getMoleculeDiffusionWait(m);
            if (wait > 0) this._queueMoleculeDiffusionAt(m, (tick + wait) >>> 0);
        }
    }

    diffuseEnval() {
        const tileCount = this._tileCount;
        const tiles = this._tilesFlat;
        const eCur = this._eCur;
        const eNext = this._eNext;
        const tileLeft = this._tileLeft;
        const tileRight = this._tileRight;
        const tileUp = this._tileUp;
        const tileDown = this._tileDown;
        const tileUpLeft = this._tileUpLeft;
        const tileUpRight = this._tileUpRight;
        const tileDownLeft = this._tileDownLeft;
        const tileDownRight = this._tileDownRight;

        const alpha = 0.18;
        const oneMinusAlpha = 1 - alpha;
        const inv9 = 1 / 9;

        for (let i = 0; i < tileCount; i++) {
            const eCenter = eCur[i];
            const eSum = eCenter
                + eCur[tileLeft[i]]
                + eCur[tileRight[i]]
                + eCur[tileUp[i]]
                + eCur[tileDown[i]]
                + eCur[tileUpLeft[i]]
                + eCur[tileUpRight[i]]
                + eCur[tileDownLeft[i]]
                + eCur[tileDownRight[i]];
            const eAvg = eSum * inv9;
            eNext[i] = alpha * eAvg + oneMinusAlpha * eCenter;
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
        this._eCur = eNext;
        this._eNext = eCur;
    }

    adjustTileEnval(tile, delta) {
        if (!tile) return;
        if (!Number.isFinite(delta) || delta === 0) return;
        tile.enval += delta;
        const tileIndex = tile.__i;
        if (Number.isInteger(tileIndex) && tileIndex >= 0) this._eCur[tileIndex] = tile.enval;
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
            for (let yi = 0; yi < yIndices.length; yi++) {
                const tile = col[yIndices[yi]];
                tile.enval += intensity;
                const tileIndex = tile.__i;
                if (Number.isInteger(tileIndex) && tileIndex >= 0) this._eCur[tileIndex] = tile.enval;
            }
        }

        const affectedTileCount = xIndices.length * yIndices.length;
        this.envalSum += intensity * affectedTileCount;
        this.avgEnval = this._tileCount > 0 ? (this.envalSum / this._tileCount) : (this.baseEnval ?? 0);
        return affectedTileCount;
    }

    getLocalEnvalAverage(centerX, centerY, radius = 2) {
        if (radius === LOCAL_ENVAL_RADIUS) {
            const width = this.width;
            const height = this.height;
            const x = ((centerX % width) + width) % width;
            const y = ((centerY % height) + height) % height;
            const xPrev = this._xPrev;
            const xNext = this._xNext;
            const yPrev = this._yPrev;
            const yNext = this._yNext;
            const xM1 = xPrev[x];
            const xP1 = xNext[x];
            const xM2 = xPrev[xM1];
            const xP2 = xNext[xP1];
            const yM1 = yPrev[y];
            const yP1 = yNext[y];
            const yM2 = yPrev[yM1];
            const yP2 = yNext[yP1];
            const eCur = this._eCur;
            const rowM2 = xM2 * height;
            const rowM1 = xM1 * height;
            const row0 = x * height;
            const rowP1 = xP1 * height;
            const rowP2 = xP2 * height;
            const sum =
                eCur[rowM2 + yM2] + eCur[rowM2 + yM1] + eCur[rowM2 + y] + eCur[rowM2 + yP1] + eCur[rowM2 + yP2]
                + eCur[rowM1 + yM2] + eCur[rowM1 + yM1] + eCur[rowM1 + y] + eCur[rowM1 + yP1] + eCur[rowM1 + yP2]
                + eCur[row0 + yM2] + eCur[row0 + yM1] + eCur[row0 + y] + eCur[row0 + yP1] + eCur[row0 + yP2]
                + eCur[rowP1 + yM2] + eCur[rowP1 + yM1] + eCur[rowP1 + y] + eCur[rowP1 + yP1] + eCur[rowP1 + yP2]
                + eCur[rowP2 + yM2] + eCur[rowP2 + yM1] + eCur[rowP2 + y] + eCur[rowP2 + yP1] + eCur[rowP2 + yP2];
            return sum * LOCAL_ENVAL_INV_WINDOW_AREA;
        }

        let sum = 0;
        let count = 0;
        const eCur = this._eCur;
        const height = this.height;
        for (let dx = -radius; dx <= radius; dx++) {
            const x = (centerX + dx + this.width) % this.width;
            const row = x * height;
            for (let dy = -radius; dy <= radius; dy++) {
                const y = (centerY + dy + this.height) % this.height;
                sum += eCur[row + y];
                count++;
            }
        }
        return count > 0 ? (sum / count) : 0;
    }

    addProductAround(centerX, centerY, product) {
        const choice = randomInt(9);
        const nx = (centerX + MOORE_WITH_CENTER_DX[choice] + this.width) % this.width;
        const ny = (centerY + MOORE_WITH_CENTER_DY[choice] + this.height) % this.height;
        const cloneComp = Object.assign({}, product.composition || {});
        const clone = createMolecule(cloneComp, product.bondMultiplier || 1.0);
        const tile = this.grid[nx][ny];
        this._addMoleculeToTile(tile, clone);
    }

    addEnvalAround(centerX, centerY, envalDelta) {
        if (!Number.isFinite(envalDelta) || envalDelta === 0) return;
        const choice = randomInt(9);
        const nx = (centerX + MOORE_WITH_CENTER_DX[choice] + this.width) % this.width;
        const ny = (centerY + MOORE_WITH_CENTER_DY[choice] + this.height) % this.height;
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
