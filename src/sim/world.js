import { createMolecule } from "./chem.js";
import { Cell } from "./cell.js";

export class World {
    constructor(width, height) {
        this.width = width;
        this.height = height;
        this.dt = 10;

        this.baseTemperature = 0.45 + Math.random() * 0.1;
        this.baseSolute = 0.2 + Math.random() * 0.2;
        this.temperatureRelaxation = 0.0015;

        this.grid = [];
        for (let x = 0; x < width; x++) {
            this.grid[x] = [];
            for (let y = 0; y < height; y++) {
                const tile = this.createTile();
                tile.__x = x;
                tile.__y = y;
                tile.__world = this;
                this.grid[x][y] = tile;
            }
        }

        this._tileCount = this.width * this.height;

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

        this._tCur = new Float32Array(this._tileCount);
        this._sCur = new Float32Array(this._tileCount);
        this._tNext = new Float32Array(this._tileCount);
        this._sNext = new Float32Array(this._tileCount);

        this._moleculeMoveMolecules = [];
        this._moleculeMoveDestTiles = [];

        this.temperatureSum = this.baseTemperature * this._tileCount;
        this.avgTemperature = this._tileCount > 0 ? (this.temperatureSum / this._tileCount) : (this.baseTemperature ?? 0.5);
    }

    createTile() {
        return {
            molecules: this.seedMolecules(),
            cells: [],
            temperature: this.baseTemperature,
            solute: this.baseSolute
        };
    }

    seedMolecules() {
        const arr = [];
        arr.push(createMolecule({ A: 1 }));
        if (Math.random() < 0.6) arr.push(createMolecule({ B: 1 }));
        if (Math.random() < 0.45) arr.push(createMolecule({ C: 1 }));
        if (Math.random() < 0.12) arr.push(createMolecule({ D: 1 }));
        if (Math.random() < 0.08) arr.push(createMolecule({ E: 1 }));
        if (Math.random() < 0.05) arr.push(createMolecule({ X: 1 }));
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
        const env = this._sharedEnv || (this._sharedEnv = { temperature: 0, pH: 0.5, dt: dtSec });
        env.pH = 0.5;
        env.dt = dtSec;
        for (let x = 0; x < this.width; x++) {
            const col = grid[x];
            for (let y = 0; y < this.height; y++) {
                const tile = col[y];
                const cells = tile.cells;
                if (!cells || cells.length === 0) continue;
                env.temperature = tile.temperature;
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
        this.diffuseScalars();
    }

    diffuseMolecules() {
        const w = this.width;
        const h = this.height;
        const grid = this.grid;

        const xPrev = this._xPrev;
        const xNext = this._xNext;
        const yPrev = this._yPrev;
        const yNext = this._yNext;

        const moved = this._moleculeMoveMolecules || (this._moleculeMoveMolecules = []);
        const destArrays = this._moleculeMoveDestTiles || (this._moleculeMoveDestTiles = []);
        moved.length = 0;
        destArrays.length = 0;

        const rand = Math.random;

        for (let x = 0; x < w; x++) {
            const col = grid[x];
            for (let y = 0; y < h; y++) {
                const tile = col[y];
                const mols = tile.molecules;
                const n = mols.length;
                if (n === 0) continue;
                let write = 0;
                let anyMoved = false;
                for (let read = 0; read < n; read++) {
                    const m = mols[read];
                    let rate = m.diffusionRate;
                    if (rand() < rate) {
                        anyMoved = true;
                        const dir = (rand() * 4) | 0;
                        let nx = x;
                        let ny = y;
                        if (dir === 0) nx = xNext[x];
                        else if (dir === 1) nx = xPrev[x];
                        else if (dir === 2) ny = yNext[y];
                        else ny = yPrev[y];
                        moved.push(m);
                        destArrays.push(grid[nx][ny].molecules);
                    } else {
                        if (anyMoved) mols[write] = m;
                        write++;
                    }
                }
                if (anyMoved) mols.length = write;
            }
        }
        for (let i = 0; i < moved.length; i++) {
            destArrays[i].push(moved[i]);
        }
    }

    diffuseScalars() {
        const w = this.width;
        const h = this.height;
        const grid = this.grid;

        const tileCount = w * h;

        if (!this._tCur || this._tCur.length !== tileCount) {
            this._tileCount = tileCount;
            this._xPrev = new Int32Array(w);
            this._xNext = new Int32Array(w);
            for (let x = 0; x < w; x++) {
                this._xPrev[x] = (x === 0) ? (w - 1) : (x - 1);
                this._xNext[x] = (x === w - 1) ? 0 : (x + 1);
            }
            this._yPrev = new Int32Array(h);
            this._yNext = new Int32Array(h);
            for (let y = 0; y < h; y++) {
                this._yPrev[y] = (y === 0) ? (h - 1) : (y - 1);
                this._yNext[y] = (y === h - 1) ? 0 : (y + 1);
            }
            this._tCur = new Float32Array(tileCount);
            this._sCur = new Float32Array(tileCount);
            this._tNext = new Float32Array(tileCount);
            this._sNext = new Float32Array(tileCount);
        }

        const tCur = this._tCur;
        const sCur = this._sCur;
        const tNext = this._tNext;
        const sNext = this._sNext;

        let idx = 0;
        for (let x = 0; x < w; x++) {
            const col = grid[x];
            for (let y = 0; y < h; y++) {
                const tile = col[y];
                tCur[idx] = tile.temperature;
                sCur[idx] = tile.solute;
                idx++;
            }
        }

        const xPrev = this._xPrev;
        const xNext = this._xNext;
        const yPrev = this._yPrev;
        const yNext = this._yNext;

        const alpha = 0.18;
        const oneMinusAlpha = 1 - alpha;
        const inv9 = 1 / 9;

        const relaxRatePerSecond = 0.02;
        const relax = relaxRatePerSecond * (this.dt / 1000);

        for (let x = 0; x < w; x++) {
            const row = x * h;
            const rowL = xPrev[x] * h;
            const rowR = xNext[x] * h;

            for (let y = 0; y < h; y++) {
                const yU = yPrev[y];
                const yD = yNext[y];

                const iC = row + y;

                const iLU = rowL + yU;
                const iLC = rowL + y;
                const iLD = rowL + yD;

                const iCU = row + yU;
                const iCD = row + yD;

                const iRU = rowR + yU;
                const iRC = rowR + y;
                const iRD = rowR + yD;

                const tSum =
                    tCur[iC] +
                    tCur[iLU] + tCur[iLC] + tCur[iLD] +
                    tCur[iCU] + tCur[iCD] +
                    tCur[iRU] + tCur[iRC] + tCur[iRD];

                const sSum =
                    sCur[iC] +
                    sCur[iLU] + sCur[iLC] + sCur[iLD] +
                    sCur[iCU] + sCur[iCD] +
                    sCur[iRU] + sCur[iRC] + sCur[iRD];

                const tAvg = tSum * inv9;
                const sAvg = sSum * inv9;

                const tDiffused = tCur[iC] * oneMinusAlpha + tAvg * alpha;
                tNext[iC] = tDiffused + (this.baseTemperature - tDiffused) * relax;

                sNext[iC] = sCur[iC] * oneMinusAlpha + sAvg * alpha;
            }
        }

        let sumT = 0;
        idx = 0;
        for (let x = 0; x < w; x++) {
            const col = grid[x];
            for (let y = 0; y < h; y++) {
                const tile = col[y];

                let t = tNext[idx];
                if (t < 0) t = 0;
                else if (t > 5) t = 5;

                let s = sNext[idx];
                if (s < 0) s = 0;
                else if (s > 1) s = 1;

                tile.temperature = t;
                tile.solute = s;

                sumT += t;
                idx++;
            }
        }
        this.temperatureSum = sumT;
        this.avgTemperature = tileCount > 0 ? (sumT / tileCount) : (this.baseTemperature ?? 0.5);
    }

    addProductAround(centerX, centerY, product) {
        const ring = this.mooreNeighbors(centerX, centerY).concat([[centerX, centerY]]);
        const [nx, ny] = ring[Math.floor(Math.random() * ring.length)];
        const cloneComp = Object.assign({}, product.composition || {});
        const clone = createMolecule(cloneComp, product.bondMultiplier || 1.0);
        this.grid[nx][ny].molecules.push(clone);
        this.grid[nx][ny].solute = Math.min(1, this.grid[nx][ny].solute + 0.01);
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

    randomNeighbor(x, y) {
        const dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]];
        const [dx, dy] = dirs[Math.floor(Math.random() * dirs.length)];
        return [(x + dx + this.width) % this.width, (y + dy + this.height) % this.height];
    }
}
