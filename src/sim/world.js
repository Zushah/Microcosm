import { createMolecule } from "./chem.js";
import { Cell } from "./cell.js";

export class World {
    constructor(width, height) {
        this.width = width;
        this.height = height;
        this.dt = 10;

        this.baseTemperature = 0.45 + Math.random() * 0.1;
        this.baseSolute = 0.2 + Math.random() * 0.2;

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
        const x = Math.floor(Math.random() * this.width);
        const y = Math.floor(Math.random() * this.height);
        const cell = new Cell(genomeFactory());
        this.grid[x][y].cells.push(cell);
        cell.birthSimTime = window.SIM_TIME ?? 0;
        cell._worldRef = this;
        cell._tileX = x;
        cell._tileY = y;
    }

    step() {
        this.diffuseMolecules();

        for (let x = 0; x < this.width; x++) {
            for (let y = 0; y < this.height; y++) {
                const tile = this.grid[x][y];
                const env = {
                    temperature: tile.temperature,
                    pH: 0.5,
                    dt: this.dt / 1000
                };
                const cells = tile.cells.slice();
                for (const cell of cells) {
                    cell._worldRef = this;
                    cell.step(env, tile);
                }
                tile.cells = tile.cells.filter(c => c.state !== "dead");
            }
        }

        this.diffuseScalars();
    }

    diffuseMolecules() {
        const transfers = [];
        for (let x = 0; x < this.width; x++) {
            for (let y = 0; y < this.height; y++) {
                const tile = this.grid[x][y];
                tile.molecules.forEach(molecule => {
                    const rate = diffusionRate(molecule);
                    if (Math.random() < rate) {
                        const [nx, ny] = this.randomNeighbor(x, y);
                        transfers.push({ x, y, nx, ny, molecule });
                    }
                });
            }
        }
        transfers.forEach(t => {
            const src = this.grid[t.x][t.y].molecules;
            const dst = this.grid[t.nx][t.ny].molecules;
            const idx = src.indexOf(t.molecule);
            if (idx >= 0) {
                src.splice(idx, 1);
                dst.push(t.molecule);
            }
        });
    }

    diffuseScalars() {
        const tNext = [];
        const sNext = [];
        for (let x = 0; x < this.width; x++) {
            tNext[x] = [];
            sNext[x] = [];
            for (let y = 0; y < this.height; y++) {
                tNext[x][y] = this.grid[x][y].temperature;
                sNext[x][y] = this.grid[x][y].solute;
            }
        }

        const alpha = 0.18;
        for (let x = 0; x < this.width; x++) {
            for (let y = 0; y < this.height; y++) {
                const neighbors = this.mooreNeighbors(x, y);
                let tSum = this.grid[x][y].temperature;
                let sSum = this.grid[x][y].solute;
                for (const [nx, ny] of neighbors) {
                    tSum += this.grid[nx][ny].temperature;
                    sSum += this.grid[nx][ny].solute;
                }
                const count = neighbors.length + 1;
                const tAvg = tSum / count;
                const sAvg = sSum / count;
                tNext[x][y] = this.grid[x][y].temperature * (1 - alpha) + tAvg * alpha;
                sNext[x][y] = this.grid[x][y].solute * (1 - alpha) + sAvg * alpha;
            }
        }

        for (let x = 0; x < this.width; x++) {
            for (let y = 0; y < this.height; y++) {
                this.grid[x][y].temperature = Math.max(0, Math.min(5, tNext[x][y]));
                this.grid[x][y].solute = Math.max(0, Math.min(1, sNext[x][y]));
            }
        }
    }

    addProductAround(centerX, centerY, product) {
        const ring = this.mooreNeighbors(centerX, centerY).concat([[centerX, centerY]]);
        const k = 1 + Math.floor(Math.random() * 3);
        for (let i = 0; i < k; i++) {
            const [nx, ny] = ring[Math.floor(Math.random() * ring.length)];
            const cloneComp = Object.assign({}, product.composition || {});
            const clone = createMolecule(cloneComp, product.bondMultiplier || 1.0);
            this.grid[nx][ny].molecules.push(clone);
            this.grid[nx][ny].solute = Math.min(1, this.grid[nx][ny].solute + 0.01);
        }
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

function diffusionRate(molecule) {
    return Math.min(0.04, 0.01 + (1 / (molecule.size + 1)) * molecule.polarity * 0.04);
}
