import { createMolecule } from "./chem.js";
import { Cell } from "./cell.js";

export class World {
	constructor(width, height) {
		this.width = width;
		this.height = height;
		this.dt = 10;
		this.grid = [];
		for (let x = 0; x < width; x++) {
			this.grid[x] = [];
			for (let y = 0; y < height; y++) {
				this.grid[x][y] = this.createTile();
			}
		}
	}

	createTile() {
		return {
			molecules: this.seedMolecules(),
			cells: [],
			temperature: 0.5 + (Math.random() - 0.5) * 0.1,
			pH: 0.5 + (Math.random() - 0.5) * 0.1
		};
	}

	seedMolecules() {
		return [
			createMolecule({ A: 2 }),
			createMolecule({ B: 1 }),
			createMolecule({ A: 1, B: 1 })
		];
	}

	spawnRandomCell(genomeFactory) {
		const x = Math.floor(Math.random() * this.width);
		const y = Math.floor(Math.random() * this.height);
		this.grid[x][y].cells.push(new Cell(genomeFactory()));
	}

	step() {
		this.diffuseMolecules();
		for (let x = 0; x < this.width; x++) {
			for (let y = 0; y < this.height; y++) {
				const tile = this.grid[x][y];
				const env = {
					temperature: tile.temperature,
					pH: tile.pH,
					dt: this.dt
				};
				tile.cells.forEach(cell => cell.step(env, tile));
				tile.cells = tile.cells.filter(c => c.state !== "dead");
			}
		}
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

	randomNeighbor(x, y) {
		const dirs = [
			[1, 0], [-1, 0], [0, 1], [0, -1]
			];
			const [dx, dy] = dirs[Math.floor(Math.random() * dirs.length)];
			return [
			(x + dx + this.width) % this.width,
			(y + dy + this.height) % this.height
		];
	}
}

function diffusionRate(molecule) {
	return Math.min(0.2, (1 / (molecule.size + 1)) * molecule.polarity);
}
