import { attemptReaction } from "./bio.js";

export class Cell {
	constructor(genome) {
		this.genome = genome;
		this.energy = 0;
		this.molecules = [];
		this.timeWithoutFood = 0;
		this.state = "active";
	}

	step(env, tile) {
		if (this.state !== "active") return;
		let gainedEnergy = 0;
		for (const enzyme of this.genome.enzymes) {
			const result = attemptReaction(
				enzyme,
				tile.molecules,
				env
			);
			if (result) {
				gainedEnergy += result.energyDelta;
				this.consume(tile, result);
			}
		}

		if (gainedEnergy > 0) {
			this.energy += gainedEnergy;
			this.timeWithoutFood = 0;
		} else {
			this.timeWithoutFood += env.dt;
			this.checkStarvation();
		}
	}

	consume(tile, reaction) {
		reaction.consumed.forEach(m => {
			const idx = tile.molecules.indexOf(m);
			if (idx >= 0) tile.molecules.splice(idx, 1);
		});
		tile.molecules.push(reaction.produced);
	}

	checkStarvation() {
		if (this.timeWithoutFood > this.genome.decayTime) {
			this.state = Math.random() < 0.5 ? "dead" : "dormant";
		}
	}
}
