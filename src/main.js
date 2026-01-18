import { World } from "./sim/world.js";
import { CanvasRenderer } from "./render/canvas.js";

const world = new World(200, 200);

function randomGenome() {
	return {
		enzymes: [
			{
				type: "ligase",
				affinity: { B: 1, C: 1 },
				tOpt: Math.random(),
				pHOpt: Math.random()
			}
		],
		decayTime: 1000 + Math.random() * 2000
	};
}

const canvas = document.getElementById("canvas");
const renderer = new CanvasRenderer(canvas, world);

setInterval(() => {
	if (Math.random() < 0.05) {
		world.spawnRandomCell(randomGenome);
	}
	world.step();
	renderer.render();
}, world.dt);
