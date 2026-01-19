import { World } from "./sim/world.js";
import { CanvasRenderer } from "./render/canvas.js";

const world = new World(200, 200);

function randomGenome() {
    return {
        enzymes: [
            {
                type: "ligase",
                affinity: { B: 1.0, C: 0.5 },
                tOpt: Math.random(),
                pHOpt: Math.random(),
                secretionProb: 0.5
            }
        ],
        reproThreshold: 6 + Math.random() * 6,
        initialEnergy: 1.0 + Math.random() * 1.5,
        decayTime: 500 + Math.random() * 2000,
        dormancyBias: 0.6,
        defaultSecretionProb: 0.15,
        mutationRate: 0.06,
        postDivideMortality: 0.0,
        lineageId: Math.floor(Math.random() * 1e9)
    };
}

const canvas = document.getElementById("canvas");
const renderer = new CanvasRenderer(canvas, world);

for (let i = 0; i < 40; i++) {
    world.spawnRandomCell(randomGenome);
}

setInterval(() => {
    if (Math.random() < 0.02) {
        world.spawnRandomCell(randomGenome);
    }
    world.step();
    renderer.render();
}, world.dt);
