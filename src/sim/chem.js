export const ELEMENTS = {
    A: { mass: 1.0, polarity: 0.9, energy: 0.5 },
    B: { mass: 1.2, polarity: 0.4, energy: 1.0 },
    C: { mass: 1.4, polarity: 0.2, energy: 0.9 },
    D: { mass: 1.8, polarity: 0.1, energy: 2.2 },
    E: { mass: 0.8, polarity: 1.0, energy: 1.6 },
    X: { mass: 1.0, polarity: 0.6, energy: -0.2 }
};

export function createMolecule(composition) {
	let mass = 0;
	let polarity = 0;
	let energy = 0;
	let size = 0;

	for (const el in composition) {
		const count = composition[el];
		const props = ELEMENTS[el];

		size += count;
		mass += props.mass * count;
		polarity += props.polarity * count;
		energy += props.energy * count;
	}

	polarity /= Math.max(size, 1);

	return {
		composition,
		size,
		mass,
		polarity,
		energy,
		stability: computeStability(size, polarity),
	};
}

function computeStability(size, polarity) {
	return Math.max(
		0,
		1.0 - 0.1 * size - Math.abs(polarity - 0.5)
	);
}
