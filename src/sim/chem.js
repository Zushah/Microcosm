export const ELEMENTS = {
    A: { mass: 1.0, polarity: 0.9, energy: 0.5 },
    B: { mass: 1.2, polarity: 0.4, energy: 0.9 },
    C: { mass: 1.4, polarity: 0.2, energy: 0.85 },
    D: { mass: 1.8, polarity: 0.1, energy: 3.0 },
    E: { mass: 0.8, polarity: 1.0, energy: 2.2 },
    X: { mass: 1.0, polarity: 0.6, energy: -0.2 }
};

export function createMolecule(composition) {
    const comp = Object.assign({}, composition);
    let size = 0;
    let polarity = 0;
    let energy = 0;
    for (const el in comp) {
        const count = comp[el];
        size += count;
        polarity += (ELEMENTS[el].polarity || 0) * count;
        energy += (ELEMENTS[el].energy || 0) * count;
    }
    polarity = size > 0 ? polarity / size : 0;
    return {
        composition: comp,
        size,
        polarity,
        energy
    };
}

function computeStability(size, polarity) {
	return Math.max(
		0,
		1.0 - 0.1 * size - Math.abs(polarity - 0.5)
	);
}
