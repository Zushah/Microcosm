import { createMolecule } from "./chem.js";

export const ENZYME_CLASSES = {
	ligase: {
		maxInputs: 3,
		baseRate: 0.6,
		energyCost: 1.0
	},
	hydrolase: {
		maxInputs: 1,
		baseRate: 0.8,
		energyCost: 0.2
	},
	isomerase: {
		maxInputs: 1,
		baseRate: 0.9,
		energyCost: 0.1
	}
};

export function attemptReaction(enzyme, localMolecules, env) {
	const cls = ENZYME_CLASSES[enzyme.type];
	if (!cls) return null;
	const substrates = localMolecules
		.filter(m => enzymeAccepts(enzyme, m))
		.slice(0, cls.maxInputs);

	if (substrates.length === 0) return null;

	const rate =
		cls.baseRate *
		temperatureFactor(env.temperature, enzyme.tOpt) *
		pHFactor(env.pH, enzyme.pHOpt);

	if (Math.random() > rate) return null;

	return performTransformation(enzyme, substrates, cls);
}

function enzymeAccepts(enzyme, molecule) {
	for (const el in enzyme.affinity) {
		if (molecule.composition[el]) return true;
	}
	return false;
}

function performTransformation(enzyme, substrates, cls) {
	let newComposition = {};

	for (const m of substrates) {
		for (const el in m.composition) {
			newComposition[el] = (newComposition[el] || 0) + m.composition[el];
		}
	}

	mutateComposition(newComposition);

	const product = createMolecule(newComposition);

	return {
		consumed: substrates,
		produced: product,
		energyDelta: product.energy - cls.energyCost
	};
}

function mutateComposition(comp) {
	if (Math.random() < 0.2) {
		const keys = Object.keys(comp);
		if (keys.length === 0) return;
		const k = keys[Math.floor(Math.random() * keys.length)];
		comp[k] = Math.max(0, comp[k] - 1);
	}
}

function temperatureFactor(T, Topt = 0.5) {
  	return Math.exp(-Math.abs(T - Topt));
}

function pHFactor(pH, pHopt = 0.5) {
  	return Math.exp(-Math.abs(pH - pHopt));
}
