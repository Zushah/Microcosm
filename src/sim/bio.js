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

    const candidates = localMolecules.filter(m => enzymeAccepts(enzyme, m));
    if (candidates.length === 0) return null;

    shuffleArray(candidates);
    const substrates = candidates.slice(0, cls.maxInputs);

    const rate =
        Math.min(1, cls.baseRate * 1.2) *
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
    let substrateEnergySum = 0;
    for (const m of substrates) {
        for (const el in m.composition) {
            newComposition[el] = (newComposition[el] || 0) + m.composition[el];
        }
        substrateEnergySum += (m.energy || 0);
    }

    mutateComposition(newComposition);

    const product = createMolecule(newComposition);

    const yieldFactor = 0.6 + Math.random() * 0.2;
    const energyReleased = substrateEnergySum * yieldFactor;

    const energyDelta = energyReleased - (cls.energyCost || 0);

    return {
        consumed: substrates,
        produced: product,
        energyDelta
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

function shuffleArray(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
    }
}
