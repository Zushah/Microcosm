import { createMolecule } from "./chem.js";

export const ENZYME_CLASSES = {
    ligase: {
        maxInputs: 3,
        baseRate: 0.65,
        energyCost: 0.4
    },
    hydrolase: {
        maxInputs: 1,
        baseRate: 0.85,
        energyCost: 0.15
    },
    isomerase: {
        maxInputs: 1,
        baseRate: 0.9,
        energyCost: 0.1
    }
};

export let reactionsThisTick = 0;
export function resetReactionCounter() { reactionsThisTick = 0; }

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

    const reaction = performTransformation(enzyme, substrates, cls);
    if (reaction) reactionsThisTick++;
    return reaction;
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

    const primaryProduct = createMolecule(newComposition);

    let byproducts = [];
    const byprodChance = enzyme.type === "hydrolase" ? 0.6 : 0.12;
    if (Math.random() < byprodChance) {
        const fragCount = enzyme.type === "hydrolase" ? 2 : (Math.random() < 0.5 ? 1 : 2);
        for (let f = 0; f < fragCount; f++) {
            const fragComp = {};
            const keys = Object.keys(newComposition);
            if (keys.length === 0) continue;
            const picks = 1 + Math.floor(Math.random() * Math.min(2, keys.length));
            for (let p = 0; p < picks; p++) {
                const k = keys[Math.floor(Math.random() * keys.length)];
                const available = Math.max(1, Math.floor(Math.random() * Math.max(1, newComposition[k])));
                fragComp[k] = (fragComp[k] || 0) + Math.min(available, newComposition[k]);
            }
            if (Object.keys(fragComp).length > 0) {
                byproducts.push(createMolecule(fragComp));
            }
        }
    }

    const yieldFactor = 0.55 + Math.random() * 0.25;
    const energyReleased = substrateEnergySum * yieldFactor;
    const energyDelta = energyReleased - (cls.energyCost || 0);

    const heatDelta = energyDelta * 0.06;

    return {
        consumed: substrates,
        produced: primaryProduct,
        byproducts,
        energyDelta,
        heatDelta
    };
}

function enzymeAccepts(enzyme, molecule) {
    if (!enzyme.affinity) return true;
    for (const el in enzyme.affinity) {
        if (molecule.composition && molecule.composition[el]) return true;
    }
    return false;
}

function mutateComposition(comp) {
    if (Math.random() < 0.18) {
        const keys = Object.keys(comp);
        if (keys.length === 0) return;
        const k = keys[Math.floor(Math.random() * keys.length)];
        comp[k] = Math.max(0, comp[k] - 1);
        if (comp[k] === 0) delete comp[k];
    }
}

function temperatureFactor(T, Topt = 0.5) {
    return Math.exp(-Math.abs(T - (Topt ?? 0.5)));
}
function pHFactor(pH, pHopt = 0.5) {
    return Math.exp(-Math.abs(pH - (pHopt ?? 0.5)));
}

function shuffleArray(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
    }
}
