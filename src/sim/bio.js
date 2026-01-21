import { createMolecule, ELEMENTS } from "./chem.js";

export const ENZYME_CLASSES = {
    ligase: { maxInputs: 3, baseRate: 0.65, energyCost: 0.3, denatureT: 1.2 },
    hydrolase: { maxInputs: 1, baseRate: 0.85, energyCost: 0.12, denatureT: 1.1 },
    isomerase: { maxInputs: 1, baseRate: 0.9, energyCost: 0.08, denatureT: 1.15 }
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

    const T = env.temperature ?? 0.5;
    const tOpt = enzyme.tOpt ?? 0.5;
    const tempSigma = enzyme.tempSigma ?? 0.18;
    const tempFactor = Math.exp(-Math.pow(T - tOpt, 2) / (2 * tempSigma * tempSigma));

    const pHFactor = Math.exp(-Math.abs((env.pH ?? 0.5) - (enzyme.pHOpt ?? 0.5)));

    const rate = Math.min(1, cls.baseRate * 1.2) * tempFactor * pHFactor;

    if (Math.random() > rate) return null;

    const reaction = performTransformation(enzyme, substrates, cls);
    if (reaction) reactionsThisTick++;
    return reaction;
}

function performTransformation(enzyme, substrates, cls) {
    const totalComp = {};
    let substrateAtomEnergy = 0;
    let totalAtoms = 0;
    for (const m of substrates) {
        for (const el in m.composition) {
            totalComp[el] = (totalComp[el] || 0) + m.composition[el];
            substrateAtomEnergy += (ELEMENTS[el].energy || 0) * m.composition[el];
            totalAtoms += m.composition[el];
        }
    }

    if (totalAtoms === 0) return null;

    const primaryFraction = enzyme.type === "hydrolase" ? 0.45 : 0.75;
    const primaryTargetAtoms = Math.max(1, Math.round(totalAtoms * primaryFraction));

    const atomList = [];
    for (const el in totalComp) {
        for (let i = 0; i < totalComp[el]; i++) atomList.push(el);
    }

    shuffleArray(atomList);

    const primaryAtoms = atomList.slice(0, primaryTargetAtoms);
    const remainderAtoms = atomList.slice(primaryTargetAtoms);

    const primaryComp = {};
    for (const a of primaryAtoms) primaryComp[a] = (primaryComp[a] || 0) + 1;

    const byproducts = [];
    if (remainderAtoms.length > 0) {
        const fragCount = enzyme.type === "hydrolase" ? Math.min(2, remainderAtoms.length) : (Math.random() < 0.35 ? 1 : 0);
        if (fragCount > 0) {
            for (let f = 0; f < fragCount; f++) {
                const fragComp = {};
                for (let i = 0; i < Math.ceil(remainderAtoms.length / fragCount); i++) {
                    const idx = Math.floor(Math.random() * remainderAtoms.length);
                    if (idx >= 0) {
                        const sym = remainderAtoms.splice(idx, 1)[0];
                        if (!sym) continue;
                        fragComp[sym] = (fragComp[sym] || 0) + 1;
                    }
                }
                if (Object.keys(fragComp).length > 0) byproducts.push(createMolecule(fragComp));
            }
            if (remainderAtoms.length > 0) {
                const lastFrag = {};
                for (const a of remainderAtoms) lastFrag[a] = (lastFrag[a] || 0) + 1;
                byproducts.push(createMolecule(lastFrag));
            }
        }
    }

    const primaryProduct = createMolecule(primaryComp);

    let productAtomEnergy = 0;
    for (const el in primaryComp) productAtomEnergy += (ELEMENTS[el].energy || 0) * primaryComp[el];
    for (const bp of byproducts) {
        for (const el in bp.composition) productAtomEnergy += (ELEMENTS[el].energy || 0) * bp.composition[el];
    }

    const rawDelta = substrateAtomEnergy - productAtomEnergy - (cls.energyCost || 0);

    const harvestFraction = 0.65;
    const usableEnergy = Math.max(0, rawDelta * harvestFraction);
    const heatDelta = rawDelta - usableEnergy;

    return {
        consumed: substrates,
        produced: primaryProduct,
        byproducts,
        energyDelta: usableEnergy,
        heatDelta
    };
}

// helpers
function enzymeAccepts(enzyme, molecule) {
    if (!enzyme.affinity) return true;
    for (const el in enzyme.affinity) {
        if (molecule.composition && molecule.composition[el]) return true;
    }
    return false;
}
function shuffleArray(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
    }
}
