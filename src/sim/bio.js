import { createMolecule, ELEMENTS } from "./chem.js";

export const ENZYME_CLASSES = {
    ligase: { maxInputs: 3, baseRate: 0.65, energyCost: 0.15 },
    hydrolase: { maxInputs: 1, baseRate: 0.85, energyCost: 0.05 },
    isomerase: { maxInputs: 1, baseRate: 0.9, energyCost: 0.02 }
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

    const perSubAtoms = substrates.map(m => {
        const arr = [];
        for (const el in m.composition) {
            for (let i = 0; i < m.composition[el]; i++) arr.push(el);
        }
        return arr;
    });

    const primaryComp = {};
    if (perSubAtoms.length >= 2) {
        const primaryTargetAtoms = Math.max(1, Math.round(totalAtoms * 0.7));
        const primaryAtoms = [];
        for (let i = 0; i < perSubAtoms.length; i++) {
            if (perSubAtoms[i].length > 0) {
                const idx = Math.floor(Math.random() * perSubAtoms[i].length);
                primaryAtoms.push(perSubAtoms[i].splice(idx, 1)[0]);
            }
        }
        const allRemaining = [].concat(...perSubAtoms);
        shuffleArray(allRemaining);
        while (primaryAtoms.length < primaryTargetAtoms && allRemaining.length > 0) {
            primaryAtoms.push(allRemaining.shift());
        }
        for (const a of primaryAtoms) primaryComp[a] = (primaryComp[a] || 0) + 1;
        const remainderAtoms = allRemaining;

        const byproducts = [];
        if (remainderAtoms.length > 0) {
            const fragCount = Math.min(2, Math.max(1, Math.floor(remainderAtoms.length / 2)));
            for (let f = 0; f < fragCount; f++) {
                const fragComp = {};
                const picks = Math.ceil(remainderAtoms.length / fragCount);
                for (let p = 0; p < picks && remainderAtoms.length > 0; p++) {
                    const el = remainderAtoms.shift();
                    fragComp[el] = (fragComp[el] || 0) + 1;
                }
                if (Object.keys(fragComp).length > 0) byproducts.push(createMolecule(fragComp));
            }
        }
        const primaryProduct = createMolecule(primaryComp);

        const transmuteDefault = (enzyme.type === "hydrolase") ? 0.45 : (enzyme.type === "ligase" ? 0.35 : 0.12);
        const transmuteProb = enzyme.transmuteProb ?? transmuteDefault;
        let transmutedEnergyGain = 0;
        if (Math.random() < transmuteProb) {
            const targets = [primaryProduct, ...byproducts];
            for (const targ of targets) {
                for (const el of ["D", "E"]) {
                    while (targ.composition[el] > 0 && Math.random() < 0.7) {
                        targ.composition[el] -= 1;
                        if (targ.composition[el] === 0) delete targ.composition[el];
                        targ.composition["X"] = (targ.composition["X"] || 0) + 1;
                        const before = ELEMENTS[el].energy || 0;
                        const after = ELEMENTS["X"].energy || 0;
                        transmutedEnergyGain += (before - after);
                    }
                }
            }
        }

        let productAtomEnergy = 0;
        for (const el in primaryProduct.composition) productAtomEnergy += (ELEMENTS[el].energy || 0) * primaryProduct.composition[el];
        for (const bp of byproducts) {
            for (const el in bp.composition) {
                productAtomEnergy += (ELEMENTS[el].energy || 0) * bp.composition[el];
            }
        }

        const rawDelta = substrateAtomEnergy - productAtomEnergy - (cls.energyCost || 0) + transmutedEnergyGain;
        const harvestFraction = enzyme.harvestFraction ?? 0.80;
        const usableEnergy = Math.max(0, rawDelta * harvestFraction);
        const heatDelta = rawDelta - usableEnergy;

        return {
            consumed: substrates,
            produced: primaryProduct,
            byproducts,
            energyDelta: usableEnergy,
            heatDelta,
            substrateAtomEnergy,
            productAtomEnergy,
            rawDelta
        };
    }

    const atomList = [];
    for (const el in totalComp) {
        for (let i = 0; i < totalComp[el]; i++) atomList.push(el);
    }
    shuffleArray(atomList);

    const primaryCount = Math.max(1, Math.round(atomList.length * 0.5));
    const pAtoms = atomList.slice(0, primaryCount);
    const rAtoms = atomList.slice(primaryCount);
    const pComp = {};
    for (const a of pAtoms) pComp[a] = (pComp[a] || 0) + 1;
    const byps = [];
    if (rAtoms.length > 0) {
        const frag = {};
        for (const a of rAtoms) frag[a] = (frag[a] || 0) + 1;
        byps.push(createMolecule(frag));
    }
    const primaryProduct2 = createMolecule(pComp);

    let transgain2 = 0;
    const transmuteProb2 = enzyme.transmuteProb ?? 0.25;
    if (Math.random() < transmuteProb2) {
        for (const el of ["D", "E"]) {
            while (primaryProduct2.composition[el] > 0 && Math.random() < 0.6) {
                primaryProduct2.composition[el] -= 1;
                if (primaryProduct2.composition[el] === 0) delete primaryProduct2.composition[el];
                primaryProduct2.composition["X"] = (primaryProduct2.composition["X"] || 0) + 1;
                const before = ELEMENTS[el].energy || 0;
                const after = ELEMENTS["X"].energy || 0;
                transgain2 += (before - after);
            }
        }
    }

    let productAtomEnergy2 = 0;
    for (const el in primaryProduct2.composition) productAtomEnergy2 += (ELEMENTS[el].energy || 0) * primaryProduct2.composition[el];
    for (const bp of byps) {
        for (const el in bp.composition) productAtomEnergy2 += (ELEMENTS[el].energy || 0) * bp.composition[el];
    }

    const rawDelta2 = substrateAtomEnergy - productAtomEnergy2 - (cls.energyCost || 0) + transgain2;
    const harvestFraction2 = enzyme.harvestFraction ?? 0.75;
    const usableEnergy2 = Math.max(0, rawDelta2 * harvestFraction2);
    const heatDelta2 = rawDelta2 - usableEnergy2;

    return {
        consumed: substrates,
        produced: primaryProduct2,
        byproducts: byps,
        energyDelta: usableEnergy2,
        heatDelta: heatDelta2,
        substrateAtomEnergy,
        productAtomEnergy: productAtomEnergy2,
        rawDelta: rawDelta2
    };
}

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
