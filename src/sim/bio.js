import {
    createMolecule,
    ELEMENTS,
    ALL_ELEMENT_MASK,
    compositionToElementMask,
    elementToMask,
    normalizeSpecificityMask
} from "./chem.js";

export const ENZYME_CLASSES = {
    anabolase: {
        maxInputs: 3,
        baseRate: 0.85,
        energyCost: 0.005,
        envalThroughput: 0.18,
        envalPump: 0.12,
        bondMultiplier: 1.15
    },
    catabolase: {
        maxInputs: 1,
        baseRate: 0.98,
        energyCost: 0.006,
        envalThroughput: 0.12,
        envalPump: 0.12,
        transmuteProb: 0.35,
        bondHarvestFraction: 0.96,
        transmuteHarvestFraction: 0.80
    },
    transmutase: {
        maxInputs: 1,
        baseRate: 0.90,
        energyCost: 0.010,
        envalThroughput: 0.08,
        envalPump: 0.03,
        downhillHarvestFraction: 0.90
    }
};

export let reactionsThisTick = 0;
export const resetReactionCounter = () => reactionsThisTick = 0;

const DEFAULT_ENVAL_SIGMA = 0.18;
const DEFAULT_ENVAL_ENERGY_FRACTION = 2 / 3;
const DEFAULT_ENVAL_RELEASE_FRACTION = 1 / 3;
const DEFAULT_ENVAL_PUMP = 0.1;

const _elementDeltaScratch = new Int32Array(6);

const TRANSMUTATION_UP = {
    F: "A",
    A: "C",
    C: "B",
    B: "E",
    E: "D"
};

const TRANSMUTATION_DOWN = {
    A: "F",
    B: "F",
    C: "F",
    D: "F",
    E: "F"
};


const computeCompositionMask = (comp) => {
    let mask = 0;
    if (!comp) return mask;
    for (const el in comp) {
        if (!comp[el]) continue;
        if (el === "A") mask |= 1;
        else if (el === "B") mask |= 2;
        else if (el === "C") mask |= 4;
        else if (el === "D") mask |= 8;
        else if (el === "E") mask |= 16;
        else if (el === "F") mask |= 32;
    }
    return mask;
};

const computeCompositionSize = (comp) => {
    let size = 0;
    if (!comp) return size;
    for (const el in comp) size += comp[el] || 0;
    return size;
};

const _accumulateCompDelta = (scratch, comp, sign) => {
    if (!comp) return;
    for (const el in comp) {
        const v = comp[el] || 0;
        if (v === 0) continue;
        if (el === "A") scratch[0] += sign * v;
        else if (el === "B") scratch[1] += sign * v;
        else if (el === "C") scratch[2] += sign * v;
        else if (el === "D") scratch[3] += sign * v;
        else if (el === "E") scratch[4] += sign * v;
        else if (el === "F") scratch[5] += sign * v;
    }
};

const _sampleAcceptedSubstrates = (enzyme, poolA, poolB, maxCount) => {
    const out = [];
    let seen = 0;
    const rand = Math.random;
    const scan = (arr) => {
        for (let i = 0; i < arr.length; i++) {
            const m = arr[i];
            if (!enzymeAccepts(enzyme, m)) continue;
            seen++;
            if (out.length < maxCount) {
                out.push(m);
            } else {
                const j = (rand() * seen) | 0;
                if (j < maxCount) out[j] = m;
            }
        }
    };
    if (poolA && poolA.length) scan(poolA);
    if (poolB && poolB.length) scan(poolB);
    return out;
};

const clamp01 = (v) => Math.max(0, Math.min(1, v));

const specificityMaskFromLegacyAffinity = (affinity) => {
    if (!affinity) return 0;
    return compositionToElementMask(affinity);
};

const getSpecificityMask = (enzyme) => {
    if (!enzyme) return ALL_ELEMENT_MASK;

    let mask = enzyme.specificityMask;
    if (!Number.isFinite(mask) && enzyme.affinity) {
        mask = specificityMaskFromLegacyAffinity(enzyme.affinity);
    }

    mask = normalizeSpecificityMask(mask, ALL_ELEMENT_MASK);
    enzyme.specificityMask = mask;

    delete enzyme.affinity;
    delete enzyme._affinityMask;

    return mask;
};


const computeEnvalFactor = (cell, enzyme, env) => {
    const localEnval = env.localEnval ?? env.enval ?? 0;
    const optimalEnval = (cell && cell.genome && typeof cell.genome.optimalEnval === "number")
        ? cell.genome.optimalEnval
        : (typeof enzyme.optimalEnval === "number" ? enzyme.optimalEnval : 0);
    const sigma = Math.max(1e-6, enzyme.envalSigma ?? DEFAULT_ENVAL_SIGMA);
    const d = localEnval - optimalEnval;
    const denom = 2 * sigma * sigma;
    return Math.exp(-(d * d) / denom);
};

const resolveEnvalPolarity = (cell) => {
    const opt = cell && cell.genome ? (cell.genome.optimalEnval ?? 0) : 0;
    if (opt > 0) return 1;
    if (opt < 0) return -1;
    return Math.random() < 0.5 ? -1 : 1;
};

const computeEnvalExchange = (enzyme, cls, cell, tile, env) => {
    if (!cell) {
        return {
            energyBonus: 0,
            envalInput: 0,
            envalOutput: 0,
            envalPump: 0,
            deltaEnval: 0
        };
    }

    const polarity = resolveEnvalPolarity(cell);
    const localEnval = env.localEnval ?? env.enval ?? 0;
    const availableAligned = Math.max(0, polarity * localEnval);
    const maxInput = Math.max(0, enzyme.envalThroughput ?? cls.envalThroughput ?? 0);
    const inputMagnitude = Math.min(availableAligned, maxInput);

    const energyFraction = clamp01(enzyme.envalEnergyFraction ?? DEFAULT_ENVAL_ENERGY_FRACTION);
    const releaseFraction = clamp01(Math.min(
        1 - energyFraction,
        enzyme.envalReleaseFraction ?? DEFAULT_ENVAL_RELEASE_FRACTION
    ));

    const pumpMagnitude = Math.max(0, enzyme.envalPump ?? cls.envalPump ?? DEFAULT_ENVAL_PUMP);
    const energyBonus = inputMagnitude * energyFraction;
    const recycledOutputMagnitude = inputMagnitude * releaseFraction;
    const outputMagnitude = recycledOutputMagnitude + pumpMagnitude;

    const envalInput = polarity * inputMagnitude;
    const envalOutput = -polarity * outputMagnitude;

    return {
        energyBonus,
        envalInput,
        envalOutput,
        envalPump: -polarity * pumpMagnitude,
        deltaEnval: envalOutput - envalInput
    };
};

export const attemptReaction = (enzyme, localMolecules, env, cell = null, tile = null, cellMolecules = null) => {
    const cls = ENZYME_CLASSES[enzyme.type];
    if (!cls) return null;

    const envalFactor = computeEnvalFactor(cell, enzyme, env);
    const baseRate = (typeof cls.baseRate === "number") ? cls.baseRate : 0.8;
    const rate = Math.min(1, baseRate * 1.2) * envalFactor;
    if (Math.random() > rate) return null;

    const maxInputs = (typeof cls.maxInputs === "number" && cls.maxInputs > 0) ? cls.maxInputs : 0;
    let substrates = [];
    if (maxInputs > 0) {
        const poolA = localMolecules;
        const poolB = cellMolecules;
        substrates = _sampleAcceptedSubstrates(enzyme, poolA, poolB, maxInputs);
        if (substrates.length === 0) return null;
        if (enzyme.type === "anabolase" && substrates.length < 2) return null;
    }

    const envalExchange = computeEnvalExchange(enzyme, cls, cell, tile, env);

    let result = null;
    if (enzyme.type === "anabolase") {
        result = doAnabolase(enzyme, substrates, cls, cell, tile, env, envalExchange);
    } else if (enzyme.type === "catabolase") {
        result = doCatabolase(enzyme, substrates, cls, cell, tile, env, envalExchange);
    } else if (enzyme.type === "transmutase") {
        result = doTransmutase(enzyme, substrates, cls, cell, tile, env, envalExchange);
    } else {
        result = genericTransform(enzyme, substrates, cls, envalExchange);
    }
    if (!result) return null;

    const scratch = _elementDeltaScratch;
    scratch[0] = 0;
    scratch[1] = 0;
    scratch[2] = 0;
    scratch[3] = 0;
    scratch[4] = 0;
    scratch[5] = 0;

    const consumed = result.consumed || [];
    for (let i = 0; i < consumed.length; i++) {
        const m = consumed[i];
        if (!m || !m.composition) continue;
        _accumulateCompDelta(scratch, m.composition, -1);
    }
    if (result.produced && result.produced.composition) {
        _accumulateCompDelta(scratch, result.produced.composition, 1);
    }
    const byps = result.byproducts || [];
    for (let i = 0; i < byps.length; i++) {
        const bp = byps[i];
        if (!bp || !bp.composition) continue;
        _accumulateCompDelta(scratch, bp.composition, 1);
    }

    let elementDelta = null;
    let sameComposition = true;
    if (scratch[0] !== 0) { sameComposition = false; (elementDelta || (elementDelta = {})).A = scratch[0]; }
    if (scratch[1] !== 0) { sameComposition = false; (elementDelta || (elementDelta = {})).B = scratch[1]; }
    if (scratch[2] !== 0) { sameComposition = false; (elementDelta || (elementDelta = {})).C = scratch[2]; }
    if (scratch[3] !== 0) { sameComposition = false; (elementDelta || (elementDelta = {})).D = scratch[3]; }
    if (scratch[4] !== 0) { sameComposition = false; (elementDelta || (elementDelta = {})).E = scratch[4]; }
    if (scratch[5] !== 0) { sameComposition = false; (elementDelta || (elementDelta = {})).F = scratch[5]; }
    if (elementDelta) result.elementDelta = elementDelta;

    result.efficiencyFactor = envalFactor;

    const raw = (typeof result.rawDelta === "number") ? result.rawDelta : NaN;
    const deltaEnval = (typeof result.deltaEnval === "number") ? result.deltaEnval : 0;
    const ENERGY_NOISE = 1e-2;
    if (sameComposition && Number.isFinite(raw) && Math.abs(raw) < ENERGY_NOISE && Math.abs(deltaEnval) < ENERGY_NOISE) return null;

    if (enzyme.type === "catabolase") {
        const usable = result.energyDelta || 0;
        if (usable <= 0) return null;
    }

    reactionsThisTick++;
    return result;
};

const doAnabolase = (enzyme, substrates, cls, cell, tile, env, envalExchange) => {
    if (!substrates || substrates.length === 0) return null;
    if (substrates.length < 2) return null;

    let substrateElementSum = 0;
    let substrateBondEnergy = 0;
    const compTotal = {};
    for (const m of substrates) {
        substrateElementSum += calcElementalSum(m);
        substrateBondEnergy += calcBondEnergy(m);
        for (const el in m.composition) {
            compTotal[el] = (compTotal[el] || 0) + m.composition[el];
        }
    }

    const bondMultiplier = enzyme.bondMultiplier ?? cls.bondMultiplier ?? 1.15;
    const product = createMolecule(compTotal, bondMultiplier);
    const productElementSum = calcElementalSum(product);
    const productBondEnergy = calcBondEnergy(product);

    const additionalBondStorage = productBondEnergy - substrateBondEnergy;
    const catalyticCost = cls.energyCost || 0;
    const envalEnergy = envalExchange.energyBonus || 0;
    const requiredEnergy = Math.max(0, additionalBondStorage) + catalyticCost;

    if (!cell || ((cell.energy || 0) + envalEnergy) < requiredEnergy) return null;

    const bondEnergyDelta = -additionalBondStorage;
    const chemicalDelta = bondEnergyDelta - catalyticCost;
    const netEnergyDelta = chemicalDelta + envalEnergy;

    return {
        consumed: substrates,
        produced: product,
        byproducts: [],
        energyDelta: netEnergyDelta,
        chemicalDelta,
        bondEnergyDelta,
        bondEnergyRaw: -additionalBondStorage,
        transmutationEnergyDelta: 0,
        transmutationEnergyRaw: 0,
        catalyticCost,
        envalEnergy,
        envalInput: envalExchange.envalInput,
        envalOutput: envalExchange.envalOutput,
        deltaEnval: envalExchange.deltaEnval,
        substrateAtomEnergy: substrateElementSum,
        productAtomEnergy: productElementSum,
        substrateBondEnergy,
        productBondEnergy,
        rawDelta: chemicalDelta
    };
};

const transmutaseElementWeight = (enzyme, el, count) => {
    const specificityMask = getSpecificityMask(enzyme);
    if ((specificityMask & elementToMask(el)) === 0) return 0;
    return Math.max(0, count || 0);
};

const chooseWeightedOption = (options) => {
    if (!options || options.length === 0) return null;
    let total = 0;
    for (let i = 0; i < options.length; i++) total += Math.max(0, options[i].weight || 0);
    if (total <= 0) return options[(Math.random() * options.length) | 0];

    let r = Math.random() * total;
    for (let i = 0; i < options.length; i++) {
        r -= Math.max(0, options[i].weight || 0);
        if (r <= 0) return options[i];
    }
    return options[options.length - 1];
};

const calcMoleculeTotalEnergy = (mol) => {
    if (!mol || !mol.composition) return 0;
    return calcElementalSum(mol) * (mol.bondMultiplier || 1.0);
};

const computeTransmutaseUphillBias = (enzyme, cls, cell, env) => {
    const polarity = resolveEnvalPolarity(cell);
    const localEnval = env.localEnval ?? env.enval ?? 0;
    const alignedEnval = Math.max(0, polarity * localEnval);
    const throughput = Math.max(1e-6, enzyme.envalThroughput ?? cls.envalThroughput ?? 0.08);
    const alignedFrac = clamp01(alignedEnval / throughput);
    const reserveTarget = Math.max(0.25, ((cell && cell.genome ? cell.genome.reproThreshold : 1) || 1) * 0.5);
    const energyFrac = clamp01(((cell && typeof cell.energy === "number") ? cell.energy : 0) / reserveTarget);
    return clamp01(0.10 + 0.50 * alignedFrac + 0.25 * energyFrac);
};

const doCatabolase = (enzyme, substrates, cls, cell, tile, env, envalExchange) => {
    if (!substrates || substrates.length === 0) return null;
    const mol = substrates[0];

    const substrateElementSum = calcElementalSum(mol);
    const substrateBondEnergy = calcBondEnergy(mol);

    const transmuteProb = clamp01(enzyme.transmuteProb ?? cls.transmuteProb ?? 0.75);
    const newComp = Object.assign({}, mol.composition);
    if (Math.random() < transmuteProb) {
        for (const el of ["D", "E"]) {
            while ((newComp[el] || 0) > 0 && Math.random() < 0.6) {
                newComp[el] -= 1;
                if (newComp[el] === 0) delete newComp[el];
                newComp["F"] = (newComp["F"] || 0) + 1;
            }
        }
    }

    const byproducts = fragmentComposition(newComp);
    const productElementSum = calcCompositionSum(newComp);
    const productBondEnergy = calcBondEnergyCollection(byproducts);

    const bondEnergyReleasedRaw = Math.max(0, substrateBondEnergy - productBondEnergy);
    const transmutationEnergyRaw = substrateElementSum - productElementSum;

    const bondHarvestFraction = clamp01(
        enzyme.bondHarvestFraction ?? cls.bondHarvestFraction ?? enzyme.harvestFraction ?? 0.85
    );
    const transmuteHarvestFraction = clamp01(
        enzyme.transmuteHarvestFraction ?? enzyme.harvestFraction ?? cls.transmuteHarvestFraction ?? 0.85
    );

    const bondEnergyDelta = Math.max(0, bondEnergyReleasedRaw) * bondHarvestFraction;
    const transmutationEnergyDelta = Math.max(0, transmutationEnergyRaw) * transmuteHarvestFraction;
    const catalyticCost = cls.energyCost || 0;
    const envalEnergy = envalExchange.energyBonus || 0;
    const chemicalDelta = bondEnergyDelta + transmutationEnergyDelta - catalyticCost;
    const energyDelta = chemicalDelta + envalEnergy;

    return {
        consumed: [mol],
        produced: null,
        byproducts,
        energyDelta,
        chemicalDelta,
        bondEnergyDelta,
        bondEnergyRaw: bondEnergyReleasedRaw,
        transmutationEnergyDelta,
        transmutationEnergyRaw,
        catalyticCost,
        envalEnergy,
        envalInput: envalExchange.envalInput,
        envalOutput: envalExchange.envalOutput,
        deltaEnval: envalExchange.deltaEnval,
        substrateAtomEnergy: substrateElementSum,
        productAtomEnergy: productElementSum,
        substrateBondEnergy,
        productBondEnergy,
        rawDelta: chemicalDelta
    };
};

const doTransmutase = (enzyme, substrates, cls, cell, tile, env, envalExchange) => {
    if (!substrates || substrates.length === 0) return null;
    const mol = substrates[0];
    if (!mol || !mol.composition) return null;

    const substrateAtomEnergy = calcElementalSum(mol);
    const substrateBondEnergy = calcBondEnergy(mol);
    const substrateTotalEnergy = calcMoleculeTotalEnergy(mol);
    const bondMultiplier = mol.bondMultiplier || 1.0;

    const uphillCandidates = [];
    const downhillCandidates = [];

    for (const el in mol.composition) {
        const count = mol.composition[el] || 0;
        if (count <= 0) continue;

        const weight = transmutaseElementWeight(enzyme, el, count);
        if (weight <= 0) continue;

        const uphillTarget = TRANSMUTATION_UP[el];
        if (uphillTarget) {
            const comp = Object.assign({}, mol.composition);
            comp[el] -= 1;
            if (comp[el] <= 0) delete comp[el];
            comp[uphillTarget] = (comp[uphillTarget] || 0) + 1;
            const product = createMolecule(comp, bondMultiplier);
            uphillCandidates.push({
                source: el,
                target: uphillTarget,
                product,
                raw: substrateTotalEnergy - calcMoleculeTotalEnergy(product),
                weight: weight * count
            });
        }

        const downhillTarget = TRANSMUTATION_DOWN[el];
        if (downhillTarget) {
            const comp = Object.assign({}, mol.composition);
            comp[el] -= 1;
            if (comp[el] <= 0) delete comp[el];
            comp[downhillTarget] = (comp[downhillTarget] || 0) + 1;
            const product = createMolecule(comp, bondMultiplier);
            downhillCandidates.push({
                source: el,
                target: downhillTarget,
                product,
                raw: substrateTotalEnergy - calcMoleculeTotalEnergy(product),
                weight: weight * count
            });
        }
    }

    if (uphillCandidates.length === 0 && downhillCandidates.length === 0) return null;

    const envalEnergy = envalExchange.energyBonus || 0;
    const catalyticCost = cls.energyCost || 0;
    const availableEnergy = ((cell && typeof cell.energy === "number") ? cell.energy : 0) + envalEnergy;
    const affordableUphill = uphillCandidates.filter((candidate) => {
        const requiredEnergy = catalyticCost + Math.max(0, -candidate.raw);
        return availableEnergy >= requiredEnergy;
    });

    let chosen = null;
    if (affordableUphill.length > 0 && downhillCandidates.length > 0) {
        const uphillBias = computeTransmutaseUphillBias(enzyme, cls, cell, env);
        chosen = (Math.random() < uphillBias)
            ? chooseWeightedOption(affordableUphill)
            : chooseWeightedOption(downhillCandidates);
    } else if (affordableUphill.length > 0) {
        chosen = chooseWeightedOption(affordableUphill);
    } else {
        chosen = chooseWeightedOption(downhillCandidates);
    }

    if (!chosen) return null;

    const transmutationEnergyRaw = chosen.raw;
    const downhillHarvestFraction = clamp01(
        enzyme.downhillHarvestFraction ?? cls.downhillHarvestFraction ?? 0.90
    );
    const transmutationEnergyDelta = (transmutationEnergyRaw >= 0)
        ? transmutationEnergyRaw * downhillHarvestFraction
        : transmutationEnergyRaw;
    const requiredEnergy = catalyticCost + Math.max(0, -transmutationEnergyRaw);
    if (transmutationEnergyRaw < 0 && availableEnergy < requiredEnergy) return null;

    const chemicalDelta = transmutationEnergyDelta - catalyticCost;
    const energyDelta = chemicalDelta + envalEnergy;
    const product = chosen.product;
    const productAtomEnergy = calcElementalSum(product);
    const productBondEnergy = calcBondEnergy(product);

    return {
        consumed: [mol],
        produced: product,
        byproducts: [],
        energyDelta,
        chemicalDelta,
        bondEnergyDelta: 0,
        bondEnergyRaw: 0,
        transmutationEnergyDelta,
        transmutationEnergyRaw,
        catalyticCost,
        envalEnergy,
        envalInput: envalExchange.envalInput,
        envalOutput: envalExchange.envalOutput,
        deltaEnval: envalExchange.deltaEnval,
        substrateAtomEnergy,
        productAtomEnergy,
        substrateBondEnergy,
        productBondEnergy,
        transmutedFrom: chosen.source,
        transmutedTo: chosen.target,
        transmutedDirection: transmutationEnergyRaw >= 0 ? "downhill" : "uphill",
        rawDelta: chemicalDelta
    };
};

const genericTransform = (enzyme, substrates, cls, envalExchange) => {
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

    let atomList = [];
    for (const el in totalComp) {
        for (let i = 0; i < totalComp[el]; i++) atomList.push(el);
    }
    atomList = Chalkboard.stat.shuffle(atomList);
    const primaryCount = Math.max(1, Math.round(atomList.length * 0.7));
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
    const primaryProduct = createMolecule(pComp, 1.0);

    let transgain = 0;
    if (Math.random() < 0.06) {
        for (const el of ["D", "E"]) {
            while (primaryProduct.composition[el] > 0 && Math.random() < 0.35) {
                primaryProduct.composition[el] -= 1;
                if (primaryProduct.composition[el] === 0) delete primaryProduct.composition[el];
                primaryProduct.composition["F"] = (primaryProduct.composition["F"] || 0) + 1;
                const before = ELEMENTS[el].energy || 0;
                const after = ELEMENTS["F"].energy || 0;
                transgain += (before - after);
            }
        }
    }

    let productAtomEnergy = 0;
    for (const el in primaryProduct.composition) productAtomEnergy += (ELEMENTS[el].energy || 0) * primaryProduct.composition[el];
    for (const bp of byps) for (const el in bp.composition) productAtomEnergy += (ELEMENTS[el].energy || 0) * bp.composition[el];

    const chemicalRawDelta = substrateAtomEnergy - productAtomEnergy - (cls.energyCost || 0) + transgain;
    const harvestFraction = enzyme.harvestFraction ?? 0.7;
    const chemicalDelta = Math.max(0, chemicalRawDelta * harvestFraction);
    const envalEnergy = envalExchange.energyBonus || 0;
    const usableEnergy = chemicalDelta + envalEnergy;

    return {
        consumed: substrates,
        produced: primaryProduct,
        byproducts: byps,
        energyDelta: usableEnergy,
        chemicalDelta,
        bondEnergyDelta: 0,
        bondEnergyRaw: 0,
        transmutationEnergyDelta: Math.max(0, transgain * harvestFraction),
        transmutationEnergyRaw: transgain,
        catalyticCost: cls.energyCost || 0,
        envalEnergy,
        envalInput: envalExchange.envalInput,
        envalOutput: envalExchange.envalOutput,
        deltaEnval: envalExchange.deltaEnval,
        substrateAtomEnergy,
        productAtomEnergy,
        substrateBondEnergy: 0,
        productBondEnergy: 0,
        rawDelta: chemicalDelta
    };
};

const enzymeAccepts = (enzyme, molecule) => {
    if (!enzyme) return true;

    const specificityMask = getSpecificityMask(enzyme);
    const moleculeMask = molecule && molecule.composition
        ? compositionToElementMask(molecule.composition)
        : (molecule ? molecule.elementMask : 0);

    if (!moleculeMask) return false;

    if (enzyme.type === "transmutase") {
        return (moleculeMask & specificityMask) !== 0;
    }

    return (moleculeMask & ~specificityMask) === 0;
};

const fragmentComposition = (comp) => {
    let atoms = [];
    for (const el in comp) {
        for (let i = 0; i < comp[el]; i++) atoms.push(el);
    }
    atoms = Chalkboard.stat.shuffle(atoms);
    const firstCount = Math.ceil(atoms.length * 0.6);
    const a1 = {};
    for (let i = 0; i < firstCount; i++) a1[atoms[i]] = (a1[atoms[i]] || 0) + 1;
    const a2 = {};
    for (let i = firstCount; i < atoms.length; i++) a2[atoms[i]] = (a2[atoms[i]] || 0) + 1;
    const out = [];
    if (Object.keys(a1).length > 0) out.push(createMolecule(a1, 1.0));
    if (Object.keys(a2).length > 0) out.push(createMolecule(a2, 1.0));
    return out;
};

const calcElementalSum = (mol) => {
    let s = 0;
    if (!mol || !mol.composition) return s;
    for (const el in mol.composition) s += (ELEMENTS[el].energy || 0) * mol.composition[el];
    return s;
};

const calcCompositionSum = (comp) => {
    let s = 0;
    if (!comp) return s;
    for (const el in comp) s += (ELEMENTS[el].energy || 0) * comp[el];
    return s;
};

const calcBondEnergy = (mol) => {
    if (!mol || !mol.composition) return 0;
    const elementalSum = calcElementalSum(mol);
    const totalEnergy = elementalSum * (mol.bondMultiplier || 1.0);
    return Math.max(0, totalEnergy - elementalSum);
};

const calcBondEnergyCollection = (molecules) => {
    let s = 0;
    if (!molecules) return s;
    for (let i = 0; i < molecules.length; i++) s += calcBondEnergy(molecules[i]);
    return s;
};
