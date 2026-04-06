import { createMolecule, ELEMENTS, ALL_ELEMENT_MASK, compositionToElementMask, elementToMask, normalizeSpecificityMask } from "./chem.js";
import { chance, random, randomInt, shuffleInPlace } from "./rng.js";

export const ENZYME_CLASSES = {
    anabolase: {
        maxInputs: 3,
        baseRate: 0.85,
        energyCost: 0.005,
        envalThroughput: 0.18,
        envalPump: 0.3,
        bondMultiplier: 1.18,
        bondCostFraction: 0.70
    },
    catabolase: {
        maxInputs: 1,
        baseRate: 0.98,
        energyCost: 0.006,
        envalThroughput: 0.14,
        envalPump: 0.3,
        bondHarvestFraction: 1.0
    },
    transmutase: {
        maxInputs: 1,
        baseRate: 0.30,
        energyCost: 0.12,
        envalThroughput: 0.04,
        envalPump: 0.3,
        downhillHarvestFraction: 0.18
    }
};

const DEFAULT_ENVAL_SIGMA = 0.18;
const DEFAULT_ENVAL_ENERGY_FRACTION = 2 / 3;
const DEFAULT_ENVAL_RELEASE_FRACTION = 1 / 3;
const DEFAULT_ENVAL_PUMP = 0.3;

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

const computeCompositionSize = (comp) => {
    let size = 0;
    if (!comp) return size;
    for (const el in comp) size += comp[el] || 0;
    return size;
};

const sameComposition = (a, b) => {
    const aMask = compositionToElementMask(a);
    const bMask = compositionToElementMask(b);
    if (aMask !== bMask) return false;

    for (const el of ["A", "B", "C", "D", "E", "F"]) {
        if ((a && a[el] ? a[el] : 0) !== (b && b[el] ? b[el] : 0)) return false;
    }
    return true;
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
    const rand = random;
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
    return chance(0.5) ? -1 : 1;
};

const computeEnvalExchange = (enzyme, cls, cell, env) => {
    if (!cell) {
        return {
            energyBonus: 0,
            envalInput: 0,
            envalOutput: 0,
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
        deltaEnval: envalOutput - envalInput
    };
};

export const attemptReaction = (enzyme, localMolecules, env, cell = null, tile = null, cellMolecules = null) => {
    const cls = ENZYME_CLASSES[enzyme.type];
    if (!cls) return null;

    const envalFactor = computeEnvalFactor(cell, enzyme, env);
    const baseRate = (typeof cls.baseRate === "number") ? cls.baseRate : 0.8;
    const rate = Math.min(1, baseRate * 1.2) * envalFactor;
    if (random() > rate) return null;

    const maxInputs = (typeof cls.maxInputs === "number" && cls.maxInputs > 0) ? cls.maxInputs : 0;
    let substrates = [];
    if (maxInputs > 0) {
        const poolA = localMolecules;
        const poolB = cellMolecules;
        substrates = _sampleAcceptedSubstrates(enzyme, poolA, poolB, maxInputs);
        if (substrates.length === 0) return null;
        if (enzyme.type === "anabolase" && substrates.length < 2) return null;
    }

    const envalExchange = computeEnvalExchange(enzyme, cls, cell, env);

    let result = null;
    if (enzyme.type === "anabolase") {
        result = doAnabolase(enzyme, substrates, cls, cell, envalExchange);
    } else if (enzyme.type === "catabolase") {
        result = doCatabolase(enzyme, substrates, cls, envalExchange);
    } else if (enzyme.type === "transmutase") {
        result = doTransmutase(enzyme, substrates, cls, cell, env, envalExchange);
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
    let unchangedComposition = true;
    if (scratch[0] !== 0) { unchangedComposition = false; (elementDelta || (elementDelta = {})).A = scratch[0]; }
    if (scratch[1] !== 0) { unchangedComposition = false; (elementDelta || (elementDelta = {})).B = scratch[1]; }
    if (scratch[2] !== 0) { unchangedComposition = false; (elementDelta || (elementDelta = {})).C = scratch[2]; }
    if (scratch[3] !== 0) { unchangedComposition = false; (elementDelta || (elementDelta = {})).D = scratch[3]; }
    if (scratch[4] !== 0) { unchangedComposition = false; (elementDelta || (elementDelta = {})).E = scratch[4]; }
    if (scratch[5] !== 0) { unchangedComposition = false; (elementDelta || (elementDelta = {})).F = scratch[5]; }
    if (elementDelta) result.elementDelta = elementDelta;

    result.efficiencyFactor = envalFactor;

    const raw = (typeof result.rawDelta === "number") ? result.rawDelta : NaN;
    const deltaEnval = (typeof result.deltaEnval === "number") ? result.deltaEnval : 0;
    const ENERGY_NOISE = 1e-2;
    if (unchangedComposition && Number.isFinite(raw) && Math.abs(raw) < ENERGY_NOISE && Math.abs(deltaEnval) < ENERGY_NOISE) return null;

    if (enzyme.type === "catabolase") {
        const usable = result.energyDelta || 0;
        if (usable <= 0) return null;
    }

    return result;
};

const doAnabolase = (enzyme, substrates, cls, cell, envalExchange) => {
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
    const bondCostFraction = Math.max(0, enzyme.bondCostFraction ?? cls.bondCostFraction ?? 0.90);
    const product = createMolecule(compTotal, bondMultiplier);
    const productElementSum = calcElementalSum(product);
    const productBondEnergy = calcBondEnergy(product);

    const additionalBondStorage = Math.max(0, productBondEnergy - substrateBondEnergy);
    const bondEnergyCost = additionalBondStorage * bondCostFraction;
    const catalyticCost = cls.energyCost || 0;
    const envalEnergy = envalExchange.energyBonus || 0;
    const requiredEnergy = bondEnergyCost + catalyticCost;

    if (!cell || ((cell.energy || 0) + envalEnergy) < requiredEnergy) return null;

    const bondEnergyDelta = -bondEnergyCost;
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
        bondEnergyCost,
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
    if (total <= 0) return options[randomInt(options.length)];

    let r = random() * total;
    for (let i = 0; i < options.length; i++) {
        r -= Math.max(0, options[i].weight || 0);
        if (r <= 0) return options[i];
    }
    return options[options.length - 1];
};

const computeTransmutaseUphillBias = (enzyme, cls, cell, env) => {
    const polarity = resolveEnvalPolarity(cell);
    const localEnval = env.localEnval ?? env.enval ?? 0;
    const alignedEnval = Math.max(0, polarity * localEnval);
    const throughput = Math.max(1e-6, enzyme.envalThroughput ?? cls.envalThroughput ?? 0.04);
    const alignedFrac = clamp01(alignedEnval / throughput);
    const reserveTarget = Math.max(0.25, ((cell && cell.genome ? cell.genome.reproThreshold : 1) || 1) * 0.5);
    const energyFrac = clamp01(((cell && typeof cell.energy === "number") ? cell.energy : 0) / reserveTarget);
    return clamp01(0.04 + 0.35 * alignedFrac + 0.20 * energyFrac);
};

const doCatabolase = (enzyme, substrates, cls, envalExchange) => {
    if (!substrates || substrates.length === 0) return null;
    const mol = substrates[0];
    if (!mol || !mol.composition) return null;
    if ((mol.size ?? computeCompositionSize(mol.composition)) < 2) return null;

    const substrateElementSum = calcElementalSum(mol);
    const substrateBondEnergy = calcBondEnergy(mol);

    const byproducts = fragmentComposition(mol.composition);
    if (!byproducts || byproducts.length === 0) return null;
    if (byproducts.length === 1 && sameComposition(byproducts[0].composition, mol.composition)) return null;

    const productElementSum = calcMoleculeCollectionElementSum(byproducts);
    const productBondEnergy = calcBondEnergyCollection(byproducts);

    const bondEnergyReleasedRaw = Math.max(0, substrateBondEnergy - productBondEnergy);
    if (bondEnergyReleasedRaw <= 0) return null;

    const transmutationEnergyRaw = 0;
    const bondHarvestFraction = Math.max(1, enzyme.bondHarvestFraction ?? cls.bondHarvestFraction ?? 1.0);
    const bondEnergyDelta = Math.max(0, bondEnergyReleasedRaw) * bondHarvestFraction;
    const transmutationEnergyDelta = 0;
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

const doTransmutase = (enzyme, substrates, cls, cell, env, envalExchange) => {
    if (!substrates || substrates.length === 0) return null;
    const mol = substrates[0];
    if (!mol || !mol.composition) return null;

    const substrateAtomEnergy = calcElementalSum(mol);
    const substrateBondEnergy = calcBondEnergy(mol);
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
                raw: (ELEMENTS[el].energy || 0) - (ELEMENTS[uphillTarget].energy || 0),
                weight
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
                raw: (ELEMENTS[el].energy || 0) - (ELEMENTS[downhillTarget].energy || 0),
                weight
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
        chosen = (chance(uphillBias))
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
        enzyme.downhillHarvestFraction ?? cls.downhillHarvestFraction ?? 0.18
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

const enzymeAccepts = (enzyme, molecule) => {
    if (!enzyme) return true;

    const specificityMask = getSpecificityMask(enzyme);
    const moleculeMask = molecule && molecule.composition
        ? compositionToElementMask(molecule.composition)
        : (molecule ? molecule.elementMask : 0);

    if (!moleculeMask) return false;

    if (enzyme.type === "catabolase") {
        const size = molecule && molecule.composition
            ? computeCompositionSize(molecule.composition)
            : (molecule ? (molecule.size || 0) : 0);
        if (size < 2) return false;
    }

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

    if (atoms.length < 2) return [];

    shuffleInPlace(atoms);
    const firstCount = Math.max(1, Math.min(atoms.length - 1, Math.round(atoms.length * 0.6)));
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

const calcMoleculeCollectionElementSum = (molecules) => {
    let s = 0;
    if (!molecules) return s;
    for (let i = 0; i < molecules.length; i++) s += calcElementalSum(molecules[i]);
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
