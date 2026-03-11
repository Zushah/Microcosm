import { createMolecule, ELEMENTS } from "./chem.js";

export const ENZYME_CLASSES = {
    anabolase:  { maxInputs: 3, baseRate: 0.85, energyCost: 0.005 },
    catabolase:{ maxInputs: 1, baseRate: 0.98, energyCost: 0.006 },
    transportase: { maxInputs: 0, baseRate: 0.90, energyCost: 0.010 }
};

export let reactionsThisTick = 0;
export const resetReactionCounter = () => reactionsThisTick = 0;

const _elementDeltaScratch = new Int32Array(6);

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
        else if (el === "X") scratch[5] += sign * v;
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

export const attemptReaction = (enzyme, localMolecules, env, cell = null, tile = null, cellMolecules = null) => {
    const cls = ENZYME_CLASSES[enzyme.type];
    if (!cls) return null;
    const T = env.temperature ?? 0.5;
    const tOpt = enzyme.tOpt ?? 0.5;
    const tempSigma = enzyme.tempSigma ?? 0.18;
    const dT = T - tOpt;
    const denom = 2 * tempSigma * tempSigma;
    const tempFactor = Math.exp(-(dT * dT) / denom);
    const pHFactor = Math.exp(-Math.abs((env.pH ?? 0.5) - (enzyme.pHOpt ?? 0.5)));
    const baseRate = (typeof cls.baseRate === "number") ? cls.baseRate : 0.8;
    const rate = Math.min(1, baseRate * 1.2) * tempFactor * pHFactor;
    if (Math.random() > rate) return null;
    let substrates = [];
    if (enzyme.type !== "transportase") {
        const maxInputs = (typeof cls.maxInputs === "number" && cls.maxInputs > 0) ? cls.maxInputs : 1;
        const poolA = localMolecules;
        const poolB = cellMolecules;
        substrates = _sampleAcceptedSubstrates(enzyme, poolA, poolB, maxInputs);
        if (substrates.length === 0) return null;
        if (enzyme.type === "anabolase" && substrates.length < 2) return null;
    }
    let result = null;
    if (enzyme.type === "anabolase") {
        result = doAnabolase(enzyme, substrates, cls, cell, tile, env);
    } else if (enzyme.type === "catabolase") {
        result = doCatabolase(enzyme, substrates, cls, cell, tile, env);
    } else if (enzyme.type === "transportase") {
        result = doTransportase(enzyme, cls, cell, tile, env);
    } else {
        result = genericTransform(enzyme, substrates, cls);
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
    if (scratch[5] !== 0) { sameComposition = false; (elementDelta || (elementDelta = {})).X = scratch[5]; }
    if (elementDelta) result.elementDelta = elementDelta;
    const raw = (typeof result.rawDelta === "number") ? result.rawDelta : NaN;
    const ENERGY_NOISE = 1e-2;
    if (sameComposition && Number.isFinite(raw) && Math.abs(raw) < ENERGY_NOISE) return null;
    if (enzyme.type === "catabolase") {
        const usable = result.energyDelta || 0;
        if (usable <= 0) return null;
    }
    reactionsThisTick++;
    return result;
};

const doAnabolase = (enzyme, substrates, cls, cell, tile, env) => {
    if (!substrates || substrates.length === 0) return null;
    if (substrates.length < 2) return null;

    let elementalSum = 0;
    const compTotal = {};
    for (const m of substrates) {
        for (const el in m.composition) {
            compTotal[el] = (compTotal[el] || 0) + m.composition[el];
            elementalSum += (ELEMENTS[el].energy || 0) * m.composition[el];
        }
    }

    const bondMultiplier = enzyme.bondMultiplier ?? 1.05;
    const productEnergy = elementalSum * bondMultiplier;
    const delta = productEnergy - elementalSum;
    const totalCost = delta + (cls.energyCost || 0);

    if (!cell || (cell.energy || 0) < totalCost) return null;

    const product = createMolecule(compTotal, bondMultiplier);

    cell.energy = Math.max(0, (cell.energy || 0) - totalCost);

    const heatDelta = -delta;

    return {
        consumed: substrates,
        produced: product,
        byproducts: [],
        energyDelta: 0,
        heatDelta,
        substrateAtomEnergy: elementalSum,
        productAtomEnergy: productEnergy,
        rawDelta: -totalCost
    };
};

const doCatabolase = (enzyme, substrates, cls, cell, tile, env) => {
    if (!substrates || substrates.length === 0) return null;
    const mol = substrates[0];

    const productElementSum = mol.elementalEnergySum || calcElementalSum(mol);
    const productEnergy = mol.energy || productElementSum * (mol.bondMultiplier || 1.0);

    const rawBondEnergy = productEnergy - productElementSum;

    const transmuteProb = enzyme.transmuteProb ?? 0.75;
    let transmutedEnergyGain = 0;
    const newComp = Object.assign({}, mol.composition);
    if (Math.random() < transmuteProb) {
        for (const el of ["D", "E"]) {
            while ((newComp[el] || 0) > 0 && Math.random() < 0.6) {
                newComp[el] -= 1;
                if (newComp[el] === 0) delete newComp[el];
                newComp["X"] = (newComp["X"] || 0) + 1;
                const before = ELEMENTS[el].energy || 0;
                const after = ELEMENTS["X"].energy || 0;
                transmutedEnergyGain += (before - after);
            }
        }
    }

    const byproducts = fragmentComposition(newComp);

    const rawDelta = rawBondEnergy + transmutedEnergyGain - (cls.energyCost || 0);

    const harvestFraction = enzyme.harvestFraction ?? 0.85;
    const usableEnergy = Math.max(0, rawDelta * harvestFraction);
    const heatDelta = rawDelta - usableEnergy;

    return {
        consumed: [mol],
        produced: null,
        byproducts,
        energyDelta: usableEnergy,
        heatDelta,
        substrateAtomEnergy: productElementSum,
        productAtomEnergy: productElementSum + rawBondEnergy - transmutedEnergyGain,
        rawDelta
    };
};

const doTransportase = (enzyme, cls, cell, tile, env) => {
    if (!cell || !tile) return null;
    const wanted = ["D", "E"];
    let moved = 0;
    const maxToMove = Math.max(1, Math.floor((enzyme.transportRate || 0.4) * 2));
    for (const w of wanted) {
        for (let i = 0; i < tile.molecules.length && moved < maxToMove; i++) {
            const m = tile.molecules[i];
            if (m.composition && (m.composition[w] || 0) > 0) {
                m.composition[w] -= 1;
                if (m.composition[w] === 0) delete m.composition[w];
                const taken = createMolecule({ [w]: 1 }, 1.0);
                cell.molecules.push(taken);
                moved++;
                if (Object.keys(m.composition).length === 0) {
                    tile.molecules.splice(i, 1);
                    i--;
                }
            }
        }
    }
    if (moved === 0) return null;
    const cost = (enzyme.activeCostPerAtom || 0.25) * moved + (cls.energyCost || 0.0);
    cell.energy = Math.max(0, (cell.energy || 0) - cost);
    const heatDelta = -0.01 * moved;
    return {
        consumed: [],
        produced: null,
        byproducts: [],
        energyDelta: 0,
        heatDelta,
        substrateAtomEnergy: 0,
        productAtomEnergy: 0,
        rawDelta: -cost
    };
};

const genericTransform = (enzyme, substrates, cls) => {
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
                primaryProduct.composition["X"] = (primaryProduct.composition["X"] || 0) + 1;
                const before = ELEMENTS[el].energy || 0;
                const after = ELEMENTS["X"].energy || 0;
                transgain += (before - after);
            }
        }
    }

    let productAtomEnergy = 0;
    for (const el in primaryProduct.composition) productAtomEnergy += (ELEMENTS[el].energy || 0) * primaryProduct.composition[el];
    for (const bp of byps) for (const el in bp.composition) productAtomEnergy += (ELEMENTS[el].energy || 0) * bp.composition[el];

    const rawDelta = substrateAtomEnergy - productAtomEnergy - (cls.energyCost || 0) + transgain;
    const harvestFraction = enzyme.harvestFraction ?? 0.7;
    const usableEnergy = Math.max(0, rawDelta * harvestFraction);
    const heatDelta = rawDelta - usableEnergy;

    return {
        consumed: substrates,
        produced: primaryProduct,
        byproducts: byps,
        energyDelta: usableEnergy,
        heatDelta,
        substrateAtomEnergy,
        productAtomEnergy,
        rawDelta
    };
};

const enzymeAccepts = (enzyme, molecule) => {
    if (!enzyme) return true;
    if (enzyme.type === "catabolase") {
        const bm = molecule.bondMultiplier || 1.0;
        const size = molecule.size || 0;
        if (bm > 1.0 + 1e-6 && size > 1) {
            return true;
        }
    }
    const affinity = enzyme.affinity;
    if (!affinity) return true;
    let affinityMask = enzyme._affinityMask;
    if (affinityMask == null) {
        affinityMask = 0;
        for (const el in affinity) {
            if (el === "A") affinityMask |= 1;
            else if (el === "B") affinityMask |= 2;
            else if (el === "C") affinityMask |= 4;
            else if (el === "D") affinityMask |= 8;
            else if (el === "E") affinityMask |= 16;
            else if (el === "X") affinityMask |= 32;
        }
        enzyme._affinityMask = affinityMask;
    }
    const mMask = molecule.elementMask;
    if (typeof mMask === "number") {
        return (mMask & affinityMask) !== 0;
    }
    if (molecule.composition) {
        for (const el in affinity) {
            if (molecule.composition[el]) return true;
        }
    }
    return false;
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
    for (const el in mol.composition) s += (ELEMENTS[el].energy || 0) * mol.composition[el];
    return s;
};
