export const ELEMENTS = {
    A: { mass: 1.0, polarity: 0.9, energy: 0.5 },
    B: { mass: 1.2, polarity: 0.4, energy: 1.0 },
    C: { mass: 1.4, polarity: 0.2, energy: 0.9 },
    D: { mass: 1.8, polarity: 0.1, energy: 4.0 },
    E: { mass: 0.8, polarity: 1.0, energy: 3.0 },
    F: { mass: 1.0, polarity: 0.6, energy: -0.2 }
};

export const ELEMENT_ORDER = ["A", "B", "C", "D", "E", "F"];

export const ELEMENT_MASKS = {
    A: 1,
    B: 2,
    C: 4,
    D: 8,
    E: 16,
    F: 32
};

export const ALL_ELEMENT_MASK = ELEMENT_ORDER.reduce((mask, el) => mask | ELEMENT_MASKS[el], 0);

export const elementToMask = (el) => ELEMENT_MASKS[el] || 0;

export const compositionToElementMask = (composition) => {
    let mask = 0;
    if (!composition) return mask;
    for (const el in composition) {
        if ((composition[el] || 0) > 0) mask |= elementToMask(el);
    }
    return mask;
};

export const maskToElements = (mask) => {
    const out = [];
    const normalized = Number.isFinite(mask) ? (mask | 0) : 0;
    for (let i = 0; i < ELEMENT_ORDER.length; i++) {
        const el = ELEMENT_ORDER[i];
        if ((normalized & ELEMENT_MASKS[el]) !== 0) out.push(el);
    }
    return out;
};

export const maskToString = (mask) => {
    const els = maskToElements(mask);
    return els.length > 0 ? els.join("") : "∅";
};

export const normalizeSpecificityMask = (mask, fallbackMask = ALL_ELEMENT_MASK) => {
    const fallback = Number.isFinite(fallbackMask) ? (fallbackMask | 0) : ALL_ELEMENT_MASK;
    const normalized = Number.isFinite(mask) ? ((mask | 0) & ALL_ELEMENT_MASK) : 0;
    return normalized !== 0 ? normalized : fallback;
};

export const createMolecule = (composition, bondMultiplier = 1.0) => {
    const comp = Object.assign({}, composition);
    let size = 0;
    let polarity = 0;
    let elementalEnergySum = 0;
    for (const el in comp) {
        const count = comp[el];
        size += count;
        polarity += (ELEMENTS[el].polarity || 0) * count;
        elementalEnergySum += (ELEMENTS[el].energy || 0) * count;
    }
    const elementMask = compositionToElementMask(comp);
    polarity = size > 0 ? polarity / size : 0;
    const energy = elementalEnergySum * bondMultiplier;
    const diffusionRate = Math.min(0.04, 0.01 + (1 / (size + 1)) * polarity * 0.04);
    const diffusionThreshold = Math.min(0xFFFFFFFF, Math.max(0, Math.floor(diffusionRate * 4294967296)));
    const diffusionInvLog1mP = (diffusionRate > 0 && diffusionRate < 1) ? (1 / Math.log1p(-diffusionRate)) : 0;
    return {
        composition: comp,
        size,
        polarity,
        elementalEnergySum,
        bondMultiplier,
        energy,
        diffusionRate,
        diffusionThreshold,
        diffusionInvLog1mP,
        elementMask
    };
};
