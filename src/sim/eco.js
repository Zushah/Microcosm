import { chance, randomGaussian, randomInt } from "./rng.js";

export const COMBAT_ENZYME_TYPES = Object.freeze(["attackase", "defensase"]);
export const DEFAULT_COMBAT_LEVEL_MEAN = 100;
export const DEFAULT_COMBAT_LEVEL_SIGMA = 10;
export const COMBAT_LEVEL_MUTATION_STEP_MIN = 1;
export const COMBAT_LEVEL_MUTATION_STEP_MAX = 4;

export const isCombatEnzymeType = (type) => type === "attackase" || type === "defensase";

export const cloneEnzyme = (enzyme) => {
    if (!enzyme) return enzyme;
    return { ...enzyme };
};

export const normalizeCombatLevel = (value, fallback = DEFAULT_COMBAT_LEVEL_MEAN) => {
    const numeric = Number.isFinite(value) ? value : fallback;
    return Math.max(1, Math.round(numeric));
};

export const mutateCombatLevel = (level) => {
    const current = normalizeCombatLevel(level);
    const step = COMBAT_LEVEL_MUTATION_STEP_MIN + randomInt(COMBAT_LEVEL_MUTATION_STEP_MAX - COMBAT_LEVEL_MUTATION_STEP_MIN + 1);
    const branch = chance(0.5) ? 0 : (chance(0.5) ? -1 : 1);
    if (branch === 0) return current;
    return normalizeCombatLevel(current + branch * step);
};

export const sumCombatLevels = (enzymes, type) => {
    if (!Array.isArray(enzymes) || !isCombatEnzymeType(type)) return 0;
    let total = 0;
    for (let i = 0; i < enzymes.length; i++) {
        const enzyme = enzymes[i];
        if (!enzyme || enzyme.type !== type) continue;
        total += normalizeCombatLevel(enzyme.level);
    }
    return total;
};

const getCellCombatTotals = (cell) => {
    if (cell && Number.isFinite(cell.combatAttackTotal) && Number.isFinite(cell.combatDefenseTotal)) return { attack: cell.combatAttackTotal, defense: cell.combatDefenseTotal };
    const enzymes = cell && cell.genome ? cell.genome.enzymes : null;
    return { attack: sumCombatLevels(enzymes, "attackase"), defense: sumCombatLevels(enzymes, "defensase") };
};

export const resolvePredationOutcome = (cellA, cellB) => {
    if (!cellA || !cellB || cellA === cellB) return null;
    if (cellA.state === "dead" || cellB.state === "dead") return null;
    if (cellA.lineageId === cellB.lineageId) return null;

    const aTotals = getCellCombatTotals(cellA);
    const bTotals = getCellCombatTotals(cellB);

    const aCanKill = aTotals.attack > 0 && aTotals.attack > bTotals.defense;
    const bCanKill = bTotals.attack > 0 && bTotals.attack > aTotals.defense;

    if (!aCanKill && !bCanKill) return null;
    if (aCanKill && !bCanKill) {
        return {
            winner: cellA,
            loser: cellB,
            winnerAttack: aTotals.attack,
            winnerDefense: aTotals.defense,
            loserAttack: bTotals.attack,
            loserDefense: bTotals.defense,
            winnerMargin: aTotals.attack - bTotals.defense,
            loserMargin: bTotals.attack - aTotals.defense
        };
    }
    if (bCanKill && !aCanKill) {
        return {
            winner: cellB,
            loser: cellA,
            winnerAttack: bTotals.attack,
            winnerDefense: bTotals.defense,
            loserAttack: aTotals.attack,
            loserDefense: aTotals.defense,
            winnerMargin: bTotals.attack - aTotals.defense,
            loserMargin: aTotals.attack - bTotals.defense
        };
    }

    const aMargin = aTotals.attack - bTotals.defense;
    const bMargin = bTotals.attack - aTotals.defense;

    if (aMargin === bMargin) return null;
    if (aMargin > bMargin) {
        return {
            winner: cellA,
            loser: cellB,
            winnerAttack: aTotals.attack,
            winnerDefense: aTotals.defense,
            loserAttack: bTotals.attack,
            loserDefense: bTotals.defense,
            winnerMargin: aMargin,
            loserMargin: bMargin
        };
    }

    return {
        winner: cellB,
        loser: cellA,
        winnerAttack: bTotals.attack,
        winnerDefense: bTotals.defense,
        loserAttack: aTotals.attack,
        loserDefense: aTotals.defense,
        winnerMargin: bMargin,
        loserMargin: aMargin
    };
};
