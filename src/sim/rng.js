const UINT32_RANGE = 4294967296;

const xmur3 = (text) => {
    let h = 1779033703 ^ text.length;
    for (let i = 0; i < text.length; i++) {
        h = Math.imul(h ^ text.charCodeAt(i), 3432918353);
        h = (h << 13) | (h >>> 19);
    }
    return () => {
        h = Math.imul(h ^ (h >>> 16), 2246822507);
        h = Math.imul(h ^ (h >>> 13), 3266489909);
        return (h ^ (h >>> 16)) >>> 0;
    };
};

const sfc32 = (a, b, c, d) => {
    return () => {
        a >>>= 0;
        b >>>= 0;
        c >>>= 0;
        d >>>= 0;
        const t = (a + b + d) >>> 0;
        d = (d + 1) >>> 0;
        a = b ^ (b >>> 9);
        b = (c + (c << 3)) >>> 0;
        c = ((c << 21) | (c >>> 11)) >>> 0;
        c = (c + t) >>> 0;
        return t / UINT32_RANGE;
    };
};

let _seed = "";
let _nextFloat = null;

const ensureGenerator = () => {
    if (!_nextFloat) setSeed(createRandomSeed());
    return _nextFloat;
};

export const createRandomSeed = () => {
    const cryptoObj = globalThis.crypto;
    if (cryptoObj && typeof cryptoObj.getRandomValues === "function") {
        const buffer = new Uint32Array(4);
        cryptoObj.getRandomValues(buffer);
        return Array.from(buffer, (value) => value.toString(16).padStart(8, "0")).join("");
    }
    const now = Date.now().toString(16);
    const perf = (globalThis.performance && typeof globalThis.performance.now === "function") ? Math.floor(globalThis.performance.now() * 1000).toString(16) : "0";
    return `${now}${perf.padStart(8, "0")}`;
};

export const setSeed = (seed) => {
    _seed = `${seed ?? ""}`;
    const seedFactory = xmur3(_seed);
    _nextFloat = sfc32(seedFactory(), seedFactory(), seedFactory(), seedFactory());
    for (let i = 0; i < 12; i++) _nextFloat();
    return _seed;
};

export const getSeed = () => _seed;

export const random = () => ensureGenerator()();

export const randomRange = (min, max) => min + (max - min) * random();

export const randomInt = (maxExclusive) => {
    const max = Math.floor(Number(maxExclusive) || 0);
    if (max <= 0) return 0;
    return (random() * max) | 0;
};

export const chance = (probability) => {
    if (!(probability > 0)) return false;
    if (probability >= 1) return true;
    return random() < probability;
};

export const pick = (arr) => {
    if (!arr || arr.length === 0) return undefined;
    return arr[randomInt(arr.length)];
};

export const shuffleInPlace = (arr) => {
    if (!arr || arr.length < 2) return arr;
    for (let i = arr.length - 1; i > 0; i--) {
        const j = randomInt(i + 1);
        const tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
    return arr;
};
