const fetchText = async (url) => {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed to load shader: ${String(url)} (${res.status} ${res.statusText})`);
    return await res.text();
};

const VERT_SOURCE = await fetchText(new URL("./quad.vert.glsl", import.meta.url));
const FRAG_SOURCE = await fetchText(new URL("./quad.frag.glsl", import.meta.url));

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

const hslToRgb255 = (hDeg, s01, l01) => {
    const h = ((hDeg % 360) + 360) % 360;
    const s = clamp(s01, 0, 1);
    const l = clamp(l01, 0, 1);
    const c = (1 - Math.abs(2 * l - 1)) * s;
    const hp = h / 60;
    const x = c * (1 - Math.abs((hp % 2) - 1));
    let r1 = 0;
    let g1 = 0;
    let b1 = 0;
    if (hp >= 0 && hp < 1) {
        r1 = c; g1 = x; b1 = 0;
    } else if (hp >= 1 && hp < 2) {
        r1 = x; g1 = c; b1 = 0;
    } else if (hp >= 2 && hp < 3) {
        r1 = 0; g1 = c; b1 = x;
    } else if (hp >= 3 && hp < 4) {
        r1 = 0; g1 = x; b1 = c;
    } else if (hp >= 4 && hp < 5) {
        r1 = x; g1 = 0; b1 = c;
    } else {
        r1 = c; g1 = 0; b1 = x;
    }
    const m = l - c / 2;
    const r = Math.round((r1 + m) * 255);
    const g = Math.round((g1 + m) * 255);
    const b = Math.round((b1 + m) * 255);
    return [clamp(r, 0, 255), clamp(g, 0, 255), clamp(b, 0, 255)];
};

const compileShader = (gl, type, source) => {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, source);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        const log = gl.getShaderInfoLog(sh) || "(no shader log)";
        gl.deleteShader(sh);
        throw new Error(log);
    }
    return sh;
};

const createProgram = (gl, vertSrc, fragSrc) => {
    const vs = compileShader(gl, gl.VERTEX_SHADER, vertSrc);
    const fs = compileShader(gl, gl.FRAGMENT_SHADER, fragSrc);
    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        const log = gl.getProgramInfoLog(prog) || "(no program log)";
        gl.deleteProgram(prog);
        throw new Error(log);
    }
    return prog;
};

export class CanvasRenderer {
    constructor(canvas, world, options = {}) {
        this.canvas = canvas;
        this.gl = canvas.getContext("webgl2", {
            alpha: true,
            antialias: false,
            powerPreference: "high-performance"
        });
        if (!this.gl) throw new Error("WebGL2 is not available in this browser.");

        this.world = world;

        this.scale = options.initialScale || 4;
        this.offsetX = options.initialOffsetX || 0;
        this.offsetY = options.initialOffsetY || 0;

        this.isPanning = false;
        this.isRightPainting = false;
        this.lastMouse = null;
        this._panMoved = false;
        this._hoverTile = null;
        this._lastPaintTileKey = null;

        this.onCellClick = null;
        this.onCellRightClick = null;
        this.onEditBrushStroke = null;
        this.interactionMode = "explore";
        this.brushWidth = 10;
        this.brushHeight = 10;
        this.brushIntensity = 0;
        this._lineageRgbCache = new Map();
        this._dpr = 1;
        this._instanceStrideFloats = 8;
        this._instanceCapacity = 65536;
        this._instanceCount = 0;
        this._instanceData = new Float32Array(this._instanceCapacity * this._instanceStrideFloats);

        this._initWebGL();
        this.setupMouse();
        this.resize();
        this._applyIdleCursor();
        window.addEventListener("resize", () => this.resize());
    }

    _initWebGL() {
        const gl = this.gl;
        this._program = createProgram(gl, VERT_SOURCE, FRAG_SOURCE);
        this._uViewportPx = gl.getUniformLocation(this._program, "u_viewportPx");
        this._uDpr = gl.getUniformLocation(this._program, "u_dpr");
        this._uScale = gl.getUniformLocation(this._program, "u_scale");
        this._uOffsetPx = gl.getUniformLocation(this._program, "u_offsetPx");

        this._vao = gl.createVertexArray();
        gl.bindVertexArray(this._vao);

        this._vertexBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this._vertexBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
            0, 0,
            1, 0,
            0, 1,
            1, 1
        ]), gl.STATIC_DRAW);

        const aLocalPos = gl.getAttribLocation(this._program, "a_localPos");
        gl.enableVertexAttribArray(aLocalPos);
        gl.vertexAttribPointer(aLocalPos, 2, gl.FLOAT, false, 0, 0);
        gl.vertexAttribDivisor(aLocalPos, 0);

        this._instanceBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this._instanceBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, this._instanceData.byteLength, gl.DYNAMIC_DRAW);

        const strideBytes = this._instanceStrideFloats * 4;

        const aWorldPos = gl.getAttribLocation(this._program, "a_worldPos");
        gl.enableVertexAttribArray(aWorldPos);
        gl.vertexAttribPointer(aWorldPos, 2, gl.FLOAT, false, strideBytes, 0);
        gl.vertexAttribDivisor(aWorldPos, 1);

        const aWorldSize = gl.getAttribLocation(this._program, "a_worldSize");
        gl.enableVertexAttribArray(aWorldSize);
        gl.vertexAttribPointer(aWorldSize, 2, gl.FLOAT, false, strideBytes, 2 * 4);
        gl.vertexAttribDivisor(aWorldSize, 1);

        const aColor = gl.getAttribLocation(this._program, "a_color");
        gl.enableVertexAttribArray(aColor);
        gl.vertexAttribPointer(aColor, 4, gl.FLOAT, false, strideBytes, 4 * 4);
        gl.vertexAttribDivisor(aColor, 1);

        gl.bindVertexArray(null);
        gl.bindBuffer(gl.ARRAY_BUFFER, null);

        gl.disable(gl.DEPTH_TEST);
        gl.disable(gl.CULL_FACE);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

        gl.clearColor(250 / 255, 250 / 255, 250 / 255, 1);
    }

    setInteractionMode(mode) {
        this.interactionMode = mode === "edit" ? "edit" : "explore";
        if (this.interactionMode !== "edit") {
            this.isRightPainting = false;
            this._hoverTile = null;
            this._lastPaintTileKey = null;
        }
        this._applyIdleCursor();
    }

    setEditBrush(width, height, intensity = 0) {
        this.brushWidth = Math.max(1, Math.min(this.world.width, Math.round(Number(width) || 1)));
        this.brushHeight = Math.max(1, Math.min(this.world.height, Math.round(Number(height) || 1)));
        this.brushIntensity = Number.isFinite(intensity) ? intensity : 0;
    }

    _applyIdleCursor() {
        if (this.isPanning) {
            this.canvas.style.cursor = "grabbing";
            return;
        }
        this.canvas.style.cursor = this.interactionMode === "edit" ? "crosshair" : "grab";
    }

    resize() {
        const dpr = Math.min(2, window.devicePixelRatio || 1);
        this._dpr = dpr;
        this.canvas.width = Math.floor(window.innerWidth * dpr);
        this.canvas.height = Math.floor(window.innerHeight * dpr);
        this.canvas.style.width = `${window.innerWidth}px`;
        this.canvas.style.height = `${window.innerHeight}px`;
        this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    }

    _canvasPointFromEvent(e) {
        const rect = this.canvas.getBoundingClientRect();
        return {
            rect,
            cx: e.clientX - rect.left,
            cy: e.clientY - rect.top
        };
    }

    _isEventInsideCanvas(e) {
        const rect = this.canvas.getBoundingClientRect();
        return e.clientX >= rect.left && e.clientX < rect.right && e.clientY >= rect.top && e.clientY < rect.bottom;
    }

    _tileHitFromCanvasPoint(cx, cy) {
        const gx = Math.floor((cx - this.offsetX) / this.scale);
        const gy = Math.floor((cy - this.offsetY) / this.scale);

        const x = ((gx % this.world.width) + this.world.width) % this.world.width;
        const y = ((gy % this.world.height) + this.world.height) % this.world.height;
        const tile = this.world.grid[x][y];
        const topCell = tile.cells.length > 0 ? tile.cells[tile.cells.length - 1] : null;

        return { x, y, tile, topCell };
    }

    _tileHitFromEvent(e) {
        const { cx, cy } = this._canvasPointFromEvent(e);
        return this._tileHitFromCanvasPoint(cx, cy);
    }

    _paintEditStrokeFromHit(hit) {
        if (!hit || this.interactionMode !== "edit" || !this.onEditBrushStroke) return;
        const key = `${hit.x},${hit.y}`;
        if (key === this._lastPaintTileKey) return;
        this._lastPaintTileKey = key;
        this.onEditBrushStroke(hit.x, hit.y, hit.tile, hit.topCell);
    }

    setupMouse() {
        this.canvas.addEventListener("mousedown", (e) => {
            if (e.button === 0) {
                this.isPanning = true;
                this._panMoved = false;
                this.lastMouse = { x: e.clientX, y: e.clientY };
                this._applyIdleCursor();
                return;
            }

            if (e.button === 2 && this.interactionMode === "edit") {
                e.preventDefault();
                this.isRightPainting = true;
                this._lastPaintTileKey = null;
                const hit = this._tileHitFromEvent(e);
                this._hoverTile = hit ? { x: hit.x, y: hit.y } : null;
                this._paintEditStrokeFromHit(hit);
            }
        });

        window.addEventListener("mouseup", (e) => {
            if (e.button === 0) {
                this.isPanning = false;
                this.lastMouse = null;
                this._applyIdleCursor();
            }
            if (e.button === 2) {
                this.isRightPainting = false;
                this._lastPaintTileKey = null;
                this._applyIdleCursor();
            }
        });

        window.addEventListener("mousemove", (e) => {
            if (this.isPanning && this.lastMouse) {
                const dx = e.clientX - this.lastMouse.x;
                const dy = e.clientY - this.lastMouse.y;
                if (dx !== 0 || dy !== 0) this._panMoved = true;
                this.offsetX += dx;
                this.offsetY += dy;
                this.lastMouse = { x: e.clientX, y: e.clientY };
            }

            if (this.isPanning) return;

            const inside = this._isEventInsideCanvas(e);
            if (!inside) {
                if (this.interactionMode === "edit" && !this.isRightPainting) this._hoverTile = null;
                return;
            }

            const hit = this._tileHitFromEvent(e);
            if (this.interactionMode === "edit") {
                this._hoverTile = hit ? { x: hit.x, y: hit.y } : null;
                if (this.isRightPainting) this._paintEditStrokeFromHit(hit);
            }
        });

        this.canvas.addEventListener("mouseleave", () => {
            if (this.interactionMode === "edit" && !this.isRightPainting) this._hoverTile = null;
        });

        this.canvas.addEventListener("wheel", (e) => {
            e.preventDefault();
            const { cx, cy } = this._canvasPointFromEvent(e);

            const worldX = (cx - this.offsetX) / this.scale;
            const worldY = (cy - this.offsetY) / this.scale;

            const zoom = e.deltaY < 0 ? 1.125 : 0.8888889;
            const newScale = Math.max(1, Math.min(40, this.scale * zoom));

            this.offsetX = cx - worldX * newScale;
            this.offsetY = cy - worldY * newScale;
            this.scale = newScale;
        }, { passive: false });

        this.canvas.addEventListener("click", (e) => {
            if (this._panMoved) {
                this._panMoved = false;
                return;
            }
            const hit = this._tileHitFromEvent(e);
            if (this.onCellClick) this.onCellClick(hit.x, hit.y, hit.tile, hit.topCell);
        });

        this.canvas.addEventListener("contextmenu", (e) => {
            e.preventDefault();
            if (this.interactionMode === "edit") return;
            const hit = this._tileHitFromEvent(e);
            if (this.onCellRightClick) this.onCellRightClick(hit.x, hit.y, hit.tile, hit.topCell);
        });
    }

    _ensureInstanceCapacity(nextCount) {
        if (nextCount <= this._instanceCapacity) return;
        const gl = this.gl;
        let newCap = this._instanceCapacity;
        while (newCap < nextCount) newCap *= 2;
        const next = new Float32Array(newCap * this._instanceStrideFloats);
        next.set(this._instanceData);
        this._instanceData = next;
        this._instanceCapacity = newCap;
        gl.bindBuffer(gl.ARRAY_BUFFER, this._instanceBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, this._instanceData.byteLength, gl.DYNAMIC_DRAW);
        gl.bindBuffer(gl.ARRAY_BUFFER, null);
    }

    _pushRect(worldX, worldY, worldW, worldH, r01, g01, b01, a01) {
        const next = this._instanceCount + 1;
        if (next > this._instanceCapacity) this._ensureInstanceCapacity(next);
        const base = this._instanceCount * this._instanceStrideFloats;
        this._instanceData[base + 0] = worldX;
        this._instanceData[base + 1] = worldY;
        this._instanceData[base + 2] = worldW;
        this._instanceData[base + 3] = worldH;
        this._instanceData[base + 4] = r01;
        this._instanceData[base + 5] = g01;
        this._instanceData[base + 6] = b01;
        this._instanceData[base + 7] = a01;
        this._instanceCount++;
    }

    _pushStrokeRect(x, y, w, h, lineWidth, r01, g01, b01, a01) {
        const lw = lineWidth;
        const half = lw * 0.5;
        const xOut = x - half;
        const yOut = y - half;
        const wOut = w + lw;
        const hOut = h + lw;
        const xIn = x + half;
        const yIn = y + half;
        const wIn = Math.max(0, w - lw);
        const hIn = Math.max(0, h - lw);
        this._pushRect(xOut, yOut, lw, lw, r01, g01, b01, a01);
        this._pushRect(xOut + wOut - lw, yOut, lw, lw, r01, g01, b01, a01);
        this._pushRect(xOut, yOut + hOut - lw, lw, lw, r01, g01, b01, a01);
        this._pushRect(xOut + wOut - lw, yOut + hOut - lw, lw, lw, r01, g01, b01, a01);
        if (wIn > 0) {
            this._pushRect(xIn, yOut, wIn, lw, r01, g01, b01, a01);
            this._pushRect(xIn, yOut + hOut - lw, wIn, lw, r01, g01, b01, a01);
        }
        if (hIn > 0) {
            this._pushRect(xOut, yIn, lw, hIn, r01, g01, b01, a01);
            this._pushRect(xOut + wOut - lw, yIn, lw, hIn, r01, g01, b01, a01);
        }
    }

    _tileRgb01(tile) {
        const v = (typeof tile.enval === "number") ? tile.enval : 0;
        const mapped = Math.atan(v * 1.25) / (Math.PI * 0.5);
        const mag = Math.min(1, Math.abs(mapped));

        let r = 246;
        let g = 246;
        let b = 246;

        if (mapped >= 0) {
            r = Math.floor(246 - 10 * mag);
            g = Math.floor(246 - 92 * mag);
            b = Math.floor(246 - 205 * mag);
        } else {
            r = Math.floor(246 - 205 * mag);
            g = Math.floor(246 - 118 * mag);
            b = Math.floor(246 - 10 * mag);
        }

        return [clamp(r, 0, 255) / 255, clamp(g, 0, 255) / 255, clamp(b, 0, 255) / 255];
    }

    _lineageRgb01(lineageId) {
        if (this._lineageRgbCache.has(lineageId)) return this._lineageRgbCache.get(lineageId);

        let x = (lineageId | 0) ^ 0x9e3779b9;
        x = Math.imul(x ^ (x >>> 16), 0x85ebca6b);
        x = Math.imul(x ^ (x >>> 13), 0xc2b2ae35);
        x = (x ^ (x >>> 16)) >>> 0;

        const hue = x % 360;
        const sat = 0.70;
        const light = 0.55;

        const [r255, g255, b255] = hslToRgb255(hue, sat, light);
        const out = [r255 / 255, g255 / 255, b255 / 255];

        this._lineageRgbCache.set(lineageId, out);
        return out;
    }

    _wrappedSegments(start, span, worldSize) {
        if (span >= worldSize) return [{ start: 0, size: worldSize }];

        const normalizedStart = ((start % worldSize) + worldSize) % worldSize;
        const end = normalizedStart + span;
        if (end <= worldSize) return [{ start: normalizedStart, size: span }];

        return [
            { start: normalizedStart, size: worldSize - normalizedStart },
            { start: 0, size: end - worldSize }
        ];
    }

    _brushPreviewColor01() {
        if (this.brushIntensity > 0) return [0.95, 0.56, 0.12];
        if (this.brushIntensity < 0) return [0.18, 0.43, 0.92];
        return [0.35, 0.35, 0.35];
    }

    _pushBrushPreview(centerX, centerY) {
        const brushWidth = Math.max(1, Math.min(this.world.width, this.brushWidth | 0));
        const brushHeight = Math.max(1, Math.min(this.world.height, this.brushHeight | 0));
        const startX = centerX - Math.floor(brushWidth * 0.5);
        const startY = centerY - Math.floor(brushHeight * 0.5);
        const xSegments = this._wrappedSegments(startX, brushWidth, this.world.width);
        const ySegments = this._wrappedSegments(startY, brushHeight, this.world.height);
        const [r, g, b] = this._brushPreviewColor01();

        for (let xi = 0; xi < xSegments.length; xi++) {
            const xSeg = xSegments[xi];
            for (let yi = 0; yi < ySegments.length; yi++) {
                const ySeg = ySegments[yi];
                this._pushRect(xSeg.start, ySeg.start, xSeg.size, ySeg.size, r, g, b, 0.12);
                this._pushStrokeRect(xSeg.start, ySeg.start, xSeg.size, ySeg.size, 0.12, r, g, b, 0.92);
                this._pushStrokeRect(xSeg.start, ySeg.start, xSeg.size, ySeg.size, 0.04, 1, 1, 1, 0.55);
            }
        }
    }

    render() {
        const gl = this.gl;
        const wCss = this.canvas.width / this._dpr;
        const hCss = this.canvas.height / this._dpr;
        this._instanceCount = 0;

        const minX = Math.max(0, Math.floor(-this.offsetX / this.scale));
        const minY = Math.max(0, Math.floor(-this.offsetY / this.scale));
        const maxX = Math.min(this.world.width - 1, Math.ceil((wCss - this.offsetX) / this.scale));
        const maxY = Math.min(this.world.height - 1, Math.ceil((hCss - this.offsetY) / this.scale));

        const tilesVisible = Math.max(0, maxX - minX + 1) * Math.max(0, maxY - minY + 1);
        this._ensureInstanceCapacity(Math.max(this._instanceCapacity, tilesVisible * 2));

        for (let x = minX; x <= maxX; x++) {
            for (let y = minY; y <= maxY; y++) {
                const tile = this.world.grid[x][y];
                const [r, g, b] = this._tileRgb01(tile);
                this._pushRect(x, y, 1, 1, r, g, b, 1);
            }
        }

        for (let x = minX; x <= maxX; x++) {
            for (let y = minY; y <= maxY; y++) {
                const tile = this.world.grid[x][y];
                if (tile.cells.length > 0) {
                    for (let i = 0; i < tile.cells.length; i++) {
                        const cell = tile.cells[i];
                        const [cr, cg, cb] = this._lineageRgb01(cell.lineageId || 0);
                        if (i === tile.cells.length - 1) {
                            this._pushRect(x, y, 1, 1, cr, cg, cb, 1);
                        } else {
                            this._pushRect(x + 0.15 * i, y + 0.15 * i, 0.7, 0.7, cr, cg, cb, 1);
                        }
                        if (cell._highlight) {
                            this._pushStrokeRect(
                                x + 0.02, y + 0.02, 0.96, 0.96,
                                0.16,
                                255 / 255, 240 / 255, 140 / 255, 0.95
                            );
                            this._pushStrokeRect(
                                x + 0.08, y + 0.08, 0.84, 0.84,
                                0.06,
                                0, 0, 0, 0.35
                            );
                        }
                    }
                }
            }
        }

        if (this.interactionMode === "edit" && this._hoverTile) {
            this._pushBrushPreview(this._hoverTile.x, this._hoverTile.y);
        }

        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.useProgram(this._program);
        gl.bindVertexArray(this._vao);
        gl.uniform2f(this._uViewportPx, this.canvas.width, this.canvas.height);
        gl.uniform1f(this._uDpr, this._dpr);
        gl.uniform1f(this._uScale, this.scale);
        gl.uniform2f(this._uOffsetPx, this.offsetX, this.offsetY);
        gl.bindBuffer(gl.ARRAY_BUFFER, this._instanceBuffer);
        gl.bufferSubData(
            gl.ARRAY_BUFFER,
            0,
            this._instanceData.subarray(0, this._instanceCount * this._instanceStrideFloats)
        );
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this._instanceCount);
        gl.bindVertexArray(null);
    }
}
