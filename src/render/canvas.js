export class CanvasRenderer {
    constructor(canvas, world, options = {}) {
        this.canvas = canvas;
        this.ctx = canvas.getContext("2d");
        this.world = world;

        this.scale = options.initialScale || 4;
        this.offsetX = options.initialOffsetX || 0;
        this.offsetY = options.initialOffsetY || 0;

        this.isDragging = false;
        this.lastMouse = null;

        this.onCellClick = null;
        this.onCellRightClick = null;
        this.lineageColorCache = new Map();

        this.setupMouse();
        this.resize();
        window.addEventListener("resize", () => this.resize());
    }

    resize() {
        const dpr = Math.min(2, window.devicePixelRatio || 1);
        this.canvas.width = Math.floor(window.innerWidth * dpr);
        this.canvas.height = Math.floor(window.innerHeight * dpr);
        this.canvas.style.width = `${window.innerWidth}px`;
        this.canvas.style.height = `${window.innerHeight}px`;
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    setupMouse() {
        this.canvas.addEventListener("mousedown", (e) => {
            this.isDragging = true;
            this.canvas.style.cursor = "grabbing";
            this.lastMouse = { x: e.clientX, y: e.clientY };
        });

        window.addEventListener("mouseup", () => {
            this.isDragging = false;
            this.canvas.style.cursor = "grab";
            this.lastMouse = null;
        });

        window.addEventListener("mousemove", (e) => {
            if (!this.isDragging) return;
            const dx = e.clientX - this.lastMouse.x;
            const dy = e.clientY - this.lastMouse.y;
            this.offsetX += dx;
            this.offsetY += dy;
            this.lastMouse = { x: e.clientX, y: e.clientY };
        });

        this.canvas.addEventListener("wheel", (e) => {
            e.preventDefault();
            const rect = this.canvas.getBoundingClientRect();
            const cx = e.clientX - rect.left;
            const cy = e.clientY - rect.top;

            const worldX = (cx - this.offsetX) / this.scale;
            const worldY = (cy - this.offsetY) / this.scale;

            const zoom = e.deltaY < 0 ? 1.125 : 0.8888889;
            const newScale = Math.max(1, Math.min(40, this.scale * zoom));

            this.offsetX = cx - worldX * newScale;
            this.offsetY = cy - worldY * newScale;
            this.scale = newScale;
        }, { passive: false });

        this.canvas.addEventListener("click", (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const cx = e.clientX - rect.left;
            const cy = e.clientY - rect.top;

            const gx = Math.floor((cx - this.offsetX) / this.scale);
            const gy = Math.floor((cy - this.offsetY) / this.scale);

            const x = ((gx % this.world.width) + this.world.width) % this.world.width;
            const y = ((gy % this.world.height) + this.world.height) % this.world.height;

            const tile = this.world.grid[x][y];
            const topCell = tile.cells.length > 0 ? tile.cells[tile.cells.length - 1] : null;

            if (this.onCellClick) this.onCellClick(x, y, tile, topCell);
        });

        this.canvas.addEventListener("contextmenu", (e) => {
            e.preventDefault();
            const rect = this.canvas.getBoundingClientRect();
            const cx = e.clientX - rect.left;
            const cy = e.clientY - rect.top;

            const gx = Math.floor((cx - this.offsetX) / this.scale);
            const gy = Math.floor((cy - this.offsetY) / this.scale);

            const x = ((gx % this.world.width) + this.world.width) % this.world.width;
            const y = ((gy % this.world.height) + this.world.height) % this.world.height;

            const tile = this.world.grid[x][y];
            const topCell = tile.cells.length > 0 ? tile.cells[tile.cells.length - 1] : null;

            if (this.onCellRightClick) this.onCellRightClick(x, y, tile, topCell);
        });
    }

    elementColor(sym) {
        const map = {
            A: [220, 240, 255],
            B: [200, 220, 160],
            C: [255, 200, 160],
            D: [250, 220, 250],
            E: [255, 235, 200],
            X: [200, 200, 200]
        };
        return map[sym] || [210, 210, 210];
    }

    render() {
        const ctx = this.ctx;
        const w = this.canvas.width / (window.devicePixelRatio || 1);
        const h = this.canvas.height / (window.devicePixelRatio || 1);

        ctx.fillStyle = "#fafafa";
        ctx.fillRect(0, 0, w, h);

        ctx.save();
        ctx.translate(this.offsetX, this.offsetY);
        ctx.scale(this.scale, this.scale);

        const minX = Math.max(0, Math.floor(-this.offsetX / this.scale));
        const minY = Math.max(0, Math.floor(-this.offsetY / this.scale));
        const maxX = Math.min(this.world.width - 1, Math.ceil((w - this.offsetX) / this.scale));
        const maxY = Math.min(this.world.height - 1, Math.ceil((h - this.offsetY) / this.scale));

        for (let x = minX; x <= maxX; x++) {
            for (let y = minY; y <= maxY; y++) {
                this.drawTile(x, y, this.world.grid[x][y]);
            }
        }

        for (let x = minX; x <= maxX; x++) {
            for (let y = minY; y <= maxY; y++) {
                const tile = this.world.grid[x][y];
                if (tile.cells.length > 0) {
                    for (let i = 0; i < tile.cells.length; i++) {
                        const cell = tile.cells[i];
                        const color = this.colorForCell(cell);
                        ctx.fillStyle = color;
                        if (i === tile.cells.length - 1) {
                            ctx.fillRect(x, y, 1, 1);
                        } else {
                            ctx.fillStyle = this.withAlpha(color, 0.6);
                            ctx.fillRect(x + 0.15 * i, y + 0.15 * i, 0.7, 0.7);
                        }
                        if (cell._highlight) {
                            ctx.strokeStyle = "rgba(255, 240, 140, 0.95)";
                            ctx.lineWidth = 0.16;
                            ctx.strokeRect(x + 0.02, y + 0.02, 0.96, 0.96);
                            ctx.strokeStyle = "rgba(0, 0, 0, 0.35)";
                            ctx.lineWidth = 0.06;
                            ctx.strokeRect(x + 0.08, y + 0.08, 0.84, 0.84);
                        }
                    }
                }
            }
        }

        ctx.restore();
    }

    drawTile(x, y, tile) {
        const ctx = this.ctx;

        const t = Math.max(0, Math.min(1, tile.temperature));
        const s = Math.max(0, Math.min(1, tile.solute));

        const cool = Math.max(0, 0.5 - t);
        const warm = Math.max(0, t - 0.1);

        const soluteDark = Math.min(0.4, s * 0.6);

        const r = Math.floor(250 - 1 * soluteDark + 100 * warm);
        const g = Math.floor(250 - 20 * soluteDark - 10 * warm + 20 * cool);
        const b = Math.floor(250 - 15 * soluteDark + 50 * cool);

        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(x, y, 1, 1);
    }

    lineageColor(lineageId) {
        if (this.lineageColorCache.has(lineageId)) return this.lineageColorCache.get(lineageId);
        
        let x = (lineageId | 0) ^ 0x9e3779b9;
        x = Math.imul(x ^ (x >>> 16), 0x85ebca6b);
        x = Math.imul(x ^ (x >>> 13), 0xc2b2ae35);
        x = (x ^ (x >>> 16)) >>> 0;

        const hue = x % 360;
        const sat = 70;
        const light = 55;

        const col = `hsl(${hue}, ${sat}%, ${light}%)`;
        this.lineageColorCache.set(lineageId, col);
        return col;
    }

    colorForCell(cell) {
        return this.lineageColor(cell.lineageId || 0);
    }

    withAlpha(rgbString, alpha) {
        const m = rgbString.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
        if (!m) return rgbString;
        return `rgba(${m[1]},${m[2]},${m[3]},${alpha})`;
    }
}
