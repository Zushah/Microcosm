export class CanvasRenderer {
    constructor(canvas, world) {
        this.canvas = canvas;
        this.ctx = canvas.getContext("2d");
        this.world = world;

        this.scale = 4;
        this.offsetX = 0;
        this.offsetY = 0;

        this.isDragging = false;
        this.lastMouse = null;

        this.setupMouse();
        this.resize();
        window.addEventListener("resize", () => this.resize());
    }

    resize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }

    setupMouse() {
        this.canvas.addEventListener("mousedown", e => {
            this.isDragging = true;
            this.lastMouse = { x: e.clientX, y: e.clientY };
        });

        window.addEventListener("mouseup", () => {
            this.isDragging = false;
        });

        window.addEventListener("mousemove", e => {
            if (!this.isDragging) return;
            const dx = e.clientX - this.lastMouse.x;
            const dy = e.clientY - this.lastMouse.y;
            this.offsetX += dx;
            this.offsetY += dy;
            this.lastMouse = { x: e.clientX, y: e.clientY };
        });

        this.canvas.addEventListener("wheel", e => {
            e.preventDefault();
            const zoom = e.deltaY < 0 ? 1.1 : 0.9;
            this.scale *= zoom;
            this.scale = Math.max(1, Math.min(20, this.scale));
        });
    }

    render() {
        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        ctx.save();
        ctx.translate(this.offsetX, this.offsetY);
        ctx.scale(this.scale, this.scale);

        for (let x = 0; x < this.world.width; x++) {
            for (let y = 0; y < this.world.height; y++) {
                this.drawTile(x, y, this.world.grid[x][y]);
            }
        }

        ctx.restore();
    }

    drawTile(x, y, tile) {
        const ctx = this.ctx;
        const moleculeDensity = tile.molecules.length / 5;
        const temp = tile.temperature;
        ctx.fillStyle = `rgb(
            ${Math.floor(50 + 200 * moleculeDensity)},
            ${Math.floor(50 + 150 * temp)},
            80
        )`;
        ctx.fillRect(x, y, 1, 1);
        if (tile.cells.length > 0) {
            ctx.fillStyle = "white";
            ctx.fillRect(x, y, 1, 1);
        }
    }
}
