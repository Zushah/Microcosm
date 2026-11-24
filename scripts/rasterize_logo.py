from __future__ import annotations
from pathlib import Path
from PIL import Image, ImageDraw

ASSETS = Path(__file__).resolve().parent.parent / "assets"
OUT_LOGO = ASSETS / "logo.png"
OUT_FAVICON = ASSETS / "favicon.png"

SVG_VIEWBOX = 100.0
CENTER_VB = (50.0, 50.0)
GRADIENT_START_VB = (15.0, 85.0)
GRADIENT_END_VB = (85.0, 15.0)
GRADIENT_STOPS = ((0.0, "#A01928"), (0.5, "#DD2E2F"), (1.0, "#F17A1E"))
RING_RADIUS_VB = 46.4
RING_STROKE_WIDTH_VB = 3.2
HALF_CELLS_VB = (
    (43.2, 55.7, 9.9),
    (49.7, 28.6, 4.7),
    (38.1, 29.3, 3.4),
    (28.5, 34.8, 4.6),
    (22.1, 43.4, 3.2),
    (19.4, 53.0, 3.6),
    (19.6, 62.5, 2.3),
    (22.4, 71.3, 3.4),
    (27.1, 78.7, 2.6),
    (32.9, 84.5, 2.2),
    (21.9, 28.5, 0.9),
    (66.0, 23.5, 1.3),
    (12.6, 56.5, 1.5),
    (30.0, 67.4, 1.3),
    (59.0, 22.7, 1.0),
    (39.3, 87.8, 1.5),
    (62.8, 79.4, 0.7),
    (83.9, 71.7, 0.7),
    (30.0, 21.1, 0.5),
    (82.4, 53.1, 0.5),
    (14.4, 45.7, 0.6),
    (34.2, 24.2, 0.6),
    (43.4, 25.7, 0.6),
    (16.6, 74.1, 0.6),
    (89.8, 39.3, 0.5),
    (22.8, 21.3, 0.6),
    (40.5, 38.0, 0.6),
    (58.1, 26.7, 0.8),
    (56.9, 77.3, 0.8),
    (38.4, 71.4, 0.5)
)
GRADIENT_LUT_SIZE = 4096

def hex_to_rgb(color: str) -> tuple[int, int, int]:
    return (int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16))

def gradient_lut() -> list[bytes]:
    stops = [(offset, hex_to_rgb(color)) for offset, color in GRADIENT_STOPS]
    lut = []
    for i in range(GRADIENT_LUT_SIZE):
        t = i / (GRADIENT_LUT_SIZE - 1)
        for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
            if t <= t1:
                f = (t - t0) / (t1 - t0)
                lut.append(bytes(int(round(a + (b - a) * f)) for a, b in zip(c0, c1)))
                break
    return lut

def render_gradient(render_size: int, scale: float) -> Image.Image:
    x0, y0 = GRADIENT_START_VB
    x1, y1 = GRADIENT_END_VB
    dx, dy = x1 - x0, y1 - y0
    length_sq = dx * dx + dy * dy
    lut = gradient_lut()
    top = GRADIENT_LUT_SIZE - 1
    rows = []
    for py in range(render_size):
        y = (py + 0.5) / scale
        row_base = (y - y0) * dy
        row = []
        for px in range(render_size):
            x = (px + 0.5) / scale
            t = ((x - x0) * dx + row_base) / length_sq
            row.append(lut[min(top, max(0, int(round(t * top))))])
        rows.append(b"".join(row))
    return Image.frombytes("RGB", (render_size, render_size), b"".join(rows))

def circle_box(cx_vb: float, cy_vb: float, r_vb: float, scale: float) -> list[float]:
    # Pillow's ellipse box is in pixel-center coordinates, so each edge moves half a pixel inward.
    return [(cx_vb - r_vb) * scale + 0.5, (cy_vb - r_vb) * scale + 0.5, (cx_vb + r_vb) * scale - 0.5, (cy_vb + r_vb) * scale - 0.5]

def all_cells_vb() -> list[tuple[float, float, float]]:
    cx, cy = CENTER_VB
    rotated = [(2 * cx - x, 2 * cy - y, r) for x, y, r in HALF_CELLS_VB]
    return list(HALF_CELLS_VB) + rotated

def render_png(out_path: Path, size_px: int, supersample: int) -> None:
    render_size = size_px * supersample
    scale = render_size / SVG_VIEWBOX
    mask = Image.new("L", (render_size, render_size), 0)
    d = ImageDraw.Draw(mask)
    cx, cy = CENTER_VB
    d.ellipse(circle_box(cx, cy, RING_RADIUS_VB + RING_STROKE_WIDTH_VB / 2, scale), fill=255)
    d.ellipse(circle_box(cx, cy, RING_RADIUS_VB - RING_STROKE_WIDTH_VB / 2, scale), fill=0)
    for x, y, r in all_cells_vb():
        d.ellipse(circle_box(x, y, r, scale), fill=255)
    img = render_gradient(render_size, scale).convert("RGBA")
    img.putalpha(mask)
    if supersample != 1:
        img = img.resize((size_px, size_px), Image.Resampling.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, optimize=True)

def main() -> None:
    render_png(OUT_LOGO, size_px=1024, supersample=2)
    render_png(OUT_FAVICON, size_px=144, supersample=6)
    print("Wrote:")
    print(f" - {OUT_LOGO}")
    print(f" - {OUT_FAVICON}")

if __name__ == "__main__":
    main()
