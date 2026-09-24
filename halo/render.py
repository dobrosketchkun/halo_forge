"""SVG output (primary) and PNG preview (PIL rasterizer, no cairo needed)."""
from __future__ import annotations

import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

BG = (30, 30, 36)


def polygons(g: BaseGeometry) -> list[Polygon]:
    if g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    if isinstance(g, MultiPolygon):
        return list(g.geoms)
    return [p for p in getattr(g, "geoms", []) if isinstance(p, Polygon)]


def extent(g: BaseGeometry, margin=0.12) -> float:
    if g.is_empty:
        return 1.0
    x0, y0, x1, y1 = g.bounds
    return max(abs(x0), abs(x1), abs(y0), abs(y1)) * (1 + margin)


def to_svg(g: BaseGeometry, color="#7fe7ff", glow=0.035, E=None, size=512) -> str:
    E = E or extent(g)
    d = []
    for p in polygons(g):
        for ring in [p.exterior, *p.interiors]:
            c = np.asarray(ring.coords)
            d.append("M" + " L".join(f"{x:.4f},{-y:.4f}" for x, y in c) + " Z")
    path = " ".join(d)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="{-E} {-E} {2*E} {2*E}" width="{size}" height="{size}">
  <defs><filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
    <feGaussianBlur in="SourceGraphic" stdDeviation="{glow:.4f}" result="b"/>
    <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter></defs>
  <path d="{path}" fill="{color}" fill-rule="evenodd" filter="url(#glow)"/>
</svg>
"""


def hex_rgb(c: str):
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def mask(g: BaseGeometry, E: float, px: int, ss: int = 4):
    from PIL import Image, ImageDraw   # Pillow only needed for PNG output
    S = px * ss
    im = Image.new("L", (S, S), 0)
    dr = ImageDraw.Draw(im)

    def tr(coords):
        c = np.asarray(coords)
        return [((x + E) / (2 * E) * S, (E - y) / (2 * E) * S) for x, y in c]

    # outer-most first: a polygon's hole must never erase a smaller polygon drawn earlier
    for p in sorted(polygons(g), key=lambda p: -Polygon(p.exterior).area):
        dr.polygon(tr(p.exterior.coords), fill=255)
        for h in p.interiors:
            dr.polygon(tr(h.coords), fill=0)
    return im.resize((px, px), Image.LANCZOS)


def to_png(g: BaseGeometry, path: str | None = None, color="#7fe7ff", glow=0.035, E=None, px=512,
           bg=BG):
    from PIL import Image, ImageFilter
    E = E or extent(g)
    m = mask(g, E, px)
    col = hex_rgb(color)
    out = Image.new("RGB", (px, px), bg)
    blur_px = max(glow / (2 * E) * px, 0.5)
    halo = m.filter(ImageFilter.GaussianBlur(blur_px * 1.6))
    out.paste(Image.new("RGB", (px, px), col), (0, 0), halo.point(lambda v: int(v * 0.85)))
    # core slightly brightened toward white like the in-game emissive look
    core = tuple(int(c + (255 - c) * 0.25) for c in col)
    out.paste(Image.new("RGB", (px, px), core), (0, 0), m)
    if path:
        out.save(path)
    return out
