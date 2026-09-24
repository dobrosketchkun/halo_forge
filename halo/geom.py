"""Shape primitives for the halo grammar.

Conventions (see research/GRAMMAR.md):
- Model space is y-up, unit R = main frame radius.
- Angles: 0 deg = up (12 o'clock), clockwise positive.
- Rooted motifs have their base at the origin and their tip along +y.
  Centered motifs (dot, sparkle, star, diamond, cross, heart, ring, tomoe) are centered on the origin.

Every primitive returns a Shape: a shapely geometry plus a feature table
(name -> list of (x, y, angle_deg)) that other nodes can attach to.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

EMPTY = Polygon()


# ---------------------------------------------------------------- helpers

def ang_vec(deg: float) -> tuple[float, float]:
    r = math.radians(deg)
    return math.sin(r), math.cos(r)


def vec_ang(dx: float, dy: float) -> float:
    return math.degrees(math.atan2(dx, dy)) % 360.0


def polar(r: float, deg: float) -> tuple[float, float]:
    s, c = ang_vec(deg)
    return r * s, r * c


def quad(p0, p1, p2, n=24):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2 = map(np.asarray, (p0, p1, p2))
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def cubic(p0, p1, p2, p3, n=32):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = map(np.asarray, (p0, p1, p2, p3))
    return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)


def poly(pts) -> BaseGeometry:
    g = Polygon(np.asarray(pts))
    return g if g.is_valid else g.buffer(0)


def mirror_x(pts):
    pts = np.asarray(pts)
    return np.column_stack([-pts[:, 0], pts[:, 1]])


def annulus(r: float, w: float, aspect: float = 1.0) -> BaseGeometry:
    g = Point(0, 0).buffer(r + w / 2, 128).difference(Point(0, 0).buffer(max(r - w / 2, 0), 128))
    return affinity.scale(g, 1.0, aspect, origin=(0, 0)) if aspect != 1.0 else g


def wedge(r: float, a0: float, a1: float, n: int = 48) -> BaseGeometry:
    """Pie slice from angle a0 to a1 (clockwise, degrees) of radius r."""
    angs = np.linspace(a0, a1, n)
    pts = [(0, 0)] + [polar(r, a) for a in angs]
    return poly(pts)


def tapered_stroke(center: np.ndarray, widths: np.ndarray, round_caps: bool = False) -> BaseGeometry:
    """Polygon around a sampled centerline with per-sample widths."""
    c = np.asarray(center, float)
    d = np.gradient(c, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
    nrm = np.column_stack([-d[:, 1], d[:, 0]])
    w = np.asarray(widths, float)[:, None] / 2
    left, right = c + nrm * w, c - nrm * w
    g = poly(np.vstack([left, right[::-1]]))
    if round_caps:
        g = unary_union([g, Point(*c[0]).buffer(w[0, 0], 32), Point(*c[-1]).buffer(w[-1, 0], 32)])
    return g


def profile_widths(profile, weight: float, t: np.ndarray) -> np.ndarray:
    """profile: 'uniform' | 'taper' | 'taper_one' | 'belly' | {'points': [[t, w], ...]} (w as multiple of weight)."""
    if profile in (None, "uniform"):
        return np.full_like(t, weight)
    if profile in ("taper", "taper_ends"):
        return weight * np.sin(np.pi * t) ** 0.6 + 1e-4
    if profile == "taper_one":
        return weight * (1 - t) ** 0.8 + 1e-4
    if profile == "belly":
        return weight * (0.25 + 0.75 * np.sin(np.pi * t))
    if isinstance(profile, dict) and "points" in profile:
        pts = np.asarray(profile["points"], float)
        return weight * np.interp(t, pts[:, 0], pts[:, 1]) + 1e-4
    raise ValueError(f"unknown profile {profile!r}")


# ---------------------------------------------------------------- Shape

@dataclass
class Shape:
    geom: BaseGeometry
    feats: dict[str, list[tuple[float, float, float]]] = field(default_factory=dict)
    point_r: float | None = None   # for circle-like shapes: feature point(deg) at this radius

    def feature(self, name: str, args: list[str]):
        """Resolve e.g. vertex(*), vertex(2), tip, point(45), gap_end(0,start)."""
        if name == "point":
            r = self.point_r if self.point_r is not None else 1.0
            a = float(args[0])
            x, y = polar(r, a)
            return [(x, y, a)]
        if name == "center":
            return self.feats.get("center", [(0.0, 0.0, 0.0)])
        if name == "gap_end":
            lst = self.feats.get("gap_end", [])
            if not args or args[0] == "*":
                return lst
            i = int(args[0])
            side = args[1] if len(args) > 1 else "start"
            j = 2 * i + (0 if side == "start" else 1)
            return [lst[j]] if j < len(lst) else []
        lst = self.feats.get(name)
        if lst is None:
            raise KeyError(f"shape has no feature {name!r} (has {sorted(self.feats)})")
        if not args or args[0] == "*":
            return lst
        return [lst[int(args[0]) % len(lst)]]


def _size(p, default_len, default_w):
    s = p.get("size", {}) or {}
    return float(s.get("len", default_len)), float(s.get("w", default_w)), s


# ---------------------------------------------------------------- frames

def partition_profile(kind: str, t: np.ndarray) -> np.ndarray:
    """Heraldic lines of partition as a periodic profile over t in [0, 1) per period, output in [-1, 1].
    +1 = pushed outward."""
    t = t % 1.0
    if kind == "wavy":
        return np.sin(2 * np.pi * t)
    if kind == "indented":          # small sharp teeth
        return 1 - 4 * np.abs(t - 0.5)
    if kind == "engrailed":         # scallops whose points face outward
        return 2 * np.sqrt(np.clip(1 - (2 * t - 1) ** 2, 0, 1)) * -1 + 1
    if kind == "invected":          # scallops bulging outward
        return 2 * np.sqrt(np.clip(1 - (2 * t - 1) ** 2, 0, 1)) - 1
    if kind == "embattled":         # square battlements
        return np.where(t < 0.5, 1.0, -1.0)
    if kind == "dovetailed":        # trapezoids wider at the top
        return np.interp(t, [0, 0.08, 0.42, 0.5, 0.58, 0.92, 1.0], [-1, 1, 1, -1, -1, -1, -1]) * 0 + \
            np.where((t > 0.04) & (t < 0.46), 1.0, -1.0)
    if kind == "nebuly":            # cloud-like bulbs
        return np.sin(2 * np.pi * t) + 0.45 * np.sin(4 * np.pi * t + np.pi / 2)
    if kind == "raguly":            # slanted battlements
        return np.interp(t, [0, 0.15, 0.5, 0.65, 1.0], [-1, 1, 1, -1, -1])
    if kind == "rayonny":           # flame-like wavy rays
        return np.sign(np.sin(2 * np.pi * t)) * np.abs(np.sin(2 * np.pi * t)) ** 0.5
    raise ValueError(kind)


def profiled_annulus(r, w, edge):
    """Annulus whose outer (and/or inner) boundary follows a heraldic line of partition."""
    kind, n = edge["type"], int(edge.get("n", 16))
    amp = float(edge.get("amp", w * 0.45))
    side = edge.get("side", "outer")
    th = np.linspace(0, 2 * np.pi, 1600, endpoint=False)
    prof = partition_profile(kind, th / (2 * np.pi) * n)
    outer = r + w / 2 + (amp * prof if side in ("outer", "both") else 0)
    inner = r - w / 2 + (amp * prof if side in ("inner", "both") else 0)
    o = Polygon(np.column_stack([outer * np.sin(th), outer * np.cos(th)])).buffer(0)
    i = Polygon(np.column_stack([inner * np.sin(th), inner * np.cos(th)])).buffer(0)
    return o.difference(i)


def circle(p) -> Shape:
    r = float(p.get("r", 1.0))
    w = float(p.get("weight", 0.08))
    g = annulus(r, w, float(p.get("aspect", 1.0)))
    if isinstance(p.get("edge"), dict):
        g = profiled_annulus(r, w, p["edge"])
    ends = []
    for gap in p.get("gaps", []) or []:
        at, width = float(gap["at"]), float(gap["width"])
        a0, a1 = at - width / 2, at + width / 2
        g = g.difference(wedge(r + w + 0.5, a0, a1))
        ends += [(*polar(r, a0), (a0 + 90) % 360), (*polar(r, a1), (a1 - 90) % 360)]
    if p.get("solid"):
        g = Point(0, 0).buffer(r + w / 2, 128)
    return Shape(g, {"gap_end": ends, "center": [(0, 0, 0)]}, point_r=r)


def polygon(p) -> Shape:
    n = int(p.get("n", 6))
    r = float(p.get("r", 1.0))
    radii = p.get("radii") or [1.0] * n
    bulge = float(p.get("bulge", 0.0))  # + convex, - concave (fraction of edge length)
    verts = [polar(r * radii[i % len(radii)], i * 360.0 / n) for i in range(n)]
    pts = []
    for i in range(n):
        a, b = np.array(verts[i]), np.array(verts[(i + 1) % n])
        mid = (a + b) / 2
        out = mid / (np.linalg.norm(mid) + 1e-12)
        ctrl = mid + out * bulge * np.linalg.norm(b - a) * 2
        pts.extend(quad(a, ctrl, b, 24)[:-1])
    outline = poly(pts)
    if p.get("solid") or "weight" not in p:
        g = outline
    else:
        w = float(p["weight"])
        join = p.get("corner", "mitre")
        js = {"mitre": 2, "round": 1, "bevel": 3}[join]
        g = outline.buffer(w / 2, join_style=js, mitre_limit=4).difference(
            outline.buffer(-w / 2, join_style=js, mitre_limit=4))
    for gap in p.get("gaps", []) or []:
        at, width = float(gap["at"]), float(gap["width"])
        g = g.difference(wedge(r * 2, at - width / 2, at + width / 2))
    feats = {
        "vertex": [(x, y, vec_ang(x, y)) for x, y in verts],
        "edge": [(*(np.add(verts[i], verts[(i + 1) % n]) / 2),
                  vec_ang(*(np.add(verts[i], verts[(i + 1) % n]) / 2))) for i in range(n)],
    }
    return Shape(g, feats, point_r=r)


def star_polygon(p) -> Shape:
    n = int(p.get("n", 5))
    r = float(p.get("r", 1.0))
    ri = float(p.get("inner_r", 0.5)) * r
    radii = p.get("radii") or [1.0] * n
    concave = float(p.get("concave", 0.0))
    tips = [polar(r * radii[i % len(radii)], i * 360.0 / n) for i in range(n)]
    vals = [polar(ri, (i + 0.5) * 360.0 / n) for i in range(n)]
    pts = []
    for i in range(n):
        t, v, t2 = np.array(tips[i]), np.array(vals[i]), np.array(tips[(i + 1) % n])
        if concave:
            pts.extend(quad(t, t + (v - t) * (1 + concave) / 2 + (v * concave * 0.3), v, 12)[:-1])
            pts.extend(quad(v, t2 + (v - t2) * (1 + concave) / 2 + (v * concave * 0.3), t2, 12)[:-1])
        else:
            pts += [t, v]
    outline = poly(pts)
    if "weight" in p and not p.get("solid"):
        w = float(p["weight"])
        g = outline.buffer(w / 2, join_style=2, mitre_limit=6).difference(outline.buffer(-w / 2, join_style=2))
    else:
        g = outline
    feats = {"tip": [(x, y, vec_ang(x, y)) for x, y in tips],
             "valley": [(x, y, vec_ang(x, y)) for x, y in vals]}
    return Shape(g, feats, point_r=r)


def crescent(p) -> Shape:
    """Disc of radius r minus a disc shifted toward `opening` (deg)."""
    r = float(p.get("r", 1.0))
    thick = float(p.get("thickness", 0.35))     # fraction of r removed from the belly
    ri = float(p.get("inner_r", 1.0 - thick * 0.25)) * r
    opening = float(p.get("opening", 0.0))
    d = r - ri + thick * r                        # center offset so that belly thickness = thick*r
    cx, cy = polar(d, opening)
    g = Point(0, 0).buffer(r, 128).difference(Point(cx, cy).buffer(ri, 128))
    # horns: circle-circle intersection
    a = (r * r - ri * ri + d * d) / (2 * d)
    h = math.sqrt(max(r * r - a * a, 0))
    ux, uy = cx / d, cy / d
    px, py = a * ux, a * uy
    horns = [(px + h * uy, py - h * ux), (px - h * uy, py + h * ux)]
    feats = {"horn": [(x, y, vec_ang(x, y)) for x, y in horns],
             "belly": [(*polar(r, opening + 180), (opening + 180) % 360)]}
    return Shape(g, feats, point_r=r)


def gear(p) -> Shape:
    n = int(p.get("n", 8))
    r = float(p.get("r", 1.0))
    rim = float(p.get("weight", 0.2))
    th = float(p.get("tooth_h", 0.18))
    tw = float(p.get("tooth_w", 0.45))  # fraction of pitch
    body = annulus(r - rim / 2, rim) if not p.get("solid") else Point(0, 0).buffer(r, 128)
    pitch = 2 * math.pi * r / n
    teeth = []
    for i in range(n):
        tooth = poly([(-pitch * tw / 2, r - 0.02), (pitch * tw / 2, r - 0.02),
                      (pitch * tw * 0.4, r + th), (-pitch * tw * 0.4, r + th)])
        teeth.append(affinity.rotate(tooth, -i * 360.0 / n, origin=(0, 0)))
    g = unary_union([body] + teeth)
    feats = {"tooth": [(*polar(r + th, i * 360.0 / n), i * 360.0 / n) for i in range(n)]}
    return Shape(g, feats, point_r=r)


def arc(p) -> Shape:
    """Band along a circle of radius r, spanning `span` deg centered on angle 0 (top)."""
    r = float(p.get("r", 1.0))
    span = float(p.get("span", 90))
    w = float(p.get("weight", 0.08))
    t = np.linspace(0, 1, 96)
    angs = -span / 2 + span * t
    center = np.array([polar(r, a) for a in angs])
    widths = profile_widths(p.get("profile"), w, t)
    g = tapered_stroke(center, widths, round_caps=p.get("caps") == "round")
    a0, a1 = -span / 2, span / 2
    feats = {"end": [(*polar(r, a0), (a0 - 90) % 360), (*polar(r, a1), (a1 + 90) % 360)],
             "mid": [(*polar(r, 0), 0.0)]}
    return Shape(g, feats, point_r=r)


def sector(p) -> Shape:
    r_in = float(p.get("r_in", 0.0))
    r_out = float(p.get("r_out", 1.0))
    span = float(p.get("span", 30))
    g = wedge(r_out, -span / 2, span / 2)
    if r_in > 0:
        g = g.difference(Point(0, 0).buffer(r_in, 128))
    return Shape(g, {"tip": [(*polar(r_in, 0), 180.0)], "outer": [(*polar(r_out, 0), 0.0)]}, point_r=r_out)


def outline(p) -> Shape:
    pts = np.asarray(p["points"], float)
    if p.get("mirror"):
        pts = np.vstack([pts, mirror_x(pts)[::-1]])
    shape = poly(pts)
    if "weight" in p and not p.get("solid"):
        w = float(p["weight"])
        shape = shape.buffer(w / 2, join_style=2, mitre_limit=4).difference(shape.buffer(-w / 2, join_style=2))
    return Shape(shape, {"vertex": [(x, y, vec_ang(x, y)) for x, y in pts]})


def path(p) -> Shape:
    pts = np.asarray(p["points"], float)
    w = float(p.get("weight", 0.06))
    if len(pts) >= 3 and p.get("smooth"):
        # Catmull-Rom-ish through the points via quadratic midpoints
        mids = (pts[:-1] + pts[1:]) / 2
        seg = [pts[:1]]
        for i in range(1, len(pts) - 1):
            seg.append(quad(mids[i - 1], pts[i], mids[i], 16))
        seg.append(pts[-1:])
        pts = np.vstack(seg)
    t = np.linspace(0, 1, len(pts))
    g = tapered_stroke(pts, profile_widths(p.get("profile"), w, t), round_caps=p.get("caps") == "round")
    d0, d1 = pts[0] - pts[1], pts[-1] - pts[-2]
    feats = {"end": [(*pts[0], vec_ang(*d0)), (*pts[-1], vec_ang(*d1))]}
    return Shape(g, feats)


# ---------------------------------------------------------------- motifs (rooted: base at origin, tip +y)

def _rooted(g, L, extra=None):
    f = {"tip": [(0.0, L, 0.0)], "base": [(0.0, 0.0, 180.0)], "center": [(0.0, L / 2, 0.0)]}
    if extra:
        f.update(extra)
    return Shape(g, f)


def spike(p):
    L, w, s = _size(p, 0.4, 0.12)
    c = float(s.get("concave", p.get("concave", 0.35)))
    right = quad((w / 2, 0), (w / 4 - c * w / 4, L / 2), (0, L))
    left = mirror_x(right)[::-1]
    return _rooted(poly(np.vstack([right, left])), L)


def blade(p):
    """Long 4-point diamond with concave sides; widest at shoulder k."""
    L, w, s = _size(p, 0.5, 0.14)
    k = float(s.get("shoulder", p.get("shoulder", 0.3)))
    c = float(s.get("concave", p.get("concave", 0.3)))
    sh = np.array([w / 2, L * k])
    up = quad(sh, (w / 2 * (1 - c) * 0.6, L * (k + (1 - k) * 0.5)), (0, L))
    dn = quad((0, 0), (w / 2 * (1 - c) * 0.8, L * k * 0.5), sh)
    right = np.vstack([dn, up])
    return _rooted(poly(np.vstack([right, mirror_x(right)[::-1]])), L)


def kite(p):
    L, w, s = _size(p, 0.4, 0.2)
    k = float(s.get("shoulder", p.get("shoulder", 0.35)))
    return _rooted(poly([(0, 0), (w / 2, L * k), (0, L), (-w / 2, L * k)]), L)


def triangle(p):
    L, w, _ = _size(p, 0.3, 0.3)
    return _rooted(poly([(-w / 2, 0), (w / 2, 0), (0, L)]), L)


def bar(p):
    L, w, _ = _size(p, 0.3, 0.07)
    if p.get("caps") == "round":
        g = LineString([(0, w / 2), (0, L - w / 2)]).buffer(w / 2, 32)
    else:
        g = poly([(-w / 2, 0), (w / 2, 0), (w / 2, L), (-w / 2, L)])
    return _rooted(g, L)


def leaf(p):
    L, w, _ = _size(p, 0.4, 0.18)
    right = quad((0, 0), (w, L / 2), (0, L))
    return _rooted(poly(np.vstack([right, mirror_x(right)[::-1][1:]])), L)


def petal(p):
    """Rounded tip, narrow base."""
    L, w, s = _size(p, 0.4, 0.25)
    notch = float(s.get("notch", p.get("notch", 0.0)))
    right = cubic((0, 0), (w * 0.75, L * 0.35), (w * 0.62, L * 1.05), (0, L * (1 - notch)))
    g = poly(np.vstack([right, mirror_x(right)[::-1][1:]]))
    return _rooted(g, L)


def teardrop(p):
    """Round bulb at base, point at tip (+y)."""
    L, w, _ = _size(p, 0.35, 0.2)
    right = cubic((0, L), (w * 0.15, L * 0.55), (w * 0.72, L * -0.05), (0, 0))
    g = poly(np.vstack([right, mirror_x(right)[::-1][1:]]))
    return _rooted(g, L)


def arrow(p):
    L, w, s = _size(p, 0.3, 0.25)
    notch = float(s.get("notch", p.get("notch", 0.25)))
    barb = float(s.get("barb", p.get("barb", 0.2)))
    head = poly([(0, L), (w / 2, L * barb), (0, L * (barb + notch)), (-w / 2, L * barb)])
    shaft = float(s.get("shaft", p.get("shaft", 0.0)))
    if shaft:
        head = head.union(poly([(-shaft / 2, 0), (shaft / 2, 0), (shaft / 2, L * 0.6), (-shaft / 2, L * 0.6)]))
    return _rooted(head, L)


def chevron(p):
    L, w, s = _size(p, 0.2, 0.3)
    wt = float(s.get("weight", p.get("weight", 0.05)))
    g = LineString([(-w / 2, 0), (0, L), (w / 2, 0)]).buffer(wt / 2, join_style=2, mitre_limit=5)
    return _rooted(g, L)


def horn(p):
    L, w, s = _size(p, 0.45, 0.12)
    bend = float(s.get("bend", p.get("bend", 0.35)))  # + bends clockwise (to +x)
    t = np.linspace(0, 1, 64)
    center = quad((0, 0), (0, L * 0.6), (bend * L, L), 64)
    widths = w * (1 - t) ** 0.9 + 1e-4
    g = tapered_stroke(center, widths)
    tip = center[-1]
    return Shape(g, {"tip": [(tip[0], tip[1], vec_ang(*(center[-1] - center[-2])))],
                     "base": [(0.0, 0.0, 180.0)]})


def wing(p):
    """Fan of feathers rooted at origin, fanning clockwise from +y toward +x."""
    L, w, s = _size(p, 0.4, 0.12)
    nf = int(s.get("feathers", p.get("feathers", 3)))
    spread = float(s.get("spread", p.get("spread", 50)))
    parts = []
    for i in range(nf):
        f = i / max(nf - 1, 1)
        lf = leaf({"size": {"len": L * (1 - 0.3 * f), "w": w}}).geom
        parts.append(affinity.rotate(lf, -spread * f, origin=(0, 0)))
    g = unary_union(parts)
    return Shape(g, {"base": [(0.0, 0.0, 180.0)], "tip": [(0.0, L, 0.0)]})


def fork(p):
    L, w, s = _size(p, 0.35, 0.2)
    wt = float(s.get("weight", p.get("weight", 0.04)))
    split = float(s.get("split", p.get("split", 0.6)))
    g = unary_union([
        LineString([(0, 0), (0, L * split)]).buffer(wt / 2, cap_style=2),
        LineString([(0, L * split), (w / 2, L)]).buffer(wt / 2, cap_style=2),
        LineString([(0, L * split), (-w / 2, L)]).buffer(wt / 2, cap_style=2),
    ])
    return _rooted(g, L, {"prong": [(w / 2, L, 30.0), (-w / 2, L, 330.0)]})


def shard(p):
    L, w, _ = _size(p, 0.35, 0.16)
    return _rooted(poly([(0, 0), (w * 0.6, L * 0.45), (0.05 * w, L), (-w * 0.4, L * 0.25)]), L)


# ---------------------------------------------------------------- centered motifs

def _centered(g, feats=None):
    f = {"center": [(0.0, 0.0, 0.0)]}
    if feats:
        f.update(feats)
    return Shape(g, f)


def dot(p):
    L, _, _ = _size(p, 0.12, 0.12)
    return _centered(Point(0, 0).buffer(L / 2, 64))


def ring(p):
    L, _, s = _size(p, 0.4, 0.4)
    wt = float(s.get("weight", p.get("weight", 0.06)))
    return _centered(annulus(L / 2, wt), {"point": []})


def sparkle(p):
    """n-point concave star (default 4). tip-to-tip = len, waist radius = w/2.
    sharp: exponent of the polar profile; 2 = soft sparkle, 6+ = needle points."""
    L, w, s = _size(p, 0.3, 0.08)
    n = int(s.get("n", p.get("n", 4)))
    sharp = float(s.get("sharp", p.get("sharp", 0.8)))   # 0.3 soft .. 1.0 needle
    radii = s.get("radii", p.get("radii")) or [1.0]
    tips = [polar(L / 2 * radii[i % len(radii)], i * 360 / n) for i in range(n)]
    pts = []
    for i in range(n):
        t1, t2 = np.array(tips[i]), np.array(tips[(i + 1) % n])
        c = np.array(polar(w / 2, (i + 0.5) * 360 / n))
        pts.extend(cubic(t1, t1 + (c - t1) * sharp, t2 + (c - t2) * sharp, t2, 40)[:-1])
    return _centered(poly(pts), {"tip": [(x, y, vec_ang(x, y)) for x, y in tips]})


def star(p):
    L, w, s = _size(p, 0.3, 0.3)
    n = int(s.get("n", p.get("n", 5)))
    inner = float(s.get("inner", p.get("inner", 0.45)))
    pts = []
    for i in range(n):
        pts.append(polar(L / 2, i * 360 / n))
        pts.append(polar(L / 2 * inner, (i + 0.5) * 360 / n))
    tips = pts[::2]
    return _centered(poly(pts), {"tip": [(x, y, vec_ang(x, y)) for x, y in tips]})


def diamond(p):
    L, w, _ = _size(p, 0.2, 0.14)
    return _centered(poly([(0, L / 2), (w / 2, 0), (0, -L / 2), (-w / 2, 0)]),
                     {"tip": [(0, L / 2, 0.0), (w / 2, 0, 90.0), (0, -L / 2, 180.0), (-w / 2, 0, 270.0)]})


def cross(p):
    L, w, s = _size(p, 0.3, 0.07)
    typ = s.get("type", p.get("type", "plain"))
    if typ == "pattee":
        arm = poly([(-w / 2, 0), (w / 2, 0), (w * 1.6, L / 2), (-w * 1.6, L / 2)])
    else:
        arm = poly([(-w / 2, 0), (w / 2, 0), (w / 2, L / 2), (-w / 2, L / 2)])
    g = unary_union([affinity.rotate(arm, -90 * i, origin=(0, 0)) for i in range(4)] + [Point(0, 0).buffer(w / 2)])
    return _centered(g, {"tip": [(*polar(L / 2, 90 * i), 90.0 * i) for i in range(4)]})


def heart(p):
    L, w, _ = _size(p, 0.25, 0.25)
    t = np.linspace(0, 2 * np.pi, 160)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    x = x / 32 * w
    y = (y + 5) / 29 * L  # roughly center the shape vertically
    return _centered(poly(np.column_stack([x, y])), {"tip": [(0.0, float(y.min()), 180.0)]})


def tomoe(p):
    """Magatama comma: round head at origin, tail sweeping clockwise."""
    L, w, s = _size(p, 0.35, 0.35)
    h = L * 0.32
    sweep = float(s.get("sweep", p.get("sweep", 200)))
    t = np.linspace(0, 1, 80)
    rr = h * (1 + 0.9 * t)
    angs = 90 + sweep * t
    center = np.array([(rr[i] * math.sin(math.radians(angs[i])) - h * 0.0,
                        rr[i] * math.cos(math.radians(angs[i])) - h * 0.0) for i in range(len(t))])
    center = center - center[0] + np.array([0.0, 0.0])
    widths = 2 * h * (1 - t) ** 1.1 + 1e-4
    g = unary_union([Point(0, 0).buffer(h, 64), tapered_stroke(center + [0, 0], widths)])
    return _centered(g)


def flower(p):
    """n petals around the center (kamon-like). size.len = diameter."""
    L, w, s = _size(p, 0.5, 0.2)
    n = int(s.get("n", p.get("n", 5)))
    notch = float(s.get("notch", p.get("notch", 0.0)))
    pw = float(s.get("petal_w", p.get("petal_w", 0.5)))  # petal width relative to len/2
    petals = []
    for i in range(n):
        pt = petal({"size": {"len": L / 2, "w": L / 2 * pw, "notch": notch}}).geom
        petals.append(affinity.rotate(pt, -i * 360 / n, origin=(0, 0)))
    g = unary_union(petals)
    hole = float(s.get("center_hole", p.get("center_hole", 0.0)))
    if hole:
        g = g.difference(Point(0, 0).buffer(hole * L / 2, 64))
    tips = [polar(L / 2, i * 360 / n) for i in range(n)]
    return _centered(g, {"tip": [(x, y, vec_ang(x, y)) for x, y in tips]})


# ---------------------------------------------------------------- signature motifs (BA-derived, curated)

def _stroke_along(pts, w, profile="taper_one", caps=False):
    t = np.linspace(0, 1, len(pts))
    return tapered_stroke(pts, profile_widths(profile, w, t), round_caps=caps)


def bolt(p):
    """Lightning zig-zag (Haruka, Misaka): thick middle, sharp ends."""
    L, w, s = _size(p, 0.6, 0.14)
    zig = float(s.get("zig", p.get("zig", 0.35)))
    k = [(0, 0), (zig * L, L * 0.38), (-zig * L * 0.25, L * 0.55), (zig * L * 0.6, L)]
    pts = np.vstack([np.linspace(k[i], k[i + 1], 20)[:-1] for i in range(3)] + [np.array([k[3]])])
    g = _stroke_along(pts, w, {"points": [[0, 0.1], [0.3, 1], [0.6, 0.9], [1, 0.02]]})
    return _rooted(g, L)


def flame(p):
    """S-curved tongue with wide base and flicking tip (Kasumi, Megu)."""
    L, w, s = _size(p, 0.55, 0.26)
    bend = float(s.get("bend", p.get("bend", 0.25)))
    c = cubic((0, 0), (-bend * L, L * 0.35), (bend * L * 1.2, L * 0.65), (bend * L * 0.3, L), 60)
    g = _stroke_along(c, w, {"points": [[0, 0.8], [0.25, 1.0], [0.7, 0.45], [1, 0.02]]})
    return Shape(g, {"tip": [(*c[-1], vec_ang(*(c[-1] - c[-2])))], "base": [(0.0, 0.0, 180.0)]})


def hook(p):
    """Spike that curls back at the tip (Junko, Kotama)."""
    L, w, s = _size(p, 0.5, 0.12)
    curl = float(s.get("curl", p.get("curl", 1.0)))  # + curls clockwise
    t = np.linspace(0, 1, 70)
    straight = np.column_stack([np.zeros(40), np.linspace(0, L * 0.7, 40)])
    r = L * 0.18
    a = np.linspace(0, np.pi * 1.1, 30)
    arc_ = np.column_stack([curl * (r - r * np.cos(a)), L * 0.7 + r * np.sin(a)])
    pts = np.vstack([straight, arc_[1:]])
    g = _stroke_along(pts, w, {"points": [[0, 1.0], [0.6, 0.8], [1, 0.05]]})
    return _rooted(g, L)


def curl(p):
    """Scroll: stem ending in a spiral (Eri, Tsumugi, kamon 蕨)."""
    L, w, s = _size(p, 0.5, 0.09)
    turns = float(s.get("turns", p.get("turns", 1.2)))
    d = float(s.get("dir", p.get("dir", 1)))
    stem = np.column_stack([np.zeros(20), np.linspace(0, L * 0.45, 20)])
    th = np.linspace(0, turns * 2 * np.pi, 90)
    r0 = L * 0.28
    rr = r0 * (1 - th / (turns * 2 * np.pi) * 0.85)
    cx, cy = d * r0, L * 0.45
    spiral = np.column_stack([cx - d * rr * np.cos(th), cy + rr * np.sin(th)])
    pts = np.vstack([stem, spiral[1:]])
    g = _stroke_along(pts, w, {"points": [[0, 0.7], [0.3, 1.0], [1, 0.25]]}, caps=True)
    return _rooted(g, L)


def fleur(p):
    """Fleur-de-lis-like finial: central blade + two outward curled petals (Mine, Serina, Hiromi)."""
    L, w, s = _size(p, 0.5, 0.4)
    mid = blade({"size": {"len": L, "w": w * 0.38, "shoulder": 0.35}}).geom
    side = cubic((0, L * 0.3), (w * 0.95, L * 0.18), (w * 0.95, L * 0.8), (w * 0.42, L * 0.72), 40)
    petal_r = _stroke_along(side, w * 0.22, {"points": [[0, 1.0], [0.6, 0.8], [1, 0.1]]})
    band = poly([(-w * 0.35, L * 0.14), (w * 0.35, L * 0.14), (w * 0.35, L * 0.22), (-w * 0.35, L * 0.22)])
    g = unary_union([mid, petal_r, affinity.scale(petal_r, -1, 1, origin=(0, 0)), band])
    return _rooted(g, L)


def drip(p):
    """Hanging drip: stem that swells into a drop (Tsurugi, Miyu)."""
    L, w, _ = _size(p, 0.4, 0.14)
    stem = _stroke_along(np.column_stack([np.zeros(20), np.linspace(0, L * 0.7, 20)]), w * 0.5,
                         {"points": [[0, 1.0], [1, 0.6]]})
    drop = Point(0, L * 0.78).buffer(w / 2, 48)
    tip = poly([(-w / 2, L * 0.78), (w / 2, L * 0.78), (0, L)])
    g = unary_union([stem, drop, tip.difference(Point(0, L).buffer(1e-3))])
    return _rooted(affinity.scale(g, 1, 1, origin=(0, 0)), L)


def feather(p):
    """Hawk feather (kamon 鷹の羽): long leaf with knocked-out barb slits."""
    L, w, s = _size(p, 0.6, 0.2)
    slits = int(s.get("slits", p.get("slits", 3)))
    g = leaf({"size": {"len": L, "w": w}}).geom
    g = g.difference(LineString([(0, L * 0.08), (0, L * 0.9)]).buffer(w * 0.05))
    for i in range(slits):
        y = L * (0.35 + 0.15 * i)
        g = g.difference(LineString([(0, y), (w * 0.6, y + L * 0.08)]).buffer(w * 0.045))
    return _rooted(g, L)


def branch(p):
    """Laurel/branch: curved stem with alternating leaves (Airi, Azusa)."""
    L, w, s = _size(p, 0.7, 0.16)
    n = int(s.get("leaves", p.get("leaves", 4)))
    bend = float(s.get("bend", p.get("bend", 0.2)))
    c = quad((0, 0), (bend * L, L * 0.5), (0, L), 60)
    parts = [_stroke_along(c, w * 0.22, {"points": [[0, 1], [1, 0.4]]})]
    for i in range(n):
        j = int((0.25 + 0.7 * i / max(n - 1, 1)) * (len(c) - 1))
        x, y = c[j]
        side = 1 if i % 2 == 0 else -1
        lf = leaf({"size": {"len": L * 0.3, "w": w}}).geom
        lf = affinity.rotate(lf, -side * 45, origin=(0, 0))
        parts.append(affinity.translate(lf, x, y))
    parts.append(affinity.translate(leaf({"size": {"len": L * 0.28, "w": w}}).geom, *c[-1]))
    return _rooted(unary_union(parts), L * 1.25)


def wave(p):
    """Breaking wave crest (Umika, Minato, kamon 波): tapered curl."""
    L, w, s = _size(p, 0.5, 0.2)
    th = np.linspace(0, 1.5 * np.pi, 80)
    rr = L * 0.35 * (1 - th / (1.5 * np.pi) * 0.7)
    spiral = np.column_stack([L * 0.35 - rr * np.cos(th), L * 0.55 + rr * np.sin(th)])
    base = quad((-L * 0.1, 0), (-L * 0.1, L * 0.4), spiral[0], 20)
    pts = np.vstack([base, spiral[1:]])
    g = _stroke_along(pts, w, {"points": [[0, 0.6], [0.3, 1.0], [1, 0.05]]})
    return _rooted(g, L)


def eye(p):
    """Vesica with a pupil (Kazusa, Tsubasa, Kikyou)."""
    L, w, s = _size(p, 0.5, 0.28)
    pupil = s.get("pupil", p.get("pupil", "slit"))
    right = quad((0, -L / 2), (w, 0), (0, L / 2))
    g = poly(np.vstack([right, mirror_x(right)[::-1][1:]]))
    inner = affinity.scale(g, 0.72, 0.72, origin=(0, 0))
    g = g.difference(inner)
    if pupil == "slit":
        g = g.union(affinity.scale(inner, 0.25, 0.85, origin=(0, 0)))
    else:
        g = g.union(Point(0, 0).buffer(w * 0.2, 48))
    return _centered(g, {"tip": [(0.0, L / 2, 0.0), (0.0, -L / 2, 180.0)]})


def cloud(p):
    """Scalloped cloud (Kaguya, Shuro): overlapping circles on a flat-ish base."""
    L, w, s = _size(p, 0.5, 0.3)
    k = int(s.get("puffs", p.get("puffs", 3)))
    xs = np.linspace(-L / 2 + w / 2, L / 2 - w / 2, k)
    parts = [Point(x, (w * 0.35 if 0 < i < k - 1 else 0)).buffer(w / 2 * (1.25 if 0 < i < k - 1 else 1), 48)
             for i, x in enumerate(xs)]
    return _centered(unary_union(parts))


def triquetra(p):
    """Three interlocked vesica loops (Ayame): outlines with over/under gaps."""
    L, w, s = _size(p, 0.8, 0.05)
    loops = []
    for i in range(3):
        right = quad((0, 0), (L * 0.33, L * 0.25), (0, L * 0.5))
        v = poly(np.vstack([right, mirror_x(right)[::-1][1:]]))
        ring_ = v.boundary.buffer(w / 2, 16)
        loops.append(affinity.rotate(ring_, -120 * i, origin=(0, 0)))
    g = EMPTY
    for i, lp in enumerate(loops):   # simple cyclic over/under: each loop cuts a gap in the previous one
        g = g.difference(lp.buffer(w * 0.8)).union(lp) if i else lp
    return _centered(g)


def flower_typed(p):
    """flower with petal type: round | pointed | notched | heart."""
    L, w, s = _size(p, 0.5, 0.2)
    n = int(s.get("n", p.get("n", 5)))
    typ = s.get("petal", p.get("petal", "round"))
    pw = float(s.get("petal_w", p.get("petal_w", 0.55)))
    parts = []
    for i in range(n):
        if typ == "pointed":
            pt = kite({"size": {"len": L / 2, "w": L / 2 * pw, "shoulder": 0.55}}).geom
        elif typ == "heart":
            pt = heart({"size": {"len": L / 2 * 0.9, "w": L / 2 * pw * 1.2}}).geom
            pt = affinity.translate(pt, 0, L / 4)
        else:
            pt = petal({"size": {"len": L / 2, "w": L / 2 * pw, "notch": 0.18 if typ == "notched" else 0.0}}).geom
        parts.append(affinity.rotate(pt, -i * 360 / n, origin=(0, 0)))
    g = unary_union(parts)
    tips = [polar(L / 2, i * 360 / n) for i in range(n)]
    return _centered(g, {"tip": [(x, y, vec_ang(x, y)) for x, y in tips]})


# ---------------------------------------------------------------- kamon-derived shapes

def embrace_arm(p):
    """One arm of a kamon 抱き (embracing) pair: a tapered arc up the RIGHT side, mirrored by the symmetry.
    deco: plain | leaves | feather | curl | buds"""
    r0 = float(p.get("r", 0.85))
    a0, a1 = float(p.get("from", 172)), float(p.get("to", 12))    # clockwise degrees: bottom -> top
    w = float(p.get("weight", 0.14))
    deco = p.get("deco", "plain")
    t = np.linspace(0, 1, 90)
    angs = a0 + (a1 - a0) * t
    c = np.array([polar(r0, a) for a in angs])
    prof = {"points": [[0, 0.55], [0.2, 1.0], [0.75, 0.7], [1, 0.08]]}
    parts = [tapered_stroke(c, profile_widths(prof, w, t))]
    feats = {"end": [(*c[0], (a0 + 90) % 360), (*c[-1], (a1 - 90) % 360)]}
    if deco in ("leaves", "feather", "buds"):
        k = int(p.get("count", 5))
        for i in range(k):
            f = 0.18 + 0.72 * i / max(k - 1, 1)
            a = a0 + (a1 - a0) * f
            x, y = polar(r0, a)
            travel = a - 90                                  # direction of travel along the arm (toward the top)
            for side in ((1, -1) if deco == "feather" else (1,) if deco == "buds" else (1, -1) if i % 2 else (1,)):
                if deco == "buds":
                    g = Point(0, 0).buffer(w * 0.55, 32)
                    g = affinity.translate(g, *polar(w * 1.1, a))
                else:
                    L = w * (3.2 if deco == "leaves" else 2.2) * (1 - 0.35 * f)
                    g = leaf({"size": {"len": L, "w": L * 0.42}}).geom
                    g = affinity.rotate(g, -(travel + side * (40 if deco == "leaves" else 70)), origin=(0, 0))
                parts.append(affinity.translate(g, x, y))
    if deco == "curl":
        x, y = c[-1]
        cr = w * 1.4
        th = np.linspace(0, 1.6 * np.pi, 50)
        rr = cr * (1 - th / (1.6 * np.pi) * 0.7)
        base_dir = math.radians(a1 - 90)
        spiral = np.column_stack([x + rr * np.sin(base_dir + th) - cr * math.sin(base_dir),
                                  y + rr * np.cos(base_dir + th) - cr * math.cos(base_dir)])
        parts.append(tapered_stroke(spiral, np.linspace(w * 0.5, w * 0.2, len(spiral))))
    return Shape(unary_union(parts), feats)


def sumikiri(p):
    """Cut-corner square (隅切り角) outline."""
    r = float(p.get("r", 1.0))
    cut = float(p.get("cut", 0.3))
    s = r * 0.8
    pts = [(-s + cut * s, s), (s - cut * s, s), (s, s - cut * s), (s, -s + cut * s),
           (s - cut * s, -s), (-s + cut * s, -s), (-s, -s + cut * s), (-s, s - cut * s)]
    shape = poly(pts)
    w = float(p.get("weight", 0.1))
    g = shape.buffer(w / 2, join_style=2).difference(shape.buffer(-w / 2, join_style=2))
    return Shape(g, {"vertex": [(x, y, vec_ang(x, y)) for x, y in pts]}, point_r=r)


def igeta(p):
    """Well frame (井桁): two horizontal + two vertical bars with overhanging ends, over/under weave."""
    r = float(p.get("r", 1.0))
    w = float(p.get("weight", 0.1))
    d, L = r * 0.55, r * 1.0
    bars = [LineString([(-L, y), (L, y)]).buffer(w / 2, cap_style=2) for y in (d, -d)] + \
           [LineString([(x, -L), (x, L)]).buffer(w / 2, cap_style=2) for x in (d, -d)]
    g = bars[0]
    for i, b in enumerate(bars[1:], 1):   # alternate over/under at crossings
        g = g.difference(b.buffer(w * 0.35)).union(b) if i % 2 else g.union(b.difference(g.buffer(w * 0.35)))
    return Shape(g, {"vertex": [(sx * d, sy * d, vec_ang(sx, sy)) for sx in (1, -1) for sy in (1, -1)]}, point_r=r)


def mark(p):
    """House mark / tamga glyph (halo.marks), scaled so its height = size.len."""
    from .marks import mark_geom
    L = float((p.get("size") or {}).get("len", 1.0))
    g = mark_geom(p["mark"], w=float(p.get("stroke", 0.09)))
    x0, y0, x1, y1 = g.bounds
    g = affinity.translate(g, -(x0 + x1) / 2, -(y0 + y1) / 2)
    k = L / max(y1 - y0, x1 - x0)
    return _centered(affinity.scale(g, k, k, origin=(0, 0)))


def plait(p):
    """Celtic plait ring: `strands` sinusoidal bands braided around a circle with alternating over/under crossings.
    k = crossings per strand pair around the ring (even, so over/under alternation closes)."""
    r = float(p.get("r", 1.0))
    w = float(p.get("weight", 0.07))
    amp = float(p.get("amp", 0.09))
    k = int(p.get("k", 12))
    strands = int(p.get("strands", 2))
    gap = float(p.get("gap", w * 0.6))
    edge_line = bool(p.get("border", False))
    th = np.linspace(0, 2 * np.pi, 1440, endpoint=False)
    paths = []
    for s in range(strands):
        ph = 2 * np.pi * s / strands
        rr = r + amp * np.sin(k / 2 * th + ph)
        paths.append(np.column_stack([rr * np.sin(th), rr * np.cos(th)]))
    bands = [LineString(np.vstack([P, P[:1]])).buffer(w / 2, 8) for P in paths]
    # crossings: where strand radii are equal; alternate which strand is on top along the ring
    out = unary_union(bands)
    for i in range(strands):
        for j in range(i + 1, strands):
            ri = np.sin(k / 2 * th + 2 * np.pi * i / strands)
            rj = np.sin(k / 2 * th + 2 * np.pi * j / strands)
            diff = ri - rj
            idx = np.nonzero(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
            for c, t in enumerate(idx):
                top, bot = (i, j) if c % 2 == 0 else (j, i)
                x, y = paths[top][t]
                window = Point(x, y).buffer(w * 2.2, 16)
                over = bands[top].intersection(window)
                out = out.difference(over.buffer(gap / 2, 8)).union(over)
    if edge_line:
        for rr in (r - amp - w * 1.4, r + amp + w * 1.4):
            out = out.union(annulus(rr, w * 0.45))
    return Shape(out, {"center": [(0, 0, 0)]}, point_r=r)


def ornament(p):
    """Procedural ornaments from halo.ornament: kind = rosette | fret | knot."""
    from . import ornament as O
    kind, r, w = p["kind"], float(p.get("r", 1.0)), float(p.get("weight", 0.06))
    if kind == "rosette":
        g = O.rosette(int(p.get("n", 10)), r, w, p.get("star", "fill"))
    elif kind == "fret":
        g = O.fret(p.get("cell", "key"), int(p.get("n", 16)), r, float(p.get("h", 0.2)), w, p.get("rails", "in"))
    elif kind == "knot":
        g = O.knot_rosette(p.get("curve", "7/4"), r, w)
    else:
        raise ValueError(kind)
    return Shape(g, {"center": [(0, 0, 0)]}, point_r=r)


def library(p):
    """Traced motif from halo.library.LIB, centered, diameter = size.len."""
    from .library import LIB
    g = LIB[p["name"]]
    L = float((p.get("size") or {}).get("len", 1.0))
    fx = -1 if p.get("flip") else 1
    return _centered(affinity.scale(g, fx * L / 2, L / 2, origin=(0, 0)))


REGISTRY = {
    "library": library, "mark": mark, "plait": plait, "ornament": ornament,
    "embrace_arm": embrace_arm, "sumikiri": sumikiri, "igeta": igeta,
    # signature motifs
    "bolt": bolt, "flame": flame, "hook": hook, "curl": curl, "fleur": fleur, "drip": drip,
    "feather": feather, "branch": branch, "wave": wave, "eye": eye, "cloud": cloud, "triquetra": triquetra,
    "flower_typed": flower_typed,
    # frames
    "circle": circle, "ring_band": circle, "polygon": polygon, "arc_polygon": polygon,
    "star_polygon": star_polygon, "crescent": crescent, "gear": gear, "arc": arc,
    "sector": sector, "outline": outline, "path": path,
    # rooted motifs
    "spike": spike, "blade": blade, "kite": kite, "triangle": triangle, "bar": bar, "tick": bar,
    "leaf": leaf, "petal": petal, "teardrop": teardrop, "arrow": arrow, "chevron": chevron,
    "horn": horn, "wing": wing, "fork": fork, "shard": shard,
    # centered motifs
    "dot": dot, "orb": dot, "ring": ring, "sparkle": sparkle, "star": star, "diamond": diamond,
    "cross": cross, "heart": heart, "tomoe": tomoe, "flower": flower,
}


def make_shape(p: dict) -> Shape:
    kind = p["shape"]
    if kind not in REGISTRY:
        raise KeyError(f"unknown shape {kind!r}")
    return REGISTRY[kind](p)
