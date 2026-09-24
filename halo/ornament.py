"""Procedural ornaments: Islamic rosettes, meander / fret rings, Celtic knot rosettes (alternating over/under).

All return shapely geometry in unit-radius space (outer radius ~= r / 1.0).
Safety: fret rings always keep at least one rail (loose T-fret hooks echo swastika arms);
knot curves that come out as a pentagram (5/2) or atom-like (5/3) are not offered.
"""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

from .geom import polar


def _strokes(polylines, w):
    return unary_union([LineString(p).buffer(w / 2, 8, join_style=2, mitre_limit=3) for p in polylines])


# ---------------------------------------------------------------- Islamic rosette

def rosette(n=10, r=1.0, w=0.06, star="fill", spread=0.55):
    """n-point star with a ring of petals (girih-style). star: fill | outline | knock (star cut from a disc)."""
    r_star, r_valley, r_petal = 0.5 * r, 0.3 * r, 0.78 * r
    tips = [polar(r_star, i * 360 / n) for i in range(n)]
    vals = [polar(r_valley, (i + 0.5) * 360 / n) for i in range(n)]
    ring = []
    for i in range(n):
        ring += [tips[i], vals[i]]
    half = 180 / n
    lines = [ring + [ring[0]]]
    for i in range(n):
        a = i * 360 / n
        A, B, C = polar(r_petal, a - half * spread), polar(r, a), polar(r_petal, a + half * spread)
        lines += [[A, tips[i], C], [A, B, C],
                  [C, polar(r_petal + (r - r_petal) * 0.35, a + half), polar(r_petal, a + 360 / n - half * spread)]]
    g = _strokes(lines, w)
    star_poly = Polygon(ring)
    if star == "fill":
        g = g.union(star_poly)
    elif star == "knock":
        g = g.union(Point(0, 0).buffer(r_star * 1.05, 64).difference(star_poly.buffer(-w * 0.2)))
    return g


# ---------------------------------------------------------------- meander / fret ring

FRET_CELLS = {
    "key": [[(0.0, 0.0), (0.0, 0.85), (0.72, 0.85), (0.72, 0.3), (0.32, 0.3), (0.32, 0.58)]],
    "key2": [[(0.0, 0.0), (0.0, 0.9), (0.8, 0.9), (0.8, 0.2), (0.25, 0.2), (0.25, 0.7), (0.58, 0.7), (0.58, 0.42)]],
    "T": [[(0.1, 0.0), (0.1, 0.8), (0.5, 0.8)], [(0.9, 1.0), (0.9, 0.2), (0.5, 0.2)]],
    "battlement": [[(0.0, 0.0), (0.0, 0.8), (0.5, 0.8), (0.5, 0.0)]],
}


def fret(kind="key", n=16, r=1.0, h=0.2, w=0.035, rails="in"):
    """Band of fret cells between r-h and r. rails: in | both (never none - see module doc)."""
    r_in = r - h
    lines = []
    for c in range(n):
        for pl in FRET_CELLS[kind]:
            lines.append([polar(r_in + v * h, (c + u) / n * 360) for u, v in pl])
    g = _strokes(lines, w)
    th = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    for rr in ([r_in] if rails == "in" else [r_in, r]):
        pts = np.column_stack([rr * np.sin(th), rr * np.cos(th)])
        g = g.union(LineString(np.vstack([pts, pts[:1]])).buffer(w / 2, 8))
    return g


# ---------------------------------------------------------------- Celtic knot rosette

KNOTS = {"7/4": (7, 4, 2), "8/3": (8, 3, 2), "9/4": (9, 4, 2), "loops6": (6, 5, 3)}   # hypotrochoid R, r, d


def _hypotrochoid(R, rr, d, samples=2400):
    turns = rr // math.gcd(R, rr)
    t = np.linspace(0, 2 * np.pi * turns, samples, endpoint=False)
    P = np.column_stack([(R - rr) * np.cos(t) + d * np.cos((R - rr) / rr * t),
                         (R - rr) * np.sin(t) - d * np.sin((R - rr) / rr * t)])
    P = P / np.abs(P).max()
    # rotate so a lobe points up (angle 0 = 12 o'clock)
    a = math.atan2(P[np.argmax(np.hypot(*P.T))][0], P[np.argmax(np.hypot(*P.T))][1])
    c, s = math.cos(a), math.sin(a)
    return P @ np.array([[c, -s], [s, c]]).T


def knot_rosette(kind="7/4", r=1.0, w=0.07, gap=None):
    """Closed interlaced curve with alternating over/under crossings (always consistent for a closed curve)."""
    gap = w * 0.5 if gap is None else gap
    P = _hypotrochoid(*KNOTS[kind]) * r
    n = len(P)
    segs = [LineString([P[i], P[(i + 1) % n]]) for i in range(n)]
    tree = STRtree(segs)
    cross = []
    for i, s in enumerate(segs):
        for j in tree.query(s):
            j = int(j)
            if j <= i + 1 or (i == 0 and j == n - 1):
                continue
            ip = s.intersection(segs[j])
            if ip.geom_type == "Point":
                cross.append((i, j, ip))
    passes = sorted([(i, k, 0) for k, (i, j, _) in enumerate(cross)] + [(j, k, 1) for k, (i, j, _) in enumerate(cross)])
    over = {}
    for idx, (_, k, which) in enumerate(passes):
        if k not in over:
            over[k] = which if idx % 2 == 0 else 1 - which
    out = LineString(np.vstack([P, P[:1]])).buffer(w / 2, 8)
    for k, (i, j, ip) in enumerate(cross):
        seg = (i, j)[over[k]]
        local = LineString(P[[q % n for q in range(seg - 12, seg + 13)]]).buffer(w / 2, 8)
        piece = local.intersection(ip.buffer(w * 2.0, 16))
        out = out.difference(piece.buffer(gap, 8)).union(piece)
    return out
