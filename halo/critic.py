"""Measured-style critic (thresholds from research/STYLE.md, BA p10..p90).

score(spec) -> (ok, metrics, reasons)
"""
from __future__ import annotations

import itertools
import math

from shapely.geometry import Point

from .build import Builder, _op
from .render import polygons

COVERAGE = (0.10, 0.75)          # ink / disc of the outer radius (BA p10 .13, p90 .50; solid kamon-style emblems reach ~.7)
HERO_MIN_SHARE = 0.15            # hero (+support) ink vs all ink — rejects "ring with tiny things"
FRAME_MAX_SHARE = 0.72
SPECK = 0.004                    # piece area / total ink below which a piece is a speck
NEAR_MISS = (0.012, 0.03)
COLLIDE = 0.08                   # overlap / smaller element area above which two separate elements "collide"        # gap band (×R_ext) that is neither a separation line nor open space


def outer_radius(g):
    x0, y0, x1, y1 = g.bounds
    return max(math.hypot(x, y) for x, y in [(x0, y0), (x0, y1), (x1, y0), (x1, y1)]) / math.sqrt(2) * 1.0


def score(spec: dict):
    b = Builder(spec)
    g = b.build()
    reasons = []
    if g.is_empty:
        return False, {}, ["empty"], g
    # outer radius = farthest ink point
    R = max(Point(0, 0).distance(Point(c)) for p in polygons(g) for c in p.exterior.coords)
    ink = g.area
    cov = ink / (math.pi * R * R)
    roles = {k: v.area / ink for k, v in b.role_geoms.items()}
    frame = roles.get("frame", 0.0)
    heroish = roles.get("hero", 0.0) + roles.get("support", 0.0)
    # traced crests carry deliberate kamon separation lines: close them before judging gaps / specks
    g_check = g
    lib = getattr(b, "lib_geom", None)
    if lib is not None and not lib.is_empty:
        close = 0.035 * R
        g_check = g.union(lib.buffer(close, 16).buffer(-close, 16).intersection(lib.convex_hull))
    pieces = polygons(g_check)
    specks = sum(1 for p in pieces if p.area / ink < SPECK)
    # near-miss: closest gap between distinct pieces
    gaps = []
    if 1 < len(pieces) <= 40:
        for a, c in itertools.combinations(pieces, 2):
            d = a.distance(c) / R
            if d > 0:
                gaps.append(d)
    near = [d for d in gaps if NEAR_MISS[0] < d < NEAR_MISS[1]]
    m = {"coverage": round(cov, 3), "frame_share": round(frame, 3), "hero_share": round(heroish, 3),
         "pieces": len(pieces), "specks": specks, "near_miss": len(near), "R": round(R, 2)}
    if not COVERAGE[0] <= cov <= COVERAGE[1]:
        reasons.append(f"coverage {cov:.2f}")
    if "hero" in b.role_geoms and heroish < HERO_MIN_SHARE:
        reasons.append(f"hero too weak {heroish:.2f}")
    if frame > FRAME_MAX_SHARE:
        reasons.append(f"frame dominates {frame:.2f}")
    if specks:
        reasons.append(f"{specks} specks")
    if near:
        reasons.append(f"{len(near)} near-miss gaps")
    # an idea = a distinct shape family; knocked-out motifs count, small punched holes do not
    ideas = {n["shape"] for n in _walk(spec.get("root", []))
             if not (_op_name(n) == "subtract" and n["shape"] == "dot")}
    m["ideas"] = len(ideas)
    # elements the generator flags as "must stay separate" may not run into each other (taste review:
    # polygon frames with elements inside read as clutter when they overlap; elsewhere overlap is often intended)
    rg = b.role_geoms
    for r1, r2 in (spec.get("separate") or ()):
        if r1 in rg and r2 in rg and not rg[r1].is_empty and not rg[r2].is_empty:
            ov = rg[r1].intersection(rg[r2]).area / max(min(rg[r1].area, rg[r2].area), 1e-9)
            if ov > COLLIDE:
                reasons.append(f"elements collide {r1}/{r2} {ov:.2f}")
                break
    if m["ideas"] < 2 and not spec.get("allow_single"):
        reasons.append("single idea")
    return not reasons, m, reasons, g


def _walk(nodes):
    for n in nodes:
        yield n
        yield from _walk(n.get("children", []) or [])


def _op_name(n):
    return _op(n.get("combine", "union"))[0]
