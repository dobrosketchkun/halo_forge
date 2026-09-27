"""2.5D staging of a flat halo: split the flat design into planar parts placed in 3D, then project / export.

Halo local space: x right, y = "up" of the flat icon (12 o'clock), z = halo normal (up, away from the head).
A halo floating above a head is this plane seen from slightly above. Every part stays an exact planar vector
shape (shapely polygon) with a 4x4 placement, so any view is an exact projection at any resolution.

Staging = ordered list of operations; each takes a region of the still-unassigned flat geometry:
  {"sel": {...}, "act": {...}}
  sel:  "sector": [center_deg, half_width_deg]   (0 = 12 o'clock, clockwise)
        "r": [rmin, rmax]                        (fractions of the outer radius R)
        "mode": "clip" (cut the region out) | "comp" (take whole connected pieces whose centre lies in the region)
        "max_area": fraction of total ink (comp mode: only small pieces)
        "all": true                              (everything that is left)
  act:  "z": dz                                  (parallel offset along the normal, fraction of R; negative = down)
        "upright": deg, "hinge_r": frac          (rotate outward-from-hinge up out of the plane; negative = hang)
        "billboard": true                        (keeps facing the camera)
        "thick": t                               (extruded along the normal, fraction of R)
        "fan": deg                               (each piece rotated about its own radial axis, alternating sign)
Pre-op "untilt": y-scale factor for source art that is already drawn in perspective.
"""
from __future__ import annotations

import hashlib
import json
import math
import random

import numpy as np
from shapely import affinity
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from .render import polygons

# typical viewing positions: (azimuth deg around the normal, 0 = from the front / 6 o'clock side;
#                             elevation deg above the halo plane; roll deg)
VIEWS = {
    "top": (0.0, 90.0, 0.0),        # flat icon
    "front": (0.0, 35.0, 0.0),      # halo above a head seen from the front (tilt 55 deg, BA median)
    "iso": (45.0, 35.264, 0.0),     # isometric
    "side": (60.0, 25.0, 0.0),      # three-quarter side
    "edge": (0.0, 12.0, 0.0),       # almost edge-on
}


# ---------------------------------------------------------------- geometry helpers

def outer_radius(g):
    return max((math.hypot(*c) for p in polygons(g) for c in p.exterior.coords), default=1.0)


def _wedge(center, half, R):
    if half >= 180:
        return Point(0, 0).buffer(R, 128)
    a = np.radians(np.linspace(center - half, center + half, 64))
    pts = [(0, 0)] + [(R * math.sin(t), R * math.cos(t)) for t in a]
    return Polygon(pts)


def _region(sel, R):
    reg = Point(0, 0).buffer(R * 1.5, 128)
    if "sector" in sel:
        c, h = sel["sector"]
        reg = reg.intersection(_wedge(c, h, R * 1.5))
    if "r" in sel:
        r0, r1 = sel["r"]
        ring = Point(0, 0).buffer(r1 * R, 128)
        if r0 > 0:
            ring = ring.difference(Point(0, 0).buffer(r0 * R, 128))
        reg = reg.intersection(ring)
    return reg


def _main_radius(g):
    """Outer radius of the largest piece (the main ring / body): unit for stacking selections."""
    ps = polygons(g)
    if not ps:
        return 1.0
    big = max(ps, key=lambda p: p.area)
    return max(math.hypot(*c) for c in big.exterior.coords)


def _isolated(p, g, R, gap=0.035):
    """True if the piece is clearly separated from all other ink (a free sprite, not a segment of a woven /
    broken ring separated only by hairline gaps)."""
    others = g.difference(p.buffer(1e-6))
    return others.is_empty or p.distance(others) > gap * R


def _select(rest, sel, R, total, Rm=None):
    if sel.get("all") or sel.get("rest"):
        return rest
    if sel.get("rel") == "main" and Rm:
        R = Rm
    reg = _region(sel, R)
    if sel.get("mode", "clip") == "clip":
        return rest.intersection(reg)
    pick = []
    for p in polygons(rest):
        if sel.get("max_area") is not None and p.area > sel["max_area"] * total:
            continue
        if sel.get("isolated") and not _isolated(p, rest, R):
            continue
        if sel.get("compact"):
            x0, y0, x1, y1 = p.bounds
            w, h = x1 - x0, y1 - y0
            if not (w > 0 and h > 0 and 0.6 < w / h < 1.67 and p.area / (w * h) > 0.17):
                continue
        if reg.contains(p.representative_point()):
            pick.append(p)
    return unary_union(pick)


def _rot(axis, deg):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def _about(axis, deg, point):
    M = np.eye(4)
    Rm = _rot(axis, deg)
    p = np.asarray(point, float)
    M[:3, :3] = Rm
    M[:3, 3] = p - Rm @ p
    return M


def _up_about_hinge(theta, hinge, deg):
    """Rotate so the outward radial direction at angle theta tilts toward +z by deg, about the tangent line at hinge."""
    u = np.array([math.sin(math.radians(theta)), math.cos(math.radians(theta)), 0.0])
    k = np.cross(u, [0.0, 0.0, 1.0])
    return _about(k, deg, u * hinge)


# ---------------------------------------------------------------- staging

def apply(geom, ops, untilt=1.0):
    """-> list of parts {geom, M (4x4), billboard, thick}"""
    if untilt != 1.0:
        geom = affinity.scale(geom, 1.0, untilt, origin=(0, 0))
    R = outer_radius(geom)
    Rm = _main_radius(geom)
    total = geom.area or 1.0
    rest, parts = geom, []
    for op in ops or []:
        sel, act = op.get("sel", {}), op.get("act", {})
        g = _select(rest, sel, R, total, Rm)
        if g.is_empty or g.area < 1e-5 * total:
            continue
        rest = rest.difference(g)
        if "fan" in act:
            for i, piece in enumerate(polygons(g)):
                c = piece.representative_point()
                theta = math.degrees(math.atan2(c.x, c.y))
                u = (math.sin(math.radians(theta)), math.cos(math.radians(theta)), 0.0)
                parts.append({"geom": piece, "M": _about(u, act["fan"] * (1 if i % 2 == 0 else -1), (0, 0, 0)),
                              "billboard": False, "thick": 0.0})
            continue
        if act.get("billboard"):   # every sprite is its own part, anchored where it sits on the halo
            for piece in polygons(g):
                parts.append({"geom": piece, "M": np.eye(4), "billboard": True, "thick": 0.0})
            continue
        M = np.eye(4)
        if "upright" in act:
            theta = sel.get("sector", [0, 0])[0]
            hinge = act.get("hinge_r", sel.get("r", [0.7, 1])[0]) * R
            M = _up_about_hinge(theta, hinge, act["upright"])
        if "z" in act:
            M = M.copy()
            M[2, 3] += act["z"] * R
        parts.append({"geom": g, "M": M, "billboard": bool(act.get("billboard")), "thick": act.get("thick", 0.0) * R})
    if not rest.is_empty and rest.area > 1e-6 * total:
        parts.insert(0, {"geom": rest, "M": np.eye(4), "billboard": False, "thick": 0.0})
    return parts


# ---------------------------------------------------------------- projection

def camera(az, el, roll=0.0):
    a, e = math.radians(az), math.radians(el)
    c = np.array([math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)])   # toward the camera
    r = np.array([math.cos(a), math.sin(a), 0.0])
    u = np.cross(c, r)
    if roll:
        q = math.radians(roll)
        r, u = r * math.cos(q) + u * math.sin(q), -r * math.sin(q) + u * math.cos(q)
    return r, u, c


def project(parts, az=0.0, el=90.0, roll=0.0):
    """Exact orthographic projection of all parts to 2D (shapely), same units as the flat design."""
    r, u, c = camera(az, el, roll)
    out = []
    for p in parts:
        g, M = p["geom"], p["M"]
        if p["billboard"]:
            cen = g.centroid
            a3 = M @ np.array([cen.x, cen.y, 0.0, 1.0])
            out.append(affinity.translate(g, a3[:3] @ r - cen.x, a3[:3] @ u - cen.y))
            continue
        zs = np.linspace(-p["thick"] / 2, p["thick"] / 2, 9) if p["thick"] > 0 else [0.0]
        for z in zs:
            A = M[:3, :3]
            t = M[:3, 3] + A[:, 2] * z
            # image = [r;u] . (A [x y 0] + t)
            a, b, d, e = r @ A[:, 0], r @ A[:, 1], u @ A[:, 0], u @ A[:, 1]
            proj = affinity.affine_transform(g, [a, b, d, e, float(r @ t), float(u @ t)])
            if proj.area < 1e-4 * max(g.area, 1e-9):   # plane seen edge-on: keep it visible as a thin line
                proj = proj.buffer(0.004 * outer_radius(g) + 1e-4)
            out.append(proj)
    return unary_union(out).buffer(0)


# ---------------------------------------------------------------- serialisation (web preview, JSON, glTF)

def _rings(poly):
    return [[[round(x, 5), round(y, 5)] for x, y in poly.exterior.coords[:-1]]] + \
           [[[round(x, 5), round(y, 5)] for x, y in h.coords[:-1]] for h in poly.interiors]


def to_json(parts, color):
    """Vector 3D description: every part = planar polygons (with holes) in local 2D + column-major 4x4 matrix."""
    return {"format": "halo3d/1", "color": color, "units": "halo radius ~1, z = halo normal",
            "parts": [{"polygons": [_rings(q) for q in polygons(p["geom"])],
                       "matrix": [round(v, 6) for v in np.asarray(p["M"]).T.reshape(-1)],
                       "billboard": p["billboard"], "thickness": round(p["thick"], 5)} for p in parts]}


def to_gltf(parts, color):
    """Minimal glTF 2.0 (embedded buffer): triangulated faces per part, extruded walls for thick parts."""
    import base64
    import struct

    import shapely
    rgb = [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    pos, idx, nodes, meshes, accessors, views = [], [], [], [], [], []
    blob = b""

    def add_mesh(verts, tris):
        nonlocal blob
        vb = struct.pack(f"<{len(verts) * 3}f", *[c for v in verts for c in v])
        ib = struct.pack(f"<{len(tris) * 3}I", *[i for t in tris for i in t])
        off = len(blob)
        blob += vb + ib
        views.extend([{"buffer": 0, "byteOffset": off, "byteLength": len(vb), "target": 34962},
                      {"buffer": 0, "byteOffset": off + len(vb), "byteLength": len(ib), "target": 34963}])
        vs = np.array(verts)
        accessors.extend([{"bufferView": len(views) - 2, "componentType": 5126, "count": len(verts), "type": "VEC3",
                           "min": vs.min(0).tolist(), "max": vs.max(0).tolist()},
                          {"bufferView": len(views) - 1, "componentType": 5125, "count": len(tris) * 3, "type": "SCALAR"}])
        meshes.append({"primitives": [{"attributes": {"POSITION": len(accessors) - 2}, "indices": len(accessors) - 1,
                                       "material": 0}]})
        return len(meshes) - 1

    for p in parts:
        verts, tris = [], []
        h = p["thick"] / 2
        for poly in polygons(p["geom"]):
            tri = shapely.constrained_delaunay_triangles(poly)
            for z in ([-h, h] if h > 0 else [0.0]):
                for t in tri.geoms:
                    base = len(verts)
                    verts += [(x, y, z) for x, y in list(t.exterior.coords)[:3]]
                    tris.append((base, base + 1, base + 2))
            if h > 0:
                for ring in [poly.exterior, *poly.interiors]:
                    cs = list(ring.coords)
                    for (x0, y0), (x1, y1) in zip(cs, cs[1:]):
                        b = len(verts)
                        verts += [(x0, y0, -h), (x1, y1, -h), (x1, y1, h), (x0, y0, h)]
                        tris += [(b, b + 1, b + 2), (b, b + 2, b + 3)]
        if not tris:
            continue
        mi = add_mesh(verts, tris)
        node = {"mesh": mi, "matrix": [float(v) for v in np.asarray(p["M"]).T.reshape(-1)]}
        if p["billboard"]:
            node["extras"] = {"billboard": True}
        nodes.append(node)
    root = {"children": list(range(1, len(nodes) + 1)), "name": "halo",
            "rotation": [-0.7071068, 0, 0, 0.7071068]}   # halo normal (local z) -> glTF +Y up
    gltf = {"asset": {"version": "2.0", "generator": "Halo Forge"}, "scene": 0, "scenes": [{"nodes": [0]}],
            "nodes": [root] + nodes, "meshes": meshes, "accessors": accessors, "bufferViews": views,
            "buffers": [{"byteLength": len(blob),
                         "uri": "data:application/octet-stream;base64," + base64.b64encode(blob).decode()}],
            "materials": [{"pbrMetallicRoughness": {"baseColorFactor": rgb + [1.0], "metallicFactor": 0.0},
                           "emissiveFactor": rgb, "doubleSided": True}]}
    return json.dumps(gltf)


# ---------------------------------------------------------------- staging choice for generated halos

def _rng(seed):
    return random.Random(int.from_bytes(hashlib.blake2b(f"stage:{seed}".encode(), digest_size=8).digest(), "big"))


def choose(geom, desc, seed):
    """Sample a staging for a generated halo, with the frequencies measured on official halos
    (research: .ignore/research/depth/DEPTH.md). Returns (ops, label)."""
    rng = _rng(seed)
    R = outer_radius(geom)
    total = geom.area or 1.0
    pieces = polygons(geom)
    Rm = _main_radius(geom)
    inner = [p for p in pieces if max(math.hypot(*c) for c in p.exterior.coords) < 0.85 * Rm]
    top = geom.intersection(_region({"sector": [0, 22], "r": [0.72, 1.01]}, R))
    has_top = top.area > 0.006 * total
    def compact(p):   # sparkle / dot / diamond-like: fills a good share of its (roughly square) bounding box
        x0, y0, x1, y1 = p.bounds
        w, h = x1 - x0, y1 - y0
        return w > 0 and h > 0 and 0.6 < w / h < 1.67 and p.area / (w * h) > 0.17
    small = [p for p in pieces if p.area < 0.025 * total and compact(p)
             and p.representative_point().distance(Point(0, 0)) > 0.3 * R and _isolated(p, geom, R)]
    t = desc.get("skeleton", "")
    weights = {"flat": 59, "crossed": 15 if has_top else 0, "stacked": 13 if inner else 0,
               "solid": 4 if t in ("reticle", "segmented", "crescent", "flanked_ring") else 0,
               "fan": 3 if t in ("pinwheel", "flower", "star") else 0}
    kind = rng.choices(list(weights), weights=list(weights.values()))[0]
    ops = []
    if kind == "crossed":
        ang = rng.choice([60, 75, 90])
        ops.append({"sel": {"sector": [0, 22], "r": [0.72, 1.01]}, "act": {"upright": ang, "hinge_r": 0.72}})
        label = f"crossed {ang}"
    elif kind == "stacked":
        d = rng.choice([0.14, 0.2, 0.26])
        ops.append({"sel": {"r": [0, 0.5], "mode": "comp", "rel": "main"}, "act": {"z": -2 * d}})
        ops.append({"sel": {"r": [0, 0.85], "mode": "comp", "rel": "main"}, "act": {"z": -d}})
        label = f"stacked {d}"
    elif kind == "solid":
        th = rng.choice([0.08, 0.12])
        ops.append({"sel": {"rest": True}, "act": {"thick": th}})
        label = f"solid {th}"
    elif kind == "fan":
        ang = rng.choice([25, 35])
        ops.append({"sel": {"r": [0.15, 1.01], "mode": "comp"}, "act": {"fan": ang}})
        label = f"fan {ang}"
    else:
        label = "flat"
    woven = any(str(v).startswith(("plait", "fret_", "loops")) for v in desc.values())
    if 1 <= len(small) <= 8 and not woven and rng.random() < 0.35:
        ops.insert(0, {"sel": {"r": [0.3, 1.5], "mode": "comp", "max_area": 0.025, "compact": True, "isolated": True},
                       "act": {"billboard": True}})
        label += " +billboard"
    return ops, label


# ================================================================ element-aware staging (generated halos)
# The generator knows what every piece is (frame, main-element copies, supports, centre, markers). Staging moves
# WHOLE elements / copies and hinges them where they attach, so frames are never cut and nothing is split.

from shapely.ops import nearest_points

ROLE_ORDER = ["frame", "center", "support", "hero"]      # who owns overlapping ink (frame first: never cut)
KINDS = {  # arrangement -> weight (flat reduced from the measured 59% to make room for the richer arrangements)
    "flat": 40, "stand_up": 13, "stacked": 12, "wings_out": 8, "hang": 6, "crown": 6, "orbit": 6,
    "counter_tilt": 5, "lotus": 4,
}


APPLICABLE = {"stand_up": 0.32, "stacked": 0.93, "hang": 0.29, "crown": 0.27, "lotus": 0.38,
              "counter_tilt": 0.19, "wings_out": 0.38, "orbit": 0.2}   # measured on 300 generated halos


def _angle(g):
    c = g.representative_point()
    return math.degrees(math.atan2(c.x, c.y)) % 360.0


def _adiff(a, b):
    return abs((a - b + 180) % 360 - 180)


def own_parts(G, elements):
    """Split the finished flat geometry G among the elements' copies. Returns parts with role / element / copy."""
    rest = G
    parts = []
    order = sorted(elements, key=lambda e: ROLE_ORDER.index(e["role"]) if e["role"] in ROLE_ORDER else 9)
    for e in order:
        for i, (M, g) in enumerate(e["insts"]):
            piece = rest.intersection(g)
            if piece.is_empty or piece.area < 1e-6:
                continue
            rest = rest.difference(piece)
            parts.append({"geom": piece, "M": np.eye(4), "billboard": False, "thick": 0.0,
                          "role": e["role"], "elem": e["k"], "shape": e["shape"], "inst": i,
                          "n_inst": len(e["insts"]), "node": e["node"], "angle": _angle(piece),
                          "rmin": piece.distance(Point(0, 0)),
                          "rmax": max((math.hypot(*c) for q in polygons(piece) for c in q.exterior.coords), default=0)})
    if not rest.is_empty and rest.area > 1e-6:
        parts.append({"geom": rest, "M": np.eye(4), "billboard": False, "thick": 0.0, "role": "frame", "elem": -1,
                      "shape": "rest", "inst": 0, "n_inst": 1, "node": {}, "angle": 0.0, "rmin": 0.0, "rmax": 0.0})
    return parts


def _hinged(part, deg):
    """Rotate a part about the tangent line through its point nearest the centre (where it attaches),
    tilting its outward direction toward +z by deg (negative = hang down)."""
    g = part["geom"]
    m = nearest_points(g, Point(0, 0))[0]
    theta = part["angle"]
    u = np.array([math.sin(math.radians(theta)), math.cos(math.radians(theta)), 0.0])
    k = np.cross(u, [0.0, 0.0, 1.0])
    return _about(k, deg, (m.x, m.y, 0.0))


def _crown(part, height, Rm):
    """Replace a flat ring frame by a vertical band (a crown / short cylinder wall) of the given height."""
    node = part["node"]
    r = float(node.get("r", 1.0)) if node.get("shape") == "circle" else Rm
    K = 72
    seg = 2 * math.pi * r / K * 1.04
    out = []
    probe = part["geom"].buffer(0.02 * Rm)
    for i in range(K):
        th = 2 * math.pi * (i + 0.5) / K
        if not probe.contains(Point(r * math.sin(th), r * math.cos(th))):
            continue   # keep the frame's gaps
        t = np.array([math.cos(th), -math.sin(th), 0.0])
        z = np.array([0.0, 0.0, 1.0])
        n = np.cross(t, z)
        M = np.eye(4)
        M[:3, 0], M[:3, 1], M[:3, 2] = t, z, n
        M[:3, 3] = [r * math.sin(th), r * math.cos(th), 0.0]
        out.append({"geom": Polygon([(-seg / 2, 0), (seg / 2, 0), (seg / 2, height), (-seg / 2, height)]),
                    "M": M, "billboard": False, "thick": 0.0, "role": "frame", "elem": part["elem"], "shape": "band",
                    "inst": i, "n_inst": K, "node": node, "angle": math.degrees(th), "rmin": r, "rmax": r})
    return out


def choose_elements(G, elements, desc, seed):
    """Sample an arrangement for a generated halo and return (parts, label)."""
    rng = _rng(seed)
    parts = own_parts(G, elements)
    R = outer_radius(G)
    frames = [p for p in parts if p["role"] == "frame" and p["shape"] != "rest"]
    Rm = max((p["rmax"] for p in frames), default=R) or R
    movable = [p for p in parts if p["role"] != "frame"]

    top = [p for p in movable if _adiff(p["angle"], 0) < 15 and p["rmin"] > 0.45 * Rm and p["n_inst"] <= 8]
    bottom = [p for p in movable if _adiff(p["angle"], 180) < 20 and p["rmin"] > 0.55 * Rm]
    sides = [p for p in movable if (_adiff(p["angle"], 90) < 35 or _adiff(p["angle"], 270) < 35) and p["rmin"] > 0.55 * Rm]
    elem_rmax = {}
    for p in parts:
        elem_rmax[p["elem"]] = max(elem_rmax.get(p["elem"], 0), p["rmax"])
    inner = [p for p in parts if elem_rmax[p["elem"]] < 0.85 * Rm and p["shape"] != "rest"]   # whole elements only
    ring_frame = [p for p in frames if p["shape"] == "circle" and p["node"].get("r", 1.0) > 0.8]
    orbiters = [p for p in movable if p["n_inst"] >= 3 and p["rmin"] > 0.75 * Rm]
    inner_rings = [p for p in parts if p["shape"] == "circle" and 0.2 * Rm < p["rmax"] < 0.85 * Rm]
    lotus = [p for p in movable if p["role"] == "hero" and p["n_inst"] >= 4]

    ok = {"flat": True, "stand_up": bool(top), "stacked": bool(inner), "wings_out": len(sides) >= 2,
          "hang": bool(bottom), "crown": bool(ring_frame), "orbit": len(orbiters) >= 3,
          "counter_tilt": bool(inner_rings), "lotus": len(lotus) >= 4}
    kinds = [k for k in KINDS if ok[k]]
    # target shares are overall shares: an arrangement that few halos can take gets a proportionally larger weight
    # when it is possible (APPLICABLE = measured fraction of halos where it applies); flat takes the remainder
    w = {k: 1.25 * KINDS[k] / APPLICABLE[k] for k in kinds if k != "flat"}   # 1.25: compensates overlap
    flat_w = max(100 - sum(w.values()), KINDS["flat"] * 0.25)
    kind = rng.choices(["flat"] + list(w), weights=[flat_w] + list(w.values()))[0]
    label = kind

    if kind == "stand_up":
        deg = rng.choice([60, 75, 90])
        for p in top:
            p["M"] = _hinged(p, deg)
        label = f"stand_up {deg}"
    elif kind == "stacked":
        # layers = groups of elements that touch each other (touching parts must stay together); one depth per layer,
        # ranked by size (smaller = lower, an inverted cone as in Hoshino / Asuna); total depth capped at 35% of R
        elems = sorted({p["elem"] for p in inner})
        geo = {e: unary_union([p["geom"] for p in inner if p["elem"] == e]) for e in elems}
        grp = {e: e for e in elems}

        def find(e):
            while grp[e] != e:
                e = grp[e]
            return e
        for i, a_ in enumerate(elems):
            for b_ in elems[i + 1:]:
                if geo[a_].distance(geo[b_]) < 0.01 * Rm:
                    grp[find(b_)] = find(a_)
        size = {}
        for p in inner:
            g_ = find(p["elem"])
            size[g_] = max(size.get(g_, 0), p["rmax"])
        levels = sorted({round(v / Rm, 1) for v in size.values()}, reverse=True)
        d = min(rng.choice([0.12, 0.18, 0.24]), 0.35 / len(levels)) * Rm
        for p in inner:
            step = levels.index(round(size[find(p["elem"])] / Rm, 1)) + 1
            p["M"] = p["M"].copy()
            p["M"][2, 3] -= d * step
        label = f"stacked {len(levels)}"
    elif kind == "wings_out":
        deg = rng.choice([40, 55, 70])
        for p in sides:
            p["M"] = _hinged(p, deg)
        label = f"wings_out {deg}"
    elif kind == "hang":
        for p in bottom:
            p["M"] = _hinged(p, -90)
        label = "hang"
    elif kind == "crown":
        h = rng.choice([0.16, 0.26]) * Rm
        keep = [p for p in parts if p not in ring_frame]
        band = [q for p in ring_frame for q in _crown(p, h, Rm)]
        for p in keep:   # everything else floats at mid-height of the band
            p["M"] = p["M"].copy()
            p["M"][2, 3] += h / 2
        parts = keep + band
        label = "crown"
    elif kind == "orbit":
        amp, phase = rng.choice([0.12, 0.2]) * Rm, rng.random() * 360
        for p in orbiters:
            p["M"] = p["M"].copy()
            p["M"][2, 3] += amp * math.sin(math.radians(2 * p["angle"] + phase))
        label = "orbit"
    elif kind == "counter_tilt":
        deg, axis_a = rng.choice([22, 32]), rng.choice([0, 45, 90])
        a = math.radians(axis_a)
        for p in inner_rings:
            p["M"] = _about((math.cos(a), -math.sin(a), 0.0), deg, (0, 0, 0))
        label = f"counter_tilt {deg}"
    elif kind == "lotus":
        deg = rng.choice([25, 35, 45])
        for p in lotus:
            p["M"] = _hinged(p, deg)
        label = f"lotus {deg}"

    # add-on: isolated small sprites (sparkles, dots) of unmoved supports / centres face the camera
    woven = any(str(v).startswith(("plait", "fret_", "loops")) for v in desc.values())
    total = G.area or 1.0
    sprites = []
    for p in parts:
        if p["role"] in ("support", "center") and np.allclose(p["M"], np.eye(4)):
            for q in polygons(p["geom"]):
                x0, y0, x1, y1 = q.bounds
                w, hh = x1 - x0, y1 - y0
                if (q.area < 0.025 * total and w > 0 and hh > 0 and 0.6 < w / hh < 1.67 and q.area / (w * hh) > 0.17
                        and q.distance(Point(0, 0)) > 0.3 * R and _isolated(q, G, R)):
                    sprites.append((p, q))
    if 1 <= len(sprites) <= 8 and not woven and rng.random() < 0.35:
        for p, q in sprites:
            p["geom"] = p["geom"].difference(q.buffer(1e-6))
            parts.append({"geom": q, "M": np.eye(4), "billboard": True, "thick": 0.0, "role": p["role"],
                          "elem": p["elem"], "shape": p["shape"], "inst": p["inst"], "n_inst": p["n_inst"],
                          "node": p["node"], "angle": _angle(q), "rmin": 0, "rmax": 0})
        label += " +billboard"
    parts = [p for p in parts if not p["geom"].is_empty and p["geom"].area > 1e-7]
    return parts, label
