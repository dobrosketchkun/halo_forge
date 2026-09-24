"""Interpret a halo spec (grammar v2 subset) into a single shapely geometry.

Supported now:
  sym: D|C|mirror_v|mirror_h|none  (unified with repeat: root nodes default to `repeat: group`)
  place: polar | cartesian | attach | span
  repeat: group | none | orbit | mirror_pair | tiers | row   (+ `group: true` to also apply the symmetry group)
  alternate / break (per-instance field overrides, `drop: true`)
  combine: union(fillet) | subtract | overlay(gap) | interrupt(clear) | underlay(gap) | intersect | exclusive
  render: fill | line (per node, or global)
"""
from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass

import numpy as np
from shapely import affinity
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .geom import EMPTY, Shape, make_shape, polar, vec_ang

GAP = {"hairline": 0.012, "thin": 0.02, "medium": 0.035, "wide": 0.06}


# ---------------------------------------------------------------- affine helpers (3x3, y-up, angles clockwise)

def T(x, y):
    return np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], float)


def Rot(deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    # clockwise rotation in y-up coordinates
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], float)


MIRROR_X = np.diag([-1.0, 1.0, 1.0])
MIRROR_Y = np.diag([1.0, -1.0, 1.0])


def apply_geom(g: BaseGeometry, M) -> BaseGeometry:
    return affinity.affine_transform(g, [M[0, 0], M[0, 1], M[1, 0], M[1, 1], M[0, 2], M[1, 2]])


def apply_feat(f, M):
    x, y, a = f
    p = M @ np.array([x, y, 1.0])
    dx, dy = math.sin(math.radians(a)), math.cos(math.radians(a))
    d = M[:2, :2] @ np.array([dx, dy])
    return float(p[0]), float(p[1]), vec_ang(d[0], d[1])


def group(sym) -> list[np.ndarray]:
    if not sym:
        return [np.eye(3)]
    kind = sym.get("kind", "none")
    n = int(sym.get("n", 1))
    if kind == "none":
        return [np.eye(3)]
    if kind == "mirror_v":
        return [np.eye(3), MIRROR_X]
    if kind == "mirror_h":
        return [np.eye(3), MIRROR_Y]
    rots = [Rot(k * 360.0 / n) for k in range(n)]
    if kind == "C":
        return rots
    if kind == "D":
        return rots + [R @ MIRROR_X for R in rots]
    raise ValueError(f"unknown sym {sym}")


# ---------------------------------------------------------------- instances

@dataclass
class Inst:
    M: np.ndarray
    shape: Shape
    index: int

    def features(self, ref: str):
        m = re.fullmatch(r"(\w+)(?:\(([^)]*)\))?", ref.strip())
        if not m:
            raise ValueError(f"bad feature ref {ref!r}")
        name, args = m.group(1), [a.strip() for a in (m.group(2) or "").split(",") if a.strip()]
        return [apply_feat(f, self.M) for f in self.shape.feature(name, args)]


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def orient_angle(orient, phi):
    if orient is None:
        return None
    if isinstance(orient, (int, float)):
        return float(orient)
    o = str(orient)
    m = re.fullmatch(r"(\w+)\s*([+-]\s*\d+(?:\.\d+)?)?", o.replace("+-", "-"))
    base, extra = m.group(1), float((m.group(2) or "0").replace(" ", ""))
    table = {"radial_out": phi, "radial_in": phi + 180, "tangent_cw": phi + 90, "tangent_ccw": phi - 90,
             "upright": 0.0, "follow": phi, "normal": phi, "tangent": phi + 90, "down": 180.0}
    return table[base] + extra


class Builder:
    def __init__(self, spec: dict):
        self.spec = spec
        self.G = group(spec.get("sym"))
        self.registry: dict[str, list[Inst]] = {}

    # -- placement -> list of base transforms (before repeat)
    def base_transforms(self, node, parent: Inst | None):
        place = node.get("place") or {"polar": {"r": 0, "phase": 0}}
        (kind, P), = place.items()
        own_phase = float(node.get("phase", 0))
        if kind == "polar":
            r, phi = float(P.get("r", 0)), float(P.get("phase", 0))
            o = orient_angle(P.get("orient", "radial_out" if r > 0 else "upright"), phi)
            M = T(*polar(r, phi)) @ Rot(o + own_phase)
            return [parent.M @ M if parent else M], False
        if kind == "cartesian":
            o = orient_angle(P.get("orient", "upright"), 0.0)
            M = T(float(P.get("x", 0)), float(P.get("y", 0))) @ Rot(o + own_phase)
            return [parent.M @ M if parent else M], False
        if kind == "attach":
            targets = self._targets(P.get("to"), parent)
            out = []
            for tgt in targets:
                for (x, y, a) in tgt.features(P.get("at", "center")):
                    o = orient_angle(P.get("orient", "follow"), a)
                    off = P.get("offset") or {}
                    ax, ay = math.sin(math.radians(a)), math.cos(math.radians(a))
                    x2 = x + ax * off.get("along", 0) + ay * off.get("normal", 0)
                    y2 = y + ay * off.get("along", 0) - ax * off.get("normal", 0)
                    out.append(T(x2, y2) @ Rot(o + own_phase))
            return out, True
        if kind == "span":
            tgt = self._targets(P.get("to_node") or P.get("of"), parent)
            out = []
            for t in tgt:
                (x0, y0, _), = t.features(P["from"])[:1]
                (x1, y1, _), = t.features(P["to"])[:1]
                ang = vec_ang(x1 - x0, y1 - y0)
                inset = float(P.get("inset", 0))
                L = math.hypot(x1 - x0, y1 - y0) - 2 * inset
                ux, uy = (x1 - x0) / (L + 2 * inset), (y1 - y0) / (L + 2 * inset)
                M = T(x0 + ux * inset, y0 + uy * inset) @ Rot(ang + own_phase)
                M = M.copy()
                M_len = L
                out.append((M, M_len))
            node["_span_len"] = [l for _, l in out]
            return [m for m, _ in out], True
        raise ValueError(f"unknown place {kind}")

    def _targets(self, to, parent):
        if to in (None, "parent"):
            if parent is None:
                raise ValueError("attach without parent needs `to`")
            return [parent]
        insts = self.registry.get(to)
        if insts is None:
            raise KeyError(f"attach target {to!r} not built yet (order matters)")
        if parent is not None and len(insts) > parent.index and to != "all":
            return [insts[parent.index]]
        return insts

    def transforms(self, node, parent):
        bases, from_attach = self.base_transforms(node, parent)
        rep = node.get("repeat", "none" if (parent is not None or from_attach) else "group")
        if isinstance(rep, str):
            rep = {rep: {}}
        (rk, RP), = rep.items() if isinstance(rep, dict) and len(rep) == 1 else (("custom", rep),)
        RP = RP or {}
        out = []
        pre = parent.M if (parent is not None and not from_attach) else np.eye(3)
        pre_inv = np.linalg.inv(pre)
        for B in bases:
            local = pre_inv @ B   # transform relative to parent frame
            if rk == "group":
                out += [pre @ g @ local for g in self.G]
            elif rk == "none":
                out.append(B)
            elif rk == "orbit":
                n = int(RP.get("n", 4))
                skip = set(RP.get("skip", []) or [])
                span = RP.get("span")
                for k in range(n):
                    if k in skip:
                        continue
                    ang = (span[0] + (span[1] - span[0]) * k / max(n - 1, 1)) if span else k * 360.0 / n
                    out.append(pre @ Rot(ang) @ local)
            elif rk == "mirror_pair":
                out += [pre @ local, pre @ MIRROR_X @ local]
            elif rk == "tiers":
                for tier in (RP if isinstance(RP, list) else RP.get("list", [])):
                    sc = float(tier.get("scale", 1))
                    tl = T(*polar(tier["r"], tier.get("phase", 0))) @ Rot(orient_angle(tier.get("orient", "radial_out"), tier.get("phase", 0))) \
                        if "r" in tier else T(tier.get("x", 0), tier.get("y", 0))
                    out.append(pre @ tl @ np.diag([sc, sc, 1]))
            elif rk == "row":
                n, gap = int(RP.get("n", 3)), float(RP.get("gap", 0.1))
                for k in range(n):
                    out.append(pre @ local @ T((k - (n - 1) / 2) * gap, 0))
            else:
                raise ValueError(f"unknown repeat {rk}")
        if isinstance(RP, dict) and RP.get("group") and rk != "group":
            out = [g @ M for M in out for g in self.G] if parent is None else out
        return out

    # -- build
    def build_node(self, node, parent: Inst | None = None) -> BaseGeometry:
        node = dict(node)
        Ms = self.transforms(node, parent)
        span_lens = node.pop("_span_len", None)
        alts = node.get("alternate") or []
        breaks = {}
        for b in node.get("break") or []:
            breaks.setdefault(b["instance"], {}).update(b.get("set", {}))
        geoms, insts = [], []
        for i, M in enumerate(Ms):
            n_i = node
            if alts:
                n_i = deep_merge(n_i, alts[i % len(alts)])
            if i in breaks:
                n_i = deep_merge(n_i, breaks[i])
            if n_i.get("drop"):
                continue
            if span_lens:
                n_i = deep_merge(n_i, {"size": {"len": span_lens[i % len(span_lens)]}})
            if "scale" in n_i:
                M = M @ np.diag([float(n_i["scale"]), float(n_i["scale"]), 1])
            if "phase_extra" in n_i:
                M = M @ Rot(float(n_i["phase_extra"]))
            shape = make_shape(n_i)
            if n_i["shape"] in ("library", "plait", "ornament"):   # crests / weaves: their inner gaps are intended (critic)
                self.lib_geom = getattr(self, "lib_geom", EMPTY).union(apply_geom(shape.geom, M))
            inst = Inst(M, shape, i)
            insts.append(inst)
            g = apply_geom(shape.geom, M)
            # register before children so siblings/children can reference it
            self.registry.setdefault(node.get("id", f"_{id(node)}"), [])
            for child in n_i.get("children", []) or []:
                cg = self.build_node(child, inst)
                g = combine(g, cg, child.get("combine", "union"))
            if n_i.get("render") == "line":
                g = g.boundary.buffer(float(n_i.get("line_weight", 0.04)) / 2, 32)
            geoms.append(g)
        if node.get("id"):
            self.registry[node["id"]] = insts
        return unary_union(geoms) if geoms else EMPTY

    def build(self) -> BaseGeometry:
        acc = EMPTY
        self.role_geoms: dict[str, BaseGeometry] = {}
        for node in self.spec.get("root", []):
            g = self.build_node(node)
            acc = combine(acc, g, node.get("combine", "union"))
            role = node.get("role")
            if role and _op(node.get("combine", "union"))[0] != "subtract":
                self.role_geoms[role] = self.role_geoms.get(role, EMPTY).union(g)
        if self.spec.get("render") == "line":
            acc = acc.boundary.buffer(float(self.spec.get("line_weight", 0.04)) / 2, 32)
        return acc.buffer(0)


def _op(combine_spec):
    if isinstance(combine_spec, str):
        return combine_spec, {}
    (k, v), = combine_spec.items()
    return k, (v or {})


def combine(acc: BaseGeometry, g: BaseGeometry, spec) -> BaseGeometry:
    op, P = _op(spec)
    gap = P.get("gap", P.get("clear", "thin"))
    gap = GAP.get(gap, gap) if isinstance(gap, str) else float(gap)
    if op == "union":
        out = acc.union(g)
        f = float(P.get("fillet", 0))
        return out.buffer(f, 32).buffer(-f, 32) if f else out
    if op == "subtract":
        return acc.difference(g)
    if op in ("overlay", "interrupt"):
        return acc.difference(g.buffer(gap, 32)).union(g)
    if op == "underlay":
        return acc.union(g.difference(acc.buffer(gap, 32)))
    if op == "intersect":
        return acc.intersection(g)
    if op == "exclusive":
        return acc.symmetric_difference(g)
    raise ValueError(f"unknown combine {op}")


def build(spec: dict) -> BaseGeometry:
    return Builder(spec).build()
