"""Layered halo sampler: preset constraints x shared layers -> grammar spec (v2 subset).

Every halo is built from the same independent LAYERS, each a discrete, visible choice:
  global   : symmetry kind + order, stroke weight
  frame    : shape, treatment (closed / gapped at heroes / gapped between / broken), edge (plain / teeth / dots / spikes)
  hero     : family, placement, cardinal/diagonal phase, tilt, size, render (fill / outline / split)
  support  : family, placement
  center   : motif
  mods     : tip accent, emphasis (near-symmetry), alternating sizes
A PRESET (reticle, crest, flower...) only restricts which values each layer may take — so the type keeps
its character while the number of distinct combinations multiplies across layers.

Priors come from the BA decomposition (research/ba_decomp, research/STYLE.md).
Each node carries a `role` (frame / hero / support / center) so the critic can check hierarchy.
"""
from __future__ import annotations

# Bump whenever sampling changes (weights, pools, presets): the same seed then yields a different halo.
# Issued halos are frozen in the registry (halo/issue.py), so a bump never changes anyone's halo.
GENERATOR_VERSION = "2026.09.24-taste4"

import colorsys
import math

from .library import load_kamon_lib, load_lib
from .marks import random_mark

KAMON_ROLES = load_kamon_lib()   # curated traced public-domain kamon, by role ("kamon:<crest>#<item>")
KAMON_LIB = KAMON_ROLES["interior"]
KAMON_CENTERS = KAMON_ROLES["center"]
KAMON_PARTS = KAMON_ROLES["part"]
KAMON_LIB_CONTAINERS = KAMON_ROLES["container"]
HER = load_lib("heraldry_lib", "her")
# flank charges drawn with their base on the right: mirror them so they grow OUT of the halo (right-side copy)
FLANK_FLIP = {"her:ramsHorn"}   # curated Armoria charges ("her:<name>")
import random

WEIGHTS = {"thin": 0.075, "medium": 0.11, "thick": 0.16}     # main stroke, ×ring radius (STYLE.md)
SIZES = {"small": 0.3, "medium": 0.52, "large": 0.72}
WIDTH = {"spike": 0.34, "blade": 0.34, "kite": 0.5, "triangle": 0.75, "sparkle": 0.2, "arrow": 0.8,
         "diamond": 0.6, "petal": 0.6, "leaf": 0.42, "teardrop": 0.55, "dot": 1.0, "heart": 1.0,
         "horn": 0.3, "wing": 0.32, "fork": 0.6, "bar": 0.26, "shard": 0.45,
         "bolt": 0.28, "flame": 0.45, "hook": 0.22, "curl": 0.18, "fleur": 0.8, "drip": 0.3,
         "feather": 0.34, "branch": 0.3, "wave": 0.36, "eye": 0.55, "chevron": 0.8}
ROOTED = {"spike", "blade", "kite", "triangle", "arrow", "petal", "leaf", "teardrop", "horn", "fork", "shard",
          "bolt", "flame", "hook", "fleur", "feather", "bar", "wing", "curl", "wave", "drip", "chevron", "branch"}

POINTY = ["spike", "blade", "kite", "triangle", "arrow", "bar", "fleur", "fork", "feather", "diamond", "sparkle", "chevron"]
EARS = ["petal", "leaf", "wing", "spike", "sparkle", "teardrop", "horn", "blade", "feather", "flame", "curl", "wave",
        "hook", "branch"]
PETALS = ["petal", "leaf", "teardrop", "heart", "kite", "feather", "flame", "eye", "blade", "diamond"]
SWIRLS = ["horn", "blade", "leaf", "petal", "tomoe", "flame", "wave", "curl", "shard"]   # no bolt/hook: see forbidden()
SUPPORTS = ["dot", "sparkle", "spike", "teardrop", "bar", "diamond", "petal", "curl", "chevron", "triangle"]
CENTERS = ["empty", "ring", "dot", "sparkle", "diamond", "flower_round", "flower_pointed", "flower_notched",
           "flower_heart", "knock_ring", "eye", "star5", "cross"]
PETAL_TYPES = ["round", "pointed", "notched", "heart"]
LINES = ["wavy", "indented", "engrailed", "invected", "embattled", "nebuly", "raguly", "rayonny"]
EDGE_LINES = [f"L:{k}:{side}" for k in LINES for side in ("outer", "both")]   # heraldic lines of partition
RING_EDGES = ["plain"] * 10 + ["teeth", "dots", "spikes"] + EDGE_LINES

# preset -> allowed values per layer (+ BA-derived frequency)
PRESETS = {
    "reticle": dict(freq=32, sym=["D"], n=[4, 4, 4, 2, 3, 6, 8], frame=["ring", "double_ring", "thick_ring", "polygon", "plait2", "plait2_border"],
                    frame_mod=["closed", "closed", "gap_hero", "gap_between"], edge=RING_EDGES,
                    hero=POINTY, place=["crossing", "outside", "inward", "on_ring"], tilt=[0], support=["none", "inner_ring"] + SUPPORTS,
                    support_place=["between_out", "between_in"], center=CENTERS + ["mark"] * 3),
    "flanked_ring": dict(freq=16, sym=["mirror_v"], n=[1], frame=["ring", "double_ring", "thick_ring", "plait2", "plait2_wide", "plait3", "fret_key", "loops6"],
                         frame_mod=["closed", "closed", "gap_between"], edge=RING_EDGES,
                         hero=EARS, place=["flank_90", "flank_60", "flank_45", "flank_120", "flank_135"], tilt=[0, 15, 30, -15],
                         support=["none", "none"] + SUPPORTS, support_place=["crown", "base", "crown_base"], center=CENTERS * 6 + KAMON_CENTERS + HER["center"] + ["mark"] * 12),
    "crest": dict(freq=14, sym=["mirror_v"], n=[1], frame=["ring", "crescent", "none", "polygon", "thick_ring"],
                  frame_mod=["closed"], edge=["plain"], hero=["kite", "spike", "blade", "sparkle", "diamond", "fleur", "flame", "fork"],
                  place=["crown"], tilt=[0], support=["horn", "hook", "curl", "flame", "wing", "feather", "leaf", "spike", "wave"],
                  support_place=["flank_45", "flank_60", "flank_75", "flank_120"], center=CENTERS,
                  extra=dict(base=["none", "spike", "dot", "triangle", "drip", "teardrop", "fleur"])),
    "flower": dict(freq=22, sym=["D", "D", "C"], n=[5, 5, 6, 8, 3, 4, 7], frame=["none", "none", "ring", "polygon", "double_ring", "snow_ring", "mokko", "mokko5", "kikko", "plait2", "plait2_border"],
                   frame_mod=["closed"], edge=["plain"] * 6 + ["dots"] + EDGE_LINES, hero=PETALS, place=["radial_0", "radial_1"],
                   tilt=[0, 0, 25], support=["none", "none", "dot", "sparkle", "teardrop", "diamond", "spike"],
                   support_place=["between_out", "between_in"], center=CENTERS),
    "star": dict(freq=20, sym=["D"], n=[4, 4, 5, 6, 8, 3, 12], frame=["none", "none", "ring", "thick_ring", "lozenge", "sumikiri", "kikko", "plait2", "plait2_wide", "fret_key2", "fret_T"], frame_mod=["closed"],
                 edge=["plain"] * 8 + EDGE_LINES, hero=["sparkle_one", "spike", "blade", "kite", "flame", "fleur", "feather"],
                 place=["radial_0"], tilt=[0], support=["none", "sparkle", "dot", "spike", "second_star", "inner_ring"],
                 support_place=["between_out", "between_in"], center=["empty", "dot", "ring", "knock_ring", "diamond", "sparkle"]),
    "pinwheel": dict(freq=5, sym=["C"], n=[3, 3, 4, 5, 6], frame=["none", "ring", "double_ring", "snow_ring", "mokko5", "kikko", "plait2", "plait3"], frame_mod=["closed", "gap_between"],
                     edge=["plain"] * 8 + EDGE_LINES, hero=SWIRLS, place=["radial_0", "radial_1"], tilt=[35, 55, 75],
                     support=["none", "dot", "sparkle", "teardrop"], support_place=["between_out", "between_in"], center=CENTERS),
    "polygon_frame": dict(freq=4, sym=["D"], n=[3, 4, 5, 6, 8], frame=["polygon", "double_polygon"], frame_mod=["closed"],
                          edge=["plain"], hero=["spike", "dot", "sparkle", "teardrop", "triangle", "diamond", "fleur", "leaf"],
                          place=["vertex", "inside"], tilt=[0], support=["none", "dot", "diamond", "sparkle"],
                          support_place=["between_in"], center=[c for c in CENTERS if c != "empty"] + ["mark"] * 3),
    "emblem": dict(freq=15, sym=["D"], n=[4, 5, 6, 8, 3], frame=["disc", "thick_ring", "star_polygon", "gear"],
                   frame_mod=["closed"], edge=["plain", "plain", "outer_ring"], hero=["flower_round", "flower_pointed", "flower_notched",
                   "flower_heart", "sparkle", "star", "petals", "teardrops", "dots"],
                   place=["knockout"], tilt=[0], support=["none"], support_place=["na"], center=["empty", "dot", "ring"]),
    "segmented": dict(freq=8, sym=["C"], n=[2, 3, 4, 6], frame=["segmented_2", "segmented_3"], frame_mod=["gap_20", "gap_45"],
                      edge=["plain"], hero=["arc_offset_0", "arc_offset_half", "arc_offset_quarter"], place=["na"], tilt=[0],
                      support=["none", "dot", "sparkle", "bar"], support_place=["between_out"], center=CENTERS),
    "crescent": dict(freq=5, sym=["none"], n=[1], frame=["crescent_0", "crescent_45", "crescent_315", "crescent_90"],
                     frame_mod=["thin", "mid", "fat"], edge=["plain"], hero=["sparkle", "dot", "star", "eye", "flower_round", "diamond"],
                     place=["inside", "horns"], tilt=[0], support=["none", "sparkle", "dot"], support_place=["horns", "outside"],
                     center=["empty"]),
    "knot": dict(freq=5, sym=["C", "D"], n=[3, 4, 5, 6], frame=["none", "ring"], frame_mod=["closed"], edge=["plain"],
                 hero=["triquetra"] * 2 + ["loop_leaf", "loop_petal", "loop_eye"] * 2 + ["knot_rosette"], place=["radial_0"], tilt=[0],
                 support=["none", "dot", "sparkle"], support_place=["between_out"], center=CENTERS),
    "scatter": dict(freq=3, sym=["D"], n=[4, 5, 6, 8], frame=["none"], frame_mod=["closed"], edge=["plain"],
                    hero=["sparkle", "diamond", "petal", "teardrop", "star"], place=["orbit"], tilt=[0],
                    support=["none", "dot", "sparkle"], support_place=["between_out"], center=["empty", "dot", "sparkle"]),
}
MODS = ["tip_accent", "emphasis", "alternate"]


def pick(rng, table):
    if isinstance(table, dict):
        keys = list(table)
        return rng.choices(keys, weights=[table[k] for k in keys])[0]
    return rng.choice(table)


def motif(family, L, role="hero", variant=0):
    node = {"shape": family, "role": role, "size": {"len": L, "w": L * WIDTH.get(family, 0.4)}}
    s = node["size"]
    flip = -1 if variant % 2 else 1
    if family == "sparkle":
        s.update({"w": L * 0.2, "sharp": 0.95 if variant % 2 else 0.75})
    elif family == "horn":
        s["bend"] = 0.45 * flip
    elif family == "wing":
        s.update({"feathers": 3, "spread": 55})
    elif family == "bar":
        node["caps"] = "round"
    elif family == "flame":
        s["bend"] = 0.3 * flip
    elif family == "bolt":
        s["zig"] = 0.3 * flip
    elif family == "hook":
        s["curl"] = flip
    elif family == "curl":
        s["dir"] = flip
    elif family == "chevron":
        s["weight"] = L * 0.18
    return node


def ring(r, w, role="frame", **kw):
    return {"shape": "circle", "r": r, "weight": w, "repeat": "none", "role": role, **kw}


def center_nodes(c, W, size="small"):
    nodes = _center_nodes(c, W)
    if size == "large":
        for nd in nodes:
            if nd["shape"] == "circle":
                nd["r"] *= 1.35
            else:
                nd["size"]["len"] *= 1.45
                if "w" in nd["size"]:
                    nd["size"]["w"] *= 1.45
    return nodes


def _center_nodes(c, W):
    if c == "empty":
        return []
    if c.startswith("mark:"):
        return [{"shape": "mark", "mark": decode_mark(c), "size": {"len": 0.72}, "stroke": 0.075,
                 "repeat": "none", "role": "center"}]
    if c.startswith(("kamon:", "her:")):
        return [{"shape": "library", "name": c, "size": {"len": 0.95}, "repeat": "none", "role": "center"}]
    one = {"repeat": "none", "role": "center"}
    if c == "ring":
        return [ring(0.4, W * 0.8, role="center")]
    if c == "dot":
        return [{"shape": "dot", "size": {"len": 0.26}, **one}]
    if c == "sparkle":
        return [{"shape": "sparkle", "size": {"len": 0.6, "w": 0.12, "sharp": 0.9}, **one}]
    if c == "diamond":
        return [{"shape": "diamond", "size": {"len": 0.42, "w": 0.26}, **one}]
    if c.startswith("flower_"):
        return [{"shape": "flower_typed", "size": {"len": 0.62, "n": 5, "petal_w": 0.6, "petal": c[7:]}, **one}]
    if c == "knock_ring":
        return [{"shape": "dot", "size": {"len": 0.46}, **one, "children": [{"shape": "dot", "size": {"len": 0.22}, "combine": "subtract"}]}]
    if c == "eye":
        return [{"shape": "eye", "size": {"len": 0.55, "w": 0.3}, **one}]
    if c == "star5":
        return [{"shape": "star", "size": {"len": 0.5, "n": 5, "inner": 0.45}, **one}]
    if c == "cross":
        return [{"shape": "cross", "size": {"len": 0.46, "w": 0.1, "type": "plain"}, **one}]   # no pattee: Iron Cross
    raise ValueError(c)


# ---------------------------------------------------------------- layer builders

KAMON_CONTAINERS = ["snow_ring", "mokko", "mokko5", "kikko", "lozenge", "sumikiri", "igeta"]
PLAITS = {"plait2": (12, 2, 0.085, False), "plait2_wide": (8, 2, 0.12, False),
          "plait2_border": (16, 2, 0.075, True), "plait3": (12, 3, 0.1, False)}   # k, strands, amp, border
PLAIT_KINDS = list(PLAITS)


def container_nodes(kind, W, r=1.0):
    """Kamon containers (輪 / 雪輪 / 木瓜 / 五瓜 / 亀甲 / 菱 / 隅切り角 / 井桁) as frame nodes."""
    base = {"role": "frame", "repeat": "none", "weight": W}
    if kind.startswith("fret_"):   # meander / fret band ring (always railed)
        cell = kind[5:]
        return [{"shape": "ornament", "kind": "fret", "cell": cell, "n": 16 if cell != "key2" else 12, "r": r * 1.1,
                 "h": 0.22, "weight": max(W * 0.4, 0.035), "rails": "both" if cell == "T" else "in",
                 "role": "frame", "repeat": "none"}]
    if kind == "loops6":   # ring of woven loops (Celtic knot)
        return [{"shape": "ornament", "kind": "knot", "curve": "loops6", "r": r * 1.12, "weight": max(W * 0.6, 0.055),
                 "role": "frame", "repeat": "none"}]
    if kind in PLAITS:   # Celtic plait ring (woven over/under)
        k, strands, amp, border = PLAITS[kind]
        return [{"shape": "plait", "r": r, "weight": max(W * 0.6, 0.055), "amp": amp, "k": k, "strands": strands,
                 "border": border, "role": "frame", "repeat": "none"}]
    if kind.startswith("kamon:"):   # traced container (snow ring, stepped lozenge, double hexagon...)
        return [{"shape": "library", "name": kind, "size": {"len": 2.15 * r}, "role": "frame", "repeat": "none"}]
    if kind == "ring":
        return [ring(r, W)]
    if kind == "double_ring":
        return [ring(r, W), ring(r - W * 2.0, W * 0.5)]
    if kind == "snow_ring":     # six soft lobes
        return [{"shape": "polygon", "n": 6, "r": r * 0.95, "bulge": 0.2, **base}]
    if kind == "mokko":         # four lobes
        return [{"shape": "polygon", "n": 4, "r": r * 0.95, "bulge": 0.24, "phase": 45, **base}]
    if kind == "mokko5":        # five lobes (五瓜)
        return [{"shape": "polygon", "n": 5, "r": r * 0.95, "bulge": 0.22, "phase": 36, **base}]
    if kind == "kikko":         # tortoiseshell hexagon, flat top
        return [{"shape": "polygon", "n": 6, "r": r, "phase": 30, **base}]
    if kind == "lozenge":
        return [{"shape": "polygon", "n": 4, "r": r * 1.35, "radii": [1.15, 0.8, 1.15, 0.8], **base}]
    if kind == "sumikiri":
        return [{"shape": "sumikiri", "r": r * 1.15, "cut": 0.3, **base}]
    if kind == "igeta":
        return [{"shape": "igeta", "r": r * 1.45, **base}]
    return []


def build_frame(ch, n, W, hero_phase):
    f, mod, edge = ch["frame"], ch["frame_mod"], ch["edge"]
    step = 360.0 / max(n, 1)
    gaps = None
    if mod == "gap_hero":
        gaps = [{"at": hero_phase + k * step, "width": 24} for k in range(n)]
    elif mod == "gap_between":
        gaps = [{"at": hero_phase + step / 2 + k * step, "width": 20} for k in range(n)] if n > 1 else \
            [{"at": 90, "width": 30}, {"at": 270, "width": 30}]
    nodes = []
    if f in ("ring", "double_ring", "thick_ring"):
        w = W * (1.9 if f == "thick_ring" else 1.0)
        nodes.append(ring(1.0, w, **({"gaps": gaps} if gaps else {})))
        if f == "double_ring":
            nodes.append(ring(1.0 - W * 2.0, W * 0.5, role="frame"))
    elif f in ("polygon", "double_polygon"):
        k = n if n >= 3 else 6
        nodes.append({"id": "frame", "role": "frame", "shape": "polygon", "n": k, "r": 1.0, "weight": W,
                      "phase": 0 if ch.get("frame_phase", "pointy") == "pointy" else 180 / k, "repeat": "none"})
        if f == "double_polygon":
            nodes.append({"shape": "polygon", "role": "frame", "n": k, "r": 1.0 - W * 2.2, "weight": W * 0.55,
                          "phase": nodes[0]["phase"], "repeat": "none"})
    elif f == "crescent":
        nodes.append({"shape": "crescent", "role": "frame", "r": 1.0, "thickness": 0.35, "opening": 0, "repeat": "none"})
    elif f in KAMON_CONTAINERS or f in PLAITS or f.startswith("fret_") or f == "loops6":
        nodes += container_nodes(f, W)
    if edge.startswith("L:") and nodes and nodes[0]["shape"] == "circle":
        _, kind, side = edge.split(":")
        k = max(n, 2)
        nodes[0]["edge"] = {"type": kind, "n": k * max(round(16 / k), 1), "side": side,
                            "amp": nodes[0]["weight"] * (0.45 if side == "outer" else 0.32)}
    if edge in ("teeth", "dots", "spikes") and nodes and nodes[0]["shape"] == "circle":
        fam = {"teeth": "bar", "dots": "dot", "spikes": "triangle"}[edge]
        nodes.append({"shape": fam, "role": "frame", "size": {"len": W * 1.1, "w": W},
                      "place": {"polar": {"r": 1.0 + nodes[0]["weight"] * 0.45, "phase": 0}},
                      "repeat": {"orbit": {"n": 16 if n in (2, 4, 8) else 12 if n in (3, 6) else 15}}})
    return nodes


def hero_node(ch, W, frame_w):
    fam, place, size = ch["hero"], ch["place"], ch["size"]
    L = SIZES[size]
    if fam == "sparkle_one":   # one n-point sparkle as the whole hero
        return {"shape": "sparkle", "role": "hero", "repeat": "none",
                "size": {"len": 2.2 if size == "large" else 1.8, "n": max(ch["n"], 4), "w": 0.3, "sharp": 0.95 if ch["variant"] else 0.8}}
    m = motif(fam, L, variant=ch["variant"])
    m["size"]["w"] *= {"slim": 0.7, "normal": 1.0, "wide": 1.35, "na": 1.0}[ch["hero_width"]]
    phase = ch["hero_phase_deg"]
    tilt = ch["tilt"]
    o = lambda base: f"{base}+{tilt}" if tilt >= 0 else f"{base}{tilt}"
    fw = (frame_w or W) + (0.24 if ch["hero_gap"] == "float" else 0.0)   # floating heroes sit clear of the frame
    if place == "crossing":
        m["place"] = {"polar": {"r": 1 - L * 0.45, "phase": phase, "orient": o("radial_out")}}
    elif place == "outside":
        m["place"] = {"polar": {"r": 1 + fw * 0.5, "phase": phase, "orient": o("radial_out")}}
    elif place == "inward":
        m["place"] = {"polar": {"r": 1 + fw * 0.5 + L, "phase": phase, "orient": "radial_in"}}
    elif place == "on_ring":
        m["place"] = {"polar": {"r": 1 - fw * 0.3, "phase": phase, "orient": o("radial_out")}}
    elif place == "inside":
        m["place"] = {"polar": {"r": max(1 - fw - L - 0.06, 0.12), "phase": phase, "orient": o("radial_out")}}
    elif place.startswith("flank_"):
        m["place"] = {"polar": {"r": 1 + fw * 0.4, "phase": int(place[6:]), "orient": o("radial_out")}}
    elif place == "crown":
        m["place"] = {"polar": {"r": 0.7, "phase": 0}}
        m["repeat"] = "none"
        m["size"]["len"] = 0.9
    elif place.startswith("radial_"):
        m["place"] = {"polar": {"r": 0.08 if place == "radial_0" else 0.3, "phase": phase, "orient": o("radial_out")}}
        m["size"]["w"] *= 1.25
    elif place == "vertex":
        m["place"] = {"attach": {"to": "frame", "at": "vertex(*)"}}
        m["size"]["len"] = L * 0.6
        m["size"]["w"] = L * 0.6 * WIDTH.get(fam, 0.4)
    elif place == "orbit":
        m["place"] = {"polar": {"r": 1.0, "phase": phase, "orient": o("radial_out")}}
    if ch["hero_render"] == "outline" and fam not in ("dot",):
        m["render"] = "line"
        m["line_weight"] = W * 0.55
    elif ch["hero_render"] == "split" and fam in ROOTED:
        m.setdefault("children", []).append({"shape": "bar", "size": {"len": L * 0.55, "w": max(W * 0.28, 0.02)},
                                             "place": {"polar": {"r": L * 0.18, "phase": 0}}, "combine": "subtract"})
    return m


def support_nodes(ch, n, W, hero_phase):
    s, sp = ch["support"], ch["support_place"]
    if s == "none":
        return []
    if s == "inner_ring":
        return [ring(0.6, W * 0.6, role="support")]
    if s == "second_star":
        return [{"shape": "sparkle", "role": "support", "size": {"len": 1.1, "n": n, "w": 0.2, "sharp": 0.8},
                 "phase": 180 / max(n, 1), "repeat": "none"}]
    L = 0.3 if s not in ("dot",) else W * 1.6
    m = motif(s, L, role="support", variant=ch["variant"])
    between = hero_phase + 180.0 / max(n, 1)
    if sp == "between_out":
        m["place"] = {"polar": {"r": 1 + W + 0.12, "phase": between}}
    elif sp == "between_in":
        m["place"] = {"polar": {"r": 1 - W - 0.12 - L, "phase": between}}
    elif sp == "crown":
        m["place"] = {"polar": {"r": 1 + W * 0.6, "phase": 0}}
        m["repeat"] = "none"
    elif sp == "base":
        m["place"] = {"polar": {"r": 1 + W * 0.6, "phase": 180}}
        m["repeat"] = "none"
    elif sp == "crown_base":
        m["place"] = {"polar": {"r": 1 + W * 0.6, "phase": 0}}
        m["repeat"] = {"orbit": {"n": 2}}
    elif sp.startswith("flank_"):
        m["size"]["len"] = SIZES["large"]
        m["size"]["w"] = SIZES["large"] * WIDTH.get(s, 0.4)
        m["place"] = {"polar": {"r": 0.95, "phase": int(sp[6:])}}
    else:
        return []
    return [m]


# ---------------------------------------------------------------- special presets (frame-as-hero)

def emblem_nodes(ch, n, W):
    f = ch["frame"]
    nodes = []
    if f == "disc":
        ds = ch["disc_shape"]
        if ds == "circle":
            nodes.append({"shape": "dot", "role": "hero", "size": {"len": 2.0}, "repeat": "none"})
        elif ds == "lobed":
            nodes.append({"shape": "polygon", "role": "hero", "n": max(n, 5), "r": 0.95, "bulge": 0.22, "solid": True, "repeat": "none"})
        else:
            k = 6 if ds == "hexagon" else 8
            nodes.append({"shape": "polygon", "role": "hero", "n": k, "r": 1.05, "solid": True, "phase": 180 / k, "repeat": "none"})
    elif f == "thick_ring":
        nodes.append(ring(0.8, 0.42, role="hero"))
    elif f == "star_polygon":
        nodes.append({"shape": "star_polygon", "role": "hero", "n": max(n, 5), "r": 1.1, "inner_r": 0.68,
                      "concave": 0.4 if ch["variant"] % 2 else 0.0, "repeat": "none"})
    elif f == "gear":
        nodes.append({"shape": "gear", "role": "hero", "n": n * 2 if n < 6 else n, "r": 0.95, "weight": 0.3, "tooth_h": 0.2,
                      "tooth_w": 0.5, "solid": True, "repeat": "none"})
    cut = ch["hero"]
    sub = {"combine": "subtract", "repeat": "none"}
    if cut.startswith("flower_"):
        k = {"shape": "flower_typed", "size": {"len": 1.3, "n": n, "petal_w": 0.6, "petal": cut[7:]}, **sub}
    elif cut == "sparkle":
        k = {"shape": "sparkle", "size": {"len": 1.5, "n": n, "w": 0.3, "sharp": 0.85}, **sub}
    elif cut == "star":
        k = {"shape": "star", "size": {"len": 1.35, "n": n, "inner": 0.45}, **sub}
    elif cut == "cross":
        k = {"shape": "cross", "size": {"len": 1.4, "w": 0.24, "type": "pattee"}, **sub}
    elif cut == "petals":
        k = {"shape": "petal", "size": {"len": 0.5, "w": 0.3}, "place": {"polar": {"r": 0.22, "phase": 0}}, "combine": "subtract"}
    elif cut == "teardrops":
        k = {"shape": "teardrop", "size": {"len": 0.45, "w": 0.24}, "place": {"polar": {"r": 0.3, "phase": 180 / n}}, "combine": "subtract"}
    else:
        k = {"shape": "dot", "size": {"len": 0.3}, "place": {"polar": {"r": 0.45, "phase": 0}}, "combine": "subtract"}
    if f == "gear" and not cut.startswith("flower"):
        k = {"shape": "dot", "size": {"len": 0.9}, **sub}
        nodes.append(k)
        nodes.append({"shape": "flower_typed", "role": "hero", "size": {"len": 0.7, "n": n, "petal_w": 0.6, "petal": "round"}, "repeat": "none"})
    else:
        nodes.append(k)
    if ch["edge"] == "outer_ring":
        nodes.append(ring(1.2 if f != "star_polygon" else 1.3, W * 0.6, role="support"))
    return nodes


def segmented_nodes(ch, n, W):
    count = int(ch["frame"][-1])
    gw = int(ch["frame_mod"][4:])
    off = {"arc_offset_0": 0, "arc_offset_half": 180 / n, "arc_offset_quarter": 90 / n}[ch["hero"]]
    nodes = []
    pattern = {"heavy_outer": [1.5, 0.8, 0.6], "uniform": [1.0, 1.0, 1.0], "heavy_inner": [0.6, 1.0, 1.7]}[ch["ring_pattern"]]
    for i, r in enumerate([1.0, 0.74, 0.5][:count]):
        ph = off if i % 2 else 0
        nodes.append(ring(r, W * pattern[i], role="frame" if i == 0 else "hero",
                          gaps=[{"at": ph + j * 360 / n, "width": gw} for j in range(n)]))
    return nodes


def crescent_nodes(ch, W):
    opening = int(ch["frame"].split("_")[1])
    thick = {"thin": 0.25, "mid": 0.4, "fat": 0.55}[ch["frame_mod"]]
    nodes = [{"id": "moon", "role": "hero", "shape": "crescent", "r": 1.0, "thickness": thick, "opening": opening, "repeat": "none"}]
    fam = ch["hero"]
    L = 0.8 if ch["place"] == "inside" else 0.4
    m = center_nodes({"flower_round": "flower_round"}.get(fam, "dot"), W)[0] if fam in ("dot", "flower_round") else \
        {"shape": fam, "size": {"len": L, "w": L * (0.18 if fam == "sparkle" else 0.55), "n": 4 if fam == "sparkle" else 5}}   # 5-point sparkle + crescent = star-and-crescent
    m = dict(m, role="hero")
    if ch["place"] == "inside":
        m.update({"place": {"polar": {"r": 0.42, "phase": opening, "orient": "upright"}}, "repeat": "none"})
    else:
        m.update({"place": {"attach": {"to": "moon", "at": "horn(*)", "orient": "upright"}}})
    nodes.append(m)
    if ch["support"] != "none":
        s = motif(ch["support"], 0.22, role="support")
        s["place"] = {"polar": {"r": 1.25, "phase": opening + 180}} if ch["support_place"] == "outside" else \
            {"attach": {"to": "moon", "at": "belly", "orient": "follow"}}
        s["repeat"] = "none"
        nodes.append(s)
    return nodes


def knot_nodes(ch, n, W):
    fam = ch["hero"]
    if fam == "triquetra":
        return [{"shape": "triquetra", "role": "hero", "size": {"len": 1.5, "w": W * 0.7}, "repeat": "none"}], 3
    if fam == "knot_rosette":   # Celtic knot rosette (former ornament preset; rosettes removed after taste pass 2)
        curve = "8/3" if ch["variant"] == 0 else "9/4"
        return [{"shape": "ornament", "kind": "knot", "curve": curve, "r": 0.85, "weight": max(W * 0.7, 0.06),
                 "role": "hero", "repeat": "none"}], int(curve[0])
    shape = {"loop_leaf": "leaf", "loop_petal": "petal", "loop_eye": "leaf"}[fam]
    return [{"shape": shape, "role": "hero", "size": {"len": 0.95, "w": 0.5 if fam != "loop_eye" else 0.7}, "render": "line",
             "line_weight": W * 0.7, "place": {"polar": {"r": 0.0, "phase": 0}}}], n


# ---------------------------------------------------------------- sample

def color(rng):
    h = rng.random()
    s = rng.choice([0.45, 0.65, 0.85])
    v = rng.choice([0.92, 1.0])
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


def choose(rng, P):
    ch = {
        "sym": rng.choice(P["sym"]), "n": rng.choice(P["n"]), "weight": pick(rng, {"thin": 2, "medium": 3, "thick": 2}),
        "frame": rng.choice(P["frame"]), "frame_mod": rng.choice(P["frame_mod"]), "edge": rng.choice(P["edge"]),
        "frame_phase": rng.choice(["pointy", "flat"]),
        "hero": rng.choice(P["hero"]), "place": rng.choice(P["place"]), "hero_phase": rng.choice(["cardinal", "cardinal", "diagonal"]),
        "tilt": rng.choice(P["tilt"]), "size": rng.choice(["medium", "large"]),
        "hero_render": pick(rng, {"fill": 6, "outline": 2, "split": 1}), "variant": rng.choice([0, 1]),
        "support": rng.choice(P["support"]), "support_place": rng.choice(P["support_place"]),
        "center": rng.choice(P["center"]),
        "_mark": encode_mark(random_mark(rng)),
        "tip_accent": pick(rng, {"none": 5, "dot": 1, "sparkle": 1, "diamond": 1}),
        "emphasis": pick(rng, {"none": 6, "1.35": 1, "1.6": 1}),
        "alternate": pick(rng, {"none": 6, "0.55": 1, "0.7": 1}),
        # always-visible proportion dims (multiply the space without adding clutter)
        "hero_width": rng.choice(["slim", "normal", "wide"]),
        "hero_gap": rng.choice(["touch", "float"]),
        "center_size": rng.choice(["small", "large"]),
        # optional single marker on the axis (BA: Nonomi bead, Arata orbs, Mashiro, Hinata)
        "marker": pick(rng, {"none": 6, "dot": 1, "sparkle": 1, "diamond": 1, "drop": 1, "spike": 1}),
        "marker_place": rng.choice(["crown", "base", "both"]),
        # emblem / segmented specifics
        "disc_shape": rng.choice(["circle", "hexagon", "octagon", "lobed"]),
        "ring_pattern": rng.choice(["heavy_outer", "uniform", "heavy_inner"]),
    }
    for k, vals in P.get("extra", {}).items():
        ch[k] = rng.choice(vals)
    mk = ch.pop("_mark")
    if ch["center"] == "mark":
        ch["center"] = mk
    return ch


DEFAULTS = {"edge": "plain", "frame_mod": "closed", "support": "none", "center": "empty", "tip_accent": "none",
            "emphasis": "none", "alternate": "none", "hero_render": "fill", "marker": "none"}
MANDATORY = {"crest": {"support"}, "polygon_frame": {"center"}, "segmented": {"frame_mod"}, "crescent": {"frame_mod"},
             "emblem": {"edge"}}


def budget(rng, preset, ch):
    """Detail budget: BA halos carry 2-3 ideas. Keep at most 1-3 optional layers active, drop the rest."""
    active = [k for k, v in DEFAULTS.items() if ch.get(k, v) != v and k not in MANDATORY.get(preset, ())
              and not (k == "frame_mod" and ch[k] in ("thin", "mid", "fat"))]
    keep = pick(rng, {1: 2, 2: 4, 3: 3})
    rng.shuffle(active)
    for k in active[keep:]:
        ch[k] = DEFAULTS[k]
    return ch


def normalize(preset, ch):
    """Zero out choices that have no visible effect, so descriptors only count real differences."""
    if preset in ("flanked_ring", "crest", "crescent") or ch["sym"] in ("mirror_v", "none"):
        ch["n"] = 1
        ch["hero_phase"] = "na"
    if ch["frame"] not in ("polygon", "double_polygon"):
        ch["frame_phase"] = "na"
    if ch["hero"] not in ROOTED:
        ch["tip_accent"] = "na"
        if ch["hero_render"] == "split":
            ch["hero_render"] = "fill"
    if ch["support"] in ("none", "inner_ring", "second_star"):
        ch["support_place"] = "na"
    if ch["sym"] != "D" or ch["n"] not in (4, 6, 8):
        ch["alternate"] = "na"
    if ch["sym"] not in ("D", "C") or ch["n"] < 3:
        ch["emphasis"] = "na"
    if ch["tilt"] and ch["sym"] == "D":   # tilted heroes break mirror symmetry -> chiral
        ch["sym"] = "C"
    if preset in ("emblem", "segmented", "knot", "crescent"):
        for k in ("tip_accent", "emphasis", "alternate", "hero_render", "size", "tilt"):
            ch[k] = "na"
    if ch["hero"] in ("dot",) and ch["hero_render"] == "outline":
        ch["hero_render"] = "fill"
    # new proportion dims only where they are visible
    if preset in ("emblem", "segmented", "knot", "crescent") or ch["hero"] == "sparkle_one":
        ch["hero_width"] = "na"
    if not (ch["place"] in ("outside", "inward", "vertex") or ch["place"].startswith("flank_")) or ch["frame"] == "none":
        ch["hero_gap"] = "na"
    if ch["center"] == "empty":
        ch["center_size"] = "na"
    if ch["marker"] == "none" or preset in ("emblem", "segmented", "crescent", "scatter", "knot") or ch["frame"] == "none":
        ch["marker"] = "none"
        ch["marker_place"] = "na"
    if preset == "crest" and ch["marker_place"] in ("crown", "both"):
        ch["marker_place"] = "base" if ch.get("base", "none") == "none" else "na"
        if ch["marker_place"] == "na":
            ch["marker"] = "none"
    if ch.get("hero") == "knot_rosette":   # a center on top of the interlace reads as mud
        ch["center"], ch["center_size"] = "empty", "na"
    if preset == "polygon_frame":
        if ch["center_size"] != "na":
            ch["center_size"] = "small"
        if ch["n"] == 3:
            ch["support"], ch["support_place"] = "none", "na"
    if not (preset == "emblem" and ch["frame"] == "disc"):
        ch["disc_shape"] = "na"
    if preset != "segmented" or ch["frame"] == "segmented_2" and ch["ring_pattern"] == "heavy_inner":
        ch["ring_pattern"] = "na" if preset != "segmented" else "heavy_outer"
    return ch


def build_spec(preset, ch):
    W = WEIGHTS[ch["weight"]]
    n = ch["n"] if isinstance(ch["n"], int) else 1
    sym = {"kind": ch["sym"], "n": n} if ch["sym"] in ("D", "C") else {"kind": ch["sym"]}
    hero_phase = 180.0 / n if ch["hero_phase"] == "diagonal" and n > 1 else 0.0
    ch["hero_phase_deg"] = hero_phase
    if preset == "emblem":
        return sym, emblem_nodes(ch, n, W) + center_nodes(ch["center"], W * 0.7, ch["center_size"])
    if preset == "segmented":
        return sym, segmented_nodes(ch, n, W) + support_nodes(ch, n, W, 0) + center_nodes(ch["center"], W * 0.8, ch["center_size"])
    if preset == "crescent":
        return sym, crescent_nodes(ch, W)
    if preset == "knot":
        nodes, n2 = knot_nodes(ch, n, W)
        sym = {"kind": "C" if ch["hero"] == "triquetra" else ch["sym"], "n": n2}
        return sym, nodes + build_frame(ch, n2, W * 0.8, 0) + support_nodes(ch, n2, W, 0) + center_nodes(ch["center"], W * 0.7, ch["center_size"])

    frame = build_frame(ch, n if ch["sym"] in ("D", "C") else 2, W, hero_phase)
    fw = frame[0]["weight"] if frame and frame[0]["shape"] == "circle" else W
    hero = hero_node(ch, W, fw if frame else None)
    nodes = frame + [hero]
    if preset == "crest":
        s = motif(ch["support"], SIZES["large"], role="support", variant=ch["variant"])
        ang = int(ch["support_place"][6:])
        s["place"] = {"polar": {"r": 0.95 if ch["frame"] != "none" else 0.45, "phase": ang,
                                "orient": "radial_out" if ch["variant"] == 0 else "radial_out+20"}}
        nodes.append(s)
        if ch.get("base", "none") != "none":
            b = motif(ch["base"], SIZES["small"], role="support")
            b["place"] = {"polar": {"r": 1.0 if ch["frame"] != "none" else 0.45, "phase": 180}}
            b["repeat"] = "none"
            nodes.append(b)
        if ch["frame"] == "none":
            hero["place"]["polar"]["r"] = 0.25
    else:
        nodes += support_nodes(ch, n, W, hero_phase)
    # mods (grammar embellishments)
    if ch["tip_accent"] not in ("none", "na"):
        acc = ch["tip_accent"]
        size = {"dot": {"len": W * 1.4}, "sparkle": {"len": 0.26, "w": 0.05, "sharp": 0.9}, "diamond": {"len": 0.2, "w": 0.12}}[acc]
        hero.setdefault("children", []).append({"shape": acc, "size": size, "place": {"attach": {"at": "tip", "offset": {"along": 0.1}}},
                                                "combine": {"overlay": {"gap": "thin"}}})
    if ch["emphasis"] not in ("none", "na") and hero.get("repeat", "group") == "group":
        idx = [0, n] if ch["sym"] == "D" else [0]
        hero["break"] = [{"instance": i, "set": {"scale": float(ch["emphasis"])}} for i in idx]
    if ch["alternate"] not in ("none", "na"):
        hero["alternate"] = [{}, {"scale": float(ch["alternate"])}]
    if ch["marker"] != "none":
        fam = {"dot": "dot", "sparkle": "sparkle", "diamond": "diamond", "drop": "teardrop", "spike": "spike"}[ch["marker"]]
        mk = motif(fam, 0.26 if fam != "dot" else 0.17, role="support")
        r_out = 1 + fw * 0.5 + 0.07
        spots = {"crown": [0], "base": [180], "both": [0, 180]}[ch["marker_place"]]
        for ph in spots:
            nodes.append(dict(mk, place={"polar": {"r": r_out + (0 if fam in ("dot", "diamond", "sparkle") else 0),
                                                   "phase": ph, "orient": "radial_out"}}, repeat="none"))
            if fam in ("dot", "diamond", "sparkle"):   # centered motifs: push their center out by half their size
                nodes[-1]["place"]["polar"]["r"] += mk["size"]["len"] / 2
    return sym, nodes + center_nodes(ch["center"], W * 0.8, ch["center_size"])


# ---------------------------------------------------------------- kamon preset: arrangement operators

KAMON_ARR = {   # arrangement -> allowed motif families (kamon blazon operators, research/kamon/OPERATORS.md)
    "embrace": ["plain", "leaves", "feather", "curl", "buds"],                        # 抱き
    "crossed": ["feather", "blade", "arrow", "branch", "leaf", "fork", "flame"],        # 違い
    "facing": ["leaf", "feather", "wing", "horn", "curl", "flame", "wave", "teardrop", "hook"],  # 対い
    "heads_together": ["leaf", "petal", "teardrop", "feather", "flame", "kite", "heart"] + KAMON_PARTS + HER["part"],   # 頭合せ
    "piled": ["dot", "flower_round", "flower_pointed", "diamond", "heart", "star", "tomoe"] + KAMON_PARTS + HER["part"],  # 三つ盛り
    "tiered": ["diamond", "diamond_outline"],   # 三階菱 (chevrons/bars dropped: they read as rank insignia)
    "twisted": ["petal", "leaf", "teardrop", "feather", "kite"] + KAMON_PARTS + HER["part"],         # 捻じ
    "charge": KAMON_LIB,                                                              # a real crest as the interior
}
KAMON_FREQ = {"embrace": 5, "crossed": 5, "facing": 4, "heads_together": 3, "piled": 2, "tiered": 1, "twisted": 2,
              "charge": 8 if KAMON_LIB else 0}
KAMON_CONTAINER_CHOICES = ["none", "ring", "double_ring"] + KAMON_CONTAINERS + KAMON_LIB_CONTAINERS


def choose_kamon(rng):
    arr = pick(rng, KAMON_FREQ)
    ch = {"arrangement": arr, "hero": rng.choice(KAMON_ARR[arr]),
          "container": pick(rng, {"none": 6 if arr == "embrace" else 3, "ring": 3, "double_ring": 1,
                                  **{c: 1 for c in KAMON_CONTAINERS}, "plait2": 1, "plait2_wide": 1,
                                  **{c: 4 / max(len(KAMON_LIB_CONTAINERS), 1) for c in KAMON_LIB_CONTAINERS}}),
          "render": pick(rng, {"fill": 3, "outline": 1}), "weight": pick(rng, {"thin": 2, "medium": 3, "thick": 2}),
          "n": rng.choice([3, 4]) if arr == "heads_together" else rng.choice([5, 6]) if arr == "twisted" else "na",
          "center": rng.choice(["empty", "empty", "dot", "ring", "sparkle", "diamond", "flower_round", "flower_pointed",
                                "knock_ring", "eye", "star5"]),
          "center_size": rng.choice(["small", "large"]), "variant": rng.choice([0, 1])}
    if arr in ("crossed", "piled", "tiered", "charge"):
        ch["center"] = "empty"
    if ch["center"] == "empty":
        ch["center_size"] = "na"
    ch["cross_angle"] = rng.choice([25, 35, 45]) if arr == "crossed" else "na"
    ch["container_edge"] = (pick(rng, {"plain": 3, "line": 2}) if ch["container"] in ("ring", "double_ring") else "na")
    if ch["container_edge"] == "line":
        ch["container_edge"] = rng.choice(EDGE_LINES)
    if arr == "charge" and ch["container"] == "none":   # always transformed: never a bare real crest
        ch["container"] = rng.choice(["ring", "double_ring"] + KAMON_CONTAINERS)
    # BA outward layer (needs a container to project from)
    if ch["container"] != "none":
        ch["outward"] = pick(rng, {"none": 0 if arr == "charge" else 3, "crown": 2, "crown_base": 1, "flank": 2, "cardinal": 1})
        pool = {"crown": ["sparkle", "diamond", "spike", "dot", "teardrop"], "crown_base": ["sparkle", "diamond", "spike", "dot"],
                "flank": ["sparkle", "leaf", "wing", "spike", "teardrop"], "cardinal": ["bar", "spike", "diamond"]}
        ch["outward_motif"] = rng.choice(pool[ch["outward"]]) if ch["outward"] != "none" else "na"
    else:
        ch["outward"], ch["outward_motif"] = "none", "na"
    return ch


def _rooted_at_center(m, L, x, y, angle):
    """Place a rooted motif so that its MIDDLE sits at (x, y), pointing along `angle`."""
    s, c = math.sin(math.radians(angle)), math.cos(math.radians(angle))
    m["place"] = {"cartesian": {"x": x - s * L / 2, "y": y - c * L / 2, "orient": angle}}
    return m


def _part(fam, L):
    return {"shape": "library", "name": fam, "role": "hero", "size": {"len": L}}


def kamon_nodes(ch):
    W = WEIGHTS[ch["weight"]]
    arr, fam = ch["arrangement"], ch["hero"]
    nodes, sym = [], {"kind": "none"}
    is_part = fam.startswith(("kamon:", "her:")) and arr != "charge"
    if arr == "embrace":
        sym = {"kind": "mirror_v"}
        nodes.append({"shape": "embrace_arm", "role": "hero", "r": 0.8, "weight": W * 1.5, "deco": fam, "count": 5})
    elif arr == "crossed":
        L, a = 1.5, ch["cross_angle"]
        for k, ang in enumerate((a, -a)):
            m = _rooted_at_center(motif(fam, L, variant=ch["variant"]), L, 0.0, 0.0, ang)
            m["size"]["w"] *= 0.55                                 # long crossed charges are slim
            if fam == "arrow":
                m["size"].update({"w": 0.32, "shaft": W * 0.7})
            m["repeat"] = "none"
            if k:
                m["combine"] = {"overlay": {"gap": "medium"}}     # over/under at the crossing
            nodes.append(m)
    elif arr == "facing":
        sym = {"kind": "mirror_v"}
        L = 0.95
        m = motif(fam, L, variant=ch["variant"])
        m["place"] = {"cartesian": {"x": 0.3, "y": -0.5, "orient": "upright-18"}}
        nodes.append(m)
    elif arr == "heads_together":
        n = ch["n"]
        sym = {"kind": "D", "n": n}
        L = 0.62
        if is_part:
            m = _part(fam, 0.44)
            m["place"] = {"polar": {"r": 0.42, "phase": 0, "orient": "radial_in"}}
        else:
            m = motif(fam, L, variant=ch["variant"])
            m["place"] = {"polar": {"r": L + 0.06, "phase": 0, "orient": "radial_in"}}
        nodes.append(m)
    elif arr == "piled":
        for (x, y) in [(0.0, 0.33), (-0.36, -0.27), (0.36, -0.27)]:
            if is_part:
                m = _part(fam, 0.58)
            elif fam.startswith("flower_"):
                m = {"shape": "flower_typed", "role": "hero", "size": {"len": 0.62, "n": 5, "petal_w": 0.6, "petal": fam[7:]}}
            elif fam == "tomoe":
                m = {"shape": "tomoe", "role": "hero", "size": {"len": 0.55}}
            elif fam == "star":
                m = {"shape": "star", "role": "hero", "size": {"len": 0.58, "n": 5, "inner": 0.45}}
            else:
                m = motif(fam, 0.5 if fam != "dot" else 0.46)
            m["place"] = {"cartesian": {"x": x, "y": y}}
            m["repeat"] = "none"
            nodes.append(m)
    elif arr == "tiered":
        for i, (y, w) in enumerate([(0.42, 0.55), (0.0, 0.8), (-0.42, 1.05)]):
            if fam.startswith("diamond"):
                m = {"shape": "diamond", "role": "hero", "size": {"len": 0.36, "w": w}}
                if fam == "diamond_outline":
                    m["children"] = [{"shape": "diamond", "size": {"len": 0.36 - W * 1.6, "w": w - W * 2.2}, "combine": "subtract"}]
                m["place"] = {"cartesian": {"x": 0, "y": y}}
            elif fam == "chevron":
                m = {"shape": "chevron", "role": "hero", "size": {"len": 0.22, "w": w, "weight": W * 1.1}}
                m["place"] = {"cartesian": {"x": 0, "y": y - 0.11}}
            else:
                m = {"shape": "bar", "role": "hero", "size": {"len": w, "w": W * 1.4}}
                m["place"] = {"cartesian": {"x": -w / 2, "y": y, "orient": 90}}
            m["repeat"] = "none"
            nodes.append(m)
    elif arr == "charge":
        size = 1.5 if ch["container"] != "none" else 2.0
        if ch["container"] in ("lozenge", "kikko", "sumikiri") or ch["container"].startswith("kamon:"):
            size = 1.25
        nodes.append({"shape": "library", "name": fam, "role": "hero", "size": {"len": size}, "repeat": "none"})
    elif arr == "twisted":
        n = ch["n"]
        sym = {"kind": "C", "n": n}
        if is_part:
            m = _part(fam, 0.38)
            m["place"] = {"polar": {"r": 0.5, "phase": 0, "orient": "radial_out+32"}}
        else:
            m = motif(fam, 0.7, variant=ch["variant"])
            m["size"]["w"] *= 1.3
            m["place"] = {"polar": {"r": 0.08, "phase": 0, "orient": "radial_out+32"}}
        nodes.append(m)
    if ch["render"] == "outline":
        for m in nodes:
            m["render"] = "line"
            m["line_weight"] = W * 0.6
    cont = container_nodes(ch["container"], W) if ch["container"] != "none" else []
    if str(ch.get("container_edge", "")).startswith("L:") and cont:
        _, kind, side = ch["container_edge"].split(":")
        cont[0]["edge"] = {"type": kind, "n": 16, "side": side, "amp": W * (0.45 if side == "outer" else 0.32)}
    nodes += cont
    nodes += center_nodes(ch["center"], W * 0.8, ch["center_size"])
    nodes += outward_nodes(ch, W)
    return sym, nodes


# outer radius of each container along the vertical / horizontal axes (where outward elements attach)
CONTAINER_R = {"fret_key": 1.12, "fret_key2": 1.12, "fret_T": 1.12, "fret_battlement": 1.12,
               "loops6": 1.14, "plait2": 1.1, "plait2_wide": 1.13, "plait2_border": 1.18, "plait3": 1.12, "ring": 1.0, "double_ring": 1.0, "snow_ring": 1.1, "mokko": 1.05, "mokko5": 1.05, "kikko": 0.87,
               "lozenge": 1.55, "sumikiri": 0.92, "igeta": 1.45}
CONTAINER_RX = {"lozenge": 1.08, "kikko": 1.0, "igeta": 1.45}


def outward_nodes(ch, W):
    """BA layer on top of a kamon: elements projecting OUT of the container (kamon never do; BA does 45% of the time)."""
    kind, fam = ch.get("outward", "none"), ch.get("outward_motif", "na")
    if kind == "none":
        return []
    Rv = CONTAINER_R.get(ch["container"], 1.0) + W * 0.5
    Rh = CONTAINER_RX.get(ch["container"], Rv - W * 0.5) + W * 0.5
    L = {"sparkle": 0.42, "dot": 0.2, "diamond": 0.34, "spike": 0.42, "leaf": 0.5, "wing": 0.55, "teardrop": 0.42,
         "bar": 0.4}[fam]
    one = lambda ph, R, rep="none": dict(motif(fam, L, role="support"), repeat=rep,
                                         place={"polar": {"r": R + (L / 2 if fam in ("sparkle", "dot", "diamond") else 0.02),
                                                          "phase": ph}})
    if kind == "crown":
        return [one(0, Rv)]
    if kind == "crown_base":
        return [one(0, Rv), one(180, Rv)]
    if kind == "flank":
        return [one(90, Rh), one(270, Rh)]
    if kind == "cardinal":   # ticks crossing the container, reticle-style
        m = motif(fam, L, role="support")
        m.update(repeat={"orbit": {"n": 4}}, place={"polar": {"r": Rv - L * 0.45, "phase": 0}})
        return [m]
    return []


# ---------------------------------------------------------------- hate-symbol safety filter
# Shapes a halo generator can produce BY ACCIDENT that read as extremist symbols (ADL hate-symbol database):
# swastika / hooked triskele (angular arms in one-way rotation), black sun (ring of bent rays), sun cross
# (plain cross touching a ring), Iron Cross (cross pattee). Checked on every sample; offending choices are resampled.
ANGULAR = {"bolt", "hook", "shard", "blade", "kite", "fork", "horn", "spike", "arrow", "triangle", "chevron", "bar"}
SOFT_ROTORS = {"tomoe", "petal", "leaf", "teardrop", "flame"}   # curl / wave have hooked tips: 4 in rotation read swastika-like


def forbidden(preset, ch) -> str | None:
    fam = str(ch.get("hero", ""))
    rotational = ch.get("sym") == "C" or ch.get("arrangement") == "twisted" or preset == "pinwheel"
    n = ch.get("n") if isinstance(ch.get("n"), int) else 0
    if rotational and fam in ("bolt", "hook"):
        return "bent arms in rotation (triskele / swastika / black sun)"
    if rotational and n == 4 and fam not in SOFT_ROTORS and not fam.startswith(("kamon:", "flower_")):
        return "four angular arms in one-way rotation (swastika)"
    if fam == "cross" and preset == "emblem":
        return "cross knocked out of a disc (sun cross)"
    crescent = preset == "crescent" or ch.get("frame") == "crescent" or ch.get("container") == "crescent"
    if crescent and ("star" in (fam, ch.get("support"), ch.get("marker")) or ch.get("center") == "star5"):
        return "crescent with a five-pointed star (religious / national symbol)"
    return None


# ---------------------------------------------------------------- heraldic preset (Armoria charges, CC0 / NC-SA)

HER_CONTAINERS = ["ring", "double_ring", "lozenge", "kikko", "sumikiri", "roundel_disc", "wreath", "wreath",
                  "plait2", "plait2_wide", "fret_key", "fret_key2", "loops6"]


def choose_heraldic(rng):
    pool = HER["interior"]
    crosses = [c for c in pool if c.startswith(("her:cross", "her:crosslet"))]
    others = [c for c in pool if c not in crosses]
    charge = rng.choice(crosses if crosses and rng.random() < 0.25 else others)
    ch = {"charge": charge, "container": rng.choice(HER_CONTAINERS),
          "weight": pick(rng, {"thin": 2, "medium": 3, "thick": 2}), "render": pick(rng, {"fill": 3, "outline": 1})}
    ch["edge"] = rng.choice(["plain"] * 3 + EDGE_LINES) if ch["container"] in ("ring", "double_ring") else "na"
    ch["flank"] = pick(rng, {"none": 3, **{f: 1 for f in HER["flank"]}})
    if ch["flank"] == "none":
        ch["outward"] = pick(rng, {"none": 2, "crown": 2, "crown_base": 1, "cardinal": 1})
    else:
        ch["outward"] = pick(rng, {"none": 3, "crown": 2})
    ch["outward_motif"] = rng.choice(["sparkle", "diamond", "spike", "dot"]) if ch["outward"] != "none" else "na"
    ch["wreath"] = "na"
    if ch["container"] == "wreath":
        wreaths = HER["container"]
        ch["wreath"] = rng.choice(wreaths) if wreaths else "na"
        if not wreaths:
            ch["container"] = "ring"
    return ch


def heraldic_nodes(ch):
    W = WEIGHTS[ch["weight"]]
    c = ch["container"]
    nodes = []
    if c == "roundel_disc":   # charge knocked out of a solid roundel
        nodes.append({"shape": "dot", "role": "frame", "size": {"len": 2.1}, "repeat": "none"})
        nodes.append({"shape": "library", "name": ch["charge"], "size": {"len": 1.5}, "repeat": "none",
                      "combine": "subtract"})
        nodes.append({"shape": "circle", "role": "support", "r": 1.2, "weight": W * 0.6, "repeat": "none"})
    else:
        if c == "wreath":
            nodes.append({"shape": "library", "name": ch["wreath"], "role": "frame", "size": {"len": 2.3}, "repeat": "none"})
        else:
            cont = container_nodes(c, W)
            if ch["edge"].startswith("L:"):
                _, kind, side = ch["edge"].split(":")
                cont[0]["edge"] = {"type": kind, "n": 16, "side": side, "amp": W * (0.45 if side == "outer" else 0.32)}
            nodes += cont
        size = 1.25 if c in ("lozenge", "kikko", "sumikiri") else 1.2 if c == "wreath" else 1.45
        m = {"shape": "library", "name": ch["charge"], "role": "hero", "size": {"len": size}, "repeat": "none"}
        if ch["render"] == "outline":
            m.update(render="line", line_weight=W * 0.55)
        nodes.append(m)
    if ch["flank"] != "none":
        R = CONTAINER_RX.get(c, 1.15 if c == "wreath" else 1.0) + W
        nodes.append({"shape": "library", "name": ch["flank"], "role": "support", "size": {"len": 0.95},
                      "place": {"cartesian": {"x": R + 0.35, "y": 0.1, "orient": "upright"}}, "flip": ch["flank"] in FLANK_FLIP})
    if ch["outward"] != "none":
        nodes += outward_nodes({"outward": ch["outward"], "outward_motif": ch["outward_motif"],
                                "container": c if c in CONTAINER_R else "ring"}, W)
    return {"kind": "mirror_v"}, nodes


# ---------------------------------------------------------------- sigil preset: house mark / tamga in a frame

def encode_mark(m):
    return "mark:" + ".".join(str(m[k]) for k in ("top", "bottom", "bars", "bar_pos", "son", "staff", "asym"))


def decode_mark(s):
    top, bottom, bars, bar_pos, son, staff, asym = s[5:].split(".")
    return {"top": top, "bottom": bottom, "bars": bars, "bar_pos": bar_pos, "son": son, "staff": staff,
            "asym": asym == "True"}


SIGIL_FRAMES = ["ring", "ring", "double_ring", "snow_ring", "mokko", "kikko", "lozenge", "sumikiri", "none",
                "plait2", "plait2_border", "fret_key", "fret_T", "loops6"]


def choose_sigil(rng):
    ch = {"mark": encode_mark(random_mark(rng)), "frame": rng.choice(SIGIL_FRAMES),
          "weight": pick(rng, {"thin": 2, "medium": 3, "thick": 2}),
          "stroke": rng.choice(["fine", "bold"])}
    ch["edge"] = rng.choice(["plain"] * 3 + EDGE_LINES) if ch["frame"] in ("ring", "double_ring") else "na"
    ch["flank"] = pick(rng, {"none": 4, "wing": 1, "sparkle": 1, "leaf": 1, **{f: 0.5 for f in HER["flank"]}})
    ch["outward"] = pick(rng, {"none": 3, "crown": 2, "crown_base": 1, "cardinal": 1}) if ch["frame"] != "none" else "none"
    ch["outward_motif"] = rng.choice(["sparkle", "diamond", "spike", "dot"]) if ch["outward"] != "none" else "na"
    if ch["frame"] == "none":
        ch["edge"], ch["flank"] = "na", pick(rng, {"wing": 1, "sparkle": 1, "leaf": 1})
        while decode_mark(ch["mark"])["bars"] == "none":   # a frameless sigil needs a substantial mark
            ch["mark"] = encode_mark(random_mark(rng))
    return ch


def sigil_nodes(ch):
    W = WEIGHTS[ch["weight"]]
    m = decode_mark(ch["mark"])
    f = ch["frame"]
    nodes = []
    if f != "none":
        cont = container_nodes(f, W)
        if str(ch["edge"]).startswith("L:"):
            _, kind, side = ch["edge"].split(":")
            cont[0]["edge"] = {"type": kind, "n": 16, "side": side, "amp": W * (0.45 if side == "outer" else 0.32)}
        nodes += cont
    size = 1.25 if f in ("lozenge", "kikko", "sumikiri") else 1.45 if f != "none" else 1.9
    if f == "loops6":   # taste review: the mark ran into the woven loops
        size = 1.0
    nodes.append({"shape": "mark", "mark": m, "role": "hero", "size": {"len": size},
                  "stroke": 0.075 if ch["stroke"] == "fine" else 0.11, "repeat": "none"})
    if ch["flank"] != "none":
        R = CONTAINER_RX.get(f, 1.0) + W if f != "none" else 0.75
        if ch["flank"].startswith("her:"):
            nodes.append({"shape": "library", "name": ch["flank"], "role": "support", "size": {"len": 0.9},
                          "place": {"cartesian": {"x": R + 0.33, "y": 0.1, "orient": "upright"}}, "flip": ch["flank"] in FLANK_FLIP})
        else:
            fl = motif(ch["flank"], 0.5, role="support")
            fl["place"] = {"polar": {"r": R + 0.05, "phase": 90}}
            nodes.append(fl)
    if ch["outward"] != "none":
        nodes += outward_nodes({"outward": ch["outward"], "outward_motif": ch["outward_motif"],
                                "container": f if f in CONTAINER_R else "ring"}, W)
    sym = {"kind": "none"} if m["asym"] else {"kind": "mirror_v"}
    if sym["kind"] == "none" and ch["flank"] != "none":   # asymmetric mark: mirror only the flank pair
        for nd in nodes:
            if nd.get("role") == "support" or nd.get("role") == "frame":
                nd.setdefault("repeat", "none")
        for nd in nodes:
            if nd.get("role") == "support" and "place" in nd and nd["repeat"] == "none":
                nd["repeat"] = "mirror_pair"
    return sym, nodes


# ---------------------------------------------------------------- ornament preset: rosette / knot interiors

ORN_INTERIORS = ["rosette8", "rosette10", "knot8/3", "knot9/4"]   # 7/4 reads as an atom symbol
ORN_FRAMES = ["ring", "double_ring", "plait2", "plait2_border", "fret_key", "fret_key2", "fret_T", "loops6",
              "snow_ring", "mokko5", "kikko"]


def choose_ornament(rng):
    ch = {"interior": rng.choice(ORN_INTERIORS), "frame": rng.choice(ORN_FRAMES),
          "weight": pick(rng, {"thin": 2, "medium": 3, "thick": 2}),
          "star": rng.choice(["fill", "outline", "knock"])}
    if not ch["interior"].startswith("rosette"):
        ch["star"] = "na"
    if ch["frame"] == "loops6" and ch["interior"].startswith("knot"):   # two knots nested = too busy
        ch["frame"] = "ring"
    ch["edge"] = rng.choice(["plain"] * 3 + EDGE_LINES) if ch["frame"] in ("ring", "double_ring") else "na"
    ch["outward"] = pick(rng, {"none": 2, "crown": 2, "crown_base": 1, "cardinal": 2, "flank": 1})
    ch["outward_motif"] = rng.choice({"crown": ["sparkle", "diamond", "spike", "dot"],
                                      "crown_base": ["sparkle", "diamond", "dot"],
                                      "cardinal": ["bar", "spike", "diamond"],
                                      "flank": ["sparkle", "leaf", "wing", "teardrop"]}.get(ch["outward"], ["na"]))
    return ch


def ornament_nodes(ch):
    W = WEIGHTS[ch["weight"]]
    f = ch["frame"]
    nodes = container_nodes(f, W)
    if str(ch["edge"]).startswith("L:"):
        _, kind, side = ch["edge"].split(":")
        nodes[0]["edge"] = {"type": kind, "n": 16, "side": side, "amp": W * (0.45 if side == "outer" else 0.32)}
    inner = 0.62 if f.startswith(("fret", "plait", "loops")) else 0.78 if f in ("kikko", "mokko5", "snow_ring") else 0.8
    it = ch["interior"]
    if it.startswith("rosette"):
        nodes.append({"shape": "ornament", "kind": "rosette", "n": int(it[7:]), "r": inner, "weight": max(W * 0.6, 0.05),
                      "star": ch["star"], "role": "hero", "repeat": "none"})
        n = int(it[7:])
    else:
        nodes.append({"shape": "ornament", "kind": "knot", "curve": it[4:], "r": inner, "weight": max(W * 0.7, 0.06),
                      "role": "hero", "repeat": "none"})
        n = int(it[4])
    if ch["outward"] != "none":
        nodes += outward_nodes({"outward": ch["outward"], "outward_motif": ch["outward_motif"],
                                "container": f if f in CONTAINER_R else "ring"}, W)
    return {"kind": "mirror_v"}, nodes


def _load_factors():
    import json
    import os
    path = os.path.join(os.path.dirname(__file__), "data", "preset_factors.json")
    return json.load(open(path)) if os.path.exists(path) else {}


PRESET_FACTORS = _load_factors()   # taste pass 2 (pairwise duel): capped 0.75-1.33 per preset


def sample(seed: int):
    rng = random.Random(seed)
    base = {**{k: v["freq"] for k, v in PRESETS.items()}, "kamon": 18,
            "heraldic": 12 if HER["interior"] else 0, "sigil": 12,
            "ornament": 0}   # taste pass 2: lost 8/8 pairs; rosettes dropped, knot rosette moved into "knot"
    preset = pick(rng, {k: v * PRESET_FACTORS.get(k, 1.0) for k, v in base.items()})
    if preset == "heraldic":
        ch = choose_heraldic(rng)
        sym, nodes = heraldic_nodes(ch)
        return {"name": f"seed{seed}", "color": color(rng), "desc": {"skeleton": preset, **ch},
                "halo": {"sym": sym, "root": nodes, "allow_single": True}}
    if preset == "ornament":
        ch = choose_ornament(rng)
        sym, nodes = ornament_nodes(ch)
        return {"name": f"seed{seed}", "color": color(rng), "desc": {"skeleton": preset, **ch},
                "halo": {"sym": sym, "root": nodes, "allow_single": True}}
    if preset == "sigil":
        ch = choose_sigil(rng)
        sym, nodes = sigil_nodes(ch)
        return {"name": f"seed{seed}", "color": color(rng), "desc": {"skeleton": preset, **ch},
                "halo": {"sym": sym, "root": nodes, "allow_single": True}}
    if preset == "kamon":
        ch = choose_kamon(rng)
        while forbidden(preset, ch):
            ch = choose_kamon(rng)
        sym, nodes = kamon_nodes(ch)
        desc = {"skeleton": preset, **ch}
        # non-radial kamon arrangements are complete with one charge (三つ盛り, 違い...)
        single_ok = ch["arrangement"] in ("crossed", "piled", "tiered", "facing", "charge")
        return {"name": f"seed{seed}", "color": color(rng), "desc": desc,
                "halo": {"sym": sym, "root": nodes, "allow_single": single_ok}}
    ch = normalize(preset, budget(rng, preset, choose(rng, PRESETS[preset])))
    while forbidden(preset, ch):
        ch = normalize(preset, budget(rng, preset, choose(rng, PRESETS[preset])))
    sym, nodes = build_spec(preset, ch)
    separate = []   # element-collision check available per preset; none enabled (taste review: no reliable rule)
    desc = {"skeleton": preset, **{k: v for k, v in ch.items() if k != "hero_phase_deg"}}
    return {"name": f"seed{seed}", "color": color(rng), "halo": {"sym": sym, "root": nodes, "separate": separate}, "desc": desc}
