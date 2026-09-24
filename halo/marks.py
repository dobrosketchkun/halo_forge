"""House marks / tamgas / siglas: procedural line glyphs (staff + crossbars + terminals + son-marks).

No open dataset exists (research/traditions/sources_A.md), so marks are generated from the shared stroke grammar
of these traditions. A mark is a dict of discrete choices; `mark_geom(m)` returns shapely geometry (unit height),
`mark_blocked(m)` returns a reason if the combination reads as a symbol we must not produce:

  runes appropriated by extremists (ADL hate-symbol database): Tiwaz (arrow on a bare staff), Algiz / Life rune
  (arms branching upward from a staff that continues), Yr / "death rune" (legs at the bottom), Wolfsangel (hooks
  at both ends in opposite directions), Hagal (asterisk through the staff), Othala (lozenge on legs), sowilo
  (zig-zag - never generated);
  religious / national / planetary signs: Latin cross (one crossbar high on a bare staff), ankh / Venus / globus
  cruciger (ring + single crossbar), trident (Ukrainian tryzub, Poseidon), anchor (fine, allowed).
"""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

TOPS = ["none", "fork", "cup", "ring", "hooks", "hook_left", "bar", "dot"]
BOTTOMS = ["none", "crescent", "ring", "bar", "dot"]
BARS = ["none", "one", "two", "one_short", "two_short"]   # no chevron: arrow on a staff = Tiwaz
BAR_POS = ["high", "mid", "low"]
SONS = ["none", "tick", "dot", "two_dots"]
STAFF = ["short", "long"]


def mark_blocked(m: dict) -> str | None:
    top, bot, bars, pos = m["top"], m["bottom"], m["bars"], m.get("bar_pos", "mid")
    bare_bottom = bot == "none"
    if top == "fork" and bars == "none" and bare_bottom and m["son"] == "none":
        return "Algiz / Y-rune on a bare staff"
    if top == "hooks" and bars == "none" and bot != "crescent":
        return "trident-like (tryzub / Poseidon)"
    if top == "hook_left" and bot != "none" and bars in ("one", "one_short") and m.get("asym"):
        return "Wolfsangel-like (hooks + crossbar)"
    if bars == "one" and top in ("none", "dot", "bar") and bot in ("none", "bar", "dot"):
        return "plain cross (Latin / sun cross inside a ring)"
    if bars == "two" and top in ("none", "dot") and bot in ("none", "dot"):
        return "double cross (Lorraine / patriarchal)"
    if "ring" in (top, bot) and bars in ("one", "one_short"):
        return "ankh / Venus / globus cruciger"
    if bars == "chevron" and top == "none" and bare_bottom:
        return "Tiwaz-like arrow on a bare staff"
    if sum(x != "none" for x in (top, bot, bars)) < 2:
        return "too simple (staff + one feature)"
    if top == "fork" and bot in ("none", "dot") and bars == "none":
        return "Algiz / Y-rune"
    return None


def mark_geom(m: dict, w: float = 0.09):
    """Unit mark: staff from y=-0.5 to +0.5 (long) or -0.35..0.35 (short), stroke width w."""
    h = 0.5 if m.get("staff", "long") == "long" else 0.36
    lines, extra = [LineString([(0, -h), (0, h)])], []
    arm = 0.28
    top, bot = m["top"], m["bottom"]
    # top terminals
    if top == "fork":
        lines += [LineString([(0, h), (-arm, h + 0.26)]), LineString([(0, h), (arm, h + 0.26)])]
    elif top == "cup":
        t = np.linspace(math.pi, 2 * math.pi, 40)
        cup = np.column_stack([arm * np.cos(t), h + 0.22 + 0.22 * np.sin(t)])
        lines.append(LineString(cup))
    elif top == "ring":
        extra.append(Point(0, h + 0.15).buffer(0.15 + w / 2, 48).difference(Point(0, h + 0.15).buffer(0.15 - w / 2, 48)))
    elif top == "hooks":
        for s in (-1, 1):
            t = np.linspace(0, math.pi, 30)
            lines.append(LineString(np.column_stack([s * (0.14 - 0.14 * np.cos(t)), h + 0.14 * np.sin(t) * 0.9])))
    elif top == "hook_left":
        t = np.linspace(0, math.pi * 1.1, 30)
        lines.append(LineString(np.column_stack([-(0.15 - 0.15 * np.cos(t)), h + 0.15 * np.sin(t)])))
    elif top == "bar":
        lines.append(LineString([(-arm * 0.8, h), (arm * 0.8, h)]))
    elif top == "dot":
        extra.append(Point(0, h + 0.13).buffer(w * 0.85, 32))
    # bottom terminals
    if bot == "crescent":
        t = np.linspace(math.pi * 1.05, math.pi * 1.95, 40)
        lines.append(LineString(np.column_stack([0.3 * np.cos(t), -h + 0.24 + 0.3 * np.sin(t)])))
    elif bot == "ring":
        extra.append(Point(0, -h - 0.14).buffer(0.14 + w / 2, 48).difference(Point(0, -h - 0.14).buffer(0.14 - w / 2, 48)))
    elif bot == "bar":
        lines.append(LineString([(-arm * 0.7, -h), (arm * 0.7, -h)]))
    elif bot == "dot":
        extra.append(Point(0, -h - 0.13).buffer(w * 0.85, 32))
    # crossbars
    y = {"high": h * 0.45, "mid": 0.0, "low": -h * 0.45}.get(m.get("bar_pos", "mid"), 0.0)
    bars = m["bars"]
    L = arm * (0.55 if bars.endswith("short") else 1.0)
    left = -L if not m.get("asym") else 0.0
    if bars in ("one", "one_short"):
        lines.append(LineString([(left, y), (L, y)]))
    elif bars in ("two", "two_short"):
        for dy in (0.11, -0.11):
            lines.append(LineString([(left, y + dy), (L, y + dy)]))
    elif bars == "chevron":
        lines.append(LineString([(-L, y - 0.14), (0, y), (L, y - 0.14)]))
    # son marks (siglas poveiras: the heir's mark differs by small ticks)
    son = m.get("son", "none")
    sx = arm + 0.14
    if son == "tick":
        lines.append(LineString([(sx, -0.08), (sx, 0.08)]))
        if not m.get("asym"):
            lines.append(LineString([(-sx, -0.08), (-sx, 0.08)]))
    elif son in ("dot", "two_dots"):
        pts = [(sx, 0.0)] + ([(-sx, 0.0)] if not m.get("asym") else [])
        if son == "two_dots":
            pts = [(x, y0) for x, _ in pts for y0 in (0.1, -0.1)]
        extra += [Point(p).buffer(w * 0.7, 24) for p in pts]
    g = unary_union([l.buffer(w / 2, 16, cap_style=1, join_style=1) for l in lines] + extra)
    return g


def random_mark(rng) -> dict:
    while True:
        m = {"top": rng.choice(TOPS), "bottom": rng.choice(BOTTOMS), "bars": rng.choice(BARS),
             "bar_pos": rng.choice(BAR_POS), "son": rng.choice(SONS), "staff": rng.choice(STAFF),
             "asym": rng.random() < 0.2}
        if m["bars"] == "none":
            m["bar_pos"] = "na"
        if not mark_blocked(m):
            return m
