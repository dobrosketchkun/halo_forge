"""Public entry points: halo for a seed, random seeds, SVG / PNG output. Used by the CLI and the web page.

A seed is any text ("12345", "tralala", a name). It is hashed (BLAKE2b) into the sampler's internal number, and
"halo <seed>" is the first critic-approved sample in a deterministic sequence derived from that hash, so every seed
gives a good-looking halo and the same seed gives the same halo everywhere (CLI, browser) for a GENERATOR_VERSION.
For permanent one-halo-per-person issuance with a uniqueness guarantee use halo.issue.Issuer instead.
"""
from __future__ import annotations

import hashlib
import html
import random

from .build import build
from .critic import score
from .generate import GENERATOR_VERSION, sample
from .render import extent, polygons, to_png

MAX_SEED_LEN = 200
ATTEMPTS = 200


def normalize_seed(seed) -> str:
    key = str(seed).strip()
    if not key:
        raise ValueError("seed must not be empty")
    if len(key) > MAX_SEED_LEN:
        raise ValueError(f"seed must be at most {MAX_SEED_LEN} characters")
    return key


def sub_seed(seed, attempt: int) -> int:
    """Internal sampler seed for attempt k of a public seed (any text)."""
    h = hashlib.blake2b(f"{normalize_seed(seed)}#{attempt}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") & ((1 << 62) - 1)


def random_seed() -> str:
    return str(random.SystemRandom().randrange(1, 10 ** 12))


def normalize_color(color):
    """'#ff8800', 'ff8800', '#f80' or 'f80' -> '#ff8800'; None -> None (use the halo's own color)."""
    if color is None or str(color).strip() == "":
        return None
    c = str(color).strip().lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    if len(c) != 6 or any(ch not in "0123456789abcdefABCDEF" for ch in c):
        raise ValueError(f"color must be a hex color like #ff8800, got {color!r}")
    return "#" + c.lower()


def safe_name(seed) -> str:
    """Seed as a filename-safe fragment."""
    s = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(seed))[:60]
    return s or "halo"


def halo_for(seed) -> dict:
    """Deterministic, critic-approved halo for a public seed (any text). Returns the record plus its geometry."""
    seed = normalize_seed(seed)
    from . import easter
    key = easter.match(seed)
    if key:
        rec = easter.record(key)
        if rec:
            return {"seed": seed, "attempt": 0, "generator": GENERATOR_VERSION, **rec}
    for attempt in range(ATTEMPTS):
        it = sample(sub_seed(seed, attempt))
        ok, _, _, g = score(it["halo"])
        if ok:
            return {"seed": seed, "attempt": attempt, "generator": GENERATOR_VERSION, "color": it["color"],
                    "type": it["desc"]["skeleton"], "desc": it["desc"], "halo": it["halo"], "geometry": g}
    it = sample(sub_seed(seed, 0))   # practically unreachable (~60% of samples pass); fall back to the raw sample
    return {"seed": seed, "attempt": -1, "generator": GENERATOR_VERSION, "color": it["color"],
            "type": it["desc"]["skeleton"], "desc": it["desc"], "halo": it["halo"], "geometry": build(it["halo"])}


def _lighten(color: str, k: float = 0.25) -> str:
    c = color.lstrip("#")
    r, g, b = (int(c[i:i + 2], 16) for i in (0, 2, 4))
    return "#%02x%02x%02x" % tuple(int(v + (255 - v) * k) for v in (r, g, b))


def svg(rec: dict, size: int = 512, color: str | None = None, background: str | None = None,
        glow: float = 0.035) -> str:
    """Standalone SVG matching the PNG look: soft glow in the halo color + slightly brightened core.
    background=None -> transparent. Model space is y-up, flipped for SVG."""
    g = rec["geometry"]
    color = normalize_color(color) or rec["color"]
    E = extent(g)
    d = []
    for p in polygons(g):
        for ring in [p.exterior, *p.interiors]:
            d.append("M" + " L".join(f"{x:.3f},{-y:.3f}" for x, y in ring.coords) + " Z")
    uid = "h" + hashlib.blake2b(str(rec["seed"]).encode("utf-8"), digest_size=4).hexdigest()
    bg = f'<rect x="{-E:.3f}" y="{-E:.3f}" width="{2 * E:.3f}" height="{2 * E:.3f}" fill="{background}"/>' if background else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{-E:.3f} {-E:.3f} {2 * E:.3f} {2 * E:.3f}" '
            f'width="{size}" height="{size}">'
            f'<title>Halo {html.escape(str(rec["seed"]))}</title>'
            f'<defs><filter id="{uid}g" x="-50%" y="-50%" width="200%" height="200%">'
            f'<feGaussianBlur stdDeviation="{glow * 1.6:.4f}"/></filter>'
            f'<path id="{uid}p" d="{" ".join(d)}" fill-rule="evenodd"/></defs>'
            f'{bg}<use href="#{uid}p" fill="{color}" opacity="0.85" filter="url(#{uid}g)"/>'
            f'<use href="#{uid}p" fill="{_lighten(color)}"/></svg>')


def png(rec: dict, path: str | None = None, size: int = 512, color: str | None = None,
        background: tuple = (30, 30, 36)):
    """PNG via Pillow (CLI only; the web page rasterizes the SVG in the browser)."""
    return to_png(rec["geometry"], path, normalize_color(color) or rec["color"], px=size, bg=background)
