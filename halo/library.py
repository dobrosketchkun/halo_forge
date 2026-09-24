"""Curated traced motif libraries (kamon, heraldry) loaded as shapely geometry for `shape: library`.

Libraries are built offline (see .ignore/scripts/build_*_library.py) into halo/data/*.json; at runtime they are
only loaded, so this module needs nothing beyond shapely.
"""
from __future__ import annotations

import os

LIB: dict[str, object] = {}


def load_lib(name: str, prefix: str) -> dict[str, list[str]]:
    """Load halo/data/<name>.json (curated traced motifs) into LIB as '<prefix>:<id>'.
    Returns {role: [keys]} for roles interior / center / container / part / flank."""
    from shapely import wkt
    import json
    path = os.path.join(os.path.dirname(__file__), "data", name + ".json")
    roles: dict[str, list[str]] = {"interior": [], "center": [], "container": [], "part": [], "flank": []}
    if not os.path.exists(path):
        return roles
    with open(path, encoding="utf-8") as f:
        for e in json.load(f):
            key = prefix + ":" + e.get("id", e.get("name"))
            LIB[key] = wkt.loads(e["wkt"])
            for r in e.get("roles", ["interior", "center"]):
                roles.setdefault(r, []).append(key)
    return roles



def load_kamon_lib() -> dict[str, list[str]]:
    return load_lib("kamon_lib", "kamon")
