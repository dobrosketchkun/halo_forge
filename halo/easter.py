"""Character-name seeds (loaded lazily; the halo data is only read when a name matches)."""
from __future__ import annotations

import json
import os
import sys
import unicodedata

_DATA = os.path.join(os.path.dirname(__file__), "data")
_HONORIFICS = ("senpai", "sensei", "chan", "san", "kun", "sama", "tan", "chi")
_MACRONS = str.maketrans({"ā": "aa", "ē": "ee", "ī": "ii", "ō": "oo", "ū": "uu", "Ā": "Aa", "Ē": "Ee", "Ī": "Ii",
                          "Ō": "Oo", "Ū": "Uu", "ô": "oo", "û": "uu", "Ô": "Oo", "Û": "Uu"})
_names = None
_halos = None


def norm_name(text) -> str:
    """Case-, accent-, spacing- and punctuation-insensitive form (keeps CJK / Hangul letters)."""
    t = unicodedata.normalize("NFKC", str(text))
    t = t.translate(_MACRONS)   # Hepburn long vowels: Sunaōkami == Sunaookami, Ryūge == Ryuuge
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return "".join(ch for ch in t.casefold() if ch.isalnum())


def _load_names():
    global _names
    if _names is None:
        path = os.path.join(_DATA, "ba_names.json")
        _names = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    return _names


def _load_halos():
    global _halos
    if _halos is None:
        path = os.path.join(_DATA, "ba_halos.json")
        if os.path.exists(path):
            _halos = json.load(open(path, encoding="utf-8"))
        elif sys.platform == "emscripten":          # web page: fetch on first use only
            from pyodide.http import open_url
            _halos = json.loads(open_url("halo/data/ba_halos.json").read())
        else:
            _halos = {}
    return _halos


def match(seed):
    """Gallery key for a character-name seed, or None."""
    names = _load_names()
    if not names:
        return None
    n = norm_name(seed)
    if n in names:
        return names[n]
    for h in _HONORIFICS:
        if n.endswith(h) and n[:-len(h)] in names:
            return names[n[:-len(h)]]
    return None


def record(key):
    from shapely import wkt
    h = _load_halos().get(key)
    if not h:
        return None
    return {"type": key, "color": h["color"], "desc": {"skeleton": "character"}, "halo": None,
            "geometry": wkt.loads(h["wkt"])}
