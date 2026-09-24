"""Uniqueness: salient descriptors + a min-distance issuance registry.

A descriptor is the tuple of DISCRETE, visible choices the sampler made (skeleton, symmetry order, hero family,
placement, frame, center, embellishments, weight...). Distance = weighted count of differing dimensions.
Calibration from real BA pairs (research/SPEC.md §2): d=2 same family, d=3 clearly different people, d>=4 very different.

Registry.try_issue() accepts a candidate only if its distance to EVERY issued halo is >= min_dist.
Lookup uses multi-index hashing: dimensions are split into `min_dist` blocks; since all weights are >= 1,
any pair with weighted distance < min_dist differs in < min_dist dimensions, so (pigeonhole) it agrees exactly
on at least one block. Only those candidates are compared exactly -> near O(1) per lookup at millions.
"""
from __future__ import annotations

from collections import defaultdict

# dimensions that change the overall read of a halo count double (all weights must be >= 1)
HEAVY = {"hero", "place", "frame", "sym", "n", "support"}
SKELETON_WEIGHT = 4   # a different skeleton alone is already "clearly different"
IGNORED = {"skeleton", "variant"}   # variant = handedness/bend flip: visible but too subtle to count
MARK_KEYS = {"center", "mark"}       # keys whose value can be an encoded house mark ("mark:...")
MARK_PARTS = ("top", "bottom", "bars", "bar_pos", "son", "staff", "asym")


def descriptor(desc: dict) -> tuple:
    """Flatten the sampler's desc into (skeleton, ((dim, value), ...)) with a fixed dim order."""
    d = {}
    for k, v in desc.items():
        if k in IGNORED:
            continue
        is_mark = isinstance(v, str) and v.startswith("mark:")
        if k in MARK_KEYS:   # a mark counts by its visible parts; sub-dims always present so dims stay aligned
            parts = v[5:].split(".") if is_mark else ["na"] * len(MARK_PARTS)
            for name, pv in zip(MARK_PARTS, parts):
                d[f"{k}.{name}"] = pv
            d[k] = "mark" if is_mark else v
        else:
            d[k] = v
    return desc["skeleton"], tuple(sorted((k, str(v)) for k, v in d.items()))


def weight(dim: str) -> int:
    return 2 if dim in HEAVY else 1


def distance(a: tuple, b: tuple) -> int:
    if a[0] != b[0]:
        return SKELETON_WEIGHT + 99  # different skeleton: never a conflict
    da, db = dict(a[1]), dict(b[1])
    return sum(weight(k) for k in da.keys() | db.keys() if da.get(k, "na") != db.get(k, "na"))


class Registry:
    def __init__(self, min_dist: int = 4):
        self.min_dist = min_dist
        self.items: list[tuple] = []
        self.index: dict[tuple, list[int]] = defaultdict(list)

    def _blocks(self, desc: tuple):
        sk, dims = desc
        k = self.min_dist
        for b in range(k):
            yield (sk, b, tuple(v for i, (_, v) in enumerate(dims) if i % k == b))

    def conflicts(self, desc: tuple) -> list[int]:
        seen, out = set(), []
        for key in self._blocks(desc):
            for j in self.index.get(key, ()):
                if j not in seen:
                    seen.add(j)
                    if distance(desc, self.items[j]) < self.min_dist:
                        out.append(j)
        return out

    def try_issue(self, desc: tuple) -> int | None:
        if self.conflicts(desc):
            return None
        i = len(self.items)
        self.items.append(desc)
        for key in self._blocks(desc):
            self.index[key].append(i)
        return i

    def __len__(self):
        return len(self.items)
