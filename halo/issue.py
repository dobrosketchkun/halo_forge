"""Issue halos to people: deterministic per person, critic-approved, min-distance unique, and FROZEN once issued.

    iss = Issuer("out/registry.jsonl", min_dist=4)
    halo = iss.issue("hunter-000123")      # -> dict(person, seed, desc, color, halo spec)

A person's candidate seeds are derived from a hash of their id, so a first issue is reproducible for a given
generator version. Once issued, the registry stores the FULL record (desc, color, halo spec, generator version):
later changes to the generator (new presets, re-weighted choices) never change anyone's issued halo, and the
uniqueness registry is rebuilt from the stored descriptors, not by re-sampling seeds.
Each candidate must (1) pass the style critic and (2) differ from every issued halo by >= min_dist.
The registry is an append-only JSONL file.
"""
from __future__ import annotations

import hashlib
import json
import os

from .critic import score
from .generate import GENERATOR_VERSION, sample
from .unique import Registry, descriptor

MAX_ATTEMPTS = 400


def seed_for(person: str, attempt: int) -> int:
    h = hashlib.blake2b(f"{person}#{attempt}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") & ((1 << 62) - 1)


class Issuer:
    def __init__(self, path: str | None = None, min_dist: int = 4):
        self.path = path
        self.reg = Registry(min_dist)
        self.records: dict[str, dict] = {}
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    self.reg.try_issue(descriptor(rec["desc"]))
                    self.records[rec["person"]] = rec

    def issue(self, person: str) -> dict:
        if person in self.records:
            return self.records[person]
        for attempt in range(MAX_ATTEMPTS):
            seed = seed_for(person, attempt)
            it = sample(seed)
            desc = descriptor(it["desc"])
            if self.reg.conflicts(desc):          # cheap check first
                continue
            ok, _, _, _ = score(it["halo"])        # geometry + critic (expensive)
            if not ok:
                continue
            self.reg.try_issue(desc)
            rec = {"person": person, "seed": seed, "attempts": attempt + 1, "generator": GENERATOR_VERSION,
                   "color": it["color"], "desc": it["desc"], "halo": it["halo"]}
            self.records[person] = rec
            if self.path:
                os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return rec
        raise RuntimeError(f"no unique halo found for {person!r} in {MAX_ATTEMPTS} attempts (space saturating)")
