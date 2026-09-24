# Halo Forge

Procedural generator of Blue Archive–style halos: millions of variants, each different in ways you can see.
A seed is any text (a number, a word, a name); every seed gives one halo, the same everywhere (command line,
Python, web page - https://dobrosketchkun.github.io/halo_forge/).

![24 generated halos: reticle, flanked ring, crest, flower, star, pinwheel, kamon, heraldic, sigil, knot, crescent,
segmented, polygon frame, emblem and scatter types](docs/showcase.png)

*Seen from the default "above head" view. Each label is the seed (`python halo_cli.py --seed show17`, or
`?seed=show17` on the web page), the halo type and, if it is not a single plane, its 3D arrangement: **stacked**
(inner layers sink), **crossed** (top element stands up), **solid** (thick band), **fan** (blades angled).*

## Web page


`index.html` is a static page for GitHub Pages. It runs the Python generator in the browser (Pyodide), so there is
no server. Features: random halo, halo by seed, custom color (picker, hex, presets) or the seed's own color,
SVG / PNG download (512–2048 px, transparent or dark background), shareable links:
`https://<user>.github.io/<repo>/?seed=tralala` or with a color `?seed=tralala&color=ff8800`.

Publish: push the repo, then in GitHub **Settings → Pages** choose "Deploy from a branch", branch `main`, folder
`/ (root)`. The `.nojekyll` file is required (it keeps `halo/__init__.py` from being dropped by Jekyll).
Test locally: `python -m http.server 8000`, then open http://localhost:8000/?seed=12345.

## Command line

```
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt

python halo_cli.py                               # one random halo -> halo_<seed>.svg + .png
python halo_cli.py --seed 12345                  # a specific halo (any text: --seed tralala)
python halo_cli.py --count 20 -o halos/          # 20 random halos
python halo_cli.py --seed 7 --color ff8800          # custom color (#ff8800, ff8800 or #f80)
python halo_cli.py --seed 7 --svg-only --transparent --size 1024
python halo_cli.py --seed 7 --info               # what the halo is made of (JSON)
python halo_cli.py --issue alice bob             # permanent, mutually distinct halos per person (registry.jsonl)
```

## Python

```python
from halo import api
rec = api.halo_for("tralala")      # {seed, type, color, desc, halo (spec), geometry (shapely)}
open("h.svg", "w").write(api.svg(rec, size=1024))
api.png(rec, "h.png", size=1024)
```

Geometry uses GEOS fixed-precision overlay (see `halo/__init__.py`): the browser's shapely/GEOS build is older and
would otherwise crash on some near-degenerate shapes.

`halo.issue.Issuer` issues one halo per person with a guaranteed minimum difference to every other issued halo and
freezes it in a registry, so later generator changes never alter an issued halo.

## Layout

| Path | What |
|---|---|
| `halo/api.py` | public entry: `halo_for(seed)`, `svg()`, `png()` |
| `halo/generate.py` | sampler: 15 halo types over shared layers (frame, main element, supports, center, marks, edges) |
| `halo/geom.py`, `build.py` | shape primitives and the grammar interpreter (symmetry, placement, boolean combine) |
| `halo/critic.py` | rejects off-style samples (coverage, hierarchy, near-miss gaps) |
| `halo/marks.py`, `ornament.py`, `library.py` | house marks / tamgas, fret and knot ornaments, traced kamon + heraldry |
| `halo/unique.py`, `issue.py` | uniqueness distance and per-person issuance |
| `halo/data/` | traced kamon (public domain / CC0) and heraldic charges (Armoria, CC0 / CC BY-NC-SA), taste weights |
| `halo_cli.py`, `index.html` | command line and web page |

Licenses of traced motifs are recorded per item in `halo/data/*.json`. Some heraldic charges are CC BY-NC-SA:
non-commercial use only, with attribution.
