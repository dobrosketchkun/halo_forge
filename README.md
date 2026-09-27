# Halo Forge

Procedural generator of Blue Archive–style halos: millions of variants, each different in ways you can see.
A seed is any text (a number, a word, a name); every seed gives one halo, the same everywhere (command line,
Python, web page - https://dobrosketchkun.github.io/halo_forge/).

![24 generated halos: reticle, flanked ring, crest, flower, star, pinwheel, kamon, heraldic, sigil, knot, crescent,
segmented, polygon frame, emblem and scatter types](docs/showcase.png)

*Seen from the default "above head" view. Each label is the seed (`python halo_cli.py --seed show17`, or
`?seed=show17` on the web page), the halo type and, if it is not a single plane, its 3D arrangement:
**stacked** (inner layers sink), **stand up** (top element rises), **wings** (side elements angle up),
**hang** (bottom element hangs down), **crown** (the ring is a vertical band), **orbit** (satellites at different
heights), **tilt** (inner ring tilted against the outer), **lotus** (main elements rise into a cone).*

## Web page

`index.html` is a static page for GitHub Pages. It runs the Python generator in the browser (Pyodide), so there is
no server. Features: random halo, halo by seed, a rotatable 3D preview (drag in any direction / scroll; preset views:
top, above head, side, isometric, edge), custom color (picker, hex, presets) or the seed's own color, image download
of any view (SVG, or PNG 512–2048 px, transparent or dark), 3D download (glTF binary, vector JSON), shareable links:
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
python halo_cli.py --seed 7 --view iso           # top | front (above head, default) | iso | side | edge
python halo_cli.py --seed 7 --view 30,40         # custom azimuth,elevation[,roll] in degrees
python halo_cli.py --seed 7 --gltf --json3d      # 3D model (.gltf) + vector 3D description (.halo3d.json)
python halo_cli.py --issue alice bob             # permanent, mutually distinct halos per person (registry.jsonl)
```

## Python

```python
from halo import api
rec = api.halo_for("tralala")      # {seed, type, color, desc, stage, geometry (flat), parts (3D)}
open("h.svg", "w").write(api.svg(rec, size=1024, view="iso"))
api.png(rec, "h.png", size=1024, view=(30, 40, 0))
open("h.gltf", "w").write(api.to_gltf(rec))
```

## 3D

Halos are 2.5D like the ones in Blue Archive art: the flat design is made of elements (frame, main element copies,
supports, centre), and each arrangement places whole elements in space: one tilted plane, inner layers that sink,
a top element that stands up, side elements angled out, a bottom element that hangs, a ring that becomes a vertical
band, satellites at different heights, an inner ring tilted against the outer, main elements rising into a cone,
plus occasional camera-facing sparkles. Every part stays an exact vector outline with a 3D placement, so every view
is an exact projection at any resolution. `halo3d.json`: `parts[].polygons` (outer ring + holes, local 2D) +
`parts[].matrix` (column-major 4x4, local z = halo normal) + `billboard` + `thickness`.

Geometry is made robust for the browser (see `halo/__init__.py`): the shapely/GEOS build in Pyodide is older and
would otherwise crash on near-degenerate shapes, so overlays use fixed precision and round buffers are built from
per-segment capsules. The same rules run everywhere, so a seed gives the same halo in the CLI and the browser.

`halo.issue.Issuer` issues one halo per person with a guaranteed minimum difference to every other issued halo and
freezes it in a registry, so later generator changes never alter an issued halo.

## Layout

| Path | What |
|---|---|
| `halo/api.py` | public entry: `halo_for(seed)`, `svg()`, `png()` |
| `halo/generate.py` | sampler: 15 halo types over shared layers (frame, main element, supports, center, marks, edges) |
| `halo/geom.py`, `build.py` | shape primitives and the grammar interpreter (symmetry, placement, boolean combine) |
| `halo/stage.py` | 3D arrangements (element-aware), exact projections, vector JSON and glTF export |
| `halo/critic.py` | rejects off-style samples (coverage, hierarchy, near-miss gaps) |
| `halo/marks.py`, `ornament.py`, `library.py` | house marks / tamgas, fret and knot ornaments, traced kamon + heraldry |
| `halo/unique.py`, `issue.py` | uniqueness distance and per-person issuance |
| `halo/data/` | traced kamon (public domain / CC0) and heraldic charges (Armoria, CC0 / CC BY-NC-SA), taste weights |
| `halo_cli.py`, `index.html` | command line and web page |

Licenses of traced motifs are recorded per item in `halo/data/*.json`. Some heraldic charges are CC BY-NC-SA:
non-commercial use only, with attribution.
