"""Procedural halo generator.

Robust geometry: every boolean operation (union / difference / intersection / symmetric difference, unary union)
runs with GEOS fixed-precision overlay (snap-rounding to GRID). Floating-point overlay can raise
TopologyException on near-degenerate inputs; older GEOS builds (e.g. shapely 2.0 in Pyodide, used by the web page)
hit this on some halos, and in Pyodide such a C++ exception is fatal to the whole interpreter. GRID is far below
anything visible (halo radius ~1).
"""
import shapely
import shapely.ops
from shapely.geometry.base import BaseGeometry

GRID = 1e-7
_EMPTY = shapely.Polygon()


def _grid(grid_size):
    return GRID if grid_size is None else grid_size


def _area(g):
    """Keep only the polygonal part: fixed-precision overlay rejects mixed-dimension input, and stray lines /
    points from earlier operations never draw anyway."""
    if g is None or g.is_empty:
        return _EMPTY
    t = g.geom_type
    if t in ("Polygon", "MultiPolygon"):
        return g
    if t == "GeometryCollection":
        parts = [p for p in g.geoms if p.geom_type in ("Polygon", "MultiPolygon")]
        return shapely.union_all(parts, grid_size=GRID) if parts else _EMPTY
    return _EMPTY


def _op(fn):
    def method(self, other, grid_size=None):
        a, b = _area(self), _area(other)
        if a.is_empty and self.geom_type not in ("Polygon", "MultiPolygon", "GeometryCollection"):
            # non-areal geometry (lines used for buffering etc.): fall back to the plain operation
            return fn(self, other)
        return _area(fn(a, b, grid_size=_grid(grid_size)))
    return method


def _unary_union(geoms, grid_size=None):
    if isinstance(geoms, BaseGeometry):
        geoms = [geoms]
    return _area(shapely.union_all([_area(g) for g in geoms], grid_size=_grid(grid_size)))


_union = _op(shapely.union)
_difference = _op(shapely.difference)
_intersection = _op(shapely.intersection)
_symmetric_difference = _op(shapely.symmetric_difference)


BaseGeometry.union = _union
BaseGeometry.difference = _difference
BaseGeometry.intersection = _intersection
BaseGeometry.symmetric_difference = _symmetric_difference
shapely.ops.unary_union = _unary_union


_orig_buffer = BaseGeometry.buffer


def _valid(g):
    """Repair to valid polygonal geometry without buffer(0) (old GEOS can throw on self-crossing input):
    make_valid, keep the polygonal parts, snap to GRID."""
    if shapely.is_valid(g) and g.geom_type in ("Polygon", "MultiPolygon"):
        return g
    v = shapely.make_valid(g)
    polys = []
    for p in shapely.get_parts(v):
        if p.geom_type == "Polygon":
            polys.append(p)
        elif p.geom_type == "MultiPolygon":
            polys.extend(shapely.get_parts(p))
    if not polys:
        return _EMPTY
    # make_valid pieces do not overlap: collect them, no overlay needed
    return polys[0] if len(polys) == 1 else shapely.MultiPolygon(polys)


def _buffer(self, distance, *args, **kwargs):
    """Buffer is not covered by fixed-precision overlay; old GEOS (Pyodide) throws TopologyException on many
    near-degenerate inputs, and there a C++ exception kills the interpreter. Rules (same on every platform, so a seed
    gives the same halo everywhere):
      distance 0 (repair)                   -> make_valid + snapped union
      round buffer of polygons / long lines -> union of per-segment capsules (cannot throw)
      styled buffer (mitre / flat) or short lines -> native, on grid-snapped input (simple primitives only)"""
    if self.is_empty:
        return _EMPTY
    t = self.geom_type
    if t in ("Polygon", "MultiPolygon", "GeometryCollection"):
        g = _valid(self)
        if g.is_empty or not distance:
            return g
        styled = any(k in kwargs and kwargs[k] not in (1, "round") for k in ("join_style", "cap_style"))
        if not styled:
            return _segment_buffer(g, distance)
        return _orig_buffer(shapely.set_precision(g, GRID), distance, *args, **kwargs)
    if t in ("LineString", "MultiLineString", "LinearRing"):
        g = shapely.set_precision(self, GRID)
        if g.is_empty:
            return _EMPTY
        styled = any(k in kwargs and kwargs[k] not in (1, "round") for k in ("join_style", "cap_style"))
        if not styled and shapely.get_num_coordinates(g) > 50:
            return _segment_buffer(g, distance)
        return _orig_buffer(g, distance, *args, **kwargs)
    return _orig_buffer(self, distance, *args, **kwargs)


COMPLEX = 600   # coordinates; above this, buffers are built from per-segment capsules (never throws, even in old GEOS)


def _segment_buffer(g, distance, quad_segs=8):
    """Robust buffer: union of round capsules around every boundary segment (buffering a 2-point line cannot fail),
    merged with fixed-precision overlay. Positive: shape + capsules; negative: shape - capsules. Round joins/caps."""
    import numpy as np
    parts = shapely.get_parts(g)
    rings = shapely.get_rings(parts) if g.geom_type in ("Polygon", "MultiPolygon") else parts
    segs = []
    tol = min(abs(distance) * 0.25, 0.002)   # invisible (<1 px at 1024 px); drops thousands of micro-edges
    for ring in np.atleast_1d(rings):
        c = shapely.get_coordinates(shapely.simplify(ring, tol))
        if len(c) >= 2:
            segs.append(np.stack([c[:-1], c[1:]], axis=1))
    if not segs:
        return _EMPTY if distance < 0 else g
    segs = np.concatenate(segs)
    segs = segs[np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1) > 1e-9]
    caps = shapely.buffer(shapely.linestrings(segs), abs(distance), quad_segs=3)
    stroke = shapely.union_all(caps, grid_size=GRID)
    if g.geom_type not in ("Polygon", "MultiPolygon"):
        return _area(stroke)
    if distance > 0:
        return _area(shapely.union_all([g, stroke], grid_size=GRID))
    return _area(shapely.difference(g, stroke, grid_size=GRID))


BaseGeometry.buffer = _buffer
