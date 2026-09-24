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
