"""Automatic probability -> glacier mask -> polygon conversion (no manual editing).

Steps (parameters from configs/chhota_shigri.yaml:postprocessing):
  1. threshold P(glacier) >= prob_threshold
  2. median filter (size = median_smoothing), as in the official GlaViTU deploy script
  3. restrict to the fixed analysis domain (RGI7 outline + buffer, minus neighbouring glaciers)
  4. keep connected components (8-connectivity) that intersect the *core* of the inventory
     outline (RGI7 eroded by core_erosion_m). The core only identifies *which* ice body is
     Chhota Shigri; the margins, terminus and extent are entirely model-derived.
  5. remove objects < min_object_px, fill holes < min_hole_px
  6. polygonise, make_valid, drop slivers, topology checks
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import scipy.ndimage as ndi
from rasterio.features import rasterize, shapes
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely.validation import make_valid
from skimage.morphology import remove_small_holes, remove_small_objects

from src.common import ROOT, Grid

INV = ROOT / "data" / "inventories"


def reference_masks(grid: Grid, core_erosion_m=300.0):
    gpkg = INV / "chhota_shigri.gpkg"
    dom = gpd.read_file(gpkg, layer="domain").geometry
    rgi = gpd.read_file(gpkg, layer="rgi7").geometry
    shp = (grid.height, grid.width)
    domain = rasterize(dom, out_shape=shp, transform=grid.transform).astype(bool)
    rgi_mask = rasterize(rgi, out_shape=shp, transform=grid.transform).astype(bool)
    core = rasterize(rgi.buffer(-core_erosion_m), out_shape=shp, transform=grid.transform).astype(bool)
    return domain, rgi_mask, core


def clean_mask(prob, grid: Grid, cfg_pp: dict, domain=None, core=None):
    if domain is None or core is None:
        domain, _, core = reference_masks(grid)
    m = prob >= cfg_pp["prob_threshold"]
    if cfg_pp.get("median_smoothing"):
        m = ndi.median_filter(m.astype(np.uint8), size=cfg_pp["median_smoothing"]).astype(bool)
    m &= domain
    lab, n = ndi.label(m, structure=np.ones((3, 3)))
    keep = np.unique(lab[core & m])
    keep = keep[keep > 0]
    m = np.isin(lab, keep)
    m = remove_small_objects(m, min_size=cfg_pp["min_object_px"])
    m = remove_small_holes(m, area_threshold=cfg_pp["min_hole_px"])
    return m


def polygonise(mask, grid: Grid, min_area_m2=None):
    min_area_m2 = min_area_m2 if min_area_m2 is not None else 4 * grid.res ** 2
    geoms = [shape(g) for g, v in shapes(mask.astype(np.uint8), mask=mask, transform=grid.transform) if v == 1]
    geom = make_valid(unary_union(geoms))
    if geom.geom_type == "GeometryCollection":
        geom = unary_union([g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")])
    parts = [p for p in getattr(geom, "geoms", [geom]) if p.area >= min_area_m2]
    geom = unary_union(parts)
    checks = dict(is_valid=bool(geom.is_valid), n_parts=len(parts),
                  n_holes=sum(len(p.interiors) for p in parts),
                  raster_area_km2=float(mask.sum() * grid.res ** 2 / 1e6),
                  vector_area_km2=float(geom.area / 1e6))
    checks["raster_vector_area_diff_pct"] = 100 * abs(checks["vector_area_km2"] - checks["raster_area_km2"]) / max(checks["raster_area_km2"], 1e-9)
    return geom, checks
