"""Automatic main flowline (centerline) of Chhota Shigri -- no manual drawing.

The official RGI7 centerline product (RGI2000-v7.0-L) is behind an NASA Earthdata login, so the
main flowline is generated with a least-cost-path approach in the spirit of Kienholz et al.
(2014, The Cryosphere 8, 503-519), the algorithm family used for the RGI7 centerlines:

  * glacier mask G = RGI7 (2002) outline, which encloses all later (smaller) extents
  * terminus T = RGI7 attribute (termlon, termlat), snapped to the lowest G pixel within 150 m
  * heads = local elevation maxima along the G boundary (>= 60 % of the elevation range,
    >= 500 m apart)
  * cost = 1 + 200 * ((Dmax - D) / Dmax)^4, D = distance to the glacier margin, which keeps the path
    in the middle of the ice (Kienholz et al.'s "distance-to-margin" term)
  * main flowline = the longest of the least-cost paths T -> head, smoothed (running mean, 150 m)
    and resampled to 10 m; arc length s is measured from the head (s = 0) downstream.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import scipy.ndimage as ndi
from rasterio.features import rasterize
from shapely.geometry import LineString, Point
from skimage.graph import MCP_Geometric

from src.common import ROOT, Grid, get_logger, read_raster

log = get_logger("centerline")
INV = ROOT / "data" / "inventories"


def build_centerline(grid: Grid, extra_mask=None):
    gpkg = INV / "chhota_shigri.gpkg"
    rgi = gpd.read_file(gpkg, layer="rgi7")
    elev = read_raster(ROOT / "data" / "dem" / "elevation.tif")
    G = rasterize(rgi.geometry, out_shape=(grid.height, grid.width), transform=grid.transform).astype(bool)
    if extra_mask is not None:
        G |= extra_mask
    D = ndi.distance_transform_edt(G) * grid.res
    cost = np.where(G, 1 + 200 * ((D.max() - D) / D.max()) ** 4, np.inf)

    # terminus from the inventory attribute
    t_ll = gpd.GeoSeries([Point(rgi.termlon.iloc[0], rgi.termlat.iloc[0])], crs=4326).to_crs(grid.crs).iloc[0]
    inv = ~grid.transform
    tc, tr = inv * (t_ll.x, t_ll.y)
    tr, tc = int(tr), int(tc)
    rad = int(150 / grid.res)
    rr, cc = np.mgrid[max(tr - rad, 0):tr + rad + 1, max(tc - rad, 0):tc + rad + 1]
    ok = G[rr, cc]
    k = np.argmin(np.where(ok, elev[rr, cc], np.inf))
    term = (int(rr.flat[k]), int(cc.flat[k]))

    # heads: boundary local maxima of elevation
    boundary = G & ~ndi.binary_erosion(G)
    zmin, zmax = elev[G].min(), elev[G].max()
    zb = np.where(boundary, elev, -np.inf)
    locmax = (zb == ndi.maximum_filter(zb, size=int(500 / grid.res))) & boundary & (elev >= zmin + 0.6 * (zmax - zmin))
    heads = list(zip(*np.nonzero(locmax)))
    mcp = MCP_Geometric(cost)
    mcp.find_costs([term])
    best, best_len = None, -1
    for h in heads:
        path = np.array(mcp.traceback(h))
        seg = np.hypot(*np.diff(path, axis=0).T).sum() * grid.res
        if seg > best_len:
            best, best_len = path, seg
    # head first; smooth and resample
    path = best[::-1].astype(float)
    w = max(int(150 / grid.res), 1)
    sm = np.column_stack([ndi.uniform_filter1d(path[:, i], w, mode="nearest") for i in range(2)])
    xs, ys = grid.transform * (sm[:, 1] + 0.5, sm[:, 0] + 0.5)
    line = LineString(np.column_stack([xs, ys]))
    s = np.arange(0, line.length, grid.res)
    line = LineString([line.interpolate(v) for v in s])
    log.info("centerline length %.0f m, %d head candidates, terminus at %.0f m a.s.l.", line.length, len(heads), elev[term])
    gdf = gpd.GeoDataFrame(dict(name=["main_flowline"], length_m=[line.length], method=["least-cost (Kienholz-style)"],
                                head_elev=[float(elev[tuple(best[-1])])], term_elev=[float(elev[term])]),
                           geometry=[line], crs=grid.crs)
    gdf.to_file(gpkg, layer="centerline")
    return line


def load_centerline(grid: Grid):
    try:
        return gpd.read_file(INV / "chhota_shigri.gpkg", layer="centerline").geometry.iloc[0]
    except Exception:
        return build_centerline(grid)
