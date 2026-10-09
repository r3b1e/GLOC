"""Sentinel-1 backscatter composites for M3 (optical + DEM + SAR), Chhota Shigri grid.

  python -m src.data.sentinel1 --years 2017 2019 2021 2023 2025

GlaViTU's SAR input is `co_pol_sar` = sigma0 VV in *linear* scale, two bands (ascending,
descending), normalised by compile_features.py as (log10(x + 1e-6) - min) / (max - min).
The benchmark was built from ellipsoid-based sigma0 (geometrically terrain-corrected GRD).

Planetary Computer offers `sentinel-1-rtc` (radiometrically terrain-corrected gamma0_T) and raw
`sentinel-1-grd` (needs SNAP-type processing). We use RTC and convert to an approximate sigma0:

    sigma0 ~= gamma0_T * cos(theta_inc)

with theta_inc (ellipsoid incidence angle) interpolated from the GRD annotation geolocation grid
of the same acquisition (~37-39 deg over Chhota Shigri). This undoes the gamma normalisation but
NOT the terrain-area flattening, so slope-dependent radiometry differs from the benchmark: the
Chhota Shigri M3 run is an *approximate-input* experiment (docs/experiments.md).

Composite: per orbit direction, per-pixel median of linear sigma0 over the optical ensemble
window (ensemble.window_core) of each year. Outputs data/sentinel1/<year>/co_pol_sar.tif (2 bands)
and scenes.json (IDs, orbits, incidence angles).
"""
from __future__ import annotations

import argparse
import json
import urllib.request
import xml.etree.ElementTree as ET

import geopandas as gpd
import numpy as np
import planetary_computer as pc
import pystac_client
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform as warp_transform
from scipy.interpolate import griddata

from src.common import ROOT, Grid, get_logger, load_config, write_raster

log = get_logger("sentinel1")
OUT = ROOT / "data" / "sentinel1"
INV = ROOT / "data" / "inventories"


def incidence_on_grid(cat, rtc_item, grid):
    grd_id = rtc_item.id.replace("_rtc", "")
    try:
        grd = cat.get_collection("sentinel-1-grd").get_item(grd_id)
        href = grd.assets["schema-product-vv"].href
        base, q = href.split("?")
        url = base.replace("/annotation/rfi/rfi-", "/annotation/") + "?" + q
        r = ET.fromstring(urllib.request.urlopen(url).read())
        pts = np.array([[float(p.find("longitude").text), float(p.find("latitude").text), float(p.find("incidenceAngle").text)]
                        for p in r.findall(".//geolocationGridPoint")])
    except Exception as e:  # fall back to swath-centre value, flagged
        log.warning("no annotation for %s (%s); using 38.0 deg", grd_id, e)
        return np.full((grid.height, grid.width), 38.0, np.float32), False
    cols, rows = np.meshgrid(np.arange(0, grid.width, 50), np.arange(0, grid.height, 50))
    xs = grid.xmin + (cols + 0.5) * grid.res
    ys = grid.ymax - (rows + 0.5) * grid.res
    lon, lat = warp_transform(grid.crs, "EPSG:4326", xs.ravel(), ys.ravel())
    inc = griddata(pts[:, :2], pts[:, 2], (np.array(lon), np.array(lat)), method="linear").reshape(xs.shape)
    import cv2
    return cv2.resize(inc.astype(np.float32), (grid.width, grid.height), interpolation=cv2.INTER_LINEAR), True


def read_rtc(href, grid):
    dst = np.full((grid.height, grid.width), np.nan, np.float32)
    with rasterio.open(href) as src:
        reproject(rasterio.band(src, 1), dst, dst_transform=grid.transform, dst_crs=grid.crs,
                  resampling=Resampling.bilinear, src_nodata=src.nodata if src.nodata is not None else 0, dst_nodata=np.nan)
    dst[dst <= 0] = np.nan
    return dst


def composite_year(cfg, cat, year, grid):
    win = cfg["ensemble"].get("window_core", cfg["ensemble"]["window"])
    roi = gpd.read_file(INV / "chhota_shigri.gpkg", layer="roi").to_crs(4326).geometry.iloc[0]
    items = list(cat.search(collections=["sentinel-1-rtc"], intersects=roi,
                            datetime=f"{year}-{win[0]}/{year}-{win[1]}").items())
    stacks = {"ascending": [], "descending": []}
    meta = []
    for it in items:
        direction = it.properties.get("sat:orbit_state")
        if direction not in stacks or "vv" not in it.assets:
            continue
        g0 = read_rtc(it.assets["vv"].href, grid)
        if np.isfinite(g0).mean() < 0.5:
            continue
        inc, ok = incidence_on_grid(cat, it, grid)
        s0 = g0 * np.cos(np.radians(inc))
        stacks[direction].append(s0)
        meta.append(dict(id=it.id, date=str(it.datetime.date()), orbit_state=direction,
                         relative_orbit=it.properties.get("sat:relative_orbit"),
                         incidence_mean_deg=float(np.nanmean(inc)), incidence_from_annotation=ok,
                         valid_frac=float(np.isfinite(g0).mean())))
        log.info("%d %s %s inc %.1f valid %.2f", year, it.id[:33], direction, np.nanmean(inc), np.isfinite(g0).mean())
    if not stacks["ascending"] or not stacks["descending"]:
        log.error("%d: missing an orbit direction (asc %d, desc %d)", year, len(stacks["ascending"]), len(stacks["descending"]))
        return None
    with np.errstate(all="ignore"):
        comp = np.stack([np.nanmedian(np.stack(stacks[d]), 0) for d in ("ascending", "descending")])
    comp = np.where(np.isfinite(comp), comp, np.nanmedian(comp)).astype(np.float32)  # layover/shadow gaps -> median
    out = OUT / str(year)
    out.mkdir(parents=True, exist_ok=True)
    write_raster(out / "co_pol_sar.tif", comp, grid, "float32")
    (out / "scenes.json").write_text(json.dumps(dict(year=year, window=win, quantity="sigma0 ~ gamma0_T*cos(inc), linear",
                                                     band_order=["ascending", "descending"], scenes=meta), indent=2))
    log.info("%d: co_pol_sar composite from %d asc + %d desc scenes", year, len(stacks["ascending"]), len(stacks["descending"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, required=True)
    a = ap.parse_args()
    cfg = load_config()
    grid = Grid.load(INV / "grid.json")
    cat = pystac_client.Client.open(cfg["sentinel2"]["stac"], modifier=pc.sign_inplace)
    for y in a.years:
        composite_year(cfg, cat, y, grid)


if __name__ == "__main__":
    main()
