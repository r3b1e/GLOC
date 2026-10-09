"""Copernicus DEM GLO-30 -> elevation, slope, aspect, hillshade on the analysis grid.

Terrain derivatives are computed with Horn's (1981) method on a 30 m UTM version of the grid
(close to the native DEM resolution, avoiding interpolation artefacts from 10 m resampling),
then bilinearly resampled to 10 m. Heights are orthometric (EGM2008), as distributed.
Only elevation and slope are model inputs; aspect and hillshade are used for analysis/figures.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import planetary_computer as pc
import pystac_client
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.warp import reproject

from src.common import ROOT, Grid, get_logger, load_config, write_raster

log = get_logger("dem")
OUT = ROOT / "data" / "dem"
INV = ROOT / "data" / "inventories"


def horn(z, res):
    zp = np.pad(z, 1, mode="edge")
    a, b, c = zp[:-2, :-2], zp[:-2, 1:-1], zp[:-2, 2:]
    d, f = zp[1:-1, :-2], zp[1:-1, 2:]
    g, h, i = zp[2:, :-2], zp[2:, 1:-1], zp[2:, 2:]
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * res)      # +x = east
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * res)      # +y = south (row direction)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    # aspect: direction the slope faces, clockwise from north
    aspect = (np.degrees(np.arctan2(-dzdx, dzdy)) + 180.0) % 360.0
    return slope, aspect, dzdx, dzdy


def hillshade(dzdx, dzdy, az=315.0, alt=45.0):
    az, alt = np.radians(360 - az + 90), np.radians(alt)
    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(dzdy, -dzdx)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(hs, 0, 1)


def main():
    cfg = load_config()
    grid = Grid.load(INV / "grid.json")
    OUT.mkdir(parents=True, exist_ok=True)
    cat = pystac_client.Client.open(cfg["sentinel2"]["stac"], modifier=pc.sign_inplace)
    roi = gpd.read_file(INV / "chhota_shigri.gpkg", layer="roi").to_crs(4326).geometry.iloc[0].buffer(0.02)
    items = list(cat.search(collections=[cfg["dem"]["collection"]], intersects=roi).items())
    log.info("DEM tiles: %s", [i.id for i in items])
    srcs = [rasterio.open(i.assets["data"].href) for i in items]
    mosaic, mtrans = merge(srcs)
    src_crs = srcs[0].crs

    def to_grid(res, resampling):
        h, w = int(grid.height * grid.res / res), int(grid.width * grid.res / res)
        dst = np.zeros((h, w), np.float32)
        reproject(mosaic[0], dst, src_transform=mtrans, src_crs=src_crs,
                  dst_transform=rasterio.transform.from_origin(grid.xmin, grid.ymax, res, res),
                  dst_crs=grid.crs, resampling=resampling)
        return dst

    elev10 = to_grid(grid.res, Resampling.bilinear)
    elev30 = to_grid(30.0, Resampling.cubic)
    slope30, aspect30, dx30, dy30 = horn(elev30, 30.0)
    hs30 = hillshade(dx30, dy30)

    def up(a, circular=False):
        # bilinear upsampling 30 m -> 10 m; aspect via unit vectors to respect circularity
        import cv2
        size = (grid.width, grid.height)
        if circular:
            r = np.radians(a)
            s = cv2.resize(np.sin(r), size, interpolation=cv2.INTER_LINEAR)
            c = cv2.resize(np.cos(r), size, interpolation=cv2.INTER_LINEAR)
            return (np.degrees(np.arctan2(s, c)) + 360) % 360
        return cv2.resize(a, size, interpolation=cv2.INTER_LINEAR)

    write_raster(OUT / "elevation.tif", elev10, grid, dtype="float32")
    write_raster(OUT / "slope.tif", up(slope30), grid, dtype="float32")
    write_raster(OUT / "aspect.tif", up(aspect30, circular=True), grid, dtype="float32")
    write_raster(OUT / "hillshade.tif", up(hs30), grid, dtype="float32")
    (OUT / "dem_source.txt").write_text("\n".join(i.id for i in items) + "\n")
    log.info("elevation %.0f-%.0f m, slope max %.1f deg", elev10.min(), elev10.max(), slope30.max())


if __name__ == "__main__":
    main()
