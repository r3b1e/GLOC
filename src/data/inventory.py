"""Fetch RGI6/RGI7/GLIMS outlines from the GLIMS WFS and derive the study ROI and analysis domain.

Outputs (data/inventories/):
  raw WFS extracts            rgi6_bbox.geojson, rgi7_bbox.geojson, glims_bbox.geojson
  chhota_shigri.gpkg          layers: rgi6, rgi7, glims (all CS outlines), neighbours_rgi7, domain, roi
  grid.json                   the fixed 10 m analysis grid shared by every product

The *analysis domain* is the RGI7 outline buffered by `domain_buffer_m`, minus neighbouring RGI7
glaciers. It keeps ice-divide partitioning fixed through time so that area differences reflect
margin changes, not inventory-specific basin definitions (GLIMS holds CS outlines from 11.6 to
16.8 km2 for 2000-2009, mostly due to differing divides).
"""
from __future__ import annotations

import urllib.parse
import urllib.request

import geopandas as gpd

from src.common import ROOT, Grid, get_logger, load_config

log = get_logger("inventory")
INV = ROOT / "data" / "inventories"

LAYERS = {
    "rgi6": "GLIMS:RGI_SouthAsiaWest",
    "rgi7": "GLIMS:RGI2000-v7.0-G-14_south_asia_west_epsg3857",
    "glims": "GLIMS:GLIMS_Glacier_Outlines",
}


def fetch_wfs(cfg, key):
    out = INV / f"{key}_bbox.geojson"
    if out.exists() and out.stat().st_size > 1000:
        return out
    b = cfg["inventories"]["bbox_lonlat"]
    q = dict(service="WFS", version="1.1.0", request="GetFeature", typeName=LAYERS[key],
             outputFormat="application/json", srsName="EPSG:4326",
             bbox=f"{b[0]},{b[1]},{b[2]},{b[3]},EPSG:4326")
    url = cfg["inventories"]["wfs"] + "?" + urllib.parse.urlencode(q)
    log.info("WFS %s", key)
    urllib.request.urlretrieve(url, out)
    return out


def main():
    cfg = load_config()
    site, crs = cfg["site"], cfg["grid"]["crs"]
    INV.mkdir(parents=True, exist_ok=True)
    rgi6 = gpd.read_file(fetch_wfs(cfg, "rgi6")).to_crs(crs)
    rgi7 = gpd.read_file(fetch_wfs(cfg, "rgi7")).to_crs(crs)
    glims = gpd.read_file(fetch_wfs(cfg, "glims")).to_crs(crs)

    cs6 = rgi6[rgi6.RGIID == site["rgi6_id"]]
    cs7 = rgi7[rgi7.rgi_id == site["rgi7_id"]]
    assert len(cs6) == 1 and len(cs7) == 1, "Chhota Shigri not found in inventory extract"
    cs_glims = glims[glims.glac_id.isin(site["glims_ids"]) & (glims.line_type == "glac_bound")].copy()
    cs_glims["area_km2"] = cs_glims.area / 1e6
    # GLIMS rows are split per polygon part; keep largest part per analysis as the main outline
    cs_glims = cs_glims.sort_values("area_km2", ascending=False).drop_duplicates("anlys_id")

    geom7 = cs7.geometry.iloc[0]
    neighbours = rgi7[(rgi7.rgi_id != site["rgi7_id"]) &
                      rgi7.intersects(geom7.buffer(cfg["grid"]["roi_buffer_m"]))]
    domain = geom7.buffer(cfg["grid"]["domain_buffer_m"]).difference(neighbours.union_all())
    roi = geom7.buffer(cfg["grid"]["roi_buffer_m"]).envelope
    grid = Grid.from_bounds(crs, roi.bounds, cfg["grid"]["resolution"])
    grid.save(INV / "grid.json")

    gpkg = INV / "chhota_shigri.gpkg"
    cs6.to_file(gpkg, layer="rgi6")
    cs7.to_file(gpkg, layer="rgi7")
    cs_glims.to_file(gpkg, layer="glims")
    neighbours.to_file(gpkg, layer="neighbours_rgi7")
    gpd.GeoDataFrame(geometry=[domain], crs=crs).to_file(gpkg, layer="domain")
    gpd.GeoDataFrame(geometry=[roi], crs=crs).to_file(gpkg, layer="roi")

    log.info("RGI6 %s area %.3f km2 (src %s)", site["rgi6_id"], cs6.geometry.area.iloc[0] / 1e6, cs6.BGNDATE.iloc[0])
    log.info("RGI7 %s area %.3f km2 (src %s)", site["rgi7_id"], geom7.area / 1e6, cs7.src_date.iloc[0])
    for _, r in cs_glims.sort_values("src_date").iterrows():
        log.info("GLIMS anlys %s %s %-40s %.3f km2", r.anlys_id, str(r.src_date)[:10], str(r.chief_affl).replace("\n", " ")[:40], r.area_km2)
    log.info("domain %.2f km2, %d neighbours, grid %dx%d px", domain.area / 1e6, len(neighbours), grid.width, grid.height)


if __name__ == "__main__":
    main()
