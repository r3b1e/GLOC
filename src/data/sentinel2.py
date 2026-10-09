"""Automatic end-of-ablation Sentinel-2 scene selection and L1C (TOA) extraction over the ROI.

Why two archives:
  * GlaViTU was trained on L1C top-of-atmosphere reflectance. Planetary Computer only serves L2A,
    so L2A is used for *discovery and screening* (SCL layer) and the model input is taken from the
    matching L1C product in the public Google Cloud Sentinel-2 bucket (anonymous HTTPS access).
  * Reflectance = (DN + RADIO_ADD_OFFSET) / QUANTIFICATION_VALUE, read from each product's
    MTD_MSIL1C.xml. This reproduces the harmonisation of COPERNICUS/S2_HARMONIZED
    (offset -1000 for processing baseline >= 04.00, 0 before).

Selection, per year:
  1. all L2A items in the ablation-season window; one item per datatake (latest reprocessing).
  2. SCL over the analysis domain -> cloud (8,9,10), cloud shadow (3), nodata (0/1) fractions,
     and seasonal-snow fraction (SCL 11) on *non-glacier* terrain (domain minus RGI7).
  3. the `n_confirm` best candidates are confirmed with s2cloudless on L1C bands.
  4. scenes with domain cloud+shadow > max_roi_cloud_frac are rejected; among the rest the
     lowest (cloud + shadow + off-glacier snow) wins. Fallback window used if nothing passes.
All candidate statistics are written to data/sentinel2/selection_<year>.csv.
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer as pc
import pystac_client
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject
from rasterio.windows import from_bounds

from src.common import ROOT, Grid, get_logger, load_config, write_raster

log = get_logger("sentinel2")
S2DIR = ROOT / "data" / "sentinel2"
INV = ROOT / "data" / "inventories"
GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".jp2,.tif,.xml",
                GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="3")
S2CLOUDLESS_BANDS = ["B01", "B02", "B04", "B05", "B08", "B8A", "B09", "B10", "B11", "B12"]


def domain_masks(grid: Grid):
    gpkg = INV / "chhota_shigri.gpkg"
    dom = gpd.read_file(gpkg, layer="domain").geometry
    rgi = gpd.read_file(gpkg, layer="rgi7").geometry
    shape = (grid.height, grid.width)
    domain = rasterize(dom, out_shape=shape, transform=grid.transform).astype(bool)
    glacier = rasterize(rgi, out_shape=shape, transform=grid.transform).astype(bool)
    return domain, glacier


def read_to_grid(href, grid: Grid, resampling=Resampling.nearest, res=None):
    """Read a raster window covering `grid` (optionally at a coarser `res`)."""
    res = res or grid.res
    h, w = int(round(grid.height * grid.res / res)), int(round(grid.width * grid.res / res))
    dst = np.zeros((h, w), np.float32)
    with rasterio.Env(**GDAL_ENV), rasterio.open(href) as src:
        if str(src.crs) == grid.crs:
            win = from_bounds(*grid.bounds, src.transform)
            return src.read(1, window=win, out_shape=(h, w), resampling=resampling, boundless=True).astype(np.float32)
        dst_t = rasterio.transform.from_origin(grid.xmin, grid.ymax, res, res)
        reproject(rasterio.band(src, 1), dst, dst_transform=dst_t, dst_crs=grid.crs, resampling=resampling)
    return dst


# ---------------------------------------------------------------- discovery / screening (L2A)
def search_l2a(cfg, year, window):
    cat = pystac_client.Client.open(cfg["sentinel2"]["stac"], modifier=pc.sign_inplace)
    roi = gpd.read_file(INV / "chhota_shigri.gpkg", layer="roi").to_crs(4326).geometry.iloc[0]
    items = cat.search(collections=[cfg["sentinel2"]["l2a_collection"]], intersects=roi,
                       datetime=f"{year}-{window[0]}/{year}-{window[1]}").item_collection()
    items = [i for i in items if i.properties.get("s2:mgrs_tile") == cfg["sentinel2"]["mgrs_tile"]]
    # one item per datatake: keep the latest processing
    by_take = {}
    for it in items:
        take = "_".join(it.id.split("_")[:4])
        if take not in by_take or it.id > by_take[take].id:
            by_take[take] = it
    return sorted(by_take.values(), key=lambda i: i.datetime)


def screen_scl(item, grid, domain, glacier):
    scl = read_to_grid(item.assets["SCL"].href, grid)
    d = domain
    off = domain & ~glacier
    frac = lambda m, where: float((m & where).sum() / max(where.sum(), 1))
    return dict(
        l2a_id=item.id,
        date=item.datetime.strftime("%Y-%m-%d"),
        datatake="_".join(item.id.split("_")[:3]).replace("MSIL2A", "MSIL1C"),
        scene_cloud_pct=item.properties.get("eo:cloud_cover"),
        sun_elevation=90 - float(item.properties.get("s2:mean_solar_zenith", np.nan)),
        scl_nodata=frac(np.isin(scl, [0, 1]), d),
        scl_cloud=frac(np.isin(scl, [8, 9, 10]), d),
        scl_shadow=frac(scl == 3, d),
        scl_snow_offglacier=frac(scl == 11, off),
    )


# ---------------------------------------------------------------- L1C access (Google bucket)
def find_l1c(cfg, datatake):
    """Return GCS prefix of the L1C SAFE for a datatake (latest reprocessing)."""
    tile = cfg["sentinel2"]["mgrs_tile"]
    base = f"tiles/{tile[:2]}/{tile[2]}/{tile[3:]}/"
    url = ("https://storage.googleapis.com/storage/v1/b/gcp-public-data-sentinel-2/o?"
           f"prefix={base}{datatake}&delimiter=/")
    prefixes = json.load(urllib.request.urlopen(url)).get("prefixes", [])
    prefixes = [p for p in prefixes if f"_T{tile}_" in p]
    if not prefixes:
        return None
    # prefer highest processing baseline (Nxxxx), then latest product discriminator
    return sorted(prefixes, key=lambda p: (re.search(r"_N(\d{4})_", p).group(1), p))[-1]


def l1c_files(prefix):
    url = ("https://storage.googleapis.com/storage/v1/b/gcp-public-data-sentinel-2/o?"
           f"prefix={prefix}GRANULE/&fields=items(name),nextPageToken")
    names = [i["name"] for i in json.load(urllib.request.urlopen(url)).get("items", [])]
    bands = {}
    for n in names:
        m = re.search(r"IMG_DATA/.*_(B\d[\dA])\.jp2$", n)
        if m:
            bands[m.group(1)] = n
    return bands


def l1c_radiometry(prefix):
    xml = urllib.request.urlopen(
        f"{load_config()['sentinel2']['l1c_bucket']}/{prefix}MTD_MSIL1C.xml").read()
    root = ET.fromstring(xml)
    quant = float(next(e.text for e in root.iter() if e.tag.endswith("QUANTIFICATION_VALUE")))
    offsets = {int(e.attrib["band_id"]): float(e.text) for e in root.iter() if e.tag.endswith("RADIO_ADD_OFFSET")}
    baseline = next((e.text for e in root.iter() if e.tag.endswith("PROCESSING_BASELINE")), None)
    return quant, offsets, baseline


BAND_INDEX = {"B01": 0, "B02": 1, "B03": 2, "B04": 3, "B05": 4, "B06": 5, "B07": 6, "B08": 7,
              "B8A": 8, "B09": 9, "B10": 10, "B11": 11, "B12": 12}


def fetch_l1c_bands(cfg, prefix, bands, grid, res=None, resampling=Resampling.bilinear):
    """Read L1C bands over the grid and convert to TOA reflectance."""
    files = l1c_files(prefix)
    quant, offsets, baseline = l1c_radiometry(prefix)
    bucket = cfg["sentinel2"]["l1c_bucket"]

    def one(b):
        dn = read_to_grid(f"/vsicurl/{bucket}/{files[b]}", grid, resampling=resampling, res=res)
        valid = dn > 0
        refl = (dn + offsets.get(BAND_INDEX[b], 0.0)) / quant
        return b, np.where(valid, refl, np.nan).astype(np.float32)

    with ThreadPoolExecutor(4) as ex:
        out = dict(ex.map(one, bands))
    return out, dict(quantification=quant, offset=offsets.get(1, 0.0), processing_baseline=baseline)


def s2cloudless_fraction(cfg, prefix, grid, domain):
    from s2cloudless import S2PixelCloudDetector
    res = 20.0
    refl, _ = fetch_l1c_bands(cfg, prefix, S2CLOUDLESS_BANDS, grid, res=res)
    stack = np.stack([np.nan_to_num(refl[b]) for b in S2CLOUDLESS_BANDS], -1)[None]
    det = S2PixelCloudDetector(threshold=cfg["sentinel2"]["s2cloudless_threshold"], average_over=4,
                               dilation_size=2, all_bands=False)
    prob = det.get_cloud_probability_maps(stack)[0]
    mask = det.get_mask_from_prob(prob[None])[0].astype(bool)
    dom20 = domain[::2, ::2][: mask.shape[0], : mask.shape[1]]
    return float(mask[dom20].mean()), prob


# ---------------------------------------------------------------- orchestration
def select_year(cfg, year, grid, domain, glacier, n_confirm=3):
    s2 = cfg["sentinel2"]
    rows = []
    for window in (s2["window"], s2["window_fallback"]):
        items = search_l2a(cfg, year, window)
        seen = {r["l2a_id"] for r in rows}
        rows += [screen_scl(i, grid, domain, glacier) for i in items if i.id not in seen]
        df = pd.DataFrame(rows)
        if df.empty:
            continue
        df["scl_bad"] = df.scl_cloud + df.scl_shadow + df.scl_nodata
        df = df.sort_values(["scl_bad", "scl_snow_offglacier"]).reset_index(drop=True)
        df["l1c_prefix"] = None
        df["s2cloudless_cloud"] = np.nan
        for i in df.index[:n_confirm]:
            if df.loc[i, "scl_bad"] > 0.25:
                continue
            prefix = find_l1c(cfg, df.loc[i, "datatake"])
            if prefix is None:
                log.warning("no L1C for %s", df.loc[i, "datatake"])
                continue
            df.loc[i, "l1c_prefix"] = prefix
            cached = S2DIR / "scenes" / df.loc[i, "date"] / "cloud_prob.tif"
            if cached.exists():
                from s2cloudless import S2PixelCloudDetector
                det = S2PixelCloudDetector(threshold=s2["s2cloudless_threshold"], average_over=4, dilation_size=2, all_bands=False)
                with rasterio.open(cached) as src:
                    cp20 = src.read(1)[::2, ::2]
                m20 = det.get_mask_from_prob(cp20[None])[0].astype(bool)
                df.loc[i, "s2cloudless_cloud"] = float(m20[domain[::2, ::2]].mean())
            else:
                df.loc[i, "s2cloudless_cloud"], _ = s2cloudless_fraction(cfg, prefix, grid, domain)
            log.info("%s  scl_bad=%.4f  s2cloudless=%.4f  snow_off=%.3f", df.loc[i, "date"],
                     df.loc[i, "scl_bad"], df.loc[i, "s2cloudless_cloud"], df.loc[i, "scl_snow_offglacier"])
        df["cloud_final"] = df[["s2cloudless_cloud"]].max(axis=1) + df.scl_shadow + df.scl_nodata
        df["single_scene_pass"] = (df.cloud_final <= s2["max_roi_cloud_frac"]) & df.l1c_prefix.notna()
        df["score"] = df.cloud_final + df.scl_snow_offglacier
        df["window"] = f"{window[0]}..{window[1]}"
        df.to_csv(S2DIR / f"selection_{year}.csv", index=False)
        meta = composite_year(cfg, year, df, grid)
        if meta and meta["passed"]:
            return meta, df
        log.warning("%d: composite did not pass in window %s, trying fallback", year, window)
    return meta, df


def extract_scene(cfg, row, grid):
    """Download one L1C scene over the grid (cached in data/sentinel2/scenes/<date>/)."""
    out = S2DIR / "scenes" / row.date
    meta_f = out / "scene.json"
    if meta_f.exists():
        return json.loads(meta_f.read_text()), out
    out.mkdir(parents=True, exist_ok=True)
    bands = cfg["sentinel2"]["bands"]
    # 10 m bands align exactly with the grid; 20 m bands (B11, B12) are bilinearly upsampled
    refl, radiometry = fetch_l1c_bands(cfg, row.l1c_prefix, bands, grid)
    write_raster(out / "optical_toa.tif", np.stack([refl[b] for b in bands]), grid, dtype="float32")
    _, prob = s2cloudless_fraction(cfg, row.l1c_prefix, grid, domain_masks(grid)[0])
    prob10 = np.kron(prob, np.ones((2, 2)))[: grid.height, : grid.width]
    write_raster(out / "cloud_prob.tif", prob10, grid, dtype="float32")
    meta = dict(date=row.date, l2a_id=row.l2a_id, l1c_product=row.l1c_prefix.rstrip("/").split("/")[-1],
                bands=bands, units="TOA reflectance", sun_elevation=row.sun_elevation,
                domain_cloud_frac=row.cloud_final, offglacier_snow_frac=row.scl_snow_offglacier,
                window=row.window, **radiometry)
    meta_f.write_text(json.dumps(meta, indent=2, default=str))
    log.info("scene %s -> %s", row.date, meta["l1c_product"])
    return meta, out


def composite_year(cfg, year, df, grid, max_scenes=3):
    """Best-scene primary + gap-filling of its cloudy pixels from the next-best clear scenes.

    Cloud = s2cloudless probability >= threshold, dilated by cloud_dilation_px. Pixels still cloudy
    after all fill scenes are NaN in the optical stack and flagged in invalid.tif.
    """
    from scipy.ndimage import binary_dilation
    s2 = cfg["sentinel2"]
    cand = df[df.l1c_prefix.notna() & df.cloud_final.notna()].sort_values("score").head(max_scenes)
    if cand.empty:
        return None
    thr, dil = s2["s2cloudless_threshold"], s2["cloud_dilation_px"]
    domain, _ = domain_masks(grid)
    from s2cloudless import S2PixelCloudDetector
    det = S2PixelCloudDetector(threshold=thr, average_over=4, dilation_size=2, all_bands=False)

    def cloud_mask(cp10):
        # identical to the mask used for scene scoring: 20 m prob, 4 px averaging, 2 px dilation
        m20 = det.get_mask_from_prob(cp10[::2, ::2][None])[0].astype(bool)
        return np.kron(m20, np.ones((2, 2), bool))[: grid.height, : grid.width]
    optical = source = None
    metas = []
    for k, (_, row) in enumerate(cand.iterrows()):
        meta, sdir = extract_scene(cfg, row, grid)
        opt = np.stack([np.asarray(b) for b in __import__("rasterio").open(sdir / "optical_toa.tif").read()])
        cp = __import__("rasterio").open(sdir / "cloud_prob.tif").read(1)
        cloudy = cloud_mask(cp)
        if dil:
            cloudy = binary_dilation(cloudy, iterations=dil)
        cloudy |= ~np.isfinite(opt).all(0)
        if optical is None:
            optical = np.where(cloudy, np.nan, opt)
            source = np.where(cloudy, 0, 1).astype(np.uint8)
        else:
            fill = (source == 0) & ~cloudy
            optical[:, fill] = opt[:, fill]
            source[fill] = k + 1
        metas.append(meta)
        remaining = float(((source == 0) & domain).sum() / domain.sum())
        log.info("%d: after %s, unresolved cloud in domain %.4f", year, meta["date"], remaining)
        if remaining <= s2["max_roi_cloud_frac"] / 4:
            break
    out = S2DIR / str(year)
    out.mkdir(parents=True, exist_ok=True)
    write_raster(out / "optical_toa.tif", optical, grid, dtype="float32")
    write_raster(out / "source_index.tif", source, grid, dtype="uint8")  # 0 = invalid, k = k-th scene
    write_raster(out / "invalid.tif", (source == 0), grid, dtype="uint8")
    cp_primary = __import__("rasterio").open(S2DIR / "scenes" / metas[0]["date"] / "cloud_prob.tif").read(1)
    write_raster(out / "cloud_prob.tif", cp_primary, grid, dtype="float32")
    frac_primary = float(((source == 1) & domain).sum() / domain.sum())
    meta = dict(year=year, date=metas[0]["date"], l1c_product=metas[0]["l1c_product"], l2a_id=metas[0]["l2a_id"],
                composite=[dict(index=i + 1, **m) for i, m in enumerate(metas)],
                domain_frac_from_primary=frac_primary, domain_invalid_frac=remaining,
                passed=remaining <= s2["max_roi_cloud_frac"], units="TOA reflectance", bands=s2["bands"])
    (out / "scene.json").write_text(json.dumps(meta, indent=2, default=str))
    log.info("%d composite: primary %s covers %.1f %% of domain, invalid %.2f %%, passed=%s", year, meta["date"],
             100 * frac_primary, 100 * remaining, meta["passed"])
    return meta


def select_ensemble(cfg, year, grid, domain, glacier):
    """All usable scenes of a year for the multi-date ensemble (see src/inference/ensemble.py).

    Window = ensemble.window for every year (consistent seasonality). A scene is kept if
    s2cloudless cloud over the domain <= max_cloud and off-glacier snow <= max_offglacier_snow
    (fresh-snowfall scenes hide the ice margin and are excluded rather than down-weighted).
    """
    e = cfg["ensemble"]
    items = search_l2a(cfg, year, e["window"])
    rows = [screen_scl(i, grid, domain, glacier) for i in items]
    df = pd.DataFrame(rows)
    df["window"] = f"{e['window'][0]}..{e['window'][1]}"
    df["kept"] = False
    df["reason"] = ""
    df["s2cloudless_cloud"] = np.nan
    df["l1c_prefix"] = None
    for i in df.index:
        if df.loc[i, "scl_cloud"] > e["max_scl_cloud_prescreen"]:
            df.loc[i, "reason"] = "SCL cloud prescreen"
            continue
        if df.loc[i, "scl_snow_offglacier"] > e["max_offglacier_snow"]:
            df.loc[i, "reason"] = "fresh snow"
            continue
        prefix = find_l1c(cfg, df.loc[i, "datatake"])
        if prefix is None:
            df.loc[i, "reason"] = "no L1C"
            continue
        df.loc[i, "l1c_prefix"] = prefix
        df.loc[i, "cloud_final"] = np.nan
        df.loc[i, "scl_shadow"] = df.loc[i, "scl_shadow"]
        row = df.loc[i].copy()
        row["cloud_final"] = np.nan
        meta, sdir = extract_scene(cfg, row, grid)
        from s2cloudless import S2PixelCloudDetector
        det = S2PixelCloudDetector(threshold=cfg["sentinel2"]["s2cloudless_threshold"], average_over=4,
                                   dilation_size=2, all_bands=False)
        with rasterio.open(sdir / "cloud_prob.tif") as src:
            m20 = det.get_mask_from_prob(src.read(1)[::2, ::2][None])[0].astype(bool)
        cloud = float(m20[domain[::2, ::2]].mean())
        df.loc[i, "s2cloudless_cloud"] = cloud
        if cloud > e["max_cloud"]:
            df.loc[i, "reason"] = "s2cloudless cloud"
            continue
        df.loc[i, "kept"] = True
        log.info("%d ensemble: %s cloud %.3f snow_off %.3f", year, row.date, cloud, row.scl_snow_offglacier)
    df.to_csv(S2DIR / f"ensemble_{year}.csv", index=False)
    kept = df[df.kept].sort_values("date")
    (S2DIR / f"ensemble_{year}.json").write_text(json.dumps(
        dict(year=year, window=e["window"], dates=kept.date.tolist(), l2a_ids=kept.l2a_id.tolist(),
             l1c_products=[p.rstrip("/").split("/")[-1] for p in kept.l1c_prefix]), indent=2))
    log.info("%d ensemble: %d scenes kept of %d candidates", year, len(kept), len(df))
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="*", type=int)
    ap.add_argument("--n_confirm", type=int, default=3)
    ap.add_argument("--ensemble", action="store_true", help="select all usable scenes per year (option A)")
    args = ap.parse_args()
    if args.ensemble:
        cfg = load_config()
        S2DIR.mkdir(parents=True, exist_ok=True)
        grid = Grid.load(INV / "grid.json")
        domain, glacier = domain_masks(grid)
        for year in args.years or cfg["sentinel2"]["years"]:
            select_ensemble(cfg, year, grid, domain, glacier)
        return
    cfg = load_config()
    S2DIR.mkdir(parents=True, exist_ok=True)
    grid = Grid.load(INV / "grid.json")
    domain, glacier = domain_masks(grid)
    for year in args.years or cfg["sentinel2"]["years"]:
        meta, _ = select_year(cfg, year, grid, domain, glacier, args.n_confirm)
        if meta is None or not meta["passed"]:
            log.error("%d: NO USABLE COMPOSITE (see selection_%d.csv)", year, year)


if __name__ == "__main__":
    main()
