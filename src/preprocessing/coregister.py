"""Automatic co-registration assessment of all ensemble scenes (no manual tie points).

  python -m src.preprocessing.coregister

Every core-window scene of every analysis year is compared with a reference scene
(coregistration.reference_scene, default 2023-09-11) on *stable* terrain:
outside the analysis domain + 300 m and outside all RGI7 glaciers + 150 m, cloud-free in both
(s2cloudless mask), snow-free in both (NDSI < 0.2), slope < 35 deg. Images are converted to Sobel
gradient magnitude of B08 (robust to illumination and seasonal radiometric change), and the sub-pixel
offset is found by upsampled phase cross-correlation (Guizar-Sicairos et al. 2008, upsample 20) on
256 px windows with >= 50 % stable pixels. Reported: median shift, 1.4826*MAD, number of windows.

Scenes with |shift| > apply_threshold_px are listed as needing correction (Fourier shift of the
optical stack before inference); Sentinel-2 L1C multi-temporal registration since the GRI-based
refinement (collection-1 reprocessing) is typically well below this threshold.
Output: outputs/statistics/coregistration.csv
"""
from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from rasterio.features import rasterize
from skimage.registration import phase_cross_correlation

from src.common import ROOT, Grid, get_logger, load_config, read_raster
from src.inference.ensemble import in_window, scene_cloud_mask
from src.postprocessing.mask import reference_masks

log = get_logger("coreg")
S2 = ROOT / "data" / "sentinel2"


def grad(img):
    img = np.nan_to_num(img)
    return np.hypot(ndi.sobel(img, 0), ndi.sobel(img, 1))


def ndsi(o):
    return (o[1] - o[4]) / np.maximum(o[1] + o[4], 1e-6)


def estimate_shift(ref, mov, stable, win=256, step=128, min_stable=0.5):
    g_ref, g_mov = grad(ref), grad(mov)
    shifts = []
    H, W = ref.shape
    for r in range(0, H - win + 1, step):
        for c in range(0, W - win + 1, step):
            m = stable[r:r + win, c:c + win]
            if m.mean() < min_stable:
                continue
            a = np.where(m, g_ref[r:r + win, c:c + win], 0)
            b = np.where(m, g_mov[r:r + win, c:c + win], 0)
            s, _, _ = phase_cross_correlation(a, b, upsample_factor=20, normalization=None)
            shifts.append(s)
    if not shifts:
        return np.array([np.nan, np.nan]), np.array([np.nan, np.nan]), 0
    s = np.array(shifts)
    med = np.median(s, 0)
    return med, 1.4826 * np.median(np.abs(s - med), 0), len(s)


def fourier_shift_stack(arr, shift):
    out = np.stack([np.real(np.fft.ifftn(ndi.fourier_shift(np.fft.fftn(np.nan_to_num(b)), shift))) for b in arr])
    return np.where(np.isfinite(arr), out, np.nan).astype(np.float32)


def apply_corrections(df, grid, cfg):
    """Shift flagged scenes onto the reference, keep originals, re-measure residual, drop prob caches."""
    import shutil
    from src.common import write_raster
    ref_date = df.reference.iloc[0]
    ref = read_raster(S2 / "scenes" / ref_date / "optical_toa.tif")
    for i, r in df[df.needs_correction].iterrows():
        sdir = S2 / "scenes" / r.scene
        orig = sdir / "optical_toa_uncorrected.tif"
        if not orig.exists():
            shutil.copy(sdir / "optical_toa.tif", orig)
            shutil.copy(sdir / "cloud_prob.tif", sdir / "cloud_prob_uncorrected.tif")
        mov = read_raster(orig)
        shift = (r.dy_px, r.dx_px)
        corr = fourier_shift_stack(mov, shift)
        write_raster(sdir / "optical_toa.tif", corr, grid, "float32")
        cp = read_raster(sdir / "cloud_prob_uncorrected.tif")
        write_raster(sdir / "cloud_prob.tif", np.clip(fourier_shift_stack(cp[None], shift)[0], 0, 1), grid, "float32")
        st = np.isfinite(corr[3]) & np.isfinite(ref[3])
        res, _, _ = estimate_shift(ref[3], corr[3], st & STABLE)
        df.loc[i, "residual_px"] = float(np.hypot(*res))
        df.loc[i, "residual_m"] = float(np.hypot(*res) * grid.res)
        df.loc[i, "corrected"] = True
        for cache in (ROOT / "experiments").glob(f"*/scene_probs/{r.scene}.tif"):
            cache.unlink()
        log.info("corrected %s by (%.2f, %.2f) px; residual %.2f px", r.scene, *shift, df.loc[i, "residual_px"])
    return df


def main(apply_threshold_px=0.25, apply=False):
    cfg = load_config()
    grid = Grid.load(ROOT / "data" / "inventories" / "grid.json")
    domain, _, _ = reference_masks(grid)
    slope = read_raster(ROOT / "data" / "dem" / "slope.tif")
    allg = gpd.read_file(ROOT / "data" / "inventories" / "rgi7_bbox.geojson").to_crs(grid.crs)
    G = rasterize(allg.geometry.buffer(150), out_shape=(grid.height, grid.width), transform=grid.transform).astype(bool)
    terrain = ~ndi.binary_dilation(domain, iterations=30) & ~G & (slope < 35)
    global STABLE
    STABLE = terrain
    ref_date = cfg.get("coregistration", {}).get("reference_scene", "2023-09-11")
    ref = read_raster(S2 / "scenes" / ref_date / "optical_toa.tif")
    ref_cloud = scene_cloud_mask(S2 / "scenes" / ref_date, grid, cfg)
    win = cfg["ensemble"].get("window_core", cfg["ensemble"]["window"])
    rows = []
    for year in cfg["sentinel2"]["years"]:
        dates = [d for d in json.loads((S2 / f"ensemble_{year}.json").read_text())["dates"] if in_window(d, win)]
        for d in dates:
            mov = read_raster(S2 / "scenes" / d / "optical_toa.tif")
            cm = scene_cloud_mask(S2 / "scenes" / d, grid, cfg)
            st = terrain & ~ref_cloud & ~cm & (ndsi(ref) < 0.2) & (ndsi(mov) < 0.2) & np.isfinite(mov[3]) & np.isfinite(ref[3])
            if d == ref_date:
                med, mad, n = np.zeros(2), np.zeros(2), 0
            else:
                med, mad, n = estimate_shift(ref[3], mov[3], st)
            mag_px = float(np.hypot(*med))
            rows.append(dict(year=year, scene=d, reference=ref_date, dy_px=med[0], dx_px=med[1], mad_dy_px=mad[0],
                             mad_dx_px=mad[1], shift_px=mag_px, shift_m=mag_px * grid.res, n_windows=n,
                             stable_frac=float(st.mean()), needs_correction=bool(mag_px > apply_threshold_px)))
            log.info("%s vs %s: (dy,dx)=(%.2f,%.2f) px = %.1f m [MAD %.2f,%.2f; %d win]", d, ref_date, med[0], med[1],
                     mag_px * grid.res, mad[0], mad[1], n)
    df = pd.DataFrame(rows)
    df["corrected"] = False
    df["residual_m"] = df.shift_m
    if apply and df.needs_correction.any():
        df = apply_corrections(df, grid, cfg)
    out = ROOT / "outputs" / "statistics" / "coregistration.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    # per-year registration uncertainty used for terminus error budgets: max shift of the year's scenes
    per_year = df.groupby("year").agg(residual_m=("residual_m", "max"), max_shift_before_m=("shift_m", "max"), mean_shift_m=("shift_m", "mean"),
                                      n_scenes=("scene", "count"), any_needs_correction=("needs_correction", "any"))
    per_year.to_csv(ROOT / "outputs" / "statistics" / "coregistration_by_year.csv")
    print(per_year.round(2).to_string())
    return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Fourier-shift scenes above the threshold")
    main(apply=ap.parse_args().apply)
