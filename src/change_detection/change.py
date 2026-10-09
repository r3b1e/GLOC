"""Multi-temporal change: area statistics, terminus retreat, year-of-loss map.

  python -m src.change_detection.change --model M2_cs_selftrain

Inputs: experiments/<model>/<year>/{mask.tif, prob_glacier.tif} for every processed year.
Outputs:
  outputs/statistics/<model>_area.csv        A(t), dA, annualised dA, loss %, uncertainties
  outputs/statistics/<model>_terminus.csv    centerline + box-method terminus positions/retreat
  outputs/change_maps/<model>_year_of_loss.tif
  outputs/polygons/<model>_terminus_points.gpkg

Uncertainty
  * area, delineation: perimeter x (0.5 px) buffer (Granshaw & Fountain 2006)
  * area, model: range of areas for P(glacier) thresholds 0.4 and 0.6
  * terminus: sqrt(res^2 + coregistration residual^2) per epoch, combined in quadrature for differences

Year-of-loss codes (int16):
  0 = never glacier, 1 = glacier in all years, 2 = unstable (lost then re-appeared),
  YYYY = glacier in the baseline year and non-glacier from YYYY onwards in every later year,
  -YYYY = non-glacier in the baseline and glacier from YYYY onwards (gain)
"""
from __future__ import annotations

import argparse
import json

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from src.change_detection.centerline import load_centerline
from src.common import ROOT, Grid, get_logger, load_config, read_raster, write_raster
from src.postprocessing.mask import clean_mask, polygonise, reference_masks

log = get_logger("change")
STATS = ROOT / "outputs" / "statistics"


def load_series(model):
    exp = ROOT / "experiments" / model
    years = sorted(int(p.name) for p in exp.iterdir() if p.is_dir() and p.name.isdigit() and (p / "mask.tif").exists())
    masks = {y: read_raster(exp / str(y) / "mask.tif").astype(bool) for y in years}
    probs = {y: read_raster(exp / str(y) / "prob_glacier.tif") for y in years}
    summ = {y: json.loads((exp / str(y) / "summary.json").read_text()) for y in years}
    dates = {y: summ[y]["scene"]["date"] for y in years}
    # surface-condition (between-scene) area spread from the multi-date ensemble, if available
    surface_sd = {y: (summ[y].get("ensemble") or {}).get("area_std_km2") or 0.0 for y in years}
    n_scenes = {y: (summ[y].get("ensemble") or {}).get("n_scenes", 1) for y in years}
    load_series.surface_sd, load_series.n_scenes = surface_sd, n_scenes
    return years, masks, probs, dates


def decimal_year(d):
    ts = pd.Timestamp(d)
    return ts.year + (ts.dayofyear - 0.5) / (366 if ts.is_leap_year else 365)


def load_calibrator(model):
    """Isotonic calibrator fitted on HMA val tiles (src.validation.calibration); falls back to the
    M1_hma_official calibrator (same architecture/loss) if the model has none."""
    import pickle
    base = model.replace("_fullwindow", "")
    for m in (base, "M1_hma_official"):
        f = ROOT / "models" / "calibration" / f"{m}_isotonic.pickle"
        if f.exists():
            with open(f, "rb") as fh:
                return pickle.load(fh), m
    return None, None


def area_table(model, years, masks, probs, dates, grid, cfg):
    domain, _, core = reference_masks(grid)
    iso, cal_name = load_calibrator(model)
    rows = []
    for y in years:
        geom, _ = polygonise(masks[y], grid)
        A = masks[y].sum() * grid.res ** 2 / 1e6
        sig_buf = geom.length * 0.5 * grid.res / 1e6
        # threshold sensitivity on CALIBRATED probability (raw GlaViTU softmax is compressed)
        pp = dict(cfg["postprocessing"])
        pc = iso.predict(probs[y].ravel()).reshape(probs[y].shape) if iso is not None else probs[y]
        a_thr = []
        for t in (0.4, 0.6):
            pp["prob_threshold"] = t
            a_thr.append(clean_mask(pc, grid, pp, domain, core).sum() * grid.res ** 2 / 1e6)
        sd = getattr(load_series, "surface_sd", {}).get(y, 0.0)
        n = getattr(load_series, "n_scenes", {}).get(y, 1)
        rows.append(dict(year=y, date=dates[y], t=decimal_year(dates[y]), area_km2=A, perimeter_km=geom.length / 1e3,
                         sigma_buffer_km2=sig_buf, area_p040_km2=a_thr[0], area_p060_km2=a_thr[1],
                         n_scenes=n, sigma_surface_km2=sd, calibrator=cal_name))
    df = pd.DataFrame(rows)
    # total: delineation buffer + probability-threshold sensitivity + surface conditions (between-scene sd).
    # The surface term uses the full between-scene sd (not sd/sqrt(n)): scenes within a season are not
    # independent draws, so the conservative choice is kept.
    df["sigma_threshold_km2"] = (df.area_p040_km2 - df.area_p060_km2) / 2
    df["sigma_area_km2"] = np.sqrt(df.sigma_buffer_km2 ** 2 + df.sigma_threshold_km2 ** 2 + df.sigma_surface_km2 ** 2)
    A0, t0 = df.area_km2.iloc[0], df.t.iloc[0]
    df["dA_prev_km2"] = df.area_km2.diff()
    df["dt_prev_yr"] = df.t.diff()
    df["rate_prev_km2_per_yr"] = df.dA_prev_km2 / df.dt_prev_yr
    df["dA_cum_km2"] = df.area_km2 - A0
    df["loss_cum_pct"] = (A0 - df.area_km2) / A0 * 100
    df["rate_cum_km2_per_yr"] = df.dA_cum_km2 / (df.t - t0)
    # Uncertainty of CHANGE: buffer and surface terms are independent between epochs (quadrature);
    # the threshold term is systematic (same model, same threshold shift in every year), so it is
    # propagated by recomputing dA at thresholds 0.4 and 0.6 and taking half the range.
    dA_lo = df.area_p040_km2 - df.area_p040_km2.iloc[0]
    dA_hi = df.area_p060_km2 - df.area_p060_km2.iloc[0]
    df["sigma_dA_threshold_km2"] = (dA_lo - dA_hi).abs() / 2
    indep = lambda c: df[c] ** 2 + df[c].iloc[0] ** 2
    df["sigma_dA_cum_km2"] = np.sqrt(indep("sigma_buffer_km2") + indep("sigma_surface_km2") + df.sigma_dA_threshold_km2 ** 2)
    df.loc[0, ["rate_cum_km2_per_yr", "sigma_dA_cum_km2"]] = np.nan
    df["sigma_indep_km2"] = np.sqrt(df.sigma_buffer_km2 ** 2 + df.sigma_surface_km2 ** 2)
    df.to_csv(STATS / f"{model}_area.csv", index=False)
    summary = {}
    if len(df) >= 3:
        # weighted least-squares trend, weights 1/sigma_indep^2 (systematic threshold term cancels in a slope)
        import statsmodels.api as sm
        X = sm.add_constant(df.t - df.t.mean())
        fit = sm.WLS(df.area_km2, X, weights=1 / df.sigma_indep_km2 ** 2).fit()
        summary["wls_trend_km2_per_yr"] = float(fit.params.iloc[1])
        summary["wls_trend_se_km2_per_yr"] = float(fit.bse.iloc[1])
        summary["wls_trend_ci95"] = [float(v) for v in fit.conf_int().iloc[1]]
        summary["wls_trend_pct_per_yr_of_mean_area"] = float(100 * fit.params.iloc[1] / df.area_km2.mean())
    hc = df[df.sigma_surface_km2 <= cfg.get("change", {}).get("high_confidence_surface_sd_km2", 0.35)]
    if len(hc) >= 2:
        a, b = hc.iloc[0], hc.iloc[-1]
        thr = abs((b.area_p040_km2 - a.area_p040_km2) - (b.area_p060_km2 - a.area_p060_km2)) / 2
        sig = float(np.sqrt(a.sigma_buffer_km2 ** 2 + b.sigma_buffer_km2 ** 2 + a.sigma_surface_km2 ** 2 +
                            b.sigma_surface_km2 ** 2 + thr ** 2))
        summary["high_confidence_period"] = dict(
            years=[int(a.year), int(b.year)], dA_km2=float(b.area_km2 - a.area_km2), sigma_dA_km2=sig,
            rate_km2_per_yr=float((b.area_km2 - a.area_km2) / (b.t - a.t)), loss_pct=float(100 * (a.area_km2 - b.area_km2) / a.area_km2),
            significant_2sigma=bool(abs(b.area_km2 - a.area_km2) > 2 * sig))
    (STATS / f"{model}_area_summary.json").write_text(json.dumps(summary, indent=2))
    df.attrs["summary"] = summary
    return df


def sample_along(line, mask, grid):
    s = np.arange(0, line.length + 1e-6, grid.res)
    pts = [line.interpolate(v) for v in s]
    inv = ~grid.transform
    rc = [(int((inv * (p.x, p.y))[1]), int((inv * (p.x, p.y))[0])) for p in pts]
    vals = np.array([mask[r, c] if 0 <= r < grid.height and 0 <= c < grid.width else False for r, c in rc])
    return s, vals, pts


def terminus_position(line, mask, grid, gap_m=100):
    """Downstream-most glacier point reached continuously from the head (gaps < gap_m tolerated)."""
    s, v, pts = sample_along(line, mask, grid)
    gap = int(gap_m / grid.res)
    last = None
    run = 0
    for i, g in enumerate(v):
        if g:
            last, run = i, 0
        else:
            run += 1
            if run > gap and last is not None:
                break
    return (s[last], pts[last]) if last is not None else (np.nan, None)


def box_method(line, masks, grid, length_m=2000):
    """Box method (e.g. Lea et al. 2014): effective length change = glacier area inside a fixed
    rectangle along the lowest `length_m` of the centerline divided by its width. The width is half
    the median tongue width of the 2002 RGI7 outline over the same reach, so that the box spans the
    terminus but not the lateral (debris-covered) margins."""
    from rasterio.features import rasterize
    from shapely.ops import substring
    L = line.length
    seg = substring(line, max(L - length_m, 0), L)
    rgi = gpd.read_file(ROOT / "data" / "inventories" / "chhota_shigri.gpkg", layer="rgi7").geometry.iloc[0]
    reach = seg.buffer(1500, cap_style=2).intersection(rgi)
    tongue_width = reach.area / seg.length
    width_m = max(0.5 * tongue_width, 4 * grid.res)
    box = seg.buffer(width_m / 2, cap_style=2)
    bm = rasterize([box], out_shape=(grid.height, grid.width), transform=grid.transform).astype(bool)
    box_method.width_m = width_m
    return box, {y: (m & bm).sum() * grid.res ** 2 / width_m for y, m in masks.items()}


def terminus_table(model, years, masks, dates, grid):
    line = load_centerline(grid)
    coreg = STATS / "coregistration_by_year.csv"
    creg = pd.read_csv(coreg).set_index("year") if coreg.exists() else None
    box, box_len = box_method(line, masks, grid)
    rows, pts = [], []
    for y in years:
        s, p = terminus_position(line, masks[y], grid)
        resid = 0.0
        if creg is not None and y in creg.index:
            r = creg.loc[y]
            resid = float(r.get("residual_m", 0.0) or 0.0)
        rows.append(dict(year=y, date=dates[y], t=decimal_year(dates[y]), terminus_s_m=s, box_length_m=box_len[y],
                         sigma_terminus_m=float(np.hypot(grid.res, resid))))
        if p is not None:
            pts.append(dict(year=y, date=dates[y], s_m=s, geometry=Point(p.x, p.y)))
    df = pd.DataFrame(rows)
    # retreat = upstream movement (positive = retreat)
    # retreat (positive = terminus moved upstream): centerline s grows downstream and box-effective
    # length grows with advance, so for both: retreat = -(change)
    for col, name in (("terminus_s_m", "centerline"), ("box_length_m", "box")):
        base = df[col].iloc[0]
        df[f"retreat_prev_{name}_m"] = -df[col].diff()
        df[f"retreat_cum_{name}_m"] = -(df[col] - base)
        df[f"rate_cum_{name}_m_per_yr"] = df[f"retreat_cum_{name}_m"] / (df.t - df.t.iloc[0])
        df[f"rate_prev_{name}_m_per_yr"] = df[f"retreat_prev_{name}_m"] / df.t.diff()
    df["sigma_retreat_cum_m"] = np.sqrt(df.sigma_terminus_m ** 2 + df.sigma_terminus_m.iloc[0] ** 2)
    df["box_width_m"] = box_method.width_m
    df.to_csv(STATS / f"{model}_terminus.csv", index=False)
    gpd.GeoDataFrame(pts, crs=grid.crs).to_file(ROOT / "outputs" / "polygons" / f"{model}_terminus_points.gpkg")
    df["box_width_m"] = box_method.width_m
    gpd.GeoDataFrame(dict(name=["tongue_box"], width_m=[box_method.width_m]), geometry=[box], crs=grid.crs).to_file(
        ROOT / "outputs" / "polygons" / f"{model}_terminus_box.gpkg")
    return df


def year_of_loss(model, years, masks, grid):
    stack = np.stack([masks[y] for y in years])
    base = stack[0]
    out = np.zeros(base.shape, np.int16)
    out[stack.all(0)] = 1
    for i, y in enumerate(years[1:], start=1):
        lost_here = base & (~stack[i:]).all(0) & (stack[:i].any(0))
        out[lost_here & (out == 0)] = y
        gained = ~base & stack[i:].all(0)
        out[gained & (out == 0)] = -y
    unstable = (out == 0) & stack.any(0)
    out[unstable] = 2
    write_raster(ROOT / "outputs" / "change_maps" / f"{model}_year_of_loss.tif", out, grid, "int16")
    vals, counts = np.unique(out, return_counts=True)
    summary = {int(v): float(c * grid.res ** 2 / 1e6) for v, c in zip(vals, counts) if v != 0}
    (STATS / f"{model}_year_of_loss_km2.json").write_text(json.dumps(summary, indent=2))
    return out, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    cfg = load_config()
    STATS.mkdir(parents=True, exist_ok=True)
    grid = Grid.load(ROOT / "data" / "inventories" / "grid.json")
    years, masks, probs, dates = load_series(a.model)
    at = area_table(a.model, years, masks, probs, dates, grid, cfg)
    tt = terminus_table(a.model, years, masks, dates, grid)
    _, yol = year_of_loss(a.model, years, masks, grid)
    pd.set_option("display.width", 200)
    print(at[["year", "date", "n_scenes", "area_km2", "sigma_area_km2", "dA_cum_km2", "sigma_dA_cum_km2", "loss_cum_pct", "rate_cum_km2_per_yr"]].round(3).to_string(index=False))
    print(tt[["year", "terminus_s_m", "retreat_cum_centerline_m", "retreat_cum_box_m", "sigma_retreat_cum_m"]].round(1).to_string(index=False))
    print("year-of-loss km2:", yol)
    print("area summary:", json.dumps(at.attrs.get("summary", {}), indent=1))


if __name__ == "__main__":
    main()
