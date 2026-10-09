"""Spatial pattern analysis of glacier loss and an interpretable retreat-susceptibility layer.

  python -m src.change_detection.patterns --model M2_cs_selftrain

Population: pixels that are glacier in the baseline year (first epoch). Each is either LOST
(year-of-loss >= second epoch) or RETAINED (code 1); unstable pixels (code 2) are excluded.
No vulnerable elevation/slope/aspect is assumed: loss fractions are computed per class, and
compared with the class distribution of the baseline glacier.

Outputs (outputs/statistics/, outputs/change_maps/):
  <model>_loss_by_elevation.csv / _slope.csv / _aspect.csv / _terminus_distance.csv
  <model>_hotspot_gistar.tif     Getis-Ord Gi* z-scores of loss on 100 m cells
  <model>_susceptibility.tif     logistic-regression P(loss | terrain, geometry), baseline glacier only
  <model>_susceptibility_model.json  standardized coefficients, CIs, spatial-block CV AUC

"Susceptibility" is a descriptive, interpretable summary of where loss occurred between the
epochs; it is NOT a forecast of future extent.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import scipy.ndimage as ndi

from src.common import ROOT, Grid, get_logger, read_raster, write_raster

log = get_logger("patterns")
STATS = ROOT / "outputs" / "statistics"
CM = ROOT / "outputs" / "change_maps"
ASPECT_LABELS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def layers(model, grid):
    yol = read_raster(CM / f"{model}_year_of_loss.tif")
    exp = ROOT / "experiments" / model
    years = sorted(int(p.name) for p in exp.iterdir() if p.name.isdigit())
    base_mask = read_raster(exp / str(years[0]) / "mask.tif").astype(bool)
    elev = read_raster(ROOT / "data" / "dem" / "elevation.tif")
    slope = read_raster(ROOT / "data" / "dem" / "slope.tif")
    aspect = read_raster(ROOT / "data" / "dem" / "aspect.tif")
    conf = np.mean([read_raster(exp / str(y) / "confidence.tif") for y in years], 0)
    # distance to the baseline terminus point and to the baseline glacier margin
    import geopandas as gpd
    tp = gpd.read_file(ROOT / "outputs" / "polygons" / f"{model}_terminus_points.gpkg").sort_values("year").iloc[0].geometry
    rr, cc = np.mgrid[0:grid.height, 0:grid.width]
    xs = grid.xmin + (cc + 0.5) * grid.res
    ys = grid.ymax - (rr + 0.5) * grid.res
    d_term = np.hypot(xs - tp.x, ys - tp.y)
    d_margin = ndi.distance_transform_edt(base_mask) * grid.res
    return dict(yol=yol, base=base_mask, elev=elev, slope=slope, aspect=aspect, conf=conf, d_term=d_term,
                d_margin=d_margin, years=years)


def class_table(values, lost, retained, bins, labels, grid):
    rows = []
    for lo, hi, lab in zip(bins[:-1], bins[1:], labels):
        sel = (values >= lo) & (values < hi)
        a_lost = (lost & sel).sum() * grid.res ** 2 / 1e6
        a_base = ((lost | retained) & sel).sum() * grid.res ** 2 / 1e6
        rows.append(dict(cls=lab, lo=lo, hi=hi, baseline_area_km2=a_base, lost_area_km2=a_lost,
                         loss_fraction=a_lost / a_base if a_base > 0 else np.nan))
    df = pd.DataFrame(rows)
    tot_lost = df.lost_area_km2.sum()
    df["share_of_total_loss"] = df.lost_area_km2 / tot_lost if tot_lost > 0 else np.nan
    df["share_of_baseline"] = df.baseline_area_km2 / df.baseline_area_km2.sum()
    # >1: class over-represented in loss relative to its share of the glacier
    df["loss_enrichment"] = df.share_of_total_loss / df.share_of_baseline
    return df


def getis_ord(loss_frac, valid, radius=2):
    """Gi* on a regular lattice with binary weights within `radius` cells (queen, incl. self)."""
    x = np.where(valid, loss_frac, 0.0)
    n = valid.sum()
    xbar = x[valid].mean()
    s = np.sqrt((x[valid] ** 2).mean() - xbar ** 2)
    k = np.ones((2 * radius + 1,) * 2)
    wx = ndi.convolve(x, k, mode="constant")
    w = ndi.convolve(valid.astype(float), k, mode="constant")
    num = wx - xbar * w
    den = s * np.sqrt((n * w - w ** 2) / (n - 1))
    z = np.where(valid & (den > 0), num / np.maximum(den, 1e-12), np.nan)
    return z


def susceptibility(L, lost, retained, grid, block_m=1000, seed=42):
    import statsmodels.api as sm
    from sklearn.metrics import roc_auc_score
    pop = lost | retained
    feats = {
        "elevation": L["elev"], "slope": L["slope"],
        "aspect_sin": np.sin(np.radians(L["aspect"])), "aspect_cos": np.cos(np.radians(L["aspect"])),
        "dist_terminus": L["d_term"], "dist_margin": L["d_margin"],
    }
    X = np.column_stack([f[pop] for f in feats.values()])
    y = lost[pop].astype(int)
    mu, sd = X.mean(0), X.std(0)
    Xs = (X - mu) / sd
    # spatial-block CV: 1 km blocks, 5 folds of blocks
    rr, cc = np.nonzero(pop)
    block = (rr * grid.res // block_m).astype(int) * 1000 + (cc * grid.res // block_m).astype(int)
    ub = np.unique(block)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(ub, rng.integers(0, 5, len(ub))))
    folds = np.array([fold_of[b] for b in block])
    aucs = []
    for f in range(5):
        tr, te = folds != f, folds == f
        if y[te].min() == y[te].max() or y[tr].min() == y[tr].max():
            continue
        m = sm.Logit(y[tr], sm.add_constant(Xs[tr])).fit(disp=0)
        aucs.append(roc_auc_score(y[te], m.predict(sm.add_constant(Xs[te], has_constant="add"))))
    model = sm.Logit(y, sm.add_constant(Xs)).fit(disp=0)
    ci = model.conf_int()
    coef = {n: dict(beta_std=float(model.params[i + 1]), ci_low=float(ci[i + 1][0]), ci_high=float(ci[i + 1][1]),
                    p=float(model.pvalues[i + 1])) for i, n in enumerate(feats)}
    prob = np.full(pop.shape, np.nan, np.float32)
    prob[pop] = model.predict(sm.add_constant(Xs))
    info = dict(n_pixels=int(pop.sum()), n_lost=int(y.sum()), coefficients=coef, pseudo_r2=float(model.prsquared),
                spatial_block_cv_auc_mean=float(np.mean(aucs)) if aucs else None,
                spatial_block_cv_auc_folds=[float(a) for a in aucs], block_m=block_m,
                note=("Descriptive model of observed loss between epochs; not a forecast. Pixel-level p-values and "
                      "CIs ignore spatial autocorrelation (overconfident); use the spatial-block CV AUC for skill. "
                      "dist_margin is near-tautological (loss occurs at margins); interpret the other terms."))
    return prob, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    grid = Grid.load(ROOT / "data" / "inventories" / "grid.json")
    L = layers(a.model, grid)
    yol = L["yol"]
    lost = L["base"] & (yol >= L["years"][1])
    retained = L["base"] & (yol == 1)
    log.info("baseline %.3f km2, lost %.3f km2, retained %.3f km2, unstable %.3f km2",
             L["base"].sum() * 1e-4, lost.sum() * 1e-4, retained.sum() * 1e-4, (L["base"] & (yol == 2)).sum() * 1e-4)

    zmin = np.floor(L["elev"][L["base"]].min() / 100) * 100
    zmax = np.ceil(L["elev"][L["base"]].max() / 100) * 100
    ebins = np.arange(zmin, zmax + 100, 100)
    tabs = {
        "elevation": class_table(L["elev"], lost, retained, ebins, [f"{int(b)}-{int(b + 100)}" for b in ebins[:-1]], grid),
        "slope": class_table(L["slope"], lost, retained, [0, 5, 10, 15, 20, 25, 30, 40, 90],
                             ["0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "30-40", ">40"], grid),
        "aspect": class_table((L["aspect"] + 22.5) % 360, lost, retained, np.arange(0, 361, 45), ASPECT_LABELS, grid),
        "terminus_distance": class_table(L["d_term"], lost, retained, [0, 250, 500, 1000, 2000, 4000, 8000, 20000],
                                         ["0-250", "250-500", "500-1000", "1-2 km", "2-4 km", "4-8 km", ">8 km"], grid),
    }
    for k, df in tabs.items():
        df.to_csv(STATS / f"{a.model}_loss_by_{k}.csv", index=False)

    # loss per year-of-loss vs terrain summary
    per_year = []
    for y in L["years"][1:]:
        m = L["base"] & (yol == y)
        if m.any():
            per_year.append(dict(year_of_loss=y, area_km2=m.sum() * 1e-4, elev_median=float(np.median(L["elev"][m])),
                                 slope_median=float(np.median(L["slope"][m])),
                                 dist_terminus_median_m=float(np.median(L["d_term"][m])),
                                 mean_confidence=float(L["conf"][m].mean())))
    pd.DataFrame(per_year).to_csv(STATS / f"{a.model}_loss_by_year_terrain.csv", index=False)

    # hotspots on 100 m cells
    f = 10
    H, W = grid.height // f * f, grid.width // f * f
    agg = lambda m: m[:H, :W].reshape(H // f, f, W // f, f).sum((1, 3))
    n_base = agg(lost | retained)
    frac = np.where(n_base > 0, agg(lost) / np.maximum(n_base, 1), 0)
    z = getis_ord(frac, n_base > 10)
    z10 = np.full((grid.height, grid.width), np.nan, np.float32)
    z10[:H, :W] = np.kron(z, np.ones((f, f)))
    write_raster(CM / f"{a.model}_hotspot_gistar.tif", z10, grid, "float32")

    prob, info = susceptibility(L, lost, retained, grid)
    write_raster(CM / f"{a.model}_susceptibility.tif", prob, grid, "float32")
    (STATS / f"{a.model}_susceptibility_model.json").write_text(json.dumps(info, indent=2))
    pd.set_option("display.width", 200)
    for k, df in tabs.items():
        print(f"--- loss by {k}")
        print(df[["cls", "baseline_area_km2", "lost_area_km2", "loss_fraction", "loss_enrichment"]].round(4).to_string(index=False))
    print(json.dumps({k: v for k, v in info.items() if k != "coefficients"}, indent=1))
    print(json.dumps(info["coefficients"], indent=1))


if __name__ == "__main__":
    main()
