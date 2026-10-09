"""Chhota Shigri validation without manual tracing.

  python -m src.validation.site_validation --models M0_global M1_hma_official M2_cs_selftrain

A. Inventory agreement (NOT accuracy): IoU / boundary metrics vs the 2002 RGI7 outline, globally
   and restricted to the upper glacier (> 5000 m), where 2002-2025 margin change is expected to be
   small relative to the tongue; low upper-zone agreement indicates systematic misclassification
   (snow, shadow, divides) rather than retreat.
B. Published areas: comparison with published glacier-wide areas (definitions differ; indicative).
C. Temporal consistency: IoU of consecutive epochs, fraction of "unstable" pixels, number of
   epoch-to-epoch area *gains* exceeding 2 sigma, terminus monotonicity. A physically consistent
   series on a retreating glacier should show few unstable pixels and no significant gains.
Output: outputs/statistics/site_validation.json and .csv
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.common import ROOT, Grid, read_raster
from src.postprocessing.mask import reference_masks
from src.validation.metrics import binary_metrics

STATS = ROOT / "outputs" / "statistics"

PUBLISHED = [
    dict(source="Wagnon et al. 2007, J. Glaciol. 53(183) 603-611", area_km2=15.7, epoch="c. 2003",
         note="including tributaries"),
    dict(source="Mandal et al. 2020, J. Glaciol. 66(259) 727-741", area_km2=15.5, epoch="2002-2019 study period",
         note="glacier-wide area used for mass balance"),
    dict(source="RGI 7.0 / GAMDAM (GLIMS 860910)", area_km2=16.77, epoch="2002-08-02", note="inventory outline"),
    dict(source="ICIMOD (GLIMS 154108)", area_km2=11.65, epoch="2009-10-25", note="different divides/tributaries"),
]


def temporal_consistency(model, grid):
    exp = ROOT / "experiments" / model
    years = sorted(int(p.name) for p in exp.iterdir() if p.name.isdigit())
    masks = [read_raster(exp / str(y) / "mask.tif").astype(bool) for y in years]
    ious = [binary_metrics(a, b)["iou"] for a, b in zip(masks[:-1], masks[1:])]
    out = dict(years=years, consecutive_iou=ious, mean_consecutive_iou=float(np.mean(ious)) if ious else None)
    yol_f = ROOT / "outputs" / "change_maps" / f"{model}_year_of_loss.tif"
    if yol_f.exists():
        yol = read_raster(yol_f)
        ever = (yol != 0)
        out["unstable_frac_of_ever_glacier"] = float((yol == 2).sum() / max(ever.sum(), 1))
        out["unstable_area_km2"] = float((yol == 2).sum() * grid.res ** 2 / 1e6)
    a = STATS / f"{model}_area.csv"
    if a.exists():
        df = pd.read_csv(a)
        gains = (df.dA_prev_km2 > 2 * np.sqrt(df.sigma_area_km2 ** 2 + df.sigma_area_km2.shift() ** 2))
        out["significant_area_gains"] = int(gains.sum())
    t = STATS / f"{model}_terminus.csv"
    if t.exists():
        dt = pd.read_csv(t)
        adv = dt.retreat_prev_centerline_m < -2 * dt.sigma_terminus_m * np.sqrt(2)
        out["significant_terminus_advances"] = int(adv.sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["M0_global", "M1_hma_official", "M2_cs_selftrain"])
    a = ap.parse_args()
    grid = Grid.load(ROOT / "data" / "inventories" / "grid.json")
    domain, rgi, _ = reference_masks(grid)
    elev = read_raster(ROOT / "data" / "dem" / "elevation.tif")
    upper = domain & (elev > 5000)
    rows, report = [], {"published_reference_areas": PUBLISHED, "models": {}}
    for m in a.models:
        exp = ROOT / "experiments" / m
        if not exp.exists():
            continue
        for yd in sorted(p for p in exp.iterdir() if p.name.isdigit()):
            y = int(yd.name)
            mask = read_raster(yd / "mask.tif").astype(bool)
            full = binary_metrics(mask, rgi, domain)
            up = binary_metrics(mask, rgi, upper)
            invalid_f = ROOT / "data" / "sentinel2" / str(y) / "invalid.tif"
            inv = read_raster(invalid_f).astype(bool) if invalid_f.exists() else np.zeros_like(mask)
            rows.append(dict(model=m, year=y, area_km2=mask.sum() * grid.res ** 2 / 1e6,
                             iou_vs_rgi7_2002=full["iou"], recall_vs_rgi7_2002=full["recall"],
                             precision_vs_rgi7_2002=full["precision"], upper_zone_iou=up["iou"],
                             upper_zone_recall=up["recall"], mean_boundary_dist_m=full["mean_boundary_dist_m"],
                             invalid_px_in_domain=int((inv & domain).sum()),
                             **{f"diff_vs_{p['area_km2']}km2": mask.sum() * grid.res ** 2 / 1e6 - p["area_km2"] for p in PUBLISHED[:2]}))
        report["models"][m] = temporal_consistency(m, grid)
    df = pd.DataFrame(rows)
    STATS.mkdir(parents=True, exist_ok=True)
    df.to_csv(STATS / "site_validation.csv", index=False)
    (STATS / "site_validation.json").write_text(json.dumps(report, indent=2, default=float))
    pd.set_option("display.width", 220)
    print(df.round(3).to_string(index=False))
    print(json.dumps(report["models"], indent=1, default=float))


if __name__ == "__main__":
    main()
