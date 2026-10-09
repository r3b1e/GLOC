"""Aggregate per-model Chhota Shigri results into cross-model tables (structural uncertainty).

  python -m src.validation.aggregate_results

Rationale: terminus *position* on the debris-covered snout depends on the model (up to ~800 m
between models), so period retreat is summarised across the model family
(M0, M1-official, M1-local, M2, M3): median, min, max. The primary (pre-specified) model is M2
(best held-out HMA IoU); the cross-model spread is the structural (model) uncertainty.
Outputs: outputs/statistics/summary_terminus_periods.csv, summary_area_by_year.csv, summary.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.common import ROOT

STATS = ROOT / "outputs" / "statistics"
MODELS = ["M0_global", "M1_hma_official", "M1_hma_local", "M2_cs_selftrain", "M3_hma_sar"]
PRIMARY = "M2_cs_selftrain"
PERIODS = [(2017, 2019), (2019, 2021), (2021, 2023), (2023, 2025), (2021, 2025), (2019, 2025), (2017, 2025)]


def main():
    term = {m: pd.read_csv(STATS / f"{m}_terminus.csv").set_index("year") for m in MODELS}
    area = {m: pd.read_csv(STATS / f"{m}_area.csv").set_index("year") for m in MODELS}
    rows = []
    for a, b in PERIODS:
        rec = dict(period=f"{a}-{b}", years=b - a)
        cl, bx = [], []
        for m in MODELS:
            t = term[m]
            dt = t.loc[b, "t"] - t.loc[a, "t"]
            r_cl = t.loc[a, "terminus_s_m"] - t.loc[b, "terminus_s_m"]
            r_bx = t.loc[b, "retreat_cum_box_m"] - t.loc[a, "retreat_cum_box_m"]
            rec[f"{m}_centerline_m"] = r_cl
            rec[f"{m}_box_m"] = r_bx
            cl.append(r_cl); bx.append(r_bx)
        rec["sigma_measurement_m"] = float(np.hypot(term[PRIMARY].loc[a, "sigma_terminus_m"], term[PRIMARY].loc[b, "sigma_terminus_m"]))
        for name, v in (("centerline", cl), ("box", bx)):
            rec[f"median_{name}_m"] = float(np.median(v)); rec[f"min_{name}_m"] = float(np.min(v)); rec[f"max_{name}_m"] = float(np.max(v))
            rec[f"median_{name}_m_per_yr"] = float(np.median(v) / dt)
        rec["all_models_retreat_box"] = bool(min(bx) > rec["sigma_measurement_m"])
        rows.append(rec)
    tp = pd.DataFrame(rows)
    tp.to_csv(STATS / "summary_terminus_periods.csv", index=False)

    ar = pd.DataFrame({m: area[m].area_km2 for m in MODELS})
    ar["median_km2"] = ar[MODELS].median(axis=1); ar["range_km2"] = ar[MODELS].max(axis=1) - ar[MODELS].min(axis=1)
    ar[f"{PRIMARY}_sigma_km2"] = area[PRIMARY].sigma_area_km2
    ar.to_csv(STATS / "summary_area_by_year.csv")
    trends = {m: json.loads((STATS / f"{m}_area_summary.json").read_text()) for m in MODELS}
    summary = dict(primary_model=PRIMARY, models=MODELS,
                   area_trend_km2_per_yr={m: [trends[m]["wls_trend_km2_per_yr"], trends[m]["wls_trend_ci95"]] for m in MODELS},
                   any_significant_area_trend=any(not (t["wls_trend_ci95"][0] < 0 < t["wls_trend_ci95"][1]) for t in trends.values()),
                   terminus_2021_2025=tp.set_index("period").loc["2021-2025"][[c for c in tp.columns if c.startswith(("median", "min", "max", "all_models"))]].to_dict())
    (STATS / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    pd.set_option("display.width", 220)
    print(tp[["period", "median_centerline_m", "min_centerline_m", "max_centerline_m", "median_box_m", "min_box_m", "max_box_m",
              "median_box_m_per_yr", "all_models_retreat_box"]].round(1).to_string(index=False))
    print(ar.round(2).to_string())
    print(json.dumps(summary["area_trend_km2_per_yr"], indent=1))


if __name__ == "__main__":
    main()
