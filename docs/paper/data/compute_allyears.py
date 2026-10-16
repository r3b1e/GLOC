"""Usage: PYTHONPATH=. .env/bin/python docs/paper/data/compute_allyears.py <out_dir>
All-year (2017 baseline) statistics. Writes ONLY to the scratch dir; project outputs untouched."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
gpd.GeoDataFrame.to_file = lambda *a, **k: None          # block gpkg writes into outputs/polygons
import src.change_detection.change as ch
from src.common import ROOT, Grid, load_config
OUT = Path(sys.argv[1]); ch.STATS = OUT
cfg = load_config(); grid = Grid.load(ROOT / "data/inventories/grid.json")
px = grid.res ** 2 / 1e6
res = {}
for m in ["M0_global", "M1_hma_official", "M1_hma_local", "M2_cs_selftrain", "M3_hma_sar"]:
    years, masks, probs, dates = ch.load_series(m)
    at = ch.area_table(m, years, masks, probs, dates, grid, cfg)
    tt = ch.terminus_table(m, years, masks, dates, grid)
    b = masks[years[0]]
    lg = pd.DataFrame([dict(year=y, loss_vs_2017_km2=(b & ~masks[y]).sum() * px,
                            gain_vs_2017_km2=(~b & masks[y]).sum() * px) for y in years])
    df = at.merge(tt, on=["year", "date", "t"]).merge(lg, on="year")
    rows = []
    for y in years:
        s = json.loads((ROOT / f"experiments/{m}/{y}/summary.json").read_text()); e = s["ensemble"]; a = s["agreement_vs_rgi7_2002"]
        rows.append(dict(year=y, dates=" ".join(d[5:] for d in e["scenes_used"]), sats=" ".join(p[:3] for p in s["scene"]["l1c_products"]),
                         scene_min=min(e["scene_areas_km2"].values()), scene_max=max(e["scene_areas_km2"].values()),
                         outliers=len(e["outlier_scenes"]), single_obs=e.get("single_obs_km2_domain"), mean_obs=e.get("mean_valid_obs_domain"),
                         gap=e.get("gap_filled_km2_domain"), unobs=e.get("unobserved_km2_domain"),
                         iou=a["iou"], prec=a["precision"], rec=a["recall"], mbd=a["mean_boundary_dist_m"], p95=a["p95_boundary_dist_m"],
                         parts=s["topology"]["n_parts"], holes=s["topology"]["n_holes"]))
    df = df.merge(pd.DataFrame(rows), on="year")
    df.to_csv(OUT / f"{m}_allyears.csv", index=False)
    res[m] = df
    print("==", m, at.attrs.get("summary"))
    print(df[["year","n_scenes","area_km2","sigma_area_km2","dA_cum_km2","sigma_dA_cum_km2","loss_cum_pct","rate_cum_km2_per_yr",
              "loss_vs_2017_km2","gain_vs_2017_km2","terminus_s_m","retreat_cum_centerline_m","retreat_cum_box_m","sigma_retreat_cum_m"]].round(3).to_string(index=False))
