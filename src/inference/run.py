"""Run a GlaViTU checkpoint on one or more years and write all per-year products.

  python -m src.inference.run --model M0_global --years 2023

Outputs per (model, year) in experiments/<model>/<year>/:
  prob_glacier.tif, confidence.tif, mask.tif, outline.gpkg, summary.json
and copies of the mask/outline/confidence in outputs/{masks,polygons,confidence}/.

summary.json also reports *agreement* with the 2002 RGI7 outline and an NDSI baseline. The 2002
outline is NOT ground truth for a 2017-2025 image; see docs/validation.md.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import geopandas as gpd
import numpy as np

from src.common import ROOT, Grid, get_logger, load_config, read_raster, write_raster
from src.inference import glavitu
from src.postprocessing.mask import clean_mask, polygonise, reference_masks
from src.validation.metrics import binary_metrics

log = get_logger("run")


def load_inputs(year):
    s2 = ROOT / "data" / "sentinel2" / str(year)
    # co-registered stack if src.preprocessing.coregister applied a correction
    opt_file = s2 / "optical_toa_coreg.tif" if (s2 / "optical_toa_coreg.tif").exists() else s2 / "optical_toa.tif"
    optical = np.moveaxis(read_raster(opt_file), 0, -1)
    elev = read_raster(ROOT / "data" / "dem" / "elevation.tif")
    slope = read_raster(ROOT / "data" / "dem" / "slope.tif")
    scene = json.loads((s2 / "scene.json").read_text())
    return optical, elev, slope, scene


def ndsi_baseline(optical, thr=0.4):
    g, swir = optical[..., 1], optical[..., 4]
    return (g - swir) / np.maximum(g + swir, 1e-6) >= thr


def resolve_weights(cfg, model_key):
    w = cfg["model"]["weights"].get(model_key)
    if w is None:  # locally trained checkpoints
        cand = sorted((ROOT / "models").glob(f"**/{model_key}*.h5"))
        if not cand:
            raise FileNotFoundError(model_key)
        w = cand[-1]
    return ROOT / w


def run(model_key, years, out_root=ROOT / "experiments"):
    cfg = load_config()
    grid = Grid.load(ROOT / "data" / "inventories" / "grid.json")
    domain, rgi_mask, core = reference_masks(grid)
    weights = resolve_weights(cfg, model_key)
    model = glavitu.build_model(weights, cfg["model"]["patch_size"])
    results = []
    for year in years:
        optical, elev, slope, scene = load_inputs(year)
        t = time.time()
        prob = glavitu.predict(model, glavitu.normalise(optical, elev, slope),
                               cfg["model"]["patch_size"], cfg["model"]["batch_size"])
        dt = time.time() - t
        p_glac = prob[..., 1]
        conf = glavitu.shannon_confidence(prob)
        mask = clean_mask(p_glac, grid, cfg["postprocessing"], domain, core)
        geom, checks = polygonise(mask, grid)

        out = Path(out_root) / model_key / str(year)
        out.mkdir(parents=True, exist_ok=True)
        write_raster(out / "prob_glacier.tif", p_glac, grid, "float32")
        write_raster(out / "confidence.tif", conf, grid, "float32")
        write_raster(out / "mask.tif", mask, grid, "uint8")
        gpd.GeoDataFrame(dict(year=[year], model=[model_key], date=[scene["date"]], area_km2=[geom.area / 1e6]),
                         geometry=[geom], crs=grid.crs).to_file(out / "outline.gpkg")

        ndsi = ndsi_baseline(optical)
        summary = dict(
            model=model_key, weights=str(weights.relative_to(ROOT)), year=year, scene=scene,
            area_km2=geom.area / 1e6, topology=checks, inference_seconds=round(dt, 1),
            mean_confidence_in_domain=float(conf[domain].mean()),
            frac_uncertain_in_domain=float(((p_glac > 0.05) & (p_glac < 0.95) & domain).sum() / domain.sum()),
            agreement_vs_rgi7_2002=binary_metrics(mask, rgi_mask, domain),
            ndsi_baseline_vs_rgi7_2002=binary_metrics(clean_mask(ndsi.astype(np.float32), grid, cfg["postprocessing"], domain, core), rgi_mask, domain),
        )
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        for sub, f in (("masks", "mask.tif"), ("polygons", "outline.gpkg"), ("confidence", "confidence.tif")):
            dst = ROOT / "outputs" / sub / f"{model_key}_{year}_{f}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(out / f, dst)
        log.info("%s %d (%s): area %.3f km2, IoU vs RGI7-2002 %.3f, %.1fs", model_key, year, scene["date"],
                 summary["area_km2"], summary["agreement_vs_rgi7_2002"]["iou"], dt)
        results.append(summary)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--years", nargs="+", type=int, required=True)
    a = ap.parse_args()
    run(a.model, a.years)


if __name__ == "__main__":
    main()
