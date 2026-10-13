"""Multi-date ensemble inference (mitigation "Option A" for surface-condition artefacts).

  python -m src.inference.ensemble --model M1_hma_official --years 2017 2019 2021 2023 2025

Rationale: single-scene masks of Chhota Shigri change by ~1 km2 between good scenes because thin
snow hides debris-covered margins in some scenes and not in others (2023-09-11 vs 2025-08-21).
Every usable scene of a year (data/sentinel2/ensemble_<year>.json) is therefore mapped separately:

  * per scene: GlaViTU softmax, cloudy pixels (s2cloudless mask) -> NaN
  * per year:  median P(glacier) over clear observations; pixels with < min_valid_obs are invalid
  * area spread: each scene's mask is evaluated on its own clear pixels, with cloudy pixels filled
    from the median, so scene areas differ only where surface conditions differ. The standard
    deviation of these areas is the surface-condition uncertainty used in change statistics.

Outputs (experiments/<model>/<year>/) are drop-in replacements for src.inference.run:
  prob_glacier.tif (median), prob_iqr.tif, n_valid.tif, confidence.tif, mask.tif, outline.gpkg,
  summary.json (incl. per-scene areas).
"""
from __future__ import annotations

import argparse
import json
import shutil
import time

import geopandas as gpd
import numpy as np
from s2cloudless import S2PixelCloudDetector

from src.common import ROOT, Grid, get_logger, load_config, read_raster, write_raster
from src.inference import glavitu
from src.inference.run import ndsi_baseline, resolve_weights
from src.postprocessing.mask import clean_mask, polygonise, reference_masks
from src.validation.metrics import binary_metrics

log = get_logger("ensemble")
S2 = ROOT / "data" / "sentinel2"


def scene_cloud_mask(sdir, grid, cfg):
    det = S2PixelCloudDetector(threshold=cfg["sentinel2"]["s2cloudless_threshold"], average_over=4,
                               dilation_size=2, all_bands=False)
    cp = read_raster(sdir / "cloud_prob.tif")
    m20 = det.get_mask_from_prob(cp[::2, ::2][None])[0].astype(bool)
    return np.kron(m20, np.ones((2, 2), bool))[: grid.height, : grid.width]


def scene_probs(model, dates, grid, cfg, elev, slope, cache_dir=None, sar=None):
    """Per-scene P(glacier) with NaN on cloud; cached as <cache_dir>/<date>.tif if given.
    sar: normalised co_pol_sar (H, W, 2) added to every scene's inputs (M3)."""
    out = []
    for d in dates:
        cf = cache_dir / f"{d}.tif" if cache_dir else None
        if cf is not None and cf.exists():
            out.append(read_raster(cf).astype(np.float32))
            continue
        sdir = S2 / "scenes" / d
        optical = np.moveaxis(read_raster(sdir / "optical_toa.tif"), 0, -1)
        cloudy = scene_cloud_mask(sdir, grid, cfg) | ~np.isfinite(optical).all(-1)
        feats = glavitu.normalise(optical, elev, slope)
        if sar is not None:
            feats["co_pol_sar"] = sar
        prob = glavitu.predict(model, feats,
                               cfg["model"]["patch_size"], cfg["model"]["batch_size"])[..., 1]
        out.append(np.where(cloudy, np.nan, prob).astype(np.float32))
        if cf is not None:
            write_raster(cf, out[-1], grid, "float32")
    return np.stack(out)


def in_window(date, window):
    md = date[5:]
    return window[0] <= md <= window[1]


def run(model_key, years, model=None, window=None, tag=""):
    """window: restrict ensemble scenes to MM-DD range (default: config ensemble.window_core).
    tag: suffix of the experiment folder (e.g. '_fullwindow' for the sensitivity run)."""
    cfg = load_config()
    grid = Grid.load(ROOT / "data" / "inventories" / "grid.json")
    domain, rgi_mask, core = reference_masks(grid)
    elev = read_raster(ROOT / "data" / "dem" / "elevation.tif")
    slope = read_raster(ROOT / "data" / "dem" / "slope.tif")
    weights = resolve_weights(cfg, model_key)
    from src.validation.benchmark_eval import model_features
    features = model_features(model_key)
    model = model or glavitu.build_model(weights, cfg["model"]["patch_size"], features=features)
    pp, ens = cfg["postprocessing"], cfg["ensemble"]
    results = []
    for year in years:
        meta = json.loads((S2 / f"ensemble_{year}.json").read_text())
        meta_all = dict(meta)
        win = window or ens.get("window_core", ens["window"])
        dates = [d for d in meta["dates"] if in_window(d, win)]
        meta["l1c_products"] = [p for d, p in zip(meta["dates"], meta["l1c_products"]) if in_window(d, win)]
        if len(dates) < 1:
            log.error("%d: no ensemble scenes", year)
            continue
        t = time.time()
        sar = None
        if "co_pol_sar" in features:
            sar = glavitu.normalise_sar(read_raster(ROOT / "data" / "sentinel1" / str(year) / "co_pol_sar.tif"))
        P_all = scene_probs(model, dates, grid, cfg, elev, slope, sar=sar,
                            cache_dir=ROOT / "experiments" / model_key / "scene_probs")
        # drop scenes without a single clear pixel in the domain (e.g. an empty L1C extraction,
        # 2024-08-31); they carry no information and would only shrink the between-scene spread
        has_data = np.array([np.isfinite(p[domain]).any() for p in P_all])
        if not has_data.all():
            log.warning("%d: dropping empty scenes %s", year, [d for d, k in zip(dates, has_data) if not k])
            dates = [d for d, k in zip(dates, has_data) if k]
            meta["l1c_products"] = [p for p, k in zip(meta["l1c_products"], has_data) if k]
            P_all = P_all[has_data]

        def combine(P):
            n_valid = np.isfinite(P).sum(0)
            with np.errstate(all="ignore"):
                med = np.nanmedian(P, 0)
                iqr = np.nanpercentile(P, 75, 0) - np.nanpercentile(P, 25, 0)
            invalid = n_valid < ens["min_valid_obs"]
            return med, iqr, n_valid, invalid

        def gap_fill(med, invalid):
            """Pixels without any clear core-window observation: fill from clear scenes of the SAME
            year outside the core window (no cross-epoch leakage). Remaining gaps stay invalid (P=0)."""
            extra = [d for d in meta_all["dates"] if d not in dates]
            filled = np.zeros(med.shape, bool)
            if invalid.any() and extra:
                Pe = scene_probs(model, extra, grid, cfg, elev, slope, sar=sar,
                                 cache_dir=ROOT / "experiments" / model_key / "scene_probs")
                with np.errstate(all="ignore"):
                    me = np.nanmedian(Pe, 0)
                filled = invalid & np.isfinite(me)
                med = np.where(filled, me, med)
            still = invalid & ~filled
            return np.where(still, 0.0, med).astype(np.float32), filled, still

        def areas(P, med):
            # each scene on its own clear pixels, cloudy pixels filled from the median
            return [float(clean_mask(np.where(np.isfinite(p), p, med), grid, pp, domain, core).sum() * grid.res ** 2 / 1e6)
                    for p in P]

        med0, _, _, inv0 = combine(P_all)
        med0, _, _ = gap_fill(med0, inv0)
        all_areas = areas(P_all, med0)
        # robust outlier rejection of whole scenes (e.g. fresh snowfall that s2cloudless partly masks
        # as cloud, so it escapes the snow filter): |A_i - median| > max(k*1.4826*MAD, floor)
        a = np.array(all_areas)
        mad_sd = 1.4826 * np.median(np.abs(a - np.median(a)))
        tol = max(ens.get("outlier_k", 3.0) * mad_sd, ens.get("outlier_floor_km2", 0.5))
        keep = np.abs(a - np.median(a)) <= tol if len(a) >= 3 else np.ones(len(a), bool)
        outliers = [d for d, k in zip(dates, keep) if not k]
        dates_used = [d for d, k in zip(dates, keep) if k]
        P = P_all[keep]
        med, iqr, n_valid, invalid = combine(P)
        med, gap_filled, invalid = gap_fill(med, invalid)
        mask = clean_mask(med, grid, pp, domain, core)
        geom, checks = polygonise(mask, grid)
        scene_areas = areas(P, med)
        prob2 = np.stack([1 - med, med], -1)
        conf = glavitu.shannon_confidence(prob2)

        out = ROOT / "experiments" / f"{model_key}{tag}" / str(year)
        out.mkdir(parents=True, exist_ok=True)
        write_raster(out / "prob_glacier.tif", med, grid, "float32")
        write_raster(out / "prob_iqr.tif", np.nan_to_num(iqr), grid, "float32")
        write_raster(out / "n_valid.tif", n_valid, grid, "uint8")
        write_raster(out / "gap_filled.tif", gap_filled, grid, "uint8")
        write_raster(out / "confidence.tif", conf, grid, "float32")
        write_raster(out / "mask.tif", mask, grid, "uint8")
        gpd.GeoDataFrame(dict(year=[year], model=[model_key], dates=[",".join(dates_used)], area_km2=[geom.area / 1e6]),
                         geometry=[geom], crs=grid.crs).to_file(out / "outline.gpkg")
        # representative date = median acquisition date (used for decimal-year change rates)
        rep = sorted(dates_used)[len(dates_used) // 2]
        summary = dict(
            model=model_key, weights=str(weights.relative_to(ROOT)), year=year, mode="multi-date ensemble",
            scene=dict(date=rep, dates=dates, l1c_products=meta["l1c_products"], window=win),
            area_km2=geom.area / 1e6, topology=checks, inference_seconds=round(time.time() - t, 1),
            ensemble=dict(n_scenes=len(dates_used), scenes_used=dates_used, outlier_scenes=outliers,
                          outlier_tolerance_km2=float(tol), all_scene_areas_km2=dict(zip(dates, all_areas)),
                          scene_areas_km2=dict(zip(dates_used, scene_areas)),
                          # robust between-scene spread (1.4826*MAD): single outlier scenes (e.g. low-sun
                          # shadow) must not dominate; the plain SD is reported alongside
                          area_std_km2=float(1.4826 * np.median(np.abs(np.array(scene_areas) - np.median(scene_areas))))
                          if len(dates_used) > 2 else (float(np.std(scene_areas, ddof=1)) if len(dates_used) > 1 else None),
                          area_sd_plain_km2=float(np.std(scene_areas, ddof=1)) if len(dates_used) > 1 else None,
                          area_range_km2=float(np.ptp(scene_areas)),
                          invalid_frac_domain=float((invalid & domain).sum() / domain.sum()),
                          gap_filled_km2_domain=float((gap_filled & domain).sum() * grid.res ** 2 / 1e6),
                          unobserved_km2_domain=float((invalid & domain).sum() * grid.res ** 2 / 1e6),
                          single_obs_km2_domain=float(((n_valid == 1) & domain).sum() * grid.res ** 2 / 1e6),
                          mean_valid_obs_domain=float(n_valid[domain].mean())),
            mean_confidence_in_domain=float(conf[domain].mean()),
            frac_uncertain_in_domain=float(((med > 0.05) & (med < 0.95) & domain).sum() / domain.sum()),
            agreement_vs_rgi7_2002=binary_metrics(mask, rgi_mask, domain),
        )
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        for sub, f in (("masks", "mask.tif"), ("polygons", "outline.gpkg"), ("confidence", "confidence.tif")):
            dst = ROOT / "outputs" / sub / f"{model_key}{tag}_{year}_{f}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(out / f, dst)
        log.info("%s %d: %d scenes (outliers %s), median-mask area %.3f km2, scene areas %s (robust sd %s)", model_key, year, len(dates_used), outliers,
                 summary["area_km2"], [round(a, 2) for a in scene_areas],
                 None if summary["ensemble"]["area_std_km2"] is None else round(summary["ensemble"]["area_std_km2"], 3))
        results.append(summary)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--years", nargs="+", type=int, required=True)
    ap.add_argument("--window", nargs=2, help="MM-DD MM-DD (default: ensemble.window_core)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    run(a.model, a.years, window=a.window, tag=a.tag)


if __name__ == "__main__":
    main()
