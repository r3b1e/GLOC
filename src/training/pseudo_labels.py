"""Confidence-aware, multi-temporal pseudo-labels for Chhota Shigri domain adaptation (M2).

  python -m src.training.pseudo_labels --teacher M1_hma_official --years 2018 2020 2022 2024

Leakage controls
  * Temporal: pseudo-labels use ONLY the scenes of `--years` (even years). The analysis epochs
    (odd years 2017-2025) are never used for training: M2 is applied to unseen acquisitions.
  * No inventory outline is used as a label.
  * Model selection for M2 uses HMA *validation* tiles, never Chhota Shigri.

Label rule (all thresholds in configs/chhota_shigri.yaml: pseudo_labels)
  Teacher P(glacier) is computed for every clear scene of every training year
  (data/sentinel2/ensemble_<year>.json; cloud -> NaN).
    glacier:    P >= 0.95 in >= scene_agreement (60 %) of the pixel's clear scenes,
                never P <= 0.05 in any scene, and yearly-median P >= 0.95 in
                >= min_temporal_agreement (3) of the training years
    background: the symmetric rule with P <= 0.05
    otherwise:  ignored (target [0, 0] -> zero focal loss)
  Pixels that change between years or flip with surface conditions are left unlabelled, so the
  labels encode no assumption about the amount/direction of change.

Robustness to surface conditions: the consensus label is attached to EVERY scene's imagery
(one training tile per scene). A margin pixel that is confidently glacier in most scenes is thus
also presented as glacier in the scenes where it is snow-dusted or debris-exposed. This is the
self-training analogue of the multi-date ensemble.

Output: data/labels/cs_pseudolabels.hdf5 (benchmark layout): one group per scene "CS-<date>",
plus external links to `--replay_tiles` HMA training tiles (replay against forgetting).
"""
from __future__ import annotations

import argparse
import json

import h5py
import numpy as np

from src.common import ROOT, Grid, get_logger, load_config, read_raster, write_raster
from src.inference import glavitu
from src.inference.ensemble import in_window, scene_probs
from src.inference.run import resolve_weights

log = get_logger("pseudo")
S2 = ROOT / "data" / "sentinel2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="M1_hma_official")
    ap.add_argument("--years", nargs="+", type=int, default=[2018, 2020, 2022, 2024])
    ap.add_argument("--replay", default="data/benchmark/hma_train.hdf5")
    ap.add_argument("--replay_tiles", type=int, default=40)
    ap.add_argument("--cs_repeats", type=int, default=2, help="hard-link repeats of each CS scene tile")
    ap.add_argument("--out", default="data/labels/cs_pseudolabels.hdf5")
    a = ap.parse_args()
    cfg = load_config()
    pl = cfg["pseudo_labels"]
    grid = Grid.load(ROOT / "data" / "inventories" / "grid.json")
    elev = read_raster(ROOT / "data" / "dem" / "elevation.tif")
    slope = read_raster(ROOT / "data" / "dem" / "slope.tif")
    model = glavitu.build_model(resolve_weights(cfg, a.teacher), cfg["model"]["patch_size"])

    scenes, P_all, year_of = [], [], []
    year_median = []
    for y in a.years:
        win = cfg["ensemble"].get("window_core", cfg["ensemble"]["window"])
        dates = [d for d in json.loads((S2 / f"ensemble_{y}.json").read_text())["dates"] if in_window(d, win)]
        P = scene_probs(model, dates, grid, cfg, elev, slope,
                        cache_dir=ROOT / "experiments" / a.teacher / "scene_probs")
        with np.errstate(all="ignore"):
            year_median.append(np.nanmedian(P, 0))
        scenes += dates; P_all.append(P); year_of += [y] * len(dates)
        log.info("teacher %s on %d: %d scenes", a.teacher, y, len(dates))
    P = np.concatenate(P_all)
    # thresholds apply to CALIBRATED probabilities (isotonic, fitted on HMA val tiles; see
    # src/validation/calibration.py): GlaViTU's raw softmax never leaves ~0.23-0.89
    import pickle
    with open(ROOT / "models" / "calibration" / f"{a.teacher}_isotonic.pickle", "rb") as fh:
        iso = pickle.load(fh)
    fin = np.isfinite(P)
    Pc = np.full(P.shape, np.nan, np.float32)
    Pc[fin] = iso.predict(P[fin].ravel())
    P = Pc
    with np.errstate(all="ignore"):
        Ym = np.stack([np.nanmedian(P[np.array(year_of) == y], 0) for y in a.years])
    valid = np.isfinite(P)
    n_valid = valid.sum(0)
    with np.errstate(invalid="ignore"):
        hi = (P >= pl["confident_glacier"]).sum(0)
        lo = (P <= pl["confident_background"]).sum(0)
    frac_hi = hi / np.maximum(n_valid, 1)
    frac_lo = lo / np.maximum(n_valid, 1)
    k = pl["min_temporal_agreement"]
    yrs_g = (Ym >= pl["confident_glacier"]).sum(0)
    yrs_b = (Ym <= pl["confident_background"]).sum(0)
    glacier = (frac_hi >= pl["scene_agreement"]) & (lo == 0) & (yrs_g >= k) & (n_valid >= 3)
    background = (frac_lo >= pl["scene_agreement"]) & (hi == 0) & (yrs_b >= k) & (n_valid >= 3)

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    lab_dir = ROOT / "data" / "labels"
    code = np.where(glacier, 2, np.where(background, 1, 0)).astype(np.uint8)
    write_raster(lab_dir / "pseudolabel_consensus.tif", code, grid, "uint8")
    stats = dict(teacher=a.teacher, years=a.years, scenes=scenes, thresholds=pl, calibrated=True,
                 glacier_px=int(glacier.sum()), background_px=int(background.sum()),
                 ignored_frac=float(1 - (glacier | background).mean()),
                 flipping_px=int(((hi > 0) & (lo > 0)).sum()), per_scene={})
    with h5py.File(out, "w") as h:
        for i, d in enumerate(scenes):
            sdir = S2 / "scenes" / d
            optical = np.moveaxis(read_raster(sdir / "optical_toa.tif"), 0, -1)
            f = glavitu.normalise(optical, elev, slope)
            clear = valid[i]
            lab = np.zeros(glacier.shape + (2,), np.uint8)
            lab[..., 0] = background & clear
            lab[..., 1] = glacier & clear
            grp = h.create_group(f"CS-{d}")
            grp.create_dataset("optical", data=f["optical"], compression="lzf")
            grp.create_dataset("dem", data=f["dem"], compression="lzf")
            grp.create_dataset("outlines", data=lab, compression="lzf")
            H, W = glacier.shape
            for kk, v in dict(height=H, width=W, region="CS", subregion="CS", tile_name=f"CS-{d}", date=d,
                              year=year_of[i]).items():
                grp.attrs[kk] = v
            for r in range(1, a.cs_repeats):
                h[f"CS-{d}-r{r}"] = grp
            stats["per_scene"][d] = dict(year=year_of[i], labelled_px=int(lab.sum()), clear_frac=float(clear.mean()))
        if a.replay and (ROOT / a.replay).exists():
            with h5py.File(ROOT / a.replay, "r") as src:
                keys = sorted(src.keys())
            rng = np.random.default_rng(cfg["seed"])
            keys = sorted(rng.choice(keys, min(a.replay_tiles, len(keys)), replace=False).tolist())
            for kk in keys:
                tile_file = ROOT / "data" / "benchmark" / "tiles" / "train" / f"{kk}.hdf5"
                h[kk] = h5py.ExternalLink(str(tile_file.resolve()), kk)
            stats["replay_tiles"] = keys
    (lab_dir / "pseudolabel_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    log.info("pseudo-labels: glacier %d px, background %d px, ignored %.1f %%, %d scenes -> %s",
             stats["glacier_px"], stats["background_px"], 100 * stats["ignored_frac"], len(scenes), out)


if __name__ == "__main__":
    main()
