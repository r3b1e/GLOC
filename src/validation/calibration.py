"""Probability calibration of GlaViTU outputs on HMA *validation* tiles (independent of Chhota Shigri).

  python -m src.validation.calibration --model M1_hma_official --n_tiles 20

Why: GlaViTU is trained with label smoothing (0.1) and a focal loss, which compress the softmax. On
Chhota Shigri the raw P(glacier) of M1_hma_official lies in ~0.23-0.89 for every pixel, so fixed
confidence thresholds such as 0.95/0.05 are never reached. Pseudo-label thresholds are therefore
applied to *calibrated* probabilities.

Method: sliding-window prediction on `n_tiles` official HMA val tiles (seeded random subset), padding
and unlabelled pixels removed, 20 000 random pixels per tile; isotonic regression raw P -> empirical
glacier frequency (sklearn IsotonicRegression, clipped). Reliability is reported as expected
calibration error (ECE, 15 bins) before/after on a held-out half of the tiles (split by tile).
Outputs: models/calibration/<model>_isotonic.pickle and experiments/calibration_<model>.json
"""
from __future__ import annotations

import argparse
import json
import pickle

import h5py
import numpy as np
from sklearn.isotonic import IsotonicRegression

from src.common import ROOT, get_logger, load_config
from src.inference import glavitu
from src.inference.run import resolve_weights
from src.validation.benchmark_eval import model_features, tile_valid_slices

log = get_logger("calibration")


def ece(p, y, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def collect(model, h5path, keys, features, cfg, per_tile=20000, seed=0):
    rng = np.random.default_rng(seed)
    P, Y, T = [], [], []
    with h5py.File(h5path, "r") as h:
        for k in keys:
            g = h[k]
            rs, cs = tile_valid_slices(g.attrs)
            feats = {f: g[f][()].astype(np.float32) for f in features}
            lab = g["outlines"][()][rs, cs]
            prob = glavitu.predict(model, feats, cfg["model"]["patch_size"], cfg["model"]["batch_size"])[rs, cs, 1]
            valid = lab.sum(-1) > 0
            idx = np.flatnonzero(valid)
            idx = rng.choice(idx, min(per_tile, idx.size), replace=False)
            P.append(prob.ravel()[idx]); Y.append(lab[..., 1].ravel()[idx]); T.append(np.full(idx.size, len(T)))
            log.info("%s: %d px, raw P range %.2f-%.2f", k, idx.size, prob[valid].min(), prob[valid].max())
    return np.concatenate(P), np.concatenate(Y).astype(float), np.concatenate(T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="M1_hma_official")
    ap.add_argument("--n_tiles", type=int, default=20)
    a = ap.parse_args()
    cfg = load_config()
    features = model_features(a.model)
    sub = "_sar" if any(f not in ("optical", "dem") for f in features) or "sarsubset" in a.model else ""
    h5path = ROOT / "data" / "benchmark" / f"hma_val{sub}.hdf5"
    with h5py.File(h5path, "r") as h:
        keys = sorted(h.keys())
    keys = sorted(np.random.default_rng(cfg["seed"]).choice(keys, min(a.n_tiles, len(keys)), replace=False).tolist())
    model = glavitu.build_model(resolve_weights(cfg, a.model), cfg["model"]["patch_size"], features=features)
    P, Y, T = collect(model, h5path, keys, features, cfg)
    fit_t = T % 2 == 0                     # tile-wise split: even tiles fit, odd tiles evaluate
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(P[fit_t], Y[fit_t])
    rep = dict(model=a.model, tiles=keys, n_pixels=int(P.size), raw_range=[float(P.min()), float(P.max())],
               ece_raw_heldout=ece(P[~fit_t], Y[~fit_t]), ece_calibrated_heldout=ece(iso.predict(P[~fit_t]), Y[~fit_t]))
    iso_all = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(P, Y)   # final: all tiles
    grid = np.linspace(0, 1, 21)
    rep["mapping"] = {f"{g:.2f}": float(iso_all.predict([g])[0]) for g in grid}
    for target in (0.05, 0.95):
        ok = grid[np.array([iso_all.predict([g])[0] for g in grid]) >= target] if target > 0.5 else \
             grid[np.array([iso_all.predict([g])[0] for g in grid]) <= target]
        rep[f"raw_threshold_for_calibrated_{target}"] = float(ok.min() if target > 0.5 else ok.max()) if ok.size else None
    out = ROOT / "models" / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{a.model}_isotonic.pickle", "wb") as f:
        pickle.dump(iso_all, f)
    (ROOT / "experiments" / f"calibration_{a.model}.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != "tiles"}, indent=1))


if __name__ == "__main__":
    main()
