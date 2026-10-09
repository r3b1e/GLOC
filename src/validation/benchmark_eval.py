"""Evaluate checkpoints on held-out HMA benchmark tiles (official test split, never used in training).

  python -m src.validation.benchmark_eval --models M0_global M1_hma_official M1_hma_local --split test

Per tile: sliding-window prediction (same as deployment), padding removed using the tile attrs
(padding_height/width, original_height/width), pixels without a label ([0, 0]) ignored.
Reports pooled IoU (sum of TP/FP/FN over all tiles, as in Maslov et al.) and mean per-tile IoU,
plus precision, recall, F1 and boundary F1 (2 px tolerance).
Output: experiments/benchmark_eval_<split>.csv (per tile) and _summary.json.
"""
from __future__ import annotations

import argparse
import json

import h5py
import numpy as np
import pandas as pd

from src.common import ROOT, get_logger, load_config
from src.inference import glavitu
from src.inference.run import resolve_weights
from src.validation.metrics import boundary_metrics

log = get_logger("bench_eval")


def tile_valid_slices(attrs):
    ph, pw = int(attrs.get("padding_height", 0)), int(attrs.get("padding_width", 0))
    oh, ow = int(attrs.get("original_height", attrs["height"])), int(attrs.get("original_width", attrs["width"]))
    return slice(ph, ph + oh), slice(pw, pw + ow)


def model_features(model_key):
    if "optonly" in model_key.lower():
        return ("optical",)
    return ("optical", "dem", "co_pol_sar") if "sar" in model_key.lower() and "nosar" not in model_key.lower() else ("optical", "dem")


def evaluate(model_key, h5path, features=None, max_tiles=None):
    cfg = load_config()
    features = features or model_features(model_key)
    model = glavitu.build_model(resolve_weights(cfg, model_key), cfg["model"]["patch_size"], features=features)
    rows = []
    with h5py.File(h5path, "r") as h:
        keys = sorted(h.keys())[:max_tiles]
        for k in keys:
            g = h[k]
            rs, cs = tile_valid_slices(g.attrs)
            feats = {f: g[f][()].astype(np.float32) for f in features}
            lab = g["outlines"][()]
            prob = glavitu.predict(model, feats, cfg["model"]["patch_size"], cfg["model"]["batch_size"])
            pred = prob[..., 1] >= 0.5
            ref = lab[..., 1] > 0.5
            valid = lab.sum(-1) > 0
            sl = (rs, cs)
            pred, ref, valid = pred[sl], ref[sl], valid[sl]
            tp = int((pred & ref & valid).sum()); fp = int((pred & ~ref & valid).sum()); fn = int((~pred & ref & valid).sum())
            b = boundary_metrics(pred & valid, ref & valid)
            rows.append(dict(model=model_key, tile=k, subregion=g.attrs.get("subregion"), tp=tp, fp=fp, fn=fn,
                             iou=tp / max(tp + fp + fn, 1), boundary_f1=b["boundary_f1"],
                             mean_boundary_dist_m=b["mean_boundary_dist_m"]))
            log.info("%s %s IoU %.3f", model_key, k, rows[-1]["iou"])
    return pd.DataFrame(rows)


def summarise(df):
    out = {}
    for m, d in df.groupby("model"):
        tp, fp, fn = d.tp.sum(), d.fp.sum(), d.fn.sum()
        p, r = tp / (tp + fp), tp / (tp + fn)
        with_glacier = d[(d.tp + d.fn) > 0]
        out[m] = dict(n_tiles=len(d), pooled_iou=tp / (tp + fp + fn), precision=p, recall=r, f1=2 * p * r / (p + r),
                      mean_tile_iou=float(with_glacier.iou.mean()), median_tile_iou=float(with_glacier.iou.median()),
                      mean_boundary_f1=float(d.boundary_f1.mean()),
                      by_subregion={s: float(x.tp.sum() / (x.tp.sum() + x.fp.sum() + x.fn.sum())) for s, x in d.groupby("subregion")})
    return out


def paired_bootstrap(df, a, b, n=2000, seed=0):
    """95 % CI of pooled-IoU difference (b - a), resampling tiles."""
    A = df[df.model == a].set_index("tile"); B = df[df.model == b].set_index("tile")
    tiles = A.index.intersection(B.index).values
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        t = rng.choice(tiles, len(tiles))
        ia = A.loc[t, "tp"].sum() / A.loc[t, ["tp", "fp", "fn"]].values.sum()
        ib = B.loc[t, "tp"].sum() / B.loc[t, ["tp", "fp", "fn"]].values.sum()
        diffs.append(ib - ia)
    return dict(delta=float(np.mean(diffs)), ci95=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_tiles", type=int)
    ap.add_argument("--subset", default="", help="e.g. '_sar' -> hma_<split>_sar.hdf5")
    a = ap.parse_args()
    h5path = ROOT / "data" / "benchmark" / f"hma_{a.split}{a.subset}.hdf5"
    out_csv = ROOT / "experiments" / f"benchmark_eval_{a.split}{a.subset}.csv"
    prev = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
    dfs = [prev[~prev.model.isin(a.models)]] if len(prev) else []
    dfs += [evaluate(m, h5path, max_tiles=a.max_tiles) for m in a.models]
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(out_csv, index=False)
    summ = summarise(df)
    models = list(summ)
    if "M0_global" in models:
        summ["paired_bootstrap_vs_M0"] = {m: paired_bootstrap(df, "M0_global", m) for m in models if m != "M0_global"}
    (ROOT / "experiments" / f"benchmark_eval_{a.split}{a.subset}_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
