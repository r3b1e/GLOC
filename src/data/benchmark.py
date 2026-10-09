"""Extract the High-Mountain-Asia (HMA) subset of the GlaViTU benchmark by HTTP byte ranges.

The full tile-based benchmark (Maslov et al. 2025; NIRD doi:10.11582/2024.00168) is 411 GB
(train 243 GB, val 81 GB, test 86 GB), far beyond local disk. NIRD serves the HDF5 files with
HTTP range support, so h5py + fsspec read only the requested tiles/datasets.

Local output: data/benchmark/hma_<split>.hdf5, same layout as the official files (one group per
tile, attrs preserved) but float32 and only the features we use. The official train/val/test
split is kept unchanged, so held-out HMA test tiles are never seen during M1 fine-tuning.

  python -m src.data.benchmark --splits val test train [--max_tiles N] [--features ...]
"""
from __future__ import annotations

import argparse
import json
import time

import fsspec
import h5py
import numpy as np

from src.common import ROOT, get_logger

log = get_logger("benchmark")
BENCH = ROOT / "data" / "benchmark"
URLS = {s: f"https://data.archive.sigma2.no/dataset/243c5285-2bf0-438f-bf2c-74b7bdba182e/download/20230905_{s}_global_ps384.hdf5"
        for s in ("train", "val", "test")}
DEFAULT_FEATURES = ["optical", "dem", "outlines", "co_pol_sar", "cross_pol_sar"]


def open_remote(split):
    f = fsspec.open(URLS[split], block_size=8 * 2 ** 20, cache_type="readahead").open()
    return h5py.File(f, "r")


def extract(split, region="HMA", features=DEFAULT_FEATURES, max_tiles=None, seed=0, shard=None):
    """One HDF5 file per tile, written to <tile>.tmp and atomically renamed when complete,
    so an interrupted run never corrupts finished tiles (resume = skip existing files)."""
    keys_all = json.loads((BENCH / "tile_keys.json").read_text())[split]
    keys = sorted(k for k in keys_all if k.split("-")[0] == region)
    if max_tiles and len(keys) > max_tiles:
        keys = sorted(np.random.default_rng(seed).choice(keys, max_tiles, replace=False).tolist())
    if shard:  # "k/n": this worker handles every n-th tile
        k_, n_ = map(int, shard.split("/"))
        keys = keys[k_::n_]
    tile_dir = BENCH / "tiles" / split
    tile_dir.mkdir(parents=True, exist_ok=True)
    todo = [k for k in keys if not (tile_dir / f"{k}.hdf5").exists()]
    if not todo:
        return tile_dir
    src = open_remote(split)
    for i, k in enumerate(todo):
        t = time.time()
        g_src = src[k]
        tmp = tile_dir / f"{k}.tmp"
        with h5py.File(tmp, "w") as dst:
            g = dst.create_group(k)
            for a, v in g_src.attrs.items():
                g.attrs[a] = v
            for f in features:
                if f in g_src:
                    arr = g_src[f][()]
                    g.create_dataset(f, data=arr.astype(np.float32 if f != "outlines" else np.uint8),
                                     compression="lzf", chunks=(128, 128, arr.shape[-1]))
        tmp.rename(tile_dir / f"{k}.hdf5")
        log.info("%s %d/%d %s (%s) %.0fs", split, i + 1, len(todo), k, g_src.attrs.get("subregion"), time.time() - t)
    return tile_dir


def merge_shards(split, region="HMA"):
    """Master file with external links to every completed per-tile file (transparent to h5py)."""
    master = BENCH / f"{region.lower()}_{split}.hdf5"
    tmp = master.with_suffix(".tmp")
    with h5py.File(tmp, "w") as m:
        for f in sorted((BENCH / "tiles" / split).glob(f"{region}-*.hdf5")):
            m[f.stem] = h5py.ExternalLink(str(f.resolve()), f.stem)
        n = len(m.keys())
    tmp.rename(master)
    log.info("%s: %d tiles in %s", split, n, master.name)
    return master


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val", "test", "train"])
    ap.add_argument("--region", default="HMA")
    ap.add_argument("--max_tiles", type=int)
    ap.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    ap.add_argument("--shard", help="k/n")
    ap.add_argument("--merge", action="store_true", help="only (re)build master files from shards")
    a = ap.parse_args()
    for s in a.splits:
        if a.merge:
            merge_shards(s, a.region)
        else:
            extract(s, a.region, a.features, a.max_tiles, shard=a.shard)


if __name__ == "__main__":
    main()
