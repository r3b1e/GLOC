"""Map figures: RGB composites, probability/confidence maps and outline overlays."""
from __future__ import annotations

import argparse

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common import ROOT, Grid, read_raster

FIG = ROOT / "outputs" / "figures"
INV = ROOT / "data" / "inventories"


def clearest_scene(year):
    """Clearest (lowest s2cloudless domain cloud) ensemble scene of the year inside the core window."""
    import pandas as pd
    from src.common import load_config
    from src.inference.ensemble import in_window
    cfg = load_config()
    win = cfg["ensemble"].get("window_core", cfg["ensemble"]["window"])
    d = pd.read_csv(ROOT / "data" / "sentinel2" / f"ensemble_{year}.csv")
    d = d[d.kept & d.date.apply(lambda x: in_window(x, win))].sort_values("s2cloudless_cloud")
    return d.date.iloc[0]


def rgb(year, bands=(2, 1, 0), stretch=(1, 99)):
    opt = read_raster(ROOT / "data" / "sentinel2" / "scenes" / clearest_scene(year) / "optical_toa.tif")
    img = np.moveaxis(opt[list(bands)], 0, -1)
    lo, hi = np.nanpercentile(img, stretch)
    return np.clip((img - lo) / (hi - lo), 0, 1)


def extent(grid: Grid):
    b = grid.bounds
    return (b[0], b[2], b[1], b[3])


def plot_outline(ax, gdf, **kw):
    gdf.boundary.plot(ax=ax, **kw)


def qc_figure(year, models, out=None):
    grid = Grid.load(INV / "grid.json")
    rgi = gpd.read_file(INV / "chhota_shigri.gpkg", layer="rgi7")
    ext = extent(grid)
    n = 2 + len(models)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.4), constrained_layout=True)
    axes[0].imshow(rgb(year), extent=ext)
    plot_outline(axes[0], rgi, color="yellow", lw=0.9, ls="--")
    axes[0].set_title(f"S2 L1C RGB {year} + RGI7 (2002, dashed)")
    axes[1].imshow(rgb(year, bands=(4, 3, 1)), extent=ext)
    axes[1].set_title("False colour SWIR1/NIR/G")
    colors = ["cyan", "magenta", "orange", "lime", "red"]
    for ax, m, c in zip(axes[2:], models, colors):
        exp = ROOT / "experiments" / m / str(year)
        p = read_raster(exp / "prob_glacier.tif")
        im = ax.imshow(p, extent=ext, cmap="Blues", vmin=0, vmax=1)
        plot_outline(ax, gpd.read_file(exp / "outline.gpkg"), color="red", lw=0.8)
        plot_outline(ax, rgi, color="k", lw=0.6, ls="--")
        ax.set_title(f"{m}: P(glacier) + outline")
        plot_outline(axes[0], gpd.read_file(exp / "outline.gpkg"), color=c, lw=0.8)
    fig.colorbar(im, ax=axes[-1], shrink=0.7, label="P(glacier)")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    out = out or FIG / f"qc_{year}_{'_'.join(models)}.png"
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--models", nargs="+", required=True)
    a = ap.parse_args()
    print(qc_figure(a.year, a.models))
