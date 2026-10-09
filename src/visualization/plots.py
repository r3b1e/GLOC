"""Result figures for one model's multi-temporal series.

  python -m src.visualization.plots --model M2_cs_selftrain
"""
from __future__ import annotations

import argparse
import json

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

from src.common import ROOT, Grid, read_raster
from src.visualization.maps import extent, rgb

FIG = ROOT / "outputs" / "figures"
STATS = ROOT / "outputs" / "statistics"
CM = ROOT / "outputs" / "change_maps"
INV = ROOT / "data" / "inventories"


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return FIG / name


def zoom_bounds(grid, pad=600):
    rgi = gpd.read_file(INV / "chhota_shigri.gpkg", layer="rgi7")
    b = rgi.total_bounds
    return b[0] - pad, b[2] + pad, b[1] - pad, b[3] + pad


def area_plots(model):
    df = pd.read_csv(STATS / f"{model}_area.csv")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].errorbar(df.t, df.area_km2, yerr=df.sigma_area_km2, fmt="o-", capsize=3)
    ax[0].set(xlabel="year", ylabel="area (km²)", title="Glacier area A(t) ± 1σ")
    ax[1].bar(df.year[1:], df.rate_prev_km2_per_yr[1:], width=1.2)
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set(xlabel="epoch end year", ylabel="km² / yr", title="Annualised area change between epochs")
    ax[2].errorbar(df.t, -df.dA_cum_km2, yerr=df.sigma_dA_cum_km2.fillna(0), fmt="s-", color="C3", capsize=3)
    ax[2].set(xlabel="year", ylabel="cumulative loss (km²)", title=f"Cumulative area loss since {df.year.iloc[0]}")
    for a in ax:
        a.grid(alpha=0.3)
    return save(fig, f"{model}_area_timeseries.png")


def terminus_plot(model):
    df = pd.read_csv(STATS / f"{model}_terminus.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.errorbar(df.t, df.retreat_cum_centerline_m, yerr=df.sigma_retreat_cum_m, fmt="o-", capsize=3, label="centerline intersection")
    ax.plot(df.t, df.retreat_cum_box_m, "s--", label="box method (tongue box)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set(xlabel="year", ylabel=f"cumulative retreat since {df.year.iloc[0]} (m)", title="Terminus retreat")
    ax.legend(); ax.grid(alpha=0.3)
    return save(fig, f"{model}_terminus_retreat.png")


def outlines_map(model):
    grid = Grid.load(INV / "grid.json")
    exp = ROOT / "experiments" / model
    years = sorted(int(p.name) for p in exp.iterdir() if p.name.isdigit())
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb(years[-1]), extent=extent(grid))
    cmap = plt.get_cmap("plasma", len(years))
    for i, y in enumerate(years):
        gpd.read_file(exp / str(y) / "outline.gpkg").boundary.plot(ax=ax, color=cmap(i), lw=1.0, label=str(y))
    gpd.read_file(INV / "chhota_shigri.gpkg", layer="centerline").plot(ax=ax, color="white", lw=0.8, ls=":")
    tp = ROOT / "outputs" / "polygons" / f"{model}_terminus_points.gpkg"
    if tp.exists():
        t = gpd.read_file(tp)
        ax.scatter(t.geometry.x, t.geometry.y, c=[cmap(years.index(y)) for y in t.year], s=25, edgecolor="k", zorder=5)
    x0, x1, y0, y1 = zoom_bounds(grid)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(f"{model}: automated outlines (background: S2 {years[-1]})")
    ax.set_xticks([]); ax.set_yticks([])
    return save(fig, f"{model}_yearly_outlines.png")


def terminus_zoom(model):
    grid = Grid.load(INV / "grid.json")
    exp = ROOT / "experiments" / model
    years = sorted(int(p.name) for p in exp.iterdir() if p.name.isdigit())
    t = gpd.read_file(ROOT / "outputs" / "polygons" / f"{model}_terminus_points.gpkg")
    cx, cy = t.geometry.x.mean(), t.geometry.y.mean()
    fig, axes = plt.subplots(1, len(years), figsize=(3.2 * len(years), 3.6))
    cl = gpd.read_file(INV / "chhota_shigri.gpkg", layer="centerline")
    for ax, y in zip(np.atleast_1d(axes), years):
        ax.imshow(rgb(y), extent=extent(grid))
        gpd.read_file(exp / str(y) / "outline.gpkg").boundary.plot(ax=ax, color="cyan", lw=1)
        cl.plot(ax=ax, color="yellow", lw=0.7, ls=":")
        ty = t[t.year == y]
        ax.scatter(ty.geometry.x, ty.geometry.y, c="red", s=20, zorder=5)
        ax.set_xlim(cx - 700, cx + 700); ax.set_ylim(cy - 700, cy + 700)
        ax.set_title(str(y)); ax.set_xticks([]); ax.set_yticks([])
    return save(fig, f"{model}_terminus_zoom.png")


def year_of_loss_map(model):
    grid = Grid.load(INV / "grid.json")
    yol = read_raster(CM / f"{model}_year_of_loss.tif").astype(int)
    years = sorted(v for v in np.unique(yol) if v > 1000)
    codes = [1, 2] + years
    colors = ["#cfe8ff", "#bbbbbb"] + list(plt.get_cmap("autumn_r", max(len(years), 1))(np.arange(len(years))))
    idx = np.full(yol.shape, np.nan)
    for i, c in enumerate(codes):
        idx[yol == c] = i
    gain = yol < 0
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(read_raster(ROOT / "data" / "dem" / "hillshade.tif"), cmap="gray", extent=extent(grid))
    ax.imshow(idx, cmap=ListedColormap(colors), norm=BoundaryNorm(np.arange(len(codes) + 1) - 0.5, len(codes)),
              extent=extent(grid), interpolation="nearest")
    if gain.any():
        ax.contour(np.flipud(gain) if False else gain, levels=[0.5], colors="blue", linewidths=0.6, extent=extent(grid), origin="upper")
    x0, x1, y0, y1 = zoom_bounds(grid)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    labels = ["glacier in all epochs", "unstable (lost & re-appeared)"] + [f"lost by {y}" for y in years]
    ax.legend(handles, labels, loc="lower left", fontsize=8)
    ax.set_title(f"{model}: year-of-loss"); ax.set_xticks([]); ax.set_yticks([])
    return save(fig, f"{model}_year_of_loss.png")


def pattern_plots(model):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    e = pd.read_csv(STATS / f"{model}_loss_by_elevation.csv")
    e = e[e.baseline_area_km2 > 0]
    ax[0].barh(e.cls, e.baseline_area_km2, color="#cfe8ff", label="baseline glacier")
    ax[0].barh(e.cls, e.lost_area_km2, color="C3", label="lost")
    ax2 = ax[0].twiny(); ax2.plot(100 * e.loss_fraction, e.cls, "k.-"); ax2.set_xlabel("loss fraction (%)")
    ax[0].set(xlabel="area (km²)", ylabel="elevation band (m)", title="Loss by elevation"); ax[0].legend(loc="lower right")
    s = pd.read_csv(STATS / f"{model}_loss_by_slope.csv")
    ax[1].bar(s.cls, 100 * s.loss_fraction, color="C1")
    ax[1].set(xlabel="slope class (°)", ylabel="loss fraction (%)", title="Loss by slope class")
    a = pd.read_csv(STATS / f"{model}_loss_by_aspect.csv")
    ax[2].remove()
    axp = fig.add_subplot(1, 3, 3, projection="polar")
    th = np.radians(np.arange(0, 360, 45))
    axp.bar(th, 100 * a.loss_fraction.fillna(0), width=np.radians(40), color="C2", alpha=0.8)
    axp.set_theta_zero_location("N"); axp.set_theta_direction(-1)
    axp.set_xticks(th); axp.set_xticklabels(a.cls); axp.set_title("Loss fraction (%) by aspect")
    return save(fig, f"{model}_patterns.png")


def hotspot_maps(model):
    grid = Grid.load(INV / "grid.json")
    z = read_raster(CM / f"{model}_hotspot_gistar.tif")
    p = read_raster(CM / f"{model}_susceptibility.tif")
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    for a in ax:
        a.imshow(read_raster(ROOT / "data" / "dem" / "hillshade.tif"), cmap="gray", extent=extent(grid))
    im0 = ax[0].imshow(np.ma.masked_invalid(z), cmap="RdBu_r", vmin=-4, vmax=4, extent=extent(grid), alpha=0.85)
    im1 = ax[1].imshow(np.ma.masked_invalid(p), cmap="magma", vmin=0, vmax=np.nanpercentile(p, 99), extent=extent(grid))
    fig.colorbar(im0, ax=ax[0], shrink=0.6, label="Gi* z-score (>1.96: loss hotspot)")
    fig.colorbar(im1, ax=ax[1], shrink=0.6, label="P(loss | terrain) — descriptive")
    ax[0].set_title("Retreat hotspots (Getis-Ord Gi*, 100 m cells)")
    ax[1].set_title("Retreat susceptibility (logistic, spatial-block CV)")
    x0, x1, y0, y1 = zoom_bounds(grid)
    for a in ax:
        a.set_xlim(x0, x1); a.set_ylim(y0, y1); a.set_xticks([]); a.set_yticks([])
    return save(fig, f"{model}_hotspots.png")


def study_area():
    grid = Grid.load(INV / "grid.json")
    gpkg = INV / "chhota_shigri.gpkg"
    fig, ax = plt.subplots(figsize=(8, 7.5))
    ax.imshow(read_raster(ROOT / "data" / "dem" / "hillshade.tif"), cmap="gray", extent=extent(grid))
    im = ax.imshow(read_raster(ROOT / "data" / "dem" / "elevation.tif"), cmap="terrain", alpha=0.45, extent=extent(grid))
    gpd.read_file(gpkg, layer="neighbours_rgi7").boundary.plot(ax=ax, color="0.3", lw=0.5)
    gpd.read_file(gpkg, layer="domain").boundary.plot(ax=ax, color="orange", lw=1, ls="--", label="analysis domain")
    gpd.read_file(gpkg, layer="rgi7").boundary.plot(ax=ax, color="blue", lw=1.2, label="RGI7 (2002)")
    try:
        gpd.read_file(gpkg, layer="centerline").plot(ax=ax, color="red", lw=1, label="auto centerline")
    except Exception:
        pass
    fig.colorbar(im, ax=ax, shrink=0.6, label="elevation (m, COP-DEM GLO-30)")
    ax.legend(loc="lower left")
    ax.set_title("Chhota Shigri study area (UTM 43N)")
    return save(fig, "study_area.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    for f in (study_area, lambda: area_plots(a.model), lambda: terminus_plot(a.model), lambda: outlines_map(a.model),
              lambda: terminus_zoom(a.model), lambda: year_of_loss_map(a.model), lambda: pattern_plots(a.model),
              lambda: hotspot_maps(a.model)):
        try:
            print(f())
        except FileNotFoundError as e:
            print("skip:", e)


if __name__ == "__main__":
    main()
