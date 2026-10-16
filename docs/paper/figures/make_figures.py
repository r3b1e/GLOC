"""Figures for the manuscript (docs/paper/manuscript.tex) that are not in outputs/figures/.

  .env/bin/python docs/paper/figures/make_figures.py

Inputs: experiments/*/training_log.csv, experiments/benchmark_eval_test_summary.json,
experiments/calibration_M1_hma_official.json, docs/paper/data/*_allyears.csv.
Outputs (300 dpi PNG, this folder): fig_training.png, fig_benchmark.png, fig_calibration.png,
fig_area_allyears.png, fig_terminus_allyears.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "docs" / "paper" / "data"

# validated categorical order (dataviz reference palette, light mode); marker = secondary encoding
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
SERIES = {
    "M0":           ("#2a78d6", "o", "M0 global"),
    "M1":           ("#eb6834", "s", "M1 HMA official"),
    "M1-local":     ("#1baf7a", "^", "M1 HMA local"),
    "M2":           ("#0b0b0b", "D", "M2 Chhota Shigri"),
    "M3":           ("#e87ba4", "v", "M3 HMA + SAR"),
}
MODEL_KEYS = {"M0": "M0_global", "M1": "M1_hma_official", "M1-local": "M1_hma_local",
              "M2": "M2_cs_selftrain", "M3": "M3_hma_sar"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.dpi": 300,
})


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", OUT / name)


def fig_training():
    runs = [("M1/M1_hma_local", "M1 local (HMA, 8 × 250 steps)", "#eb6834", "s"),
            ("M2/M2_cs_selftrain", "M2 (pseudo-labels + replay, 4 × 200)", INK, "D"),
            ("M1/M1_sarsubset_nosar", "Optical + DEM (SAR subset, 4 × 250)", "#2a78d6", "o"),
            ("M3/M3_hma_sar", "Optical + DEM + SAR (4 × 250)", "#e87ba4", "v"),
            ("M1/M1_sarsubset_optonly", "Optical only (4 × 250)", "#1baf7a", "^")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for path, label, c, mk in runs:
        df = pd.read_csv(ROOT / "experiments" / path / "training_log.csv")
        ep = df.epoch + 1
        axes[0].plot(ep, df.loss, color=c, marker=mk, ms=4, lw=1.5, label=label)
        axes[1].plot(ep, df.val_output_iou_1, color=c, marker=mk, ms=4, lw=1.5, label=label)
    axes[0].set(title="(a) Training loss (focal, both heads)", xlabel="Epoch", ylabel="Loss")
    axes[1].set(title="(b) Validation glacier IoU (HMA val tiles)", xlabel="Epoch", ylabel="IoU")
    for a in axes:
        a.set_xticks(range(1, 9))
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
    fig.tight_layout()
    save(fig, "fig_training.png")


def fig_benchmark():
    s = json.loads((ROOT / "experiments" / "benchmark_eval_test_summary.json").read_text())
    order = [("M0_global", "M0 global"), ("M1_hma_official", "M1 HMA official"),
             ("M1_hma_local", "M1 HMA local"), ("M2_cs_selftrain", "M2 Chhota Shigri")]
    iou = [s[k]["pooled_iou"] for k, _ in order]
    # paired-bootstrap ΔIoU vs M0 (experiments/benchmark_eval_test_pairwise.json, README §16)
    delta = {"M1 HMA official": (0.070, 0.040, 0.107), "M1 HMA local": (0.081, 0.047, 0.125),
             "M2 Chhota Shigri": (0.101, 0.064, 0.148)}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), gridspec_kw=dict(width_ratios=[1.1, 1]))
    y = range(len(order))
    axes[0].barh(list(y), iou, color="#2a78d6", height=0.6)
    for i, v in enumerate(iou):
        axes[0].text(v + 0.005, i, f"{v:.3f}", va="center", color=INK, fontsize=7.5)
    axes[0].set_yticks(list(y), [n for _, n in order])
    axes[0].invert_yaxis()
    axes[0].set(xlim=(0.5, 0.82), xlabel="Pooled IoU (57 held-out HMA tiles)", title="(a) Glacier IoU")
    axes[0].grid(axis="y", visible=False)
    names = list(delta)
    for i, n in enumerate(names):
        d, lo, hi = delta[n]
        axes[1].plot([lo, hi], [i, i], color=INK2, lw=1.5)
        axes[1].plot(d, i, "o", color="#2a78d6", ms=6)
        axes[1].text(hi + 0.004, i, f"+{d:.3f} [{lo:.3f}, {hi:.3f}]", va="center", fontsize=7, color=INK)
    axes[1].axvline(0, color=INK2, lw=0.8)
    axes[1].set_yticks(range(len(names)), names)
    axes[1].invert_yaxis()
    axes[1].set(xlim=(-0.01, 0.23), xlabel="ΔIoU vs M0 (95 % paired bootstrap CI)",
                title="(b) Gain over the global model")
    axes[1].grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig_benchmark.png")


def fig_calibration():
    c = json.loads((ROOT / "experiments" / "calibration_M1_hma_official.json").read_text())
    m = sorted((float(k), v) for k, v in c["mapping"].items())
    lo, hi = c["raw_range"]
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.axvspan(lo, hi, color="#2a78d6", alpha=0.08, lw=0)
    ax.text(lo + 0.01, 0.55, "range of raw\noutput (HMA val)", ha="left", fontsize=7, color=INK2)
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls="--")
    ax.text(0.80, 0.70, "perfect\ncalibration", fontsize=7, color=INK2, rotation=38)
    ax.step([x for x, _ in m], [y for _, y in m], where="post", color="#2a78d6", lw=2)
    for t in (0.05, 0.95):
        ax.axhline(t, color="#eb6834", lw=0.8, ls=":")
    ax.annotate("raw 0.70 → 0.97", xy=(0.70, 0.967), xytext=(0.30, 0.80), fontsize=7, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Raw GlaViTU P(glacier)", ylabel="Calibrated probability",
           title=f"M1 isotonic calibration (ECE {c['ece_raw_heldout']:.3f} → {c['ece_calibrated_heldout']:.3f})")
    fig.tight_layout()
    save(fig, "fig_calibration.png")


def load_allyears():
    return {k: pd.read_csv(DATA / f"{v}_allyears.csv") for k, v in MODEL_KEYS.items()}


def _series_plot(ax, d, col, odd_only_labels=True):
    for k, (c, mk, label) in SERIES.items():
        df = d[k]
        lw, ms = (2.2, 5.5) if k == "M2" else (1.4, 4.5)
        ax.plot(df.year, df[col], color=c, lw=lw, alpha=0.95, zorder=3 if k == "M2" else 2,
                marker=mk, ms=ms, mec=SURFACE, mew=0.8, markevery=[], label=label)
        odd, even = df[df.year % 2 == 1], df[df.year % 2 == 0]
        ax.plot(odd.year, odd[col], mk, color=c, ms=ms, mec=SURFACE, mew=0.8, zorder=4)
        ax.plot(even.year, even[col], mk, color=SURFACE, ms=ms, mec=c, mew=1.2, zorder=4)
    ax.set_xticks(range(2017, 2026))


def fig_area():
    d = load_allyears()
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    m2 = d["M2"]
    ax.fill_between(m2.year, m2.area_km2 - m2.sigma_area_km2, m2.area_km2 + m2.sigma_area_km2,
                    color=INK, alpha=0.08, lw=0, label="M2 ± σ total")
    ax.axhspan(15.5, 15.7, color="#eda100", alpha=0.18, lw=0, label="Published 15.5–15.7 km²")
    _series_plot(ax, d, "area_km2")
    ax.set(xlabel="Year (filled = analysis epoch, hollow = M2 training year)", ylabel="Glacier area (km²)",
           title="Chhota Shigri glacier area, multi-date ensembles (15 Aug – 15 Sep)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=7)
    fig.tight_layout()
    save(fig, "fig_area_allyears.png")


def fig_terminus():
    d = load_allyears()
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.axhspan(-14.3, 14.3, color=INK2, alpha=0.10, lw=0, label="± measurement σ (14.3 m)")
    ax.axhline(0, color=INK2, lw=0.8)
    _series_plot(ax, d, "retreat_cum_centerline_m")
    ax.set(xlabel="Year (filled = analysis epoch, hollow = M2 training year)",
           ylabel="Retreat since 2017 (m)", title="Terminus retreat relative to 2017, centerline method (positive = retreat)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=7)
    fig.tight_layout()
    save(fig, "fig_terminus_allyears.png")


if __name__ == "__main__":
    fig_training()
    fig_benchmark()
    fig_calibration()
    fig_area()
    fig_terminus()
