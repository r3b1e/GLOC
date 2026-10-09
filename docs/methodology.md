# Methodology

Every step below is automatic and parameterised in `configs/chhota_shigri.yaml`.
No glacier boundary, tie point or centerline is drawn by hand anywhere in the workflow.

## 1. Study-area definition (`src/data/inventory.py`)

1. RGI 6.0, RGI 7.0 and all GLIMS outlines inside lon 77.40–77.75, lat 32.10–32.40 are pulled
   from the GLIMS WFS.
2. Chhota Shigri = `RGI2000-v7.0-G-14-30579` (≡ `RGI60-14.15990`).
3. **ROI:** RGI7 outline + 2.5 km, envelope, snapped outward to the 60 m Sentinel-2 lattice.
   Grid: EPSG:32643, 10 m, 1434 × 1254 px. The 60 m snapping puts 10, 20 and 60 m bands on
   whole pixels, so no sub-pixel shift is introduced.
4. **Analysis domain:** RGI7 outline + 1 km buffer minus the 33 neighbouring RGI7 glaciers
   (47.9 km²). It fixes the ice divides through time, and it allows the glacier to be mapped
   larger or smaller than in 2002 without "stealing" ice from neighbours.

## 2. Sentinel-2 acquisition and cloud screening (`src/data/sentinel2.py`)

* **Window:** 1 Sep – 15 Oct (end of ablation season); fallback 15 Aug – 31 Oct if no usable
  composite is found. The window actually used is recorded per year.
* **Candidates:** all L2A items of tile 43SGR from Planetary Computer, one per datatake (the
  latest reprocessing).
* **Screening (fast):** SCL over the domain gives cloud (8, 9, 10), shadow (3), nodata (0, 1) and
  off-glacier snow (SCL 11 on domain minus RGI7). Off-glacier snow is the seasonal-snow indicator.
* **Confirmation:** for the 4 best candidates, s2cloudless runs on the matching L1C bands at 20 m.
  Cloud fraction = s2cloudless mask + SCL shadow + nodata.
* **Ranking score:** cloud fraction + off-glacier snow fraction. This penalises October scenes
  after early snowfall (e.g. 2023-10-06: 0.06 % cloud but 49 % off-glacier snow).
* **Multi-date ensemble (final design):** every scene in 15 Aug–15 Oct is screened.
  * Scenes are kept if s2cloudless domain cloud ≤ 0.5 and off-glacier snow (SCL) ≤ 0.45.
  * The primary analysis window is **15 Aug–15 Sep**. This was revised after diagnostics: later
    scenes map 1.5–4.6 km² too little glacier because of autumn snowfall and low sun.
  * Each scene is mapped separately, with cloud (s2cloudless mask) set to NaN.
  * Per-pixel median across scenes. A whole scene is rejected if |A_i − median| >
    max(3 × 1.4826·MAD, 0.5 km²).
  * One clear observation is enough. Pixels with none are gap-filled from the same year's clear
    scenes outside the window; anything left is reported as unobserved.
  * The robust between-scene area spread is the surface-condition uncertainty.
  * The earlier single-best-scene composite (gap-filled from the next-best scenes) is superseded,
    and its outputs are no longer used.
* **Model input:** L1C TOA reflectance from the Google public bucket, harmonised with
  `RADIO_ADD_OFFSET`. B2, B3, B4 and B8 are native 10 m; B11 and B12 are bilinear from 20 m.

## 3. DEM (`src/data/dem.py`)

Copernicus GLO-30 is bilinearly resampled to 10 m for elevation. Slope and aspect (Horn 1981)
are computed at 30 m and then upsampled; aspect is interpolated through sin/cos components.

## 4. Model and inference (`src/inference/glavitu.py`, `src/inference/run.py`)

* **Model:** GlaViTU (Maslov et al. 2025), about 10.0 M parameters. It has a per-group fusion block
  (convolution + squeeze-excitation per input group), a SETR transformer encoder (12 blocks,
  16 px patches) with a deep-supervision head, and a residual U-Net refinement with a final
  softmax (2 classes).
* **Inputs:** exactly the official 8 channels, as two groups: `optical` (B2, B3, B4, B8, B11, B12
  TOA) and `dem` (elevation in m, slope in degrees). Each is min-max normalised with the official
  `utils/dataset_stats.pickle`.
* **Inference:** 384 px patches with stride 192, Gaussian-weighted blending, and centred zero
  padding. This is identical to the official `deploy.py`, which cannot be imported directly
  because it opens the training HDF5 at import time.
* **Confidence:** 1 − Shannon entropy (base 2) of the softmax.
* **Hardware note:** on Apple M4 with `tensorflow-metal 1.1.0`, `model.predict` hits a SIGBUS
  (exit 138) on real ROI inputs. It runs on random tensors. CPU inference is therefore the default
  (`GEOAI_DEVICE=cpu`); it takes about 30 s per ROI.

## 5. Post-processing (`src/postprocessing/mask.py`)

1. P(glacier) ≥ 0.5, then a 3 × 3 median filter (as in the official `--smoothing 3`).
2. Clip to the analysis domain.
3. Keep only the 8-connected components that touch the RGI7 core (the outline eroded by 300 m).
   The core only identifies which ice body is Chhota Shigri. Margins and terminus come entirely
   from the model.
4. Remove objects < 100 px (0.01 km²) and fill holes < 50 px.
5. Polygonise, `make_valid`, drop slivers < 4 px, and check topology (validity, parts, holes,
   raster vs vector area).

## 6. Transfer learning hierarchy

| ID | Description | Training data | Selection data |
|---|---|---|---|
| M0 | official global GlaViTU | global benchmark (authors) | — |
| M1-official | official global → HMA fine-tune (`glavitu_finetuning_HMA`) | HMA train (authors) | HMA val |
| M1-local | our reproduction: global → HMA, reduced budget (`src/training/finetune.py`) | HMA train tiles | HMA val |
| M2 | M1 → Chhota Shigri self-training on even-year pseudo-labels + HMA replay | pseudo-labels 2018/20/22/24 + HMA train | HMA val |
| M3 | M1 + Sentinel-1 backscatter branches (weight transfer) | HMA train with SAR | HMA val |

**Fine-tuning recipe:** the official settings are Adam, lr 5e-5 with cosine decay, focal loss
(γ = 2) on both deep-supervision heads (weights 0.5 / 1.0), flip/rotation/crop-scale, gamma
contrast, feature occlusion, Gaussian shift/noise and label smoothing 0.1. The checkpoint is the
best val glacier IoU. Only the budget is reduced (batch, steps per epoch, epochs), and it is
recorded in `experiments/<M>/<run>/run_config.json`.

## 7. Pseudo-labels and temporal self-training (`src/training/pseudo_labels.py`)

* **Teacher:** M1. Pseudo-label years are 2018, 2020, 2022 and 2024. The analysis years
  2017–2025 (odd) are never used for training. This gives temporal hold-out.
* **Labels:** thresholds apply to **isotonic-calibrated** probabilities (fitted on HMA val; raw
  GlaViTU never leaves 0.24–0.89).
  * Glacier: calibrated p ≥ 0.95 in ≥ 60 % of a pixel's clear scenes (15 core-window scenes from
    the 4 even years), never ≤ 0.05, and the yearly median is ≥ 0.95 in ≥ 3 of 4 years.
  * Background: the symmetric rule.
  * The consensus label is attached to every scene's imagery.
  All other pixels are ignored (target [0, 0], which gives zero focal loss).
* **Consequence:** pixels that change between years get no label. The labels carry no
  assumption about the direction or amount of change, and the model must generalise to the
  changing margin from HMA supervision plus stable site-specific context (local snow, shadow,
  debris, illumination).
* **No inventory outline is used as a label.**
* **Replay:** HMA training tiles are mixed in to prevent forgetting. The CS tiles are
  hard-linked 6 times to set their sampling weight.

## 8. Co-registration (`src/preprocessing/coregister.py`)

Phase cross-correlation (upsample 20) runs on Sobel gradient magnitude of B8, in 256 px windows,
over stable terrain: outside the domain + 300 m and all RGI7 glaciers + 150 m, snow-free,
cloud-free, slope < 35°. **Every ensemble scene** is compared with 2023-09-11. Scenes with a shift
> 0.25 px are Fourier-shifted (`--apply`) and re-measured:
* 2017-08-23: 0.67 → 0.11 px,
* 2019-09-07: 0.36 → 0.14 px.

All residuals are ≤ 1.6 m. The per-year maximum residual enters the terminus error.

## 9. Change detection (`src/change_detection/change.py`, `centerline.py`)

* **Area** A(t) inside the domain; ΔA, annualised ΔA, loss %.
* **σ_A:** three terms in quadrature:
  * perimeter × 0.5 px buffer,
  * half the area range between **calibrated** probability thresholds 0.4 and 0.6 (isotonic
    calibration on HMA val, because the raw softmax is compressed),
  * the robust between-scene surface spread.
* **σ_ΔA:** the threshold term is systematic, so it is propagated by recomputing ΔA at both
  thresholds. The buffer and surface terms add in quadrature.
* **Trend:** weighted least squares with weights 1/σ_indep², reported with a 95 % CI.
* **High-confidence period:** epochs with surface SD ≤ 0.35 km².
* **Centerline:** automatic, least-cost path from the RGI7 terminus to the farthest boundary
  elevation maximum, with cost 1 + 200·((Dmax − D)/Dmax)^4 (after Kienholz et al. 2014).
* **Terminus:** (i) the downstream-most centerline point reached continuously from the head
  (gaps < 100 m tolerated); (ii) the box method, i.e. glacier area in a fixed 2 km box,
  half the 2002 tongue width wide (159 m), divided by its width.
* **Cross-model summary:** median, min and max across the 5 models
  (`src/validation/aggregate_results.py`). The model spread is the structural uncertainty of the
  debris-covered snout position.
* **σ_terminus:** √(res² + co-registration residual²).
* **Year-of-loss:** a pixel gets year Y if it is glacier at baseline and non-glacier in Y and in
  every later epoch. Code 2 marks "unstable" pixels (lost, then re-appeared). Codes −Y mark gains.

## 10. Spatial pattern analysis (`src/change_detection/patterns.py`)

* **Population:** baseline glacier pixels, classed as lost or retained (unstable excluded).
* **Classes:** 100 m elevation bands, slope classes, 8 aspect sectors and distance-to-terminus
  classes. For each class the loss fraction and the enrichment (share of loss ÷ share of glacier)
  are computed. No class is assumed vulnerable in advance.
* **Hotspots:** Getis-Ord Gi* on the loss fraction of 100 m cells (5 × 5 cell neighbourhood).
* **Susceptibility:** logistic regression on standardised elevation, slope, aspect sin/cos,
  distance to terminus and distance to margin. Validated with 5-fold spatial-block CV (1 km
  blocks); coefficients with 95 % CI. It is descriptive only, not a forecast.
