# Validation

Validation never depends on manually traced Chhota Shigri boundaries. It uses four independent
lines of evidence (A–D below), plus probability calibration and co-registration checks.

## A. Held-out benchmark tiles (Maslov et al. 2025, official split)

* **Data:** HMA test tiles from the NIRD archive (doi:10.11582/2024.00168). These were never used
  for training or checkpoint selection. No tile lies within 0.05° of Chhota Shigri.
* **Metrics** (`src/validation/benchmark_eval.py`): pooled IoU (Σ TP over Σ TP+FP+FN, as in the
  paper), precision, recall, F1, mean tile IoU, and boundary F1 with a 2 px tolerance. Padding
  and unlabelled pixels are excluded.
* **Uncertainty:** paired bootstrap over tiles (2000 resamples), 95 % CI of the pooled-IoU
  difference.

### A1. Transfer-learning hierarchy (57 HMA test tiles)

| Model | Pooled IoU | Precision | Recall | F1 | Boundary F1 |
|---|---|---|---|---|---|
| M0 global | 0.678 | 0.898 | 0.735 | 0.808 | 0.266 |
| M1 official (global → HMA, authors) | 0.748 | 0.905 | 0.811 | 0.856 | 0.321 |
| M1-local (global → HMA, this work, ≈1.7 % of official steps) | 0.758 | 0.886 | 0.841 | 0.863 | 0.335 |
| M2 (M1-official → Chhota Shigri self-training + HMA replay) | 0.779 | 0.882 | 0.870 | 0.876 | 0.381 |

| Paired difference | ΔIoU | 95 % CI |
|---|---|---|
| M1-official − M0 | +0.070 | [+0.040, +0.107] |
| M1-local − M0 | +0.081 | [+0.047, +0.125] |
| M1-local − M1-official | +0.011 | [−0.001, +0.024] (not significant) |
| M2 − M1-official | +0.031 | [+0.019, +0.046] |
| M2 − M0 | +0.101 | [+0.064, +0.148] |

* M1-official's IoU of 0.748 agrees with the paper's ">0.75 for debris-rich HMA". This is an
  independent check of our inference implementation (normalisation, padding, blending).
* M2's HMA gain cannot be attributed to Chhota Shigri adaptation alone, because M2 also saw
  40 HMA replay tiles. It shows that self-training caused **no forgetting**.

### A2. Input ablation (22 HMA test tiles that carry Sentinel-1 VV; equal budgets, same 66 train tiles)

| Inputs | Pooled IoU | Precision | Recall | Boundary F1 |
|---|---|---|---|---|
| optical | 0.676 | 0.904 | 0.729 | 0.327 |
| optical + DEM | **0.824** | 0.918 | 0.889 | 0.383 |
| optical + DEM + SAR (VV asc/desc) | 0.808 | 0.915 | 0.874 | 0.350 |

| Paired difference | ΔIoU | 95 % CI |
|---|---|---|
| + DEM (opt+DEM − opt) | +0.149 | [+0.078, +0.241] |
| + SAR (opt+DEM+SAR − opt+DEM) | −0.016 | [−0.029, −0.008] |

The DEM is the most important input after the optical bands. SAR showed no benefit under a
reduced, transfer-learning budget: the new SAR branch was randomly initialised and trained for
1000 steps. This does not contradict the paper, where SAR helped with full training.

## B. Existing outlines (inventory agreement, not accuracy)

`src/validation/site_validation.py` compares each automated mask with the 2002 RGI7 outline. It
reports IoU over the domain and over the upper glacier (> 5000 m), where 2002–2025 margin change
should be small. The 2002 outline is **not** ground truth for 2017–2025, so this measures
agreement, not accuracy. All GLIMS outlines are older than the Sentinel-2 era (2000–2009) and
use different divides (11.6–16.8 km²).

## C. Published areas

| Source | Area (km²) | Epoch / definition |
|---|---|---|
| Wagnon et al. 2007, J. Glaciol. 53(183) | 15.7 | c. 2003, including tributaries |
| Mandal et al. 2020, J. Glaciol. 66(259) | 15.5 | glacier-wide, 2002–2019 study period |
| RGI 7.0 / GAMDAM | 16.77 | 2002-08-02 |
| ICIMOD (GLIMS 154108) | 11.65 | 2009-10-25, different divides |

Published retreat for comparison: about 800 m between 1988 and 2003, 53 m a⁻¹ (Kulkarni et
al. 2007, cited in Wagnon et al. 2007); 800–850 m between 1972 and 2009 (cited in Gantayat et
al. 2017). Debris cover rose from 3.4 % (2005) to about 11 % (2014) (Garg et al. 2017, cited in
Mandal et al. 2020).

## D. Temporal consistency

Reported per model:
* consecutive-epoch IoU,
* area of "unstable" pixels (lost, then re-appeared),
* number of epoch-to-epoch area gains above 2σ,
* number of terminus advances above 2σ.

A physically consistent series on a retreating glacier should show few of each.

## Probability calibration

The raw GlaViTU softmax is compressed (all pixels within about 0.24–0.89; label smoothing plus
focal loss). Isotonic calibrators are fitted on HMA val tiles (`src/validation/calibration.py`).
For M1-official the held-out ECE goes from 0.239 (raw) to 0.011 (calibrated). Calibrated
probabilities are used for (i) the pseudo-label confidence thresholds and (ii) the
probability-threshold sensitivity of the area. Binary masks keep the official argmax/0.5 rule.

## Co-registration

Phase correlation on stable terrain against 2023-09-11 (`outputs/statistics/coregistration.csv`).
Two scenes exceeded 0.25 px and were Fourier-shifted: 2017-08-23 (0.67 px → residual 0.11 px)
and 2019-09-07 (0.36 px → 0.14 px). The maximum residual per year is ≤ 1.6 m (≤ 0.16 px),
within the < 0.5 px target. One scene (2023-08-27) had too little clear stable terrain to
measure.

## Avoiding circular validation

* No Chhota Shigri pixel label from the model is used to evaluate the same model.
* M2 was trained only on even-year scenes and selected on HMA val. The analysis epochs are odd
  years.
* Benchmark test tiles are never used for selection.
* The analysis-window revision (15 Aug–15 Sep) was made after diagnostics and is reported with
  the full-window sensitivity run.
