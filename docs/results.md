# Results

All numbers are produced by the pipeline (sources in brackets) and refer to the final run of
2026-10-03, after the minimum-observation fix (docs/experiments.md, incident 9). Areas are in km²,
inside the fixed analysis domain. The primary model, **M2_cs_selftrain**, was chosen by a rule
fixed independently of the Chhota Shigri results: the best held-out HMA test IoU.

## 1. Benchmark (held-out HMA tiles) — transfer learning works

[experiments/benchmark_eval_test_summary.json, *_pairwise.json]

| Model | Pooled IoU | Precision | Recall | Boundary F1 | ΔIoU vs M0 [95 % CI] |
|---|---|---|---|---|---|
| M0 global | 0.678 | 0.898 | 0.735 | 0.266 | — |
| M1 official | 0.748 | 0.905 | 0.811 | 0.321 | +0.070 [+0.040, +0.107] |
| M1-local | 0.758 | 0.886 | 0.841 | 0.335 | +0.081 [+0.047, +0.125] |
| **M2** | **0.779** | 0.882 | 0.870 | **0.381** | +0.101 [+0.064, +0.148] |

M1-local vs M1-official: +0.011 [−0.001, +0.024], not significant. M2 vs M1-official: +0.031
[+0.019, +0.046]. The M2 gain is partly from extra HMA replay; M2 shows no forgetting.

## 2. Input ablation (22 SAR-bearing HMA test tiles, equal budgets)

[experiments/benchmark_eval_test_sar_summary.json, *_pairwise.json]

| Inputs | IoU | Recall | Boundary F1 |
|---|---|---|---|
| optical | 0.676 | 0.729 | 0.327 |
| optical + DEM | **0.824** | 0.889 | 0.383 |
| optical + DEM + SAR | 0.808 | 0.874 | 0.350 |

* DEM: **+0.149 [+0.078, +0.241]**.
* SAR: −0.016 [−0.029, −0.008] (short budget, randomly initialised SAR branch).

## 3. Chhota Shigri glacier area (multi-date ensembles, 15 Aug–15 Sep)

[outputs/statistics/summary_area_by_year.csv, M2_cs_selftrain_area.csv]

| Year | Scenes | M0 | M1-off | M1-local | **M2** (± σ) | M3 | Cross-model range |
|---|---|---|---|---|---|---|---|
| 2017 | 2 | 13.19 | 14.16 | 14.42 | **15.00 ± 1.44** | 14.38 | 1.80 |
| 2019 | 2 | 14.47 | 14.81 | 15.43 | **15.89 ± 1.41** | 16.05 | 1.59 |
| 2021 | 2 | 14.70 | 14.78 | 15.24 | **15.38 ± 1.05** | 15.72 | 1.03 |
| 2023 | 6 | 14.44 | 14.46 | 15.26 | **15.03 ± 1.04** | 15.53 | 1.09 |
| 2025 | 2 | 13.42 | 13.79 | 14.10 | **14.30 ± 0.89** | 13.96 | 0.88 |

* **M2 σ components for 2023:** delineation buffer 0.41, calibrated-threshold sensitivity 0.88
  (systematic), surface conditions between scenes 0.38.
* **Published values:** 15.7 km² (c. 2003, Wagnon et al. 2007) and 15.5 km² (2002–2019, Mandal
  et al. 2020). The site-adapted models (M2, M3, M1-local: 14.3–16.1 km²) agree with them better
  than M0 does.

**Area change: not statistically detectable for 2017–2025.**

| Model | WLS trend (km² a⁻¹) | 95 % CI |
|---|---|---|
| M0 | −0.089 | [−0.456, +0.277] |
| M1-official | −0.180 | [−0.384, +0.024] |
| M1-local | −0.024 | [−0.317, +0.269] |
| M2 | −0.159 | [−0.402, +0.084] |
| M3 | −0.114 | [−0.611, +0.382] |

* Every CI includes zero.
* **M2, 2021 → 2025:** −1.08 ± 0.73 km², not significant at 2σ.
* The long-term published loss (1.48 ± 0.57 km² over 1962–2012, ≈ 0.03 km² a⁻¹) implies about
  0.25 km² over 8 years, well below the per-epoch uncertainty. The apparent 2017 → 2019 increase
  seen in all models comes from the weak 2017 epoch (2 scenes, Sentinel-2A era, 3.4 km² of the
  domain observed only once), not from glacier growth.

## 4. Terminus retreat — detected consistently

[outputs/statistics/summary_terminus_periods.csv, <model>_terminus.csv]

| Period | Box method, median [min, max] over 5 models | Centerline, median [min, max] | All models retreat? |
|---|---|---|---|
| 2021–2023 | 5 m [−14, 12] | 0 m [0, 40] | no (within ±14 m noise) |
| 2023–2025 | 80 m [67, 148] | 100 m [70, 110] | yes |
| **2021–2025** | **80 m [73, 160] ≈ 20 m a⁻¹** | **110 m [70, 130] ≈ 28 m a⁻¹** | **yes** |
| 2019–2025 | 190 m [33, 326] | 230 m [20, 350] | yes |
| 2017–2025 | 127 m [17, 498] | 150 m [10, 460] | yes (weak baseline) |

* Measurement σ per epoch is ≈ 10–14 m (pixel plus co-registration).
* Terminus *position* differs between models by up to about 800 m on the debris-covered snout,
  so only *changes* are interpreted. The box is 2 km × 159 m (half the tongue width).
* **M2 alone, 2021 → 2025:** 110 m (centerline), 74 m (box).
* For context, the literature reports about 800 m in 1988–2003 (≈ 53 m a⁻¹) and 800–850 m in
  1972–2009.
* **Caveat:** the 2025 scenes are both August, while 2023 includes September.

## 5. Year-of-loss (M2)

[outputs/change_maps/M2_cs_selftrain_year_of_loss.tif, M2_cs_selftrain_year_of_loss_km2.json]

| Class | km² |
|---|---|
| glacier in all epochs | 13.78 |
| lost by 2019 / 2021 / 2023 / 2025 (persistent) | 0.27 / 0.11 / 0.14 / 0.43 (Σ 0.94) |
| unstable (lost and re-appeared) | 1.74 |
| gained (non-glacier in 2017, glacier from Y) | 0.24 |

The unstable class is larger than the persistent loss, a direct measure of surface-condition
flicker on the margins. Every model has 1.4–2.1 km² of unstable pixels.

## 6. Spatial pattern of loss (M2, baseline 2017 glacier)

[outputs/statistics/M2_cs_selftrain_loss_by_*.csv, *_susceptibility_model.json, figures]

| Factor | Pattern (loss fraction; enrichment = share of loss ÷ share of glacier) |
|---|---|
| **Distance to terminus** | 0–250 m: 42 % (6.7×); 250 m–1 km: 22 % (3.4–3.5×); 1–2 km: 13 % (2.0×); > 2 km: ≈ 6 % (0.9–1.0×) |
| **Elevation** | 4300–4500 m (tongue): 28–33 % (4.4–5.1×). Minimum at 4900–5200 m (2–3 %). Moderate again at 5200–5700 m (7–12 %, 1.1–1.8×), on steep accumulation-basin margins. The 5700–5800 m band (55 %, 0.06 km²) is a gap-fill artefact in 2017. |
| **Slope** | > 40°: 36 % (5.6×); 30–40°: 14 % (2.2×); 0–5°: 1.4 % (0.2×) |
| **Aspect** | N 17 % (2.7×), NE 12 % (1.9×), S/SW/W 4–5 % (0.7×). Confounded: the tongue faces north. |

* **Getis-Ord Gi\* hotspots** (100 m cells, z > 1.96; figure `M2_cs_selftrain_hotspots.png`):
  significant clusters at (i) the snout and lowest ~1 km of the tongue, (ii) the tip of the
  north-western arm, and (iii) the margins of the eastern tributary basin. Cluster (iii) partly
  coincides with unstable (snow-affected) pixels. The interior tongue and the main accumulation
  basin are cold spots.
* The susceptibility map is a thin high-probability ring along the margins, which reflects the
  dominant distance-to-margin term.
* **Retreat susceptibility** (logistic, standardised β):
  * distance to margin −6.94 (near-tautological),
  * slope +0.47,
  * elevation −0.40,
  * cos(aspect) +0.29 (north-facing),
  * distance to terminus −0.25,
  * sin(aspect) +0.14.
  * Spatial-block CV AUC 0.94. It is descriptive only. Pixel-level CIs are overconfident because
    of spatial autocorrelation.

**Interpretation:** loss is dominated by frontal retreat of the low, north-facing, partly
debris-covered tongue and by thinning or retreat of steep lateral margins. The interior
accumulation area at 4900–5200 m is the most stable.

## 7. Site validation summary

[outputs/statistics/site_validation.csv/.json]

| Model | IoU vs RGI 2002 | Upper-zone recall | Boundary distance | Consecutive IoU | Unstable km² |
|---|---|---|---|---|---|
| optical only | 0.749 | 0.814 | 83 m | 0.882 | 1.71 |
| M0 | 0.799 | 0.840 | 71 m | 0.900 | 1.74 |
| M1-official | 0.816 | 0.833 | 61 m | 0.917 | 1.54 |
| M1-local | 0.836 | 0.852 | 43 m | 0.924 | 1.43 |
| M2 | 0.838 | 0.860 | 48 m | 0.919 | 1.74 |
| M3 | 0.837 | 0.885 | 46 m | 0.903 | 2.10 |
| M1-off, full window | 0.778 | 0.772 | 68 m | 0.882 | 1.57 |

No model has a significant area gain between consecutive epochs. Most show one apparent terminus
"advance" (2017 → 2019), which comes from the weak 2017 epoch.

## 8. Answers to the research questions

| Question | Answer (evidence) |
|---|---|
| Transfer + fine-tuning gives automatic delineation without tracing? | **Yes.** All outlines are automatic. HMA IoU 0.75–0.78. Inventory agreement 0.84 IoU. |
| Q1 Does HMA fine-tuning help? | **Yes**, +0.07–0.08 IoU, significant. |
| Q2 Does site adaptation help? | **Partly.** No forgetting (HMA IoU 0.779), better agreement with inventory and published area than M0/M1-official. A site-specific accuracy gain can't be proven without independent contemporaneous labels. |
| Q3 Does the DEM help? | **Strongly**: +0.149 IoU. |
| Q4 Does SAR help? | **Not at this budget** (−0.016). Exploratory on Chhota Shigri, with approximate σ⁰. |
| Q5 Does confidence-aware pseudo-labelling work? | **Yes, after calibration.** Raw thresholds yield 0 labels; calibrated ones yield 1.17 M labelled pixels with changing margins excluded. |
| Q6 Can the automated boundaries quantify retreat? | **Terminus: yes** for 2021–2025 (70–160 m, all models). **Area: not for 2017–2025** at the achievable ±0.9–1.4 km² per epoch. |
