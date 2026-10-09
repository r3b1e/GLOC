# Experiments

All runs are reproducible from the commands in `run_pipeline.sh`. Their parameters are recorded in
`configs/*.yaml` and in each run's `experiments/<M>/<run>/run_config.json`. Metrics are in
`experiments/benchmark_eval_*` and `outputs/statistics/`. See `docs/results.md` for the numbers.

## Hardware and software

Apple M4 (10-core CPU, 10-core GPU), 16 GB unified memory, macOS 26.2. Python 3.11, TensorFlow
2.15.1, tensorflow-metal 1.1.0. Full lock file: `requirements.lock.txt`. GlaViTU commit
`8c4eead670ef64764f8db2708bba3c8bbbfc67bb`.

* **Inference:** CPU. tensorflow-metal intermittently raises SIGBUS inside `model.predict`.
* **Training:** Metal GPU, 4.5× faster than CPU (2.4 s vs 10.8 s per step at batch 4, identical
  losses to 6 decimals). Training is wrapped in `train_with_retry.sh`, which resumes from the
  per-epoch "last" checkpoint after a Metal graph error. M1-local had one such error at epoch 3 and
  was resumed from epoch 2; the cosine schedule and the best-IoU threshold were restored.

## Run registry

| ID | Init | Train data | Features | Budget | Selection | Status |
|---|---|---|---|---|---|---|
| M0_global | authors | global benchmark | opt+DEM | authors' | — | used as-is |
| M1_hma_official | authors (global → HMA) | HMA train | opt+DEM | authors' (80 ep, bs 8) | HMA val | used as-is |
| M1_hma_regional_official | authors (HMA only) | HMA train | opt+DEM | authors' | HMA val | used as-is (single-scene runs only) |
| M1_hma_local | M0 weights | 165 HMA train tiles | opt+DEM | 8 ep × 250 steps, bs 4 (≈1.7 % of official steps) | best val IoU (epoch 6) | done |
| M2_cs_selftrain | M1_hma_official | 15 CS pseudo-label scenes (×2) + 40 HMA replay tiles | opt+DEM | 4 ep × 200 steps, bs 4, lr 2e-5 | HMA val (best epoch 4, 0.693) | done |
| M1_sarsubset_nosar | M0 weights | 66 SAR-bearing HMA train tiles | opt+DEM | 4 ep × 250, bs 4 | HMA val SAR tiles (0.689) | done |
| M3_hma_sar | M0 weights + new SAR branch | same 66 tiles | opt+DEM+VV (asc, desc) | 4 ep × 250, bs 4 | HMA val SAR tiles (0.666) | done |
| M1_sarsubset_optonly | M0 weights, DEM branch dropped | same 66 tiles | optical | 4 ep × 250, bs 4 | HMA val SAR tiles (0.551) | done |

## Ablations

| Question | Comparison | Evaluation data |
|---|---|---|
| Exp. 1: global pretrained | M0 | HMA test (57 tiles); CS series |
| Exp. 2: HMA fine-tuning | M0 vs M1_hma_official vs M1_hma_local | HMA test; CS series |
| Exp. 3: CS domain adaptation | M1_hma_official vs M2 | HMA test (forgetting); CS temporal consistency and inventory agreement |
| Exp. 4: optical only vs optical + DEM | M1_sarsubset_optonly vs M1_sarsubset_nosar (equal budgets, same tiles; weights transferred by input name) | HMA test SAR tiles (22); CS series |
| Exp. 5: optical + DEM + SAR | M1_sarsubset_nosar vs M3_hma_sar (equal budgets, same tiles) | HMA test SAR tiles (22); CS series (approximate SAR inputs) |
| Sensitivity: ensemble window | core 15 Aug–15 Sep vs full 15 Aug–15 Oct | CS series (M1_hma_official) |

## Seeds and splits

Seed 42 (numpy, Python, TensorFlow). The official benchmark train/val/test split is unchanged. No
HMA tile lies within 0.05° of Chhota Shigri (checked for all 277 tiles), and the nearest tiles are
about 0.6° away, so M0 and M1 never saw the study site. Chhota Shigri training uses only the
even-year scenes. Every analysis epoch (2017, 2019, 2021, 2023, 2025) is temporally held out.

## Incidents and design revisions (chronological, all automatic re-runs)

1. **Data source.** Planetary Computer has no L1C. L1C TOA is taken from the Google public bucket
   so the inputs match the model's training domain.
2. **Cloud mask in composites.** The first composite used the raw s2cloudless probability; the
   scoring used the averaged mask. Both now use the averaged + dilated mask.
3. **Benchmark extraction.** Shared shard HDF5 files were corrupted when the workers were stopped.
   This was replaced by atomic per-tile files, and the test tiles were re-downloaded.
4. **Surface-condition artefact (user decision: Option A).** Single-scene areas differ by about 1 km²
   between good scenes, because snow-dusted vs exposed debris margins change the result. This led to
   the multi-date ensemble.
5. **Late-season scenes.** Scenes after 15 Sep (autumn snowfall, low sun) map 1.5–4.6 km² too little
   ice. The primary window became 15 Aug–15 Sep, with the full window kept as a sensitivity run. This
   decision was made after looking at the data and is documented as such.
6. **Robust ensemble.** Whole-scene outliers are rejected with |A − median| > max(3 × 1.4826·MAD,
   0.5 km²).
7. **Uncalibrated softmax.** GlaViTU's raw P(glacier) stays in about 0.24–0.89 (label smoothing and
   focal loss), so the 0.95/0.05 pseudo-label rule produced zero labels. An isotonic calibration was
   fitted on HMA val tiles: held-out ECE went from 0.239 to 0.011. The calibrated thresholds
   correspond to raw ≈ 0.70 / 0.25–0.30.
8. **Sentinel-1.** Only RTC γ⁰ is available without SNAP. It was converted to approximate σ⁰ with
   γ⁰·cos θ_inc (θ from the GRD annotation geolocation grid, about 34–46°). The normalised value
   distributions match the HMA benchmark within about 0.05.
9. **Minimum-observation bias (found during results assembly).** With `min_valid_obs = 2`, pixels
   cloudy in either scene of 2-scene years were set to non-glacier: −0.73 km² of glacier-like
   pixels in 2017, −0.29 km² in 2019, and −0.13 km² in 2021 (M2). Changed to `min_valid_obs = 1`.
   Pixels with no clear core-window observation are now gap-filled from the same year's
   out-of-window scenes (no cross-epoch leakage), and the rest are reported as unobserved
   (≤ 0.52 km², all outside the RGI outline). All ensembles, analyses and validations were re-run.
   Residual side effect: about 0.06 km² of steep 5700–5800 m slopes enters the 2017 baseline from
   snowier out-of-window scenes and later appears as "loss".
10. **Box method.** The first version had an inverted sign and a 400 m width, so it measured
    lateral-margin variability. It now uses half the 2002 tongue width (159 m) and 2 km length.
11. **Change uncertainty.** The calibrated threshold term is systematic (same model, every year), so
    it is propagated by recomputing ΔA at the two thresholds rather than in quadrature. The area
    trend uses weighted least squares with the independent σ.
12. **Process interruptions** at the user's request (2026-09-30, 10-02, 10-03). All stages resume
    from per-scene caches, per-tile files and per-epoch checkpoints, so no result depends on an
    interrupted run.
