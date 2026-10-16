# Chhota Shigri Glacier Mapping with GlaViTU — Technical Report

**Architecture · Error sources · Preprocessing · Analysis of all years (2017 baseline)**

This report explains how the project maps Chhota Shigri Glacier (Himachal Pradesh, India) from
satellite data, and what the results show for every year from 2017 to 2025. It is written to be read
from start to end, but each section also stands on its own.

Every number in this report comes from the project's code, configuration, logs or output files.
The source of each number is given in the text or in the [file reference](#a2-file-reference).

| Item | Value |
|---|---|
| Glacier | Chhota Shigri, Chandra valley, Lahaul (`RGI60-14.15990` / `RGI2000-v7.0-G-14-30579`) |
| Satellite data | Sentinel-2 L1C (model input), Copernicus DEM GLO-30, Sentinel-1 RTC (M3 only) |
| Model | GlaViTU (Maslov et al., 2024), about 10 M parameters |
| Analysis grid | UTM 43N, 10 m, 1434 × 1254 px; analysis domain 47.9 km² |
| Years | 2017–2025; odd years = analysis epochs, even years = M2 training years |
| Primary model | M2 (best held-out HMA IoU: 0.779) |

---

## Contents

- [How to read this report](#how-to-read)
- [Pipeline at a glance](#pipeline)
- [1. GlaViTU architecture](#1-architecture)
  - [1.1 Source repository](#1-1-repository)
  - [1.2 The network in one picture](#1-2-overview)
  - [1.3 Stage 1 — Fusion block](#1-3-fusion)
  - [1.4 Stage 2 — SETR transformer](#1-4-setr)
  - [1.5 Stage 3a — Deep-supervision head](#1-5-deep-supervision)
  - [1.6 Stage 3 — Residual U-Net](#1-6-unet)
  - [1.7 Stage 4 — Output layer](#1-7-output)
  - [1.8 Layer summary](#1-8-summary)
  - [1.9 Why this design suits glaciers](#1-9-why)
  - [1.10 Model variants M0–M3](#1-10-variants)
- [2. Error sources and how they are handled](#2-errors)
  - [2.1 Principle: the images are never edited](#2-1-principle)
  - [2.2 The order of the defences](#2-2-order)
  - [2.3 Atmospheric errors](#2-3-atmospheric)
  - [2.4 Surface conditions](#2-4-surface)
  - [2.5 Illumination and timing](#2-5-illumination)
  - [2.6 Geometric and radiometric errors](#2-6-geometric)
  - [2.7 Temporal aggregation](#2-7-temporal)
  - [2.8 Spatial context and noise](#2-8-spatial)
  - [2.9 SAR (model M3 only)](#2-9-sar)
  - [2.10 Summary table](#2-10-summary)
- [3. Preprocessing (everything before training)](#3-preprocessing)
  - [3.1 Two data streams](#3-1-streams)
  - [3.2 A1 — Study area, domain and grid](#3-2-a1)
  - [3.3 A2 — Scene discovery](#3-3-a2)
  - [3.4 A3 — Scene screening](#3-4-a3)
  - [3.5 A4 — L1C download, band selection, radiometry](#3-5-a4)
  - [3.6 A5 — Cloud probability](#3-6-a5)
  - [3.7 A6 — DEM: elevation and slope](#3-7-a6)
  - [3.8 A7 — Co-registration](#3-8-a7)
  - [3.9 A8 — Normalisation](#3-9-a8)
  - [3.10 A9 — Calibration and pseudo-labels](#3-10-a9)
  - [3.11 B — HMA benchmark tiles](#3-11-b)
  - [3.12 Sentinel-1 (special case)](#3-12-s1)
  - [3.13 Training files for each model](#3-13-training-files)
  - [3.14 Error sources handled before training](#3-14-summary)
- [4. Analysis of all years (baseline 2017)](#4-analysis)
  - [4.1 How to read the tables](#4-1-read)
  - [4.2 Data and quality of each year](#4-2-data)
  - [4.3 M2 area and uncertainty](#4-3-area)
  - [4.4 M2 change relative to 2017](#4-4-change)
  - [4.5 All models: area and change](#4-5-models)
  - [4.6 Terminus retreat relative to 2017](#4-6-terminus)
  - [4.7 Agreement with the RGI7 outline](#4-7-agreement)
  - [4.8 Area trends](#4-8-trends)
  - [4.9 Key findings](#4-9-findings)
  - [4.10 Limitations of the year-by-year analysis](#4-10-limitations)
- [5. Known issues and corrections](#5-issues)
- [Appendix](#appendix)
  - [A.1 Glossary](#a1-glossary)
  - [A.2 File reference](#a2-file-reference)

---

<a id="how-to-read"></a>
## How to read this report

- **Sections 1–3** explain *how* the system works. **Section 4** shows *what* it found.
- Each section ends with a short summary table. If you are short of time, read these tables and
  [4.9 Key findings](#4-9-findings).
- Technical terms are explained in the [glossary](#a1-glossary).
- Symbols used in tables:
  - **†** = a training year of model M2. These values are **not independent** (see [4.1](#4-1-read)).
  - **‡** = scenes that were **not co-registered**, so the uncertainty is underestimated.
  - ● addressed explicitly · ◐ partly or indirectly addressed · ○ not addressed.

---

<a id="pipeline"></a>
## Pipeline at a glance

The whole workflow runs from one script, [`run_pipeline.sh`](../run_pipeline.sh). No step uses a
hand-drawn line.

```
 0. Get code and weights        third_party/glavitu (fixed commit), pretrained weights + SHA-256 check
 1. Get study area and data     inventory → DEM → Sentinel-2 → co-registration → Sentinel-1
 2. Get HMA benchmark tiles     expert-labelled training data, read by HTTP byte ranges
 3. Train models                M1 (HMA) → calibration → pseudo-labels → M2 (site) ; M3 (SAR)
 4. Test models                 57 held-out HMA tiles, paired bootstrap confidence intervals
 5. Make yearly outlines        multi-date ensemble per year → post-processing → polygon
 6. Measure change              area, terminus, year-of-loss, spatial patterns
 7. Validate site results       agreement with RGI7, published values, temporal consistency
```

Two groups of years are kept apart on purpose:

| Group | Years | Used for |
|---|---|---|
| **Odd years** | 2017, 2019, 2021, 2023, 2025 | Analysis only. No model is trained on them. |
| **Even years** | 2018, 2020, 2022, 2024 | Making pseudo-labels to train M2 only. |

This split prevents M2 from learning the exact images in which change is measured.

---

<a id="1-architecture"></a>
## 1. GlaViTU architecture

GlaViTU means **Gla**cier **Vi**sion **T**ransformer **U**-Net. It is a supervised deep-learning
model for **semantic segmentation**: it gives every pixel a class (glacier or not glacier).
All models in this project (M0, M1, M2, M3) use this same network. They differ only in their
weights, training labels and input channels (see [1.10](#1-10-variants)).

<a id="1-1-repository"></a>
### 1.1 Source repository

| Item | Value |
|---|---|
| Repository | <https://github.com/konstantin-a-maslov/towards_global_glacier_mapping> |
| Local folder | `third_party/glavitu/` |
| Fixed commit | `8c4eead670ef64764f8db2708bba3c8bbbfc67bb` (3 Jan 2025) |
| Paper | Maslov, Persello, Schellenberger & Stein (2024), *Globally scalable glacier mapping by deep learning matches expert delineation accuracy*, Nature Communications, doi:10.1038/s41467-024-54956-x |
| Licence | GNU GPL v2 |
| Weights | Downloaded from the authors' Google Drive into `models/pretrained/`, checked with SHA-256 |

The commit is fixed so that later changes by the authors cannot change our results.

From this repository the project uses:
- the model code (`models/mapping/glavitu.py`, `setr.py`, `resunet.py`, and the `layers/` folder);
- the data loader, samplers, augmentations and focal loss (reused by
  [`src/training/finetune.py`](../src/training/finetune.py));
- the normalisation statistics `utils/dataset_stats.pickle`;
- the sliding-window prediction logic of `deploy.py`, copied into
  [`src/inference/glavitu.py`](../src/inference/glavitu.py) because `deploy.py` opens the training
  file when it is imported.

> **Licence note.** GPL v2 is a copyleft licence. Code that includes or modifies GlaViTU code and
> is distributed must also be under a GPL-compatible licence.

<a id="1-2-overview"></a>
### 1.2 The network in one picture

Shapes are for one 384 × 384 px patch with the standard 8-channel input.

```
 optical (384×384×6) ──┐
                       ├─► [1] FUSION BLOCK ──► fused (384×384×64)
 dem     (384×384×2) ──┘                            │
                                                    ├──────────────┐ skip
                                                    ▼              │
                                     [2] SETR (transformer)        │
                                                    ▼              │
                                          encoded1 = SETR + fused ◄┘
                                           │        │
            [3a] deep-supervision head ◄───┘        ├──────────────┐ skip
              (training only)                       ▼              │
                                           [3] Residual U-Net      │
                                                    ▼              │
                                     encoded = U-Net + encoded1 ◄──┘
                                                    ▼
                                      [4] OUTPUT Dense(2, softmax) → 384×384×2
```

**The idea in one sentence:** the transformer looks at the whole patch to understand large shapes,
and the U-Net then makes the edges sharp at the full 10 m resolution.

<a id="1-3-fusion"></a>
### 1.3 Stage 1 — Fusion block

Code: `FusionBlock(64)` in `third_party/glavitu/layers/general/layers.py`.

Each input group (`optical`, `dem`, and for M3 `co_pol_sar`) has **its own branch**:

| Step | Layer | Output (optical) | Purpose |
|---|---|---|---|
| 1 | Conv 3×3, 64 filters | 384×384×64 | Finds local patterns: edges, texture, colour combinations |
| 2 | BatchNorm + LeakyReLU | 384×384×64 | Stabilises values, adds non-linearity |
| 3 | Conv 3×3, 64 filters | 384×384×64 | Combines patterns into more complex ones |
| 4 | BatchNorm + LeakyReLU | 384×384×64 | As step 2 |
| 5 | Global average pooling | 1×1×64 | One summary value per channel ("squeeze") |
| 6 | Dense 64 + LeakyReLU | 1×1×64 | Learns which channels matter for this patch |
| 7 | Dense 64 + sigmoid | 1×1×64 | A weight from 0 to 1 for each channel ("excitation") |
| 8 | Multiply step 4 × step 7 | 384×384×64 | Strengthens useful channels, weakens others |

Steps 5–8 form a **squeeze-and-excitation (SE)** block. In this implementation the SE hidden layer
is not reduced (64 → 64).

The branches are then **added**: `fused = optical_branch + dem_branch (+ sar_branch)` → 384×384×64.

**Why separate branches?** Reflectance, elevation and radar are different kinds of data. Each gets
its own feature extractor before they are combined. This also made M3 easy to build: a SAR branch
was added and all other layers were kept.

<a id="1-4-setr"></a>
### 1.4 Stage 2 — SETR transformer

Code: `SETR(...)` in `models/mapping/setr.py` and `ViTEncoder` in `models/misc/vit.py`.
SETR means **SE**gmentation **TR**ansformer. GlaViTU uses a small version:

| Setting | Value |
|---|---|
| Patch (token) size | 16 × 16 px (160 m × 160 m) |
| Embedding size | 64 |
| MLP size | 256 |
| Transformer blocks | 12 |
| Attention heads | 4 (16 dimensions each) |

**a) Turn the image into tokens**

| Step | Layer | Output | Purpose |
|---|---|---|---|
| 1 | Patch extraction 16×16 | 576 × 16,384 | Cuts the patch into a 24 × 24 grid = 576 squares |
| 2 | Patch embedding (Dense 64) | 576 × 64 | Compresses each square into a 64-number token |
| 3 | Learned positional embedding | 576 × 64 | Tells the model where each square is |

**b) Transformer block, repeated 12 times** (pre-norm design, `layers/transformers/layers.py`)

```
input ─┬─► LayerNorm ─► Multi-Head Self-Attention ─► (+) ─┬─► LayerNorm ─► Dense 256 (GELU) ─► Dropout ─► Dense 64 (GELU) ─► Dropout ─► (+) ─► output
       └──────────────────────────────────────────────────┘└────────────────────────────────────────────────────────────────────────┘
```

| Part | Purpose |
|---|---|
| LayerNorm | Normalises each token for stable training |
| **Self-attention (4 heads)** | Every square compares itself with every other square in the patch. A dark debris square can "see" that it connects to a large ice body upstream. |
| MLP (64 → 256 → 64) | Processes each token after attention has mixed in information. In this code the second Dense layer also uses GELU. |
| Residual connections (+) | Let gradients pass through 12 blocks without fading |
| Dropout 0.1 | Reduces overfitting |

After 12 blocks, the tokens are reshaped to a **24 × 24 × 64** grid.

**c) PUP decoder** (progressive upsampling) back to full size:

| Step | Layer | Output |
|---|---|---|
| 1 | Conv 3×3 + BN | 24×24×64 |
| 2 | Upsample ×2 → Conv 3×3 + BN | 48×48×64 |
| 3 | Upsample ×2 → Conv 3×3 + BN | 96×96×64 |
| 4 | Upsample ×2 → Conv 3×3 + BN | 192×192×64 |
| 5 | Conv 1×1 (linear) | 192×192×64 |
| 6 | Upsample ×2 | 384×384×64 |

**d) Skip connection:** `encoded1 = SETR output + fused`. The SETR output is blurry because it was
built from 160 m squares. Adding the fusion features back restores the 10 m detail.

<a id="1-5-deep-supervision"></a>
### 1.5 Stage 3a — Deep-supervision head

`Dense(2, softmax)` on `encoded1` gives a first, rough glacier map (384×384×2).

- It is used **only in training**. The focal loss is applied with weight 0.5 here and 1.0 on the
  final output.
- At prediction time it is removed. The code builds a 2-output `train_model` and a 1-output
  `test_model`.
- **Purpose:** it forces the transformer stage to learn a useful map on its own, and it gives
  gradients a short path back to early layers.

<a id="1-6-unet"></a>
### 1.6 Stage 3 — Residual U-Net

Code: `ResUNet(n_steps=3, start_n_filters=64)` in `models/mapping/resunet.py`.

**Building block** used at every level:

```
x ─┬─► Conv3×3 ─► BN ─► LeakyReLU ─► Conv3×3 ─► BN ─► LeakyReLU ─► (SpatialDropout) ─► (+) ─► y
   └─► Dense 1×1 (projection, no bias) ─────────────────────────────────────────────────┘
```

**Encoder** (image gets smaller, channels increase):

| Level | Operation | Output | Pixel size |
|---|---|---|---|
| 1 | Residual block 64 → skip 1 | 384×384×64 | 10 m |
| 2 | MaxPool, residual block 128 → skip 2 | 192×192×128 | 20 m |
| 3 | MaxPool, residual block 256 → skip 3 | 96×96×256 | 40 m |
| Bottleneck | MaxPool, residual block 512 | 48×48×512 | 80 m |

**Decoder** (image gets larger, channels decrease):

| Level | Operation | Output |
|---|---|---|
| 3 | Up-conv to 256, concatenate skip 3, residual block | 96×96×256 |
| 2 | Up-conv to 128, concatenate skip 2, residual block | 192×192×128 |
| 1 | Up-conv to 64, concatenate skip 1, residual block | 384×384×64 |
| Out | Dense 64 (linear) | 384×384×64 |

The **skip connections** give the decoder the sharp, high-resolution features of the encoder. This
is why U-Nets produce clean boundaries.

A final skip adds the transformer result: `encoded = encoded1 + U-Net output`. The U-Net therefore
learns only a **correction** to the transformer's map.

<a id="1-7-output"></a>
### 1.7 Stage 4 — Output layer

`Dense(2, softmax)` applied to every pixel gives **384×384×2**: channel 0 = P(not glacier),
channel 1 = P(glacier). The two values add up to 1.

For a full scene, 384 px patches are predicted with a stride of 192 px and blended with Gaussian
weights.

<a id="1-8-summary"></a>
### 1.8 Layer summary

| # | Stage | Main layers | Output | Job |
|---|---|---|---|---|
| 1 | Fusion | 2 × (Conv + BN + LeakyReLU) and SE gate per group; sum | 384×384×64 | Turn each data source into features and combine them |
| 2 | SETR encoder | Patch embedding, positional embedding, 12 transformer blocks | 24×24×64 | Global context over the whole 3.84 km patch |
| 2 | SETR decoder | 4 × (Conv + BN + upsample) | 384×384×64 | Back to full size |
| — | Skip | Add fusion features | 384×384×64 | Restore fine detail |
| 3a | Deep supervision | Dense(2, softmax) | 384×384×2 | Training only |
| 3 | Residual U-Net | 3 levels down, bottleneck, 3 levels up | 384×384×64 | Sharp 10 m boundaries |
| — | Skip | Add transformer result | 384×384×64 | U-Net learns a correction |
| 4 | Output | Dense(2, softmax) | 384×384×2 | P(glacier) per pixel |

Size: about 10 million parameters (ViT-Base alone has about 86 million).

<a id="1-9-why"></a>
### 1.9 Why this design suits glaciers

| Problem | Part of the network that addresses it |
|---|---|
| Debris-covered ice looks like rock up close | Transformer attention sees the connection to the main ice body |
| Optical, DEM and SAR are different data types | Separate fusion branches, weighted by SE |
| Outlines must be accurate to about 10 m | U-Net with skip connections at full resolution |
| Deep networks are hard to train | Residual connections and deep supervision |
| Some inputs exist only in some regions | Fusion by addition: branches can be added or removed |

<a id="1-10-variants"></a>
### 1.10 Model variants M0–M3

| Model | Learning type | Training labels | Starts from | Inputs | HMA test IoU |
|---|---|---|---|---|---|
| **M0** global | Supervised (by the authors) | Expert outlines, worldwide | — | 6 optical + 2 DEM | 0.678 |
| **M1** HMA official | Supervised fine-tuning (by the authors) | Expert outlines, HMA | M0 | 6 optical + 2 DEM | 0.748 |
| **M1-local** | Supervised fine-tuning (this project) | Expert outlines, 165 HMA tiles | M0 | 6 optical + 2 DEM | 0.758 |
| **M2** Chhota Shigri | **Semi-supervised (self-training)** | Pseudo-labels + HMA replay | M1 | 6 optical + 2 DEM | **0.779** |
| **M3** SAR | Supervised fine-tuning | Expert outlines, 66 HMA tiles with SAR | M0 + new SAR branch | 6 optical + 2 DEM + 2 SAR | 0.808 † |
| Optical-only (ablation) | Supervised fine-tuning | Same 66 tiles | M0 | 6 optical | 0.676 † |
| No-SAR baseline (ablation) | Supervised fine-tuning | Same 66 tiles | M0 | 6 optical + 2 DEM | 0.824 † |

† These three are measured on the smaller 22-tile SAR test set and cannot be compared directly with
the 57-tile results. On that set the DEM adds **+0.149 IoU** and SAR gives **−0.016 IoU**.

**Input channels**

| Group | Channels | Why it helps |
|---|---|---|
| `optical` | B2 blue, B3 green, B4 red, B8 NIR (10 m); B11 SWIR-1, B12 SWIR-2 (20 m → 10 m) | Ice and snow are bright in visible light and dark in SWIR; clouds are bright in SWIR |
| `dem` | Elevation (m), slope (°) | Glaciers lie at high altitude on gentler slopes; key for debris-covered ice |
| `co_pol_sar` (M3) | VV ascending, VV descending | Radar sees through cloud and responds to wet snow and roughness |

---

<a id="2-errors"></a>
## 2. Error sources and how they are handled

<a id="2-1-principle"></a>
### 2.1 Principle: the images are never edited

The pipeline **does not remove** snow, rock or shadow from the satellite images. Every Sentinel-2
image goes into GlaViTU unchanged. Obstacles are handled in two ways:

1. **Rejecting** data that cannot be trusted (cloudy images, fresh snowfall, cloudy pixels).
2. **Classifying**: the model, trained on expert labels, decides for each pixel whether it is glacier.
   Rock, debris, snow and shadow are all things the model learned to separate from ice.

Afterwards, the multi-date median and the post-processing clean up the remaining errors.

> **Key message:** *We do not edit the images. We reject what cannot be trusted, let a model trained
> on expert labels classify the rest, and use many images per year so that one-day effects are
> voted out.*

<a id="2-2-order"></a>
### 2.2 The order of the defences

```
 1. Choose the season        avoid seasonal snow, low sun and monsoon cloud
 2. Screen each image        reject cloudy or freshly snowed images
 3. Fix the radiometry       same brightness scale in every year
 4. Align the images         no false change from shifted images
 5. Model prediction         GlaViTU separates glacier from rock, snow, shadow, debris
 6. Mask cloudy pixels       cloud pixels set to "no data" after prediction
 7. Multi-date median        one-day errors are voted out
 8. Reject outlier images    whole bad images removed
 9. Fill gaps                pixels never seen clearly get a same-year value
10. Post-processing          noise, neighbouring glaciers and small blobs removed
```

<a id="2-3-atmospheric"></a>
### 2.3 Atmospheric errors

**Cloud — ● addressed**
- *Per image* ([`sentinel2.py`, `select_ensemble`](../src/data/sentinel2.py)): a cheap SCL
  pre-screen rejects an image with more than 60 % cloud in the domain. The limit is loose because
  SCL often marks bright ice as cloud. Then s2cloudless (threshold 0.4, 4 px averaging, 2 px = 40 m
  dilation) rejects an image with more than 50 % cloud.
- *Per pixel* ([`ensemble.py`](../src/inference/ensemble.py)): cloudy pixels are set to NaN after
  prediction and take no part in the median.

**Cloud shadow — ◐ partly addressed**
- The final ensemble has **no per-pixel cloud-shadow mask**. The SCL shadow class is computed but
  used only in an older selection method that is no longer used.
- Partial protection: the 40 m cloud-mask dilation covers shadow next to clouds, and the median and
  outlier rejection reduce the rest.
- Gap: a shadow far from its cloud can still enter the median.

<a id="2-4-surface"></a>
### 2.4 Surface conditions

**Seasonal and fresh snow — ● addressed**
- Only images from **15 Aug – 15 Sep** are used, in every year. Most seasonal snow has melted, the
  monsoon is ending, and autumn snowfall has not started.
- An image is rejected if SCL snow covers **more than 45 % of the ground outside the glacier**. Snow
  outside the glacier means recent snowfall.
- This was tested: scenes after 15 September map **1.5–4.6 km² too little glacier**. Extending the
  window to 15 October lowers agreement with the inventory (IoU 0.816 → 0.778) and creates a false
  −2.65 km² step in 2023–2025.

**Thin snow on debris margins — ◐ reduced, not solved**
- One image shows exposed debris, the next shows a thin snow cover. This alone changes the area by
  about **1 km²**.
- The per-pixel **median over all images of a year** reduces this to 0.3–0.8 km².
- In training, pseudo-labels **ignore** these flickering pixels, so M2 does not learn a wrong answer.
- This is the **largest remaining error** and the main reason area change is not detectable.

**Debris-covered ice — ◐ the hardest case**
- Debris on ice looks like rock in optical bands.
- Defences: the DEM (shape of the tongue), transformer context, and HMA fine-tuning on debris-rich
  glaciers. The DEM adds +0.149 IoU [0.078, 0.241].
- Remaining: the absolute terminus position differs by up to about 800 m between models, so only
  terminus *change* is reported.

**Bare rock and moraine — ● addressed by the model**
- No rule removes rock. The model separates rock from ice with the SWIR bands and the DEM.
- Post-processing removes objects smaller than 100 px and fills holes smaller than 50 px.

**Surface water (proglacial lakes) — ○ not addressed**
- There is no water mask. The model must separate water from ice by itself. This matters if the work
  is framed around GLOF risk.

<a id="2-5-illumination"></a>
### 2.5 Illumination and timing

**Terrain shadow — ◐ indirect**
- There is no explicit terrain-shadow mask. The hillshade is used only for figures.
- Indirect defences: the window closes before the low-sun period, slope is a model input, and the
  median and outlier rejection reduce shadow effects.

**Seasonal timing between years — ◐ partly addressed**
- The same calendar window is used every year. However, the actual dates differ: both 2025 scenes
  are from August, while 2023 reaches 11 September. Part of the 2023–2025 change may be seasonal.

<a id="2-6-geometric"></a>
### 2.6 Geometric and radiometric errors

**Misregistration — ● addressed (odd years)**
- A shift of one pixel moves the glacier edge and looks like retreat or advance.
- Phase correlation on stable terrain corrects shifts larger than 0.25 px (details in
  [3.8](#3-8-a7)). Two scenes were corrected: 0.67 → 0.11 px and 0.36 → 0.14 px. All residuals are
  ≤ 1.6 m.
- **The even-year scenes were not co-registered** (see [5. Known issues](#5-issues)).

**Radiometric baseline change — ● addressed**
- From January 2022 (processing baseline 04.00), ESA added a −1000 DN offset to Sentinel-2 data.
- Each product's own `RADIO_ADD_OFFSET` and `QUANTIFICATION_VALUE` are read from its metadata, so
  2017–2021 and 2022–2025 are on the same scale.

<a id="2-7-temporal"></a>
### 2.7 Temporal aggregation

**Anomalous whole scenes — ● addressed**
- An image is removed if its area differs from the year's median by more than
  max(3 × 1.4826 × MAD, 0.5 km²). Example: 2022-09-06 was rejected for M1 and M2.

**Data gaps — ◐ partly addressed**
- A pixel with no clear view in the window is filled from clear images of the **same year** outside
  the window. Pixels still unseen are reported as unobserved.
- 2017 and 2019 have only 2 scenes each, and about 3 km² of the domain was seen only once.

<a id="2-8-spatial"></a>
### 2.8 Spatial context and noise

**Neighbouring glaciers — ● addressed**
- The analysis domain is the RGI7 outline + 1 km **minus 33 neighbouring glaciers**.
- Only ice bodies touching the **RGI7 core** (outline shrunk by 300 m) are kept. The core only
  identifies which ice body is Chhota Shigri; the edges come from the model.

**Classification noise — ● addressed**
- Threshold 0.5, 3 × 3 median filter, objects < 100 px removed, holes < 50 px filled, slivers
  < 4 px removed after polygonisation, and validity checks.

<a id="2-9-sar"></a>
### 2.9 SAR (model M3 only)

**Layover and radar shadow — ◐**
- Missing backscatter on steep slopes is filled with the composite median.
- SAR gave no benefit: −0.016 IoU [−0.029, −0.008] compared with optical + DEM.

<a id="2-10-summary"></a>
### 2.10 Summary table

| Error source | How it is handled | Key parameters | Remaining effect | Status |
|---|---|---|---|---|
| Cloud | Image rejection + per-pixel NaN | s2cloudless p ≥ 0.4, 40 m dilation, > 50 % → reject | Gaps filled from same year | ● |
| Cloud shadow | Indirect | Dilation, median, outlier rejection | Not quantified | ◐ |
| Seasonal / fresh snow | Window + snow screen | 15 Aug–15 Sep; > 45 % off-glacier snow → reject | Later scenes: −1.5 to −4.6 km² | ● |
| Thin snow on debris | Median; ignored in pseudo-labels | 34.7 % of pixels ignored | ~1 km² → 0.3–0.8 km² | ◐ |
| Debris-covered ice | DEM + HMA fine-tuning | Elevation, slope | Terminus position spread ~800 m | ◐ |
| Bare rock | Model + object filter | < 100 px removed, holes < 50 px | Included in HMA IoU 0.779 | ● |
| Proglacial lakes | None | — | Not quantified | ○ |
| Terrain shadow | Indirect | Window, slope input, median | Not quantified | ◐ |
| Seasonal timing | Fixed window | Same dates every year | 2025 scenes August only | ◐ |
| Misregistration | Phase correlation | > 0.25 px corrected | Residual ≤ 1.6 m (odd years) | ● |
| Radiometric offset | Harmonisation | RADIO_ADD_OFFSET | None expected | ● |
| Outlier images | MAD rejection | max(3·1.4826·MAD, 0.5 km²) | Logged | ● |
| Data gaps | Same-year gap fill | No mixing of years | ~3 km² seen once (2017, 2019) | ◐ |
| Neighbouring glaciers | Domain + core connectivity | −33 glaciers; 300 m core | Edges from the model | ● |
| Noise | Morphological filtering | 3×3 median, slivers < 4 px | In ½ px × perimeter error | ● |
| SAR layover / shadow | Median fill | — | SAR −0.016 IoU | ◐ |

---

<a id="3-preprocessing"></a>
## 3. Preprocessing (everything before training)

<a id="3-1-streams"></a>
### 3.1 Two data streams

```
STREAM A: Chhota Shigri (no labels)               STREAM B: HMA benchmark (expert labels)
───────────────────────────────────               ───────────────────────────────────────
A1 Study area, domain, grid                       B1 Download HMA tiles from NIRD
A2 Scene discovery (Planetary Computer)           B2 Keep official train / val / test split
A3 Scene screening (SCL + s2cloudless)            B3 Make the SAR subset (for M3)
A4 L1C download → bands → TOA reflectance                  │
A5 Cloud probability per pixel                             │
A6 DEM → elevation, slope                                  │
A7 Co-registration                                         │
A8 Normalisation                                           │
A9 Calibration + pseudo-labels (even years) ──┐            │
                                              ▼            ▼
                                  TRAINING FILES (HDF5) for M1, M2, M3
```

<a id="3-2-a1"></a>
### 3.2 A1 — Study area, domain and grid

Code: [`src/data/inventory.py`](../src/data/inventory.py).
Source: the GLIMS WFS server, inside 77.40–77.75° E, 32.10–32.40° N. It provides RGI 6.0, RGI 7.0
and all GLIMS outlines of Chhota Shigri.

| Product | Definition | Purpose |
|---|---|---|
| ROI | Bounding box of RGI7 + 2.5 km | Area to download |
| Grid | UTM 43N, 10 m, **snapped outward to the 60 m lattice** → 1434 × 1254 px | One grid for every product |
| Domain | RGI7 + 1 km **minus 33 neighbouring RGI7 glaciers** → 47.9 km² | Where glacier pixels are counted |
| Core | RGI7 shrunk by 300 m | Identifies which ice body is Chhota Shigri |

Why it matters:
- **60 m snapping:** Sentinel-2 tile origins are multiples of 60 m, so the 10 m bands align exactly
  with the grid and need no resampling.
- **Fixed domain:** GLIMS outlines range from 11.6 to 16.8 km², mostly because of different ice
  divides. A fixed domain means area change comes from the margins, not from divide choices.
- **Location from the polygon:** the often-quoted 32.28° N 77.58° E is the Bara Shigri terminus.

<a id="3-3-a2"></a>
### 3.3 A2 — Scene discovery

Code: `search_l2a` in [`src/data/sentinel2.py`](../src/data/sentinel2.py).

1. Search Microsoft Planetary Computer (`sentinel-2-l2a`) for all items over the ROI,
   **15 Aug – 15 Oct**, for each year.
2. Keep only MGRS tile **43SGR** (no duplicates from overlapping tiles).
3. Keep **one item per datatake**, the latest reprocessing.

**Why L2A here but L1C later?** GlaViTU was trained on L1C (top-of-atmosphere) reflectance, so the
model input must be L1C. Planetary Computer serves only L2A, but L2A has the SCL layer, which is
useful for screening. L2A is therefore used **only to find and screen** scenes.

<a id="3-4-a3"></a>
### 3.4 A3 — Scene screening

| Gate | Test | Reject if | Error source |
|---|---|---|---|
| 1 | SCL cloud (classes 8, 9, 10) over the domain | > 60 % | Cloud (cheap first pass) |
| 2 | SCL snow (class 11) on **domain minus glacier** | > 45 % | Fresh snowfall |
| 3 | s2cloudless cloud over the domain (20 m) | > 50 % | Cloud (accurate test) |

Each decision and reason is saved to `data/sentinel2/ensemble_<year>.csv`. The **core window
15 Aug – 15 Sep** is applied when scenes are used; later scenes serve only for gap filling and the
sensitivity test.

<a id="3-5-a4"></a>
### 3.5 A4 — L1C download, band selection, radiometry

Code: `find_l1c` and `fetch_l1c_bands` in [`src/data/sentinel2.py`](../src/data/sentinel2.py).
Source: Google Cloud public Sentinel-2 bucket (anonymous HTTPS).

1. Find the L1C product with the same datatake; prefer the highest processing baseline.
2. **Select 6 bands**, exactly those GlaViTU was trained with:

   | Band | Native resolution | Resampling to 10 m |
   |---|---|---|
   | B2, B3, B4, B8 | 10 m | None (grid aligned) |
   | B11, B12 | 20 m | Bilinear |

3. Read **only the grid window** over HTTP (GDAL `/vsicurl`). Data are already in UTM 43N, so no
   reprojection is needed.
4. **Convert to TOA reflectance:**
   `reflectance = (DN + RADIO_ADD_OFFSET) / QUANTIFICATION_VALUE`, with both values read from each
   product's `MTD_MSIL1C.xml`. This handles the 2022 offset change.
5. DN = 0 → NaN (no data).
6. Save `data/sentinel2/scenes/<date>/optical_toa.tif` and `scene.json`.

<a id="3-6-a5"></a>
### 3.6 A5 — Cloud probability

1. Read 10 L1C bands at 20 m (B1, B2, B4, B5, B8, B8A, B9, B10, B11, B12).
2. Run s2cloudless: threshold 0.4, 4 px averaging, 2 px (40 m) dilation.
3. Upsample to 10 m and save `cloud_prob.tif`.

Used for the image-level test above, and later to give cloudy pixels **no label** in training.

<a id="3-7-a6"></a>
### 3.7 A6 — DEM: elevation and slope

Code: [`src/data/dem.py`](../src/data/dem.py). Source: Copernicus DEM **GLO-30** (Planetary
Computer); heights relative to EGM2008.

1. Mosaic all DEM tiles over the ROI.
2. **Elevation:** reproject to the 10 m grid (bilinear).
3. **Slope:** reproject to 30 m (cubic), calculate with **Horn's (1981) method**, upsample to 10 m
   (bilinear).
4. Aspect and hillshade are also made, for analysis and figures only.

**Why slope at 30 m?** The DEM's real resolution is 30 m. Slope from a DEM interpolated to 10 m has
artificial steps.

<a id="3-8-a7"></a>
### 3.8 A7 — Co-registration

Code: [`src/preprocessing/coregister.py`](../src/preprocessing/coregister.py).

1. Reference image: 2023-09-11.
2. **Stable terrain** only: outside the domain + 300 m, outside all glaciers + 150 m, slope < 35°,
   cloud-free in both images, snow-free in both images (NDSI < 0.2).
3. Convert band B8 to Sobel gradient magnitude (edges match better than raw brightness).
4. Phase cross-correlation (×20 upsampling, sub-pixel) in 256 px windows with ≥ 50 % stable pixels.
5. If the median shift is > 0.25 px, apply a Fourier shift to all bands and the cloud probability.

| Year | Shift before (m) | Residual after (m) | Scenes corrected |
|---|---|---|---|
| 2017 | 6.67 | 1.41 | 1 (0.67 → 0.11 px) |
| 2019 | 3.61 | 1.41 | 1 (0.36 → 0.14 px) |
| 2021 | 0.71 | 0.71 | 0 |
| 2023 | 1.41 | 1.41 | 0 |
| 2025 | 1.58 | 1.58 | 0 |

Even-year scenes were not processed by this step.

<a id="3-9-a8"></a>
### 3.9 A8 — Normalisation

Code: `normalise` in [`src/inference/glavitu.py`](../src/inference/glavitu.py).

1. Optical NaN → 0.
2. **Min-max scaling with the official global statistics** (`dataset_stats.pickle`), separately for
   each optical band, elevation and slope: `x_norm = (x − min) / (max − min)`.
3. Group into `optical` (6 channels) and `dem` (2 channels).

**Why the authors' statistics?** The pretrained weights expect exactly the scale of the global
benchmark. Statistics from Chhota Shigri would shift every input and damage the pretrained model.

<a id="3-10-a9"></a>
### 3.10 A9 — Calibration and pseudo-labels

This is the last step before M2 training. Code:
[`src/validation/calibration.py`](../src/validation/calibration.py) and
[`src/training/pseudo_labels.py`](../src/training/pseudo_labels.py).

**What is a pseudo-label?** A label made by a model instead of a person. The **teacher** (M1) makes
labels; the **student** (M2) trains on them. There is no hand-drawn outline of Chhota Shigri, so this
is the only way to train on the site without breaking the rule "no manual digitising".

**Step 1 — Teacher predictions.** M1 runs on the even-year scenes of the core window. Cloudy
pixels → NaN.

| Year | Scenes | Dates |
|---|---|---|
| 2018 | 3 | 18 Aug, 28 Aug, 12 Sep |
| 2020 | 4 | 17 Aug, 22 Aug, 1 Sep, 11 Sep |
| 2022 | 5 | 22 Aug, 27 Aug, 1 Sep, 6 Sep, 11 Sep |
| 2024 | 3 | 21 Aug, 31 Aug (empty, see [5](#5-issues)), 5 Sep |

**Step 2 — Calibration.** The raw GlaViTU output never leaves about **0.24–0.89**, because of label
smoothing (0.1) and focal loss in training. The rule "P ≥ 0.95" therefore selected **0 pixels** at
the first attempt.

*Calibration* corrects a model's numbers so that "P = 0.95" really means "correct 95 % of the time".
*Isotonic regression* learns this correction as a step curve that only goes up, so it keeps the
model's ranking of pixels. It was fitted on 400,000 pixels from 20 HMA **validation** tiles (true
labels; never site data) and tested on a separate half of the tiles.

| Raw P | 0.25 | 0.30 | 0.40 | 0.50 | 0.60 | **0.70** | 0.80 | 0.85 |
|---|---|---|---|---|---|---|---|---|
| Calibrated P | 0.00 | 0.06 | 0.33 | 0.63 | 0.86 | **0.97** | 0.999 | 1.00 |

The calibration error (ECE) dropped from **0.239 to 0.011**. Raw 0.70 already means 97 % correct.

**Step 3 — Consensus rule.** A pixel is labelled **glacier** only if all of these are true:
1. Calibrated P ≥ 0.95 in at least 60 % of its clear scenes.
2. P is never ≤ 0.05 in any scene.
3. The yearly median is ≥ 0.95 in at least 3 of the 4 years.

**Background** uses the opposite rule. **All other pixels are ignored.**

| Class | Result |
|---|---|
| Glacier | 292,012 px |
| Background | 881,732 px |
| Ignored | 34.7 % (margins, terminus, flickering snow, real change; 3,999 px flip between scenes) |

**Step 4 — Training file** `data/labels/cs_pseudolabels.hdf5` (same layout as the benchmark):

| Per scene, group `CS-<date>` | Shape | Content |
|---|---|---|
| `optical` | 1254 × 1434 × 6 | Normalised reflectance of that scene |
| `dem` | 1254 × 1434 × 2 | Normalised elevation and slope |
| `outlines` | 1254 × 1434 × 2 | [1,0] background, [0,1] glacier, **[0,0] ignored** (zero loss) |

- The **same consensus label** is attached to every scene, so the student learns not to be fooled by
  one-day snow or shadow.
- A pixel that is cloudy in a scene gets [0,0] in that scene.
- Each scene appears **twice** (`CS-<date>` and `CS-<date>-r1`) to raise its sampling weight.
- **40 HMA training tiles** are linked in for replay, so the model does not forget expert labels.
- Total: 70 entries (30 site + 40 HMA).
- No inventory outline is used as a label, and the M2 checkpoint is selected on **HMA validation**
  tiles, never site data.

**Why does a student learn from its teacher's own answers?**
1. The consensus over many scenes and years is better than any single prediction.
2. The student learns local conditions (snow, debris, shadow) of Chhota Shigri.
3. The difficult margins get no site label, so the student cannot copy the teacher's errors there.

Result: HMA test IoU 0.748 → 0.779; agreement with RGI7 0.816 → 0.838; boundary distance 61 → 48 m.

<a id="3-11-b"></a>
### 3.11 B — HMA benchmark tiles

Code: [`src/data/benchmark.py`](../src/data/benchmark.py). Source: the GlaViTU benchmark on the NIRD
archive (doi:10.11582/2024.00168), 411 GB in total.

1. Read **only the HMA tiles** with HTTP byte-range requests (no full download).
2. Four parallel shards; each tile is written as `.tmp` and renamed when complete.
3. Keep `optical`, `dem`, `outlines`, `co_pol_sar`, `cross_pol_sar` as provided by the authors.
4. Keep the **official split: 165 train / 55 val / 57 test.** No tile lies within 0.05° of the site.
5. SAR subset (tiles with `co_pol_sar`): 66 train / 22 val / 22 test.

<a id="3-12-s1"></a>
### 3.12 Sentinel-1 (special case)

Code: [`src/data/sentinel1.py`](../src/data/sentinel1.py). Site SAR is **not used in any training**;
it is used only when M3 is applied to the site.

- Source: Planetary Computer `sentinel-1-rtc` (γ⁰).
- Approximate σ⁰ = γ⁰ × cos(incidence angle), incidence about 37–39°.
- Per-pixel median per orbit direction in the core window; gaps filled with the median.
- Normalised as log10(x + 1e-6), then official min-max.
- Limitation: terrain flattening is not undone, so this is an approximate input.

<a id="3-13-training-files"></a>
### 3.13 Training files for each model

| Model | Training file | Validation file | Inputs |
|---|---|---|---|
| M1-local | `hma_train.hdf5` (165 tiles) | `hma_val.hdf5` (55) | optical + dem |
| M2 | `cs_pseudolabels.hdf5` (15 scenes × 2 + 40 HMA) | `hma_val.hdf5` | optical + dem |
| M3 | `hma_train_sar.hdf5` (66) | `hma_val_sar.hdf5` (22) | optical + dem + SAR |
| No-SAR / optical-only | `hma_train_sar.hdf5` (66) | `hma_val_sar.hdf5` (22) | optical + dem / optical |

Preprocessing ends here. During training, the official loader cuts random 384 px patches and applies
augmentations on the fly (flips, rotation, crop-and-scale, gamma 0.8–1.2, feature occlusion, Gaussian
shift and noise, label smoothing 0.1).

<a id="3-14-summary"></a>
### 3.14 Error sources handled before training

| Step | Error source handled |
|---|---|
| A1 Domain and grid | Neighbouring glaciers; differing ice divides; band alignment |
| A2 Discovery | Duplicate scenes |
| A3 Screening | Clouds; fresh snowfall |
| Core window | Seasonal snow; autumn snowfall; low sun |
| A4 Radiometry | ESA 2022 offset |
| A5 Cloud probability | Per-pixel clouds (and nearby shadow) |
| A6 DEM | Debris-covered ice vs rock; slope artefacts |
| A7 Co-registration | Image shifts (odd years) |
| A8 Normalisation | Scale mismatch with the pretrained model |
| A9 Pseudo-labels | Cloudy pixels unlabelled; flickering margins ignored; temporal leakage |
| B Benchmark | Spatial leakage (official split, no tile near the site) |

**Not handled before training:** per-pixel cloud-shadow mask, terrain-shadow mask, water mask.

---

<a id="4-analysis"></a>
## 4. Analysis of all years (baseline 2017)

<a id="4-1-read"></a>
### 4.1 How to read the tables

- **Baseline:** every change is relative to **2017**.
- **Sign convention:** ΔA negative = glacier area lost. Retreat positive = terminus moved upstream.
- **Odd years** are the analysis epochs. **Even years** were also mapped by every model (from
  11 October 2026). They add four extra points to the time series.
- **†** marks M2 in an even year. M2 was trained on exactly those scenes, so its values there are
  **not independent**. The proof: M2's between-scene spread in 2018 is 0.04 km², against 0.55 km²
  for M1 on the same scenes. For independent even-year values, use **M0 or M1-official**.
- **‡** marks even years, which were not co-registered.
- Each yearly product is a multi-date ensemble (per-pixel median) of the core-window scenes.
- Full LaTeX versions: [`docs/paper/table_allyears_baseline2017.tex`](paper/table_allyears_baseline2017.tex).
  Data: [`docs/paper/data/`](paper/data/).

<a id="4-2-data"></a>
### 4.2 Data and quality of each year

| | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|
| Role | baseline | training | analysis | training | analysis | training | analysis | training | analysis |
| Candidate scenes | 3 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 9 |
| Rejected: cloud / snow | 1 / 0 | 3 / 0 | 4 / 0 | 2 / 0 | 4 / 0 | 1 / 0 | 0 / 0 | 3 / 0 | 5 / 2 |
| Rejected: empty / outlier | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 1 | 0 / 0 | 1 / 0 | 0 / 0 |
| **Scenes used** | **2** | **3** | **2** | **4** | **2** | **4** | **6** | **2** | **2** |
| Platforms | 2 S2B | 2 S2B, S2A | 2 S2A | S2B, 3 S2A | 2 S2A | 3 S2A, S2B | 3 S2A, 3 S2B | S2A, S2B | S2C, S2B |
| Sun elevation (°) | 58–64 | 58–65 | 60–62 | 58–65 | 60–65 | 59–64 | 59–65 | 60–64 | 64–65 |
| Co-registration residual (m) | 1.41 | ‡ | 1.41 | ‡ | 0.71 | ‡ | 1.41 | ‡ | 1.58 |
| Mean clear looks / pixel | 1.90 | 2.79 | 1.91 | 3.91 | 1.96 | 3.82 | 5.27 | 1.88 | 1.98 |
| Area seen once (km²) | 3.41 | 0.57 | 3.14 | 0.00 | 1.68 | 0.03 | 0.00 | 5.64 | 0.88 |
| Gap-filled (km²) | 0.10 | 0.00 | 0.52 | 0.00 | 0.00 | 0.00 | 0.00 | 0.14 | 0.00 |
| Unobserved (km²) | 0.52 | 0.15 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**Reading:** 2023 is the best-observed year (6 scenes, every pixel seen about 5 times). 2017, 2019
and 2024 are the weakest (2 scenes, 3–6 km² seen only once).

<a id="4-3-area"></a>
### 4.3 M2 area and uncertainty

| | 2017 | 2018† | 2019 | 2020† | 2021 | 2022† | 2023 | 2024† | 2025 |
|---|---|---|---|---|---|---|---|---|---|
| **Area (km²)** | **15.00** | 14.76 | **15.89** | 15.11 | **15.38** | 14.96 | **15.03** | 14.69 | **14.30** |
| Perimeter (km) | 80.9 | 79.3 | 85.6 | 81.1 | 82.1 | 77.9 | 82.8 | 82.2 | 79.6 |
| σ buffer | 0.40 | 0.40 | 0.43 | 0.41 | 0.41 | 0.39 | 0.41 | 0.41 | 0.40 |
| σ threshold | 1.12 | 0.79 | 1.12 | 0.87 | 0.91 | 0.74 | 0.88 | 0.81 | 0.75 |
| σ surface | 0.81 | 0.04 | 0.75 | 0.07 | 0.32 | 0.13 | 0.37 | 0.37 | 0.27 |
| **σ total** | **1.44** | 0.88 | **1.41** | 0.96 | **1.05** | 0.85 | **1.04** | 0.98 | **0.89** |
| Single-scene areas | 14.55–15.70 | 14.53–14.86 | 15.27–16.32 | 15.11–15.47 | 15.07–15.53 | 14.84–15.05 | 14.50–15.67 | 14.38–14.90 | 14.12–14.50 |

The three uncertainty terms:
- **σ buffer** = ½ pixel × perimeter (edge position).
- **σ threshold** = half the area difference between thresholds 0.4 and 0.6 (calibrated probability).
- **σ surface** = spread between the scenes of a year (snow and debris conditions).

σ total = √(σ buffer² + σ threshold² + σ surface²). Published area: 15.5–15.7 km².

<a id="4-4-change"></a>
### 4.4 M2 change relative to 2017

| | 2018† | 2019 | 2020† | 2021 | 2022† | 2023 | 2024† | 2025 |
|---|---|---|---|---|---|---|---|---|
| **ΔA (km²)** | −0.24 | +0.89 | +0.12 | +0.38 | −0.04 | +0.03 | −0.30 | **−0.70** |
| σ ΔA (km²) | 1.05 | 1.25 | 1.03 | 1.07 | 1.07 | 1.09 | 1.11 | 1.09 |
| ΔA (%) | −1.6 | +6.0 | +0.8 | +2.5 | −0.2 | +0.2 | −2.0 | −4.7 |
| Rate since 2017 (km² a⁻¹) | −0.25 | +0.45 | +0.04 | +0.10 | −0.01 | +0.01 | −0.04 | −0.09 |
| Significant at 2σ | no | no | no | no | no | no | no | no |
| Pixels lost since 2017 (km²) | 0.61 | 0.50 | 0.49 | 0.51 | 0.54 | 0.66 | 0.74 | **0.94** |
| Pixels gained since 2017 (km²) | 0.37 | 1.40 | 0.60 | 0.89 | 0.51 | 0.70 | 0.44 | 0.24 |
| Terminus position (m along centerline) | 9770 | 10090 | 9850 | 9850 | 9740 | 9850 | 9830 | 9740 |
| **Retreat, centerline (m)** | +110 | −210 | +30 | +30 | +140 | +30 | +50 | **+140** |
| Retreat, box method (m) | +83 | −218 | +47 | +35 | +111 | +41 | +49 | **+108** |
| σ retreat (m) | 14.2‡ | 14.3 | 14.2‡ | 14.2 | 14.2‡ | 14.3 | 14.2‡ | 14.3 |

How change uncertainty is calculated: the buffer and surface terms of both years are combined in
quadrature; the threshold term is treated as systematic (the same model shifts every year the same
way).

**Reading:**
- No area change relative to 2017 is statistically significant.
- Lost area grows steadily from 0.5 to 0.94 km², while gained area falls from 1.4 to 0.24 km². The
  outline is slowly shrinking even though the net change is within the noise.
- The 2019 "advance" (−210 m) and "gain" (+1.40 km²) come from a weak epoch with 2 scenes and a large
  co-registration correction. They are not physical.

<a id="4-5-models"></a>
### 4.5 All models: area and change

**Area (km²)**

| Model | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|
| M0 global | 13.19 | 13.42 | 14.47 | 15.15 | 14.70 | 13.96 | 14.44 | 13.32 | 13.42 |
| M1 HMA official | 14.16 | 13.96 | 14.81 | 14.54 | 14.78 | 14.41 | 14.46 | 13.95 | 13.79 |
| M1 HMA local | 14.42 | — | 15.43 | — | 15.24 | — | 15.26 | — | 14.10 |
| M2 Chhota Shigri | 15.00 | 14.76† | 15.89 | 15.11† | 15.38 | 14.96† | 15.03 | 14.69† | 14.30 |
| M3 HMA + SAR | 14.38 | 14.83 | 16.05 | 15.62 | 15.72 | — | 15.53 | 14.29 | 13.96 |
| **Median (range)** | 14.38 (1.80) | 14.36 (1.40) | 15.43 (1.59) | 15.13 (1.09) | 15.24 (1.03) | 14.41 (1.00) | 15.03 (1.09) | 14.12 (1.38) | 13.96 (0.88) |

**ΔA relative to 2017 (km²)**

| Model | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|
| M0 global | +0.23 | +1.28 | +1.96 | +1.51 | +0.76 | +1.24 | +0.13 | +0.22 |
| M1 HMA official | −0.20 | +0.65 | +0.38 | +0.62 | +0.25 | +0.30 | −0.21 | −0.37 |
| M1 HMA local | — | +1.01 | — | +0.82 | — | +0.84 | — | −0.32 |
| M2 Chhota Shigri | −0.24† | +0.89 | +0.12† | +0.38 | −0.04† | +0.03 | −0.30† | −0.70 |
| M3 HMA + SAR | +0.44 | +1.67 | +1.24 | +1.34 | — | +1.14 | −0.10 | −0.43 |
| **Median** | +0.01 | +1.01 | +0.81 | +0.82 | +0.25 | +0.84 | −0.15 | −0.37 |

M1-local was run on odd years only. M3 has no 2022 result (no Sentinel-1 composite).

**Reading:** the 2017 baseline is low for most models (M0 especially), so most years show a positive
ΔA. This is a weakness of the 2017 epoch, not glacier growth. From 2023 to 2025 every model shows a
decrease.

<a id="4-6-terminus"></a>
### 4.6 Terminus retreat relative to 2017

Two methods are used:
- **Centerline:** the furthest point downstream reached continuously from the glacier head along an
  automatic least-cost centerline (after Kienholz et al. 2014).
- **Box method:** the glacier area inside a 2 km × 159 m box over the tongue, divided by the box width.

**Centerline (m)**

| Model | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|
| M0 global | 370 | 440 | 160 | 330 | 470 | 370 | 90 | 460 |
| M1 HMA official | 240 | 0 | 150 | 160 | 270 | 160 | 100 | 230 |
| M1 HMA local | — | −110 | — | −90 | — | −90 | — | 10 |
| M2 Chhota Shigri | 110† | −210 | 30† | 30 | 140† | 30 | 50† | 140 |
| M3 HMA + SAR | 50 | −120 | 30 | 40 | — | 50 | 100 | 150 |
| **Median [min, max]** | 175 [50, 370] | −110 [−210, 440] | 90 [30, 160] | 40 [−90, 330] | 270 [140, 470] | 50 [−90, 370] | 95 [50, 100] | **150 [10, 460]** |

**Box method (m)**

| Model | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|
| M0 global | 386 | 465 | 203 | 338 | 501 | 350 | 144 | 498 |
| M1 HMA official | 250 | 71 | 147 | 176 | 290 | 161 | 110 | 260 |
| M1 HMA local | — | −100 | — | −56 | — | −51 | — | 17 |
| M2 Chhota Shigri | 83† | −218 | 47† | 35 | 111† | 41 | 49† | 108 |
| M3 HMA + SAR | 65 | −126 | 31 | 47 | — | 47 | 97 | 127 |
| **Median [min, max]** | 167 [65, 386] | −100 [−218, 465] | 97 [31, 203] | 47 [−56, 338] | 290 [111, 501] | 47 [−51, 350] | 104 [49, 144] | **127 [17, 498]** |
| All available models retreat | yes | no | yes | no | yes | no | yes | yes |
| Models available | 4 | 5 | 4 | 5 | 3 | 5 | 4 | 5 |

Measurement σ ≈ 14.3 m (odd years) and ≥ 14.2 m (even years, underestimated).

**Reading:** by 2025 every model places the terminus upstream of its 2017 position (box method:
17–498 m). The "all models retreat" row cannot be compared between odd and even years, because
M1-local, the only model showing an advance, exists only in odd years.

**Period results (odd years, 5 models)** — the most robust terminus result of the project:

| Period | Centerline: median [min–max] | Box: median [min–max] | All 5 models retreat |
|---|---|---|---|
| 2017–2019 | −110 [−210, 440] | −100 [−218, 465] | no |
| 2019–2021 | 160 [−110, 240] | 105 [−127, 252] | no |
| 2021–2023 | 0 [0, 40] | 5 [−14, 12] | no |
| 2023–2025 | 100 [70, 110] | 80 [67, 148] | yes |
| **2021–2025** | **110 [70, 130] (≈ 28 m a⁻¹)** | **80 [73, 160] (≈ 20 m a⁻¹)** | **yes** |
| 2017–2025 | 150 [10, 460] | 127 [17, 498] | yes |

<a id="4-7-agreement"></a>
### 4.7 Agreement with the RGI7 outline

RGI7 is a 2002 outline, so these numbers measure **agreement, not accuracy**. A slow decline over time
is expected if the glacier shrinks.

| | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|
| **M2 IoU** | 0.836 | 0.845† | 0.833 | 0.851† | 0.854 | 0.848† | 0.843 | 0.839† | 0.823 |
| M2 precision | 0.965 | 0.978 | 0.934 | 0.970 | 0.963 | 0.974 | 0.968 | 0.977 | 0.981 |
| M2 recall | 0.862 | 0.861 | 0.885 | 0.874 | 0.883 | 0.868 | 0.867 | 0.856 | 0.836 |
| M2 mean boundary distance (m) | 48.9 | 44.2 | 52.2 | 41.8 | 42.6 | 43.7 | 45.0 | 45.9 | 51.0 |
| M2 P95 boundary distance (m) | 165.5 | 160.0 | 200.0 | 152.3 | 160.0 | 153.0 | 156.2 | 155.2 | 170.9 |
| **M1 official IoU** | 0.800 | 0.808 | 0.827 | 0.830 | 0.833 | 0.820 | 0.822 | 0.804 | 0.796 |
| M1 mean boundary distance (m) | 60.3 | 61.3 | 58.8 | 52.6 | 58.0 | 60.0 | 57.7 | 62.1 | 67.9 |

**Reading:** precision is high (≥ 0.93) in every year, so the model rarely marks non-glacier as
glacier. Recall falls after 2021 (0.883 → 0.836), which fits a glacier that is smaller than its 2002
outline.

<a id="4-8-trends"></a>
### 4.8 Area trends

Weighted least-squares trend of area against time, with 95 % confidence interval (km² a⁻¹):

| Model | Odd years only | All available years |
|---|---|---|
| M0 global | −0.089 [−0.456, 0.277] | −0.145 [−0.384, 0.094] |
| M1 HMA official | −0.180 [−0.384, 0.024] | −0.083 [−0.209, 0.043] |
| M1 HMA local | −0.024 [−0.317, 0.269] | (odd years only) |
| M2 Chhota Shigri | −0.159 [−0.402, 0.084] | −0.064 [−0.181, 0.052] |
| M3 HMA + SAR | −0.114 [−0.611, 0.382] | −0.098 [−0.342, 0.145] |

**Every confidence interval includes zero.** The long-term published loss (≈ 0.03 km² a⁻¹) implies
only about 0.25 km² over 2017–2025, which is below the detection limit of this data.

<a id="4-9-findings"></a>
### 4.9 Key findings

1. **Terminus retreat is detected.** Over 2021–2025, all five models agree: 70–160 m, about
   20–28 m a⁻¹. This is far larger than the ≈ 14 m measurement error.
2. **Relative to 2017, the terminus is upstream in 2025 for every model** (box method 17–498 m).
3. **Area change is not detectable.** No change relative to 2017 and no trend is significant at 95 %.
   The uncertainty (≈ 1 km²) is larger than the expected change (≈ 0.25 km²).
4. **The glacier outline is slowly shrinking.** M2 pixels lost since 2017 grow from 0.5 to
   0.94 km², while pixels gained fall from 1.4 to 0.24 km².
5. **Surface conditions are the main noise source.** Thin snow on debris margins changes single-image
   area by about 1 km²; the multi-date median reduces this to 0.3–0.8 km².
6. **The best-observed period is 2021–2025.** The early years (2017, 2019) have only 2 images each and
   needed the largest co-registration corrections.

<a id="4-10-limitations"></a>
### 4.10 Limitations of the year-by-year analysis

- **2017 is a weak baseline**: 2 scenes, 3.4 km² seen once, the largest co-registration correction.
  Every "relative to 2017" number carries this weakness.
- **M2 even-year values are not independent** (†). Use M0 or M1-official for even years.
- **Even years were not co-registered** (‡); their terminus σ is underestimated.
- **The model set differs by year** (M1-local odd years only; M3 has no 2022).
- **Seasonal timing differs**: 2025 has only August scenes; 2023 reaches 11 September.
- **Absolute terminus position** differs by up to ~800 m between models (debris-covered snout); only
  changes are meaningful.

---

<a id="5-issues"></a>
## 5. Known issues and corrections

These were found while checking the data for this report. Correct them before publishing.

| # | Issue | Evidence | Action |
|---|---|---|---|
| 1 | `docs/methodology.md` says site scenes were "hard-linked 6 times" in the M2 training file. | The file holds each scene **twice** (`CS-<date>`, `CS-<date>-r1`); code default `--cs_repeats 2`. | Write "2×" in the paper; correct `methodology.md`. |
| 2 | Scene 2024-08-31 is an **empty L1C download** (56 KB instead of ~24 MB). | Dropped by the ensemble; 0 labelled pixels in the pseudo-label file. | M2 trained on **14** even-year scenes with data, not 15. Correct the text or re-download the scene. |
| 3 | `README.md` says 2017 "was mostly Sentinel-2A only". | Both 2017 scenes used are **Sentinel-2B**. | Correct the README. |
| 4 | Even-year scenes were **not co-registered**. | `outputs/statistics/coregistration.csv` lists odd-year scenes only. | Run `coregister.py` on even-year scenes before using even-year terminus results. |
| 5 | `outputs/statistics/` contains odd years only. | Even-year ensembles were produced on 11 October, after the statistics run of 3 October. | All-year statistics are in [`docs/paper/data/`](paper/data/); re-run `change.py` if needed. |
| 6 | No per-pixel cloud-shadow, terrain-shadow or water mask. | `src/inference/ensemble.py` masks only s2cloudless cloud and missing data. | State as a limitation (rows ◐ / ○ in [2.10](#2-10-summary)). |

---

<a id="appendix"></a>
## Appendix

<a id="a1-glossary"></a>
### A.1 Glossary

| Term | Meaning |
|---|---|
| **Ablation season** | The part of the year when the glacier loses mass by melting (summer) |
| **Box method** | Terminus measure: glacier area inside a fixed box over the tongue, divided by the box width |
| **Calibration** | Correcting model probabilities so that "P = 0.95" means "correct 95 % of the time" |
| **Centerline** | A line along the glacier flow, here computed automatically on the DEM |
| **Co-registration** | Aligning images so that the same ground point has the same pixel in every image |
| **Deep supervision** | An extra output inside the network that receives its own loss during training |
| **DEM** | Digital elevation model (terrain heights) |
| **ECE** | Expected calibration error: average gap between predicted probability and real frequency |
| **Ensemble (multi-date)** | Combining many images of a year by taking the per-pixel median |
| **Focal loss** | A loss that focuses training on difficult pixels |
| **HMA** | High Mountain Asia, the region of the benchmark tiles |
| **IoU** | Intersection over union: overlap between prediction and reference, divided by their union |
| **Isotonic regression** | A calibration method that learns a correction curve that only goes up |
| **L1C / L2A** | Sentinel-2 products: top-of-atmosphere (L1C) and surface reflectance with scene classification (L2A) |
| **MAD** | Median absolute deviation, a robust measure of spread |
| **NDSI** | Normalised difference snow index, (green − SWIR) / (green + SWIR) |
| **Pseudo-label** | A training label made by a model instead of a person |
| **Replay** | Mixing old training data into new training so the model does not forget |
| **RGI** | Randolph Glacier Inventory (global glacier outlines) |
| **s2cloudless** | A pretrained cloud detector for Sentinel-2 |
| **SCL** | Scene classification layer of Sentinel-2 L2A (cloud, shadow, snow, …) |
| **Self-attention** | The transformer operation in which every token compares itself with every other token |
| **Semantic segmentation** | Giving a class to every pixel of an image |
| **Squeeze-and-excitation** | A block that learns a weight for each feature channel |
| **TOA reflectance** | Top-of-atmosphere reflectance (no atmospheric correction) |
| **WLS** | Weighted least squares, a trend fit that gives precise points more weight |

<a id="a2-file-reference"></a>
### A.2 File reference

| Topic | Files |
|---|---|
| Pipeline | [`run_pipeline.sh`](../run_pipeline.sh), [`configs/chhota_shigri.yaml`](../configs/chhota_shigri.yaml), [`configs/hma_finetune.yaml`](../configs/hma_finetune.yaml) |
| Model code | `third_party/glavitu/models/mapping/glavitu.py`, `setr.py`, `resunet.py`; [`src/inference/glavitu.py`](../src/inference/glavitu.py) |
| Data | [`src/data/inventory.py`](../src/data/inventory.py), [`sentinel2.py`](../src/data/sentinel2.py), [`dem.py`](../src/data/dem.py), [`sentinel1.py`](../src/data/sentinel1.py), [`benchmark.py`](../src/data/benchmark.py) |
| Preprocessing | [`src/preprocessing/coregister.py`](../src/preprocessing/coregister.py) |
| Training | [`src/training/finetune.py`](../src/training/finetune.py), [`pseudo_labels.py`](../src/training/pseudo_labels.py), [`src/validation/calibration.py`](../src/validation/calibration.py) |
| Inference | [`src/inference/ensemble.py`](../src/inference/ensemble.py), [`src/postprocessing/mask.py`](../src/postprocessing/mask.py) |
| Change | [`src/change_detection/change.py`](../src/change_detection/change.py), [`centerline.py`](../src/change_detection/centerline.py), [`patterns.py`](../src/change_detection/patterns.py) |
| Results | `outputs/statistics/`, `experiments/<model>/<year>/summary.json`, [`docs/paper/data/`](paper/data/) |
| Paper tables | [`table_error_sources.tex`](paper/table_error_sources.tex), [`table_epochs.tex`](paper/table_epochs.tex), [`table_allyears_baseline2017.tex`](paper/table_allyears_baseline2017.tex) |
| Other docs | [`README.md`](../README.md), [`methodology.md`](methodology.md), [`validation.md`](validation.md), [`results.md`](results.md), [`experiments.md`](experiments.md) |
