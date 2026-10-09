# Datasets

All sources below were accessed programmatically; nothing was digitised by hand.
Access dates: 2026-09-29/30.

## Glacier inventories (GLIMS WFS, `https://www.glims.org/geoserver/ows`)

| Layer | Use | Chhota Shigri record |
|---|---|---|
| `GLIMS:RGI_SouthAsiaWest` (RGI 6.0) | ID cross-reference | `RGI60-14.15990`, GLIMS `G077508E32224N`, 16.764 km², source date 2002-08-02, zmin 4280 m, zmax 5764 m, Lmax 8600 m |
| `GLIMS:RGI2000-v7.0-G-14_south_asia_west_epsg3857` (RGI 7.0) | ROI, analysis domain, terminus attribute, neighbour divides | `RGI2000-v7.0-G-14-30579`, GLIMS `G077513E32227N`, 16.772 km², 2002-08-02 |
| `GLIMS:GLIMS_Glacier_Outlines` | historical context / reference | 7 analyses (table below) |

GLIMS outlines of Chhota Shigri (area recomputed in UTM 43N):

| GLIMS analysis | Date | Institution | Area (km²) |
|---|---|---|---|
| 1065541 | "1450" (LIA reconstruction) | Univ. Leeds | 49.13 (glacier complex) |
| 1127326 | 2000-10-15 | KOPRI | 13.01 |
| 860910 / 1083522 | 2002-08-02 | Nagoya Univ. (GAMDAM) / Univ. Colorado (RGI) | 16.77 |
| 119262 | 2002-08-02 | Univ. Zurich | 14.87 |
| 60228 | 2002-09-28 | ICIMOD | 16.14 |
| 154108 | 2009-10-25 | ICIMOD | 11.65 |

**Important:** these outlines from the same 2000–2009 period differ by up to 5 km² (±30 %).
The spread is dominated by different ice-divide / tributary partitioning, not by ice loss. All
area statistics in this project are therefore computed inside one fixed *analysis domain*
(RGI7 outline + 1 km buffer, minus the 33 neighbouring RGI7 glaciers), so that only
margin changes affect the time series.

**Coordinate note:** the frequently quoted "32.28° N, 77.58° E" for Chhota Shigri coincides
with the RGI7 terminus of *Bara Shigri* (32.274° N, 77.585° E). The RGI7 centroid of Chhota
Shigri is 32.224° N, 77.508° E (RGI6 centroid 32.224° N, 77.508° E). The ROI is derived from the
inventory polygon, never from a point buffer.

**RGI7 centerlines** (`RGI2000-v7.0-L`) are distributed only by NSIDC behind a NASA Earthdata
login (HTTP 401 without credentials). A centerline is instead generated automatically
(`src/change_detection/centerline.py`). If an Earthdata account becomes available, the official
product can replace it.

## Sentinel-2

* **Discovery and screening:** Microsoft Planetary Computer STAC, collection `sentinel-2-l2a`
  (scene metadata + Scene Classification Layer, SCL).
* **Model input (TOA, L1C):** Planetary Computer serves only L2A. GlaViTU was trained on L1C
  top-of-atmosphere reflectance (`utils/dataset_stats.pickle`: optical min ≈ -0.027, max ≈ 2.7,
  i.e. harmonised TOA reflectance). L1C granules are read from the public Google Cloud bucket
  `gcp-public-data-sentinel-2` (anonymous HTTPS, windowed JPEG2000 reads over the ROI only).
* **Harmonisation:** reflectance = (DN + RADIO_ADD_OFFSET) / QUANTIFICATION_VALUE from each
  product's `MTD_MSIL1C.xml` (offset −1000 for processing baseline ≥ 04.00, 0 before), equivalent
  to `COPERNICUS/S2_HARMONIZED`.
* **Tile:** MGRS 43SGR, relative orbit 105 (covers the whole ROI).
* **Cloud masking:** s2cloudless (the algorithm behind the "Sentinel-2 Cloud Probability"
  product), run on the L1C bands at 20 m; threshold 0.4, 3 px dilation. SCL cloud/shadow is used
  only for fast pre-screening, because SCL frequently labels bright ice/snow as cloud over
  Chhota Shigri (e.g. 2019-09-17: SCL 11 % vs s2cloudless 2.3 % over the domain).
* Scene IDs, dates, radiometric offsets and per-candidate statistics are written to
  `data/sentinel2/selection_<year>.csv` and `data/sentinel2/<year>/scene.json`.

## DEM

Copernicus DEM GLO-30 (`cop-dem-glo-30` on Planetary Computer), tile
`Copernicus_DSM_COG_10_N32_00_E077_00_DEM`; bilinear to the 10 m grid; slope/aspect with Horn's
method at 30 m, then upsampled.

## GlaViTU benchmark (Maslov et al. 2025)

NIRD archive doi:10.11582/2024.00168 (CC-BY-4.0). The full tile-based dataset is 411 GB
(`20230905_{train,val,test}_global_ps384.hdf5` = 243 / 81 / 86 GB); the Google Drive "demo" copy
is 38 GB. Tile counts (full dataset):

| Split | Total | HMA |
|---|---|---|
| train | 1636 | 165 |
| val | 547 | 55 |
| test | 556 | 57 |

NIRD supports HTTP byte ranges, so only the HMA tiles are read (`src/data/benchmark.py`) and
stored locally as float32 (`data/benchmark/hma_{train,val,test}.hdf5`). The official split is
kept.

## Pretrained weights

Google Drive folder linked from the GlaViTU README (30 checkpoints). Used:
`glavitu_global_weights.h5` (M0), `glavitu_finetuning_HMA_weights.h5` (official global→HMA
fine-tuning, "M1-official"), `glavitu_regional_HMA_weights.h5` (HMA-only training). SHA-256 in
`models/pretrained/SHA256SUMS`.
