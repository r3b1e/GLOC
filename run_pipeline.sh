#!/usr/bin/env bash
# End-to-end, fully automatic reproduction of the Himalayan Glacier GeoAI workflow (as executed
# 2026-09-29 .. 2026-10-03). Every step reads configs/*.yaml. No manual digitisation anywhere.
# Training uses the Apple-Metal GPU (GEOAI_DEVICE=gpu) through train_with_retry.sh; inference is CPU.
set -euo pipefail
cd "$(dirname "$0")"
PY=./.env/bin/python
ODD="2017 2019 2021 2023 2025"      # analysis epochs (never used for training)
EVEN="2018 2020 2022 2024"          # pseudo-label epochs (M2 training only)

# ---- 0. third-party code + published weights -------------------------------------------------
[ -d third_party/glavitu ] || git clone https://github.com/konstantin-a-maslov/towards_global_glacier_mapping.git third_party/glavitu
git -C third_party/glavitu checkout -q 8c4eead670ef64764f8db2708bba3c8bbbfc67bb
for f in 1HLIVlWuGfMx_jZQleHlQmuzvzHVMRTOd:glavitu_global_weights.h5 \
         1hNRFR55MtkQjSH4ZRBcfswPNXnhafQ6i:glavitu_finetuning_HMA_weights.h5 \
         1HmulydQpPm4flN5_AaPvfMb0FEyMhYR5:glavitu_regional_HMA_weights.h5; do
  [ -f "models/pretrained/${f##*:}" ] || ./.env/bin/gdown -q "https://drive.google.com/uc?id=${f%%:*}" -O "models/pretrained/${f##*:}"
done
(cd models/pretrained && shasum -a 256 -c SHA256SUMS)

# ---- 1. study area, DEM, imagery --------------------------------------------------------------
$PY -m src.data.inventory                                   # RGI6/RGI7/GLIMS via WFS, ROI, domain, grid
$PY -m src.data.dem                                         # COP-DEM GLO-30 -> elevation/slope/aspect/hillshade
$PY -m src.data.sentinel2 --ensemble --years $ODD $EVEN     # screen + s2cloudless + L1C extraction of all usable scenes
$PY -m src.preprocessing.coregister --apply                 # stable-terrain phase correlation, sub-pixel correction
$PY -m src.data.sentinel1 --years $ODD                      # S1 RTC -> approx. sigma0 VV asc/desc composites (M3)

# ---- 2. HMA benchmark subset (HTTP range reads from NIRD) -------------------------------------
# tile_keys.json lists the remote tile names (written once by listing the remote HDF5 roots)
for k in 0 1 2 3; do $PY -m src.data.benchmark --splits test val train --shard $k/4 & done; wait
$PY -m src.data.benchmark --splits test val train --merge
$PY - <<'EOF'
import h5py
from pathlib import Path
for s in ["train", "val", "test"]:   # SAR-bearing subsets for the SAR / optical-only ablations
    src = h5py.File(f"data/benchmark/hma_{s}.hdf5", "r")
    with h5py.File(f"data/benchmark/hma_{s}_sar.hdf5", "w") as m:
        for k in [k for k in src if "co_pol_sar" in src[k]]:
            m[k] = h5py.ExternalLink(str(Path(f"data/benchmark/tiles/{s}/{k}.hdf5").resolve()), k)
EOF

# ---- 3. training ------------------------------------------------------------------------------
TR="./train_with_retry.sh"
$TR experiments/M1_hma_local_train.log --name M1_hma_local --init models/pretrained/glavitu_global_weights.h5 \
    --train data/benchmark/hma_train.hdf5 --val data/benchmark/hma_val.hdf5 --epochs 8 --steps 250 --batch 4
$PY -m src.validation.calibration --model M1_hma_official --n_tiles 20      # isotonic, HMA val tiles
$PY -m src.training.pseudo_labels --teacher M1_hma_official --years $EVEN   # calibrated, multi-temporal
$TR experiments/M2_train.log --name M2_cs_selftrain --init models/pretrained/glavitu_finetuning_HMA_weights.h5 \
    --train data/labels/cs_pseudolabels.hdf5 --val data/benchmark/hma_val.hdf5 --epochs 4 --steps 200 --batch 4 \
    --ignore_unlabelled --lr 2e-5
for spec in "M1_sarsubset_nosar:optical dem" "M3_hma_sar:optical dem co_pol_sar" "M1_sarsubset_optonly:optical"; do
  name=${spec%%:*}; feats=${spec#*:}
  $TR experiments/${name}_train.log --name $name --features $feats --init models/pretrained/glavitu_global_weights.h5 \
      --train data/benchmark/hma_train_sar.hdf5 --val data/benchmark/hma_val_sar.hdf5 --epochs 4 --steps 250 --batch 4 \
      $( [ "$name" = M3_hma_sar ] && echo "--out models/sar" )
done

# ---- 4. held-out HMA benchmark evaluation ------------------------------------------------------
$PY -m src.validation.benchmark_eval --models M0_global M1_hma_official M1_hma_local M2_cs_selftrain --split test
$PY -m src.validation.benchmark_eval --models M0_global M1_hma_official M1_sarsubset_nosar M3_hma_sar M1_sarsubset_optonly \
    --split test --subset _sar
for m in M0_global M1_hma_local M2_cs_selftrain M3_hma_sar; do $PY -m src.validation.calibration --model $m --n_tiles 10; done

# ---- 5. Chhota Shigri multi-date ensembles, change, patterns, figures -------------------------
for m in M0_global M1_hma_official M1_hma_local M2_cs_selftrain M3_hma_sar M1_sarsubset_nosar M1_sarsubset_optonly; do
  $PY -m src.inference.ensemble --model $m --years $ODD
done
for m in M0_global M1_hma_official; do $PY -m src.inference.ensemble --model $m --years $ODD --window 08-15 10-15 --tag _fullwindow; done
for m in M0_global M1_hma_official M1_hma_local M2_cs_selftrain M3_hma_sar M1_hma_official_fullwindow M0_global_fullwindow; do
  $PY -m src.change_detection.change --model $m
  $PY -m src.change_detection.patterns --model $m
  $PY -m src.visualization.plots --model $m
done
$PY -m src.validation.site_validation --models M0_global M1_hma_official M1_hma_local M2_cs_selftrain M3_hma_sar \
    M1_sarsubset_nosar M1_sarsubset_optonly M1_hma_official_fullwindow
