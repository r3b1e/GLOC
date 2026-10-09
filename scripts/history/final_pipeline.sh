#!/usr/bin/env bash
# Final stage (after resume_pipeline.sh): M3 SAR ablation, Chhota Shigri analysis for all models, validation.
cd "$(dirname "$0")"
PY=./.env/bin/python
LOG=experiments/final_pipeline.log
ODD="2017 2019 2021 2023 2025"
say() { echo "$(date '+%F %T') $*" >> $LOG; }
analyse() {  # change detection + spatial patterns + figures for one model
  $PY -m src.change_detection.change --model $1 >> experiments/analysis_$1.log 2>&1 &&
  $PY -m src.change_detection.patterns --model $1 >> experiments/analysis_$1.log 2>&1 &&
  $PY -m src.visualization.plots --model $1 >> experiments/analysis_$1.log 2>&1 && say "analysis $1 done" || say "analysis $1 FAILED"
}

until grep -q "ALL DONE" experiments/resume_pipeline.log 2>/dev/null; do sleep 60; done
say "final pipeline started"

gpu_chain() {
  ./train_with_retry.sh experiments/M1_sarsubset_nosar_train.log --name M1_sarsubset_nosar \
      --init models/pretrained/glavitu_global_weights.h5 --train data/benchmark/hma_train_sar.hdf5 \
      --val data/benchmark/hma_val_sar.hdf5 --epochs 4 --steps 250 --batch 4 && say "M1 sar-subset (no SAR) trained"
  ./train_with_retry.sh experiments/M3_train.log --name M3_hma_sar --features optical dem co_pol_sar \
      --init models/pretrained/glavitu_global_weights.h5 --train data/benchmark/hma_train_sar.hdf5 \
      --val data/benchmark/hma_val_sar.hdf5 --epochs 4 --steps 250 --batch 4 --out models/sar && say "M3 trained"
  say "GPU2 DONE"
}

cpu_chain() {
  $PY -m src.preprocessing.coregister >> experiments/coregistration.log 2>&1 && say "coregistration done"
  for m in M1_hma_local M2_cs_selftrain; do
    $PY -m src.inference.ensemble --model $m --years $ODD >> experiments/ensemble_odd_years.log 2>&1 && say "ensemble $m done"
  done
  for m in M1_hma_official M0_global M1_hma_local M2_cs_selftrain M1_hma_official_fullwindow; do analyse $m; done
  say "CPU2 DONE"
}

gpu_chain & cpu_chain & wait

$PY -m src.validation.benchmark_eval --models M0_global M1_hma_official M1_sarsubset_nosar M3_hma_sar --split test --subset _sar \
    >> experiments/benchmark_eval_test_sar.log 2>&1 && say "SAR-subset bench eval done"
until grep -q "co_pol_sar composite" data/sentinel1.log && ! pgrep -f src.data.sentinel1 >/dev/null; do sleep 30; done
$PY -m src.inference.ensemble --model M3_hma_sar --years $ODD >> experiments/ensemble_odd_years.log 2>&1 && say "ensemble M3 done"
analyse M3_hma_sar
$PY -m src.validation.site_validation --models M0_global M1_hma_official M1_hma_local M2_cs_selftrain M3_hma_sar \
    >> experiments/site_validation.log 2>&1 && say "site validation done"
say "FINAL DONE"
