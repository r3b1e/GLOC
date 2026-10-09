#!/usr/bin/env bash
# Resume orchestration (2026-10-02): CPU and GPU chains run in parallel without competing.
cd "$(dirname "$0")"
PY=./.env/bin/python
LOG=experiments/resume_pipeline.log
ODD="2017 2019 2021 2023 2025"; EVEN="2018 2020 2022 2024"
say() { echo "$(date '+%F %T') $*" >> $LOG; }

cpu_chain() {
  for m in M1_hma_official M0_global; do
    $PY -m src.inference.ensemble --model $m --years $ODD >> experiments/ensemble_odd_years.log 2>&1 && say "ensemble $m core done"
  done
  $PY -m src.training.pseudo_labels --teacher M1_hma_official --years $EVEN >> experiments/pseudo_labels.log 2>&1 && say "pseudo-labels done"
  $PY -m src.validation.benchmark_eval --models M0_global M1_hma_official --split test >> experiments/benchmark_eval_test.log 2>&1 && say "bench eval M0/M1-official done"
  for m in M1_hma_official M0_global; do
    $PY -m src.inference.ensemble --model $m --years $ODD --window 08-15 10-15 --tag _fullwindow >> experiments/ensemble_odd_years.log 2>&1
  done
  say "CPU CHAIN DONE"
}

gpu_chain() {
  ./train_with_retry.sh experiments/M1_hma_local_train.log --name M1_hma_local --init models/pretrained/glavitu_global_weights.h5 \
      --train data/benchmark/hma_train.hdf5 --val data/benchmark/hma_val.hdf5 --epochs 8 --steps 250 --batch 4 && say "M1 local trained"
  until grep -q "pseudo-labels done" $LOG; do sleep 30; done
  ./train_with_retry.sh experiments/M2_train.log --name M2_cs_selftrain --init models/pretrained/glavitu_finetuning_HMA_weights.h5 \
      --train data/labels/cs_pseudolabels.hdf5 --val data/benchmark/hma_val.hdf5 --epochs 4 --steps 200 --batch 4 \
      --ignore_unlabelled --lr 2e-5 && say "M2 trained"
  until grep -q "bench eval M0/M1-official done" $LOG; do sleep 30; done
  $PY -m src.validation.benchmark_eval --models M1_hma_local M2_cs_selftrain --split test >> experiments/benchmark_eval_test.log 2>&1 && say "bench eval M1-local/M2 done"
  say "GPU CHAIN DONE"
}

say "resume started"
cpu_chain & gpu_chain & wait
say "ALL DONE"
