#!/usr/bin/env bash
# Replacement for resume_pipeline.sh's GPU chain after the calibration fix (2026-10-02).
cd "$(dirname "$0")"
PY=./.env/bin/python
LOG=experiments/resume_pipeline.log
say() { echo "$(date '+%F %T') $*" >> $LOG; }
say "resume2: calibration fix started (raw GlaViTU softmax never reaches 0.95/0.05)"
$PY -m src.validation.calibration --model M1_hma_official --n_tiles 20 >> experiments/calibration.log 2>&1 && say "calibration done"
$PY -m src.training.pseudo_labels --teacher M1_hma_official --years 2018 2020 2022 2024 >> experiments/pseudo_labels.log 2>&1 && say "calibrated pseudo-labels done"
while pgrep -f "name M1_hma_local" >/dev/null; do sleep 30; done
say "M1 local trained"
./train_with_retry.sh experiments/M2_train.log --name M2_cs_selftrain --init models/pretrained/glavitu_finetuning_HMA_weights.h5 \
    --train data/labels/cs_pseudolabels.hdf5 --val data/benchmark/hma_val.hdf5 --epochs 4 --steps 200 --batch 4 \
    --ignore_unlabelled --lr 2e-5 && say "M2 trained"
until grep -q "CPU CHAIN DONE" $LOG; do sleep 30; done
$PY -m src.validation.benchmark_eval --models M1_hma_local M2_cs_selftrain --split test >> experiments/benchmark_eval_test.log 2>&1 && say "bench eval M1-local/M2 done"
say "ALL DONE"
