#!/usr/bin/env bash
# Finish stage A (2026-10-03): optical-only training (GPU) + all remaining HMA benchmark evaluations (CPU).
cd "$(dirname "$0")"
PY=./.env/bin/python; LOG=experiments/finish.log
say() { echo "$(date '+%F %T') $*" >> $LOG; }
say "finish A started"
( ./train_with_retry.sh experiments/M1_sarsubset_optonly_train.log --name M1_sarsubset_optonly --features optical \
    --init models/pretrained/glavitu_global_weights.h5 --train data/benchmark/hma_train_sar.hdf5 \
    --val data/benchmark/hma_val_sar.hdf5 --epochs 4 --steps 250 --batch 4 && say "optical-only trained" ) &
$PY -m src.validation.benchmark_eval --models M1_hma_local M2_cs_selftrain --split test >> experiments/benchmark_eval_test.log 2>&1 && say "bench eval M1-local/M2 done"
$PY -m src.validation.benchmark_eval --models M0_global M1_hma_official M1_sarsubset_nosar M3_hma_sar --split test --subset _sar \
    >> experiments/benchmark_eval_test_sar.log 2>&1 && say "SAR-subset bench eval done"
wait
$PY -m src.validation.benchmark_eval --models M1_sarsubset_optonly --split test --subset _sar >> experiments/benchmark_eval_test_sar.log 2>&1 && say "optical-only bench eval done"
say "FINISH A DONE"
