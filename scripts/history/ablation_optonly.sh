#!/usr/bin/env bash
# Experiment 4 (optical only), equal budget/tiles to the SAR ablation; runs after the final pipeline's GPU chain.
cd "$(dirname "$0")"
PY=./.env/bin/python; LOG=experiments/final_pipeline.log
say() { echo "$(date '+%F %T') $*" >> $LOG; }
until grep -q "GPU2 DONE" $LOG 2>/dev/null; do sleep 60; done
./train_with_retry.sh experiments/M1_sarsubset_optonly_train.log --name M1_sarsubset_optonly --features optical \
    --init models/pretrained/glavitu_global_weights.h5 --train data/benchmark/hma_train_sar.hdf5 \
    --val data/benchmark/hma_val_sar.hdf5 --epochs 4 --steps 250 --batch 4 && say "M1 optical-only trained"
until grep -q "SAR-subset bench eval done" $LOG; do sleep 60; done
$PY -m src.validation.benchmark_eval --models M1_sarsubset_optonly --split test --subset _sar >> experiments/benchmark_eval_test_sar.log 2>&1 && say "optical-only bench eval done"
$PY -m src.inference.ensemble --model M1_sarsubset_optonly --years 2017 2019 2021 2023 2025 >> experiments/ensemble_odd_years.log 2>&1 && say "ensemble optonly done"
say "ABLATION4 DONE"
