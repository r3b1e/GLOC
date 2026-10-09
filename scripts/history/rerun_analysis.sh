#!/usr/bin/env bash
# Re-run after the min-observation fix (2026-10-03 20:2x): all CS ensembles, analyses, aggregation, validation.
cd "$(dirname "$0")"
PY=./.env/bin/python; LOG=experiments/finish.log; ODD="2017 2019 2021 2023 2025"
say() { echo "$(date '+%F %T') $*" >> $LOG; }
say "RERUN (min-obs fix) started"
for m in M2_cs_selftrain M1_hma_official M0_global M1_hma_local M3_hma_sar M1_sarsubset_nosar M1_sarsubset_optonly; do
  $PY -m src.inference.ensemble --model $m --years $ODD >> experiments/ensemble_odd_years.log 2>&1 && say "rerun ensemble $m done" || say "rerun ensemble $m FAILED"
done
for m in M1_hma_official M0_global; do
  $PY -m src.inference.ensemble --model $m --years $ODD --window 08-15 10-15 --tag _fullwindow >> experiments/ensemble_odd_years.log 2>&1 && say "rerun fullwindow $m done"
done
for m in M2_cs_selftrain M1_hma_official M0_global M1_hma_local M3_hma_sar M1_hma_official_fullwindow M0_global_fullwindow; do
  { $PY -m src.change_detection.change --model $m && $PY -m src.change_detection.patterns --model $m && $PY -m src.visualization.plots --model $m; } \
    >> experiments/analysis_$m.log 2>&1 && say "rerun analysis $m done" || say "rerun analysis $m FAILED"
done
for m in M1_sarsubset_optonly M1_sarsubset_nosar; do $PY -m src.change_detection.change --model $m >> experiments/analysis_$m.log 2>&1 && say "rerun change $m done"; done
$PY -m src.validation.aggregate_results >> experiments/aggregate.log 2>&1 && say "aggregate done"
$PY -m src.validation.site_validation --models M0_global M1_hma_official M1_hma_local M2_cs_selftrain M3_hma_sar \
    M1_sarsubset_nosar M1_sarsubset_optonly M1_hma_official_fullwindow >> experiments/site_validation.log 2>&1 && say "site validation done"
say "RERUN DONE"
