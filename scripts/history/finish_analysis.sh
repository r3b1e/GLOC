#!/usr/bin/env bash
# Finish stage B (2026-10-03): calibration, Chhota Shigri ensembles (co-registered), change/patterns/figures, validation.
cd "$(dirname "$0")"
PY=./.env/bin/python; LOG=experiments/finish.log; ODD="2017 2019 2021 2023 2025"
say() { echo "$(date '+%F %T') $*" >> $LOG; }
analyse() {
  $PY -m src.change_detection.change --model $1 >> experiments/analysis_$1.log 2>&1 &&
  $PY -m src.change_detection.patterns --model $1 >> experiments/analysis_$1.log 2>&1 &&
  $PY -m src.visualization.plots --model $1 >> experiments/analysis_$1.log 2>&1 && say "analysis $1 done" || say "analysis $1 FAILED"
}
say "finish B started"
while pgrep -f "calibration --model M0_global" >/dev/null; do sleep 30; done
for m in M1_hma_local M2_cs_selftrain M3_hma_sar; do
  $PY -m src.validation.calibration --model $m --n_tiles 10 >> experiments/calibration.log 2>&1 && say "calibration $m done"
done
for m in M1_hma_official M0_global; do   # co-registered 2017/2019 scenes -> re-run (caches were invalidated)
  $PY -m src.inference.ensemble --model $m --years $ODD >> experiments/ensemble_odd_years.log 2>&1
  $PY -m src.inference.ensemble --model $m --years $ODD --window 08-15 10-15 --tag _fullwindow >> experiments/ensemble_odd_years.log 2>&1
  say "ensemble $m (coreg) done"
done
for m in M1_hma_local M2_cs_selftrain M3_hma_sar; do
  $PY -m src.inference.ensemble --model $m --years $ODD >> experiments/ensemble_odd_years.log 2>&1 && say "ensemble $m done"
done
for m in M0_global M1_hma_official M1_hma_local M2_cs_selftrain M3_hma_sar M1_hma_official_fullwindow M0_global_fullwindow; do analyse $m; done
until grep -q "optical-only trained" $LOG; do sleep 60; done
$PY -m src.inference.ensemble --model M1_sarsubset_optonly --years $ODD >> experiments/ensemble_odd_years.log 2>&1 && say "ensemble optonly done"
$PY -m src.inference.ensemble --model M1_sarsubset_nosar --years $ODD >> experiments/ensemble_odd_years.log 2>&1 && say "ensemble nosar done"
for m in M1_sarsubset_optonly M1_sarsubset_nosar; do $PY -m src.change_detection.change --model $m >> experiments/analysis_$m.log 2>&1 && say "change $m done"; done
$PY -m src.validation.site_validation --models M0_global M1_hma_official M1_hma_local M2_cs_selftrain M3_hma_sar \
    M1_sarsubset_nosar M1_sarsubset_optonly M1_hma_official_fullwindow >> experiments/site_validation.log 2>&1 && say "site validation done"
say "FINISH B DONE"
