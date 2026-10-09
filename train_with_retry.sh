#!/usr/bin/env bash
# Retry wrapper for tensorflow-metal's intermittent graph errors: resumes from <name>_last_weights.h5.
# Usage: ./train_with_retry.sh <log> <finetune args...>
log=$1; shift
for attempt in $(seq 1 10); do
  GEOAI_DEVICE=gpu ./.env/bin/python -m src.training.finetune "$@" --resume >> "$log" 2>&1 && { echo "TRAINING DONE (attempt $attempt)" >> "$log"; exit 0; }
  echo "RETRY $attempt after failure" >> "$log"; sleep 5
done
echo "TRAINING FAILED after 10 attempts" >> "$log"; exit 1
