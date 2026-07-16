#!/bin/bash
# Full training + eval for the depth sweep (2,3,4,5 layers), per setup.md defaults.
# Run from 01_baseline_step_ood/src/. Logs to ../results/depth_sweep.log.
set -e
cd "$(dirname "$0")"
PY=../../.venv/bin/python
LOG=../results/depth_sweep.log
mkdir -p ../results
echo "=== depth sweep started $(date) ===" >> "$LOG"

for L in 2 3 4 5; do
  TAG="L${L}_d64"
  echo "--- training n_layers=$L (tag=$TAG) $(date) ---" >> "$LOG"
  $PY train.py --n_layers $L --total_steps 12500 --warmup_steps 125 --ckpt_every 500 --tag $TAG >> "$LOG" 2>&1
  echo "--- evaluating $TAG $(date) ---" >> "$LOG"
  $PY evaluate.py --ckpt ../checkpoints/$TAG/best.pt --out ../results/$TAG/final_eval.json >> "$LOG" 2>&1
  $PY dump_val_samples.py --ckpt ../checkpoints/$TAG/best.pt --split heldout_test --n_samples 60 \
      --out ../results/$TAG/heldout_samples_readable.txt >> "$LOG" 2>&1
  echo "--- done $TAG $(date) ---" >> "$LOG"
done

echo "=== depth sweep finished $(date) ===" >> "$LOG"
