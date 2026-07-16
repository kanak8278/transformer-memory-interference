#!/bin/bash
# Wait for L3's current 12500-step run to finish, then resume from its final checkpoint
# with a doubled step budget (25000) using the fixed (data-stream-correct) resume path.
set -e
cd "$(dirname "$0")"
PY=../../.venv/bin/python
LOG=../results/L3_extend.log
CKPT=../checkpoints/L3_d64/step_12500.pt

echo "=== waiting for L3 initial run to finish $(date) ===" >> "$LOG"
while [ ! -f "$CKPT" ]; do
  sleep 15
done
echo "=== L3 initial run finished, starting extension $(date) ===" >> "$LOG"

$PY train.py --n_layers 3 --resume "$CKPT" --total_steps 25000 --warmup_steps 125 --ckpt_every 500 --tag L3_d64 >> "$LOG" 2>&1

echo "=== L3 extension finished $(date) ===" >> "$LOG"
$PY evaluate.py --ckpt ../checkpoints/L3_d64/best.pt --out ../results/L3_d64/final_eval_extended.json >> "$LOG" 2>&1
$PY dump_val_samples.py --ckpt ../checkpoints/L3_d64/best.pt --split heldout_test --n_samples 60 \
    --out ../results/L3_d64/heldout_samples_readable_extended.txt >> "$LOG" 2>&1
echo "=== L3 extension eval done $(date) ===" >> "$LOG"
