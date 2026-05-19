#!/bin/bash
# Launch K=1 SEMANTIC_MULTI ucurve sweep for all 6 models concurrently.
set -e
cd "$(dirname "$0")/.."
source .env 2>/dev/null || true

LOGDIR="experiments_cloud/results/ucurve_single_key_sem/logs"
mkdir -p "$LOGDIR"

MODELS=(claude-haiku claude-sonnet gpt-4.1 gpt-4.1-mini gemini-2.5-pro gemini-2.5-flash)

for MODEL in "${MODELS[@]}"; do
  .venv/bin/python experiments_cloud/ucurve_sweep.py \
    --model "$MODEL" \
    --dataset SEMANTIC_MULTI \
    --nk 1 \
    --nu 10 15 20 30 50 \
    --formats flat_nolabel flat_nolabel_last flat_verbose flat_verbose_last \
    --n-positions 11 \
    --ci 0.05 \
    --trials 200 \
    --save-dir experiments_cloud/results/ucurve_single_key_sem \
    > "$LOGDIR/${MODEL}.log" 2>&1 &
  echo "Launched $MODEL (PID $!)"
done

echo ""
echo "All ${#MODELS[@]} models launched. Tail logs with:"
echo "  tail -f $LOGDIR/<model>.log"
