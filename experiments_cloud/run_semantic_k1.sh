#!/bin/bash
# K=1 ablation sweep on SEMANTIC_MULTI for all 6 proprietary models.
# Same format and update levels as the main behavioral table so rows can be appended directly.
set -e
cd "$(dirname "$0")/.."
source .env 2>/dev/null || true

LOGDIR="experiments_cloud/results/semantic_k1/logs"
mkdir -p "$LOGDIR"

MODELS=(claude-haiku claude-sonnet gpt-4.1 gpt-4.1-mini gemini-2.5-pro gemini-2.5-flash)

for MODEL in "${MODELS[@]}"; do
  .venv/bin/python experiments_cloud/sweep_semantic.py \
    --model "$MODEL" \
    --key-levels 1 \
    --update-levels 1 5 10 15 20 30 50 \
    --trials 200 \
    --save-dir experiments_cloud/results/semantic_k1 \
    > "$LOGDIR/${MODEL}.log" 2>&1 &
  echo "Launched $MODEL (PID $!)"
done

echo ""
echo "All ${#MODELS[@]} models launched."
echo "Monitor: tail -f $LOGDIR/<model>.log"
