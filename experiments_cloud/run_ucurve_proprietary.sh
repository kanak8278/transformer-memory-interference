#!/bin/bash
set -e
cd "$(dirname "$0")/.."
set -a && source .env 2>/dev/null; set +a

LOGDIR="experiments_cloud/results/ucurve_proprietary/logs"
mkdir -p "$LOGDIR"

run_model() {
  local MODEL="$1"
  .venv/bin/python experiments_cloud/ucurve_sweep.py \
    --model "$MODEL" \
    --dataset SEMANTIC_MULTI \
    --nk 5 10 \
    --nu 10 20 50 \
    --formats flat_nolabel_last flat_verbose_last block_last landmark_last \
    --n-positions 16 \
    --ci 0.05 \
    --trials 200 \
    --save-dir experiments_cloud/results/ucurve_proprietary \
    > "$LOGDIR/${MODEL}.log" 2>&1
}

resume_model() {
  local MODEL="$1"
  local CKPT="experiments_cloud/results/ucurve_proprietary/${MODEL}/checkpoint.json"
  .venv/bin/python experiments_cloud/ucurve_sweep.py \
    --model "$MODEL" \
    --dataset SEMANTIC_MULTI \
    --nk 5 10 \
    --nu 10 20 50 \
    --formats flat_nolabel_last flat_verbose_last block_last landmark_last \
    --n-positions 16 \
    --ci 0.05 \
    --trials 200 \
    --save-dir experiments_cloud/results/ucurve_proprietary \
    --resume "$CKPT" \
    >> "$LOGDIR/${MODEL}.log" 2>&1
}

# Anthropic
run_model claude-haiku  &
echo "Launched claude-haiku  (PID $!)"
run_model claude-sonnet &
echo "Launched claude-sonnet (PID $!)"

# OpenAI
run_model gpt-4.1       &
echo "Launched gpt-4.1       (PID $!)"
run_model gpt-4.1-mini  &
echo "Launched gpt-4.1-mini  (PID $!)"

# Google
run_model gemini-2.5-pro   &
echo "Launched gemini-2.5-pro   (PID $!)"
run_model gemini-2.5-flash &
echo "Launched gemini-2.5-flash (PID $!)"

echo ""
echo "All 6 launched. Waiting..."
wait
echo "All done."
