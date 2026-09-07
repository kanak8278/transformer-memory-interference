#!/usr/bin/env bash
# CoT-vs-noCoT sweep on the museum M0 naturalistic narrative.
# Haiku only, 2 arms x 6 cells, prompt caching on.
#
# The two arms share a rate-limit bucket, so they run sequentially. (The plain
# cot_ivq runner parallelises across three models precisely because those sit in
# separate buckets; with one model there is nothing to overlap.)
#
# Every knob matched to run_cot_ivq_full.sh — budget 3000, 32 workers, batch 8,
# CI 0.05, min 30 / max 200 trials — so the museum and plain CoT arms differ
# only in stimulus. Deliberate: a fixed budget across both stimulus families is
# what makes the two comparable.
#
# Known, measured cost of budget 3000 on this stimulus: a 28-call probe at 16000
# on the deep positions of K10/N50 found 50% of calls exceeding 3000, and the
# excess is concentrated at depth (position 50: 6,689-9,764 tokens; `first`:
# 305-3,981). So CoT accuracy at deep positions is a LOWER BOUND here. Budget
# truncation is silent in the API response, so the sweep derives and stores
# `thinking_budget_hit` per call and `n_thinking_budget_hit` per position.
#
# Caching: all 12-18 queries in a trial share the narrative prefix (~4.7k tokens
# at K10/N50), so the breakpoint is roughly a 3.4x input-cost cut (measured). On
# for BOTH arms — the block seam can shift tokenization slightly, which is
# harmless held constant and a confound otherwise.
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-haiku}
BUDGET=${BUDGET:-3000}
WORKERS=${WORKERS:-32}
BATCH=${BATCH:-8}
CI=${CI:-0.05}
OUT=${OUT:-experiments_cloud/results/museum_cot}
LOGDIR=${LOGDIR:-experiments_cloud/results/museum_cot/_logs}
mkdir -p "$LOGDIR"

RUN=(uv run --with "anthropic>=1.4.0,<2" --with python-dotenv --with "tqdm>=4.66" python)

for arm in nocot cot_thinking; do
  echo "[$(date +%T)] START $MODEL $arm"
  "${RUN[@]}" experiments_cloud/museum_cot_sweep.py \
    --model "$MODEL" --arm "$arm" \
    --thinking-budget "$BUDGET" \
    --workers "$WORKERS" --trial-batch "$BATCH" --ci "$CI" \
    --out-dir "$OUT" \
    >> "$LOGDIR/${MODEL}_${arm}.log" 2>&1
  echo "[$(date +%T)] END   $MODEL $arm (exit $?)"
done

echo "[$(date +%T)] DONE — audit with:"
echo "  uv run python experiments_cloud/cot_ivq_inspect.py --results-dir $OUT"
