#!/usr/bin/env bash
# Full CoT-vs-noCoT IVQ sweep: 3 models x 2 arms x 6 cells, SEMANTIC_MULTI,
# flat_nolabel prompts.
#
# Parallel by model, sequential by arm. The three models sit in separate
# rate-limit buckets, so running them concurrently triples throughput at no
# risk (peak measured usage is ~1.3% of the input-token limit). The two arms
# within a model share a bucket, so they stay serialized.
#
# Concurrency: 32 workers, measured optimum. 16 gave 1.02 calls/s, 32 gives
# 1.33, 68 gives 1.37 but inflates mean latency 50% (server-side queueing).
# --trial-batch 8 keeps the pool fed once positions start retiring from the
# Wilson check, which is where a batch of 4 leaves workers idle.
#
# Thinking budget 3000 is a deliberate choice by the run owner. NOTE: the
# pilot at 15000 showed 7.4% (haiku) / 7.6% (opus) of calls exceed 3000, and
# the excess is entirely at positions 31-50 (0% at positions 1-26). Deep
# positions will therefore truncate and shallow ones will not, so CoT accuracy
# at depth in this run is a lower bound. The 15000 pilot on K10/N50 uses the
# same seeds and serves as the control for that artifact.
set -uo pipefail
cd "$(dirname "$0")/.."

BUDGET=${BUDGET:-3000}
WORKERS=${WORKERS:-32}
BATCH=${BATCH:-8}
CI=${CI:-0.05}
OUT=${OUT:-experiments_cloud/results/cot_ivq}
LOGDIR=${LOGDIR:-experiments_cloud/results/cot_ivq/_logs}
mkdir -p "$LOGDIR"

RUN=(uv run --with "anthropic>=1.4.0,<2" --with python-dotenv --with "tqdm>=4.66" python)

run_model() {
  local m=$1
  for arm in nocot cot_thinking; do
    echo "[$(date +%T)] START $m $arm"
    "${RUN[@]}" experiments_cloud/cot_ivq_sweep.py \
      --model "$m" --arm "$arm" \
      --thinking-budget "$BUDGET" \
      --workers "$WORKERS" --trial-batch "$BATCH" --ci "$CI" \
      --out-dir "$OUT" \
      >> "$LOGDIR/${m}_${arm}.log" 2>&1
    local rc=$?
    echo "[$(date +%T)] END   $m $arm (exit $rc)"
  done
}

for m in haiku sonnet opus; do
  run_model "$m" &
done
wait
echo "[$(date +%T)] ALL MODELS COMPLETE"
