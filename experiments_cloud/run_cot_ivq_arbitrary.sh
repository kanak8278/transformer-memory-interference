#!/usr/bin/env bash
# CoT IVQ N-scaling ladder — haiku, ARBITRARY_SINGLE, K=5.
#
# Question: with the thinking budget held FIXED at 3000, how far can N go before
# the model can no longer index into the stream? Budget is deliberately not
# scaled with N, so this measures compute-bounded retrieval rather than "how
# much reasoning does the task need".
#
# Dataset is ARBITRARY_SINGLE, not the SEMANTIC_MULTI used by the main sweep:
#   * SEMANTIC_MULTI pools are 45-70 real members per category, so it cannot
#     express N > 60 at all.
#   * ARBITRARY_MULTI reaches N=500 but its values encode the key in a prefix
#     ("chemical element: Elem143"), so a model can count occurrences by pattern
#     -matching the prefix without tracking the key. That inflates exactly the
#     ability under test.
#   * ARBITRARY_SINGLE values are uninformative ("tree species: dose"). The pool
#     is shared, so K*N <= 2300 — hence K=5 and N<=460.
#
# 17 positions, ci 0.05 and max 200 trials are unchanged from the main sweep so
# the per-position curves stay directly comparable.
#
# NOTE on --cache: the prompt is preamble+stream+query, so all 17 positions of a
# trial share a byte-identical prefix. The breakpoint sits after the stream. One
# call per trial is fired first to warm it; without that, 17 concurrent misses
# would each be billed at the 1.25x write rate. Haiku's minimum cacheable prefix
# is 4096 tokens, so the N=100 cell (2669) silently runs uncached — harmless,
# nothing is written, and it is the cheapest cell.
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-haiku}
BUDGET=${BUDGET:-3000}
WORKERS=${WORKERS:-32}
BATCH=${BATCH:-8}
CI=${CI:-0.05}
CELLS=${CELLS:-5_100,5_200,5_300,5_400,5_460}
DATASET=${DATASET:-ARBITRARY_SINGLE}
OUT=${OUT:-experiments_cloud/results/cot_ivq_arbitrary_single}
LOGDIR="$OUT/_logs"
mkdir -p "$LOGDIR"

RUN=(uv run --with "anthropic>=1.4.0,<2" --with python-dotenv --with "tqdm>=4.66" python)

echo "[$(date +%T)] START $MODEL cot_thinking  cells=$CELLS  dataset=$DATASET  budget=$BUDGET"
"${RUN[@]}" experiments_cloud/cot_ivq_sweep.py \
  --model "$MODEL" --arm cot_thinking --dataset "$DATASET" \
  --cells "$CELLS" --thinking-budget "$BUDGET" \
  --workers "$WORKERS" --trial-batch "$BATCH" --ci "$CI" \
  --cache --out-dir "$OUT" \
  >> "$LOGDIR/${MODEL}_cot_thinking.log" 2>&1
rc=$?
echo "[$(date +%T)] END   $MODEL cot_thinking (exit $rc)"
grep -E "DONE|billable" "$LOGDIR/${MODEL}_cot_thinking.log" | tail -3
