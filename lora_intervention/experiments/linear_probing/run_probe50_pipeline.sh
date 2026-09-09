#!/usr/bin/env bash
# Pipelined closed-pool value-identity probing.
#
# Collections are GPU-bound and run strictly one at a time -- the M5 has one
# device and it is already saturated at batch_size 4 (measured: 255 ms/trial at
# bs=4 vs 359 at bs=32, i.e. more work per call made it slower, so there is no
# idle capacity for a second process).
#
# Probe fitting is CPU/BLAS-bound, so in principle it could overlap the next
# collection. It does NOT here: this box has 24 GB and sits ~7.5 GB into swap
# under ordinary desktop load (Chrome ~1.5 GB, Teams ~0.8 GB), and the collect
# stage was already OOM-killed once. Fits therefore wait for the GPU stage to
# clear. The gemma weight download (network) is the one thing that does overlap.
#
# Restartable: every stage skips if its output already exists.
set -uo pipefail
cd "$(dirname "$0")/../../.."

PY=.venv-probe50/bin/python
EXP=lora_intervention/experiments/linear_probing
OUT=$EXP/results_probe50
LOGS=$EXP/logs
mkdir -p "$LOGS" "$OUT"
BATCH=${BATCH:-2}
GEN=${GEN:-40}

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/pipeline.log"; }

FIT_PIDS=()

collect_cell () {                       # model short cell
  local model="$1" short="$2" cell="$3"
  local nk="${cell%,*}" nu="${cell#*,}"
  local run="${short}_${nk}k_${nu}u"
  local dir="$OUT/$run" tmap="$OUT/${run}_T.json"

  if [[ -f "$dir/manifest.json" && -f "$dir/reps_CVQ.npy" ]]; then
    say "collect $run: exists, skip"; return 0
  fi
  say "collect $run: start (T=$($PY -c "import json;print(sum(json.load(open('$tmap')).values()))") passes)"
  $PY $EXP/probe50_collect.py \
      --base_model "$model" --cell "$cell" --pool_size 50 \
      --T_json "$tmap" --batch_size "$BATCH" --generate_subset "$GEN" \
      >> "$LOGS/${run}_collect.log" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then say "collect $run: FAILED rc=$rc (see $LOGS/${run}_collect.log)"; return 1; fi
  say "collect $run: done, $(du -sh "$dir" | cut -f1)"
}

fit_cell_bg () {                        # short cell  -- launches on CPU, returns immediately
  local short="$1" cell="$2"
  local nk="${cell%,*}" nu="${cell#*,}"
  local run="${short}_${nk}k_${nu}u" dir="$OUT/${short}_${nk}k_${nu}u"

  if [[ -f "$dir/probe_fits.json" ]]; then say "fit $run: exists, skip"; return 0; fi
  if [[ ! -f "$dir/manifest.json" ]]; then say "fit $run: no activations, skip"; return 0; fi
  # Overlapping fits with a collection was the original plan, but this box
  # has 24 GB and sits ~7.5 GB into swap under normal desktop load, and the
  # collect stage was already OOM-killed once. Fits therefore run only when no
  # collection is in flight.
  while pgrep -f probe50_collect >/dev/null; do sleep 30; done
  say "fit $run: launched (no collection in flight)"
  ( $PY $EXP/probe50_fit.py "$dir" --n_jobs 4 >> "$LOGS/${run}_fit.log" 2>&1 \
    && echo "[$(date '+%F %T')] fit $run: done" >> "$LOGS/pipeline.log" \
    || echo "[$(date '+%F %T')] fit $run: FAILED (see $LOGS/${run}_fit.log)" >> "$LOGS/pipeline.log" ) &
  FIT_PIDS+=($!)
}

wait_for_gpu () {
  # A collection may already be running from a previous launch; do not contend.
  if pgrep -f probe50_collect >/dev/null; then
    say "waiting for the in-flight collection to finish before touching the GPU"
    while pgrep -f probe50_collect >/dev/null; do sleep 30; done
    say "GPU free"
  fi
}

wait_for_gemma () {
  # gemma weights download in parallel; block only when gemma's turn arrives.
  local snap=~/.cache/huggingface/hub/models--google--gemma-3-4b-it
  while pgrep -f snapshot_download >/dev/null; do
    say "gemma weights still downloading ($(du -sh $snap 2>/dev/null | cut -f1)), waiting"
    sleep 60
  done
}

say "###### pipeline start (batch=$BATCH generate_subset=$GEN) ######"

wait_for_gpu

# Qwen (5,10) was launched before this script. Its fit waits for that
# collection to finish (memory, not GPU, is the binding constraint).
fit_cell_bg  "Qwen2.5-3B-Instruct" "5,10"

collect_cell "Qwen/Qwen2.5-3B-Instruct" "Qwen2.5-3B-Instruct" "10,5"
fit_cell_bg  "Qwen2.5-3B-Instruct" "10,5"

wait_for_gemma
collect_cell "google/gemma-3-4b-it" "gemma-3-4b-it" "5,10"
fit_cell_bg  "gemma-3-4b-it" "5,10"

collect_cell "google/gemma-3-4b-it" "gemma-3-4b-it" "10,5"
fit_cell_bg  "gemma-3-4b-it" "10,5"

say "all collections done; waiting on ${#FIT_PIDS[@]} background fit jobs"
for pid in "${FIT_PIDS[@]:-}"; do [[ -n "$pid" ]] && wait "$pid"; done

say "###### pipeline done ######"
say "results:"
for d in "$OUT"/*/; do
  [[ -f "$d/manifest.json" ]] || continue
  say "  $(basename "$d"): $(du -sh "$d" | cut -f1) $([[ -f "$d/probe_fits.json" ]] && echo '(fits OK)' || echo '(NO FITS)')"
done
