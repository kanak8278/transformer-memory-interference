#!/usr/bin/env bash
# End-to-end closed-pool value-identity probing.
#
# Per (model, cell): pilot -> size T from measured wrong_frac -> full collection
# -> CPU probe fits. Each stage writes into results_probe50/<run>/ and logs to
# logs/. Stages are skipped if their output already exists, so the script is
# restartable after an interruption.
#
# See PROBE50_DESIGN.md for the design. Scope: Qwen2.5-3B-Instruct base at
# (5,10) and (10,5), then gemma-3-4b-it base. LoRA is deferred -- it is
# saturated at every cell the closed pool permits.
set -uo pipefail

cd "$(dirname "$0")/../../.."          # repo root
PY=.venv-probe50/bin/python
EXP=lora_intervention/experiments/linear_probing
OUT=$EXP/results_probe50
LOGS=$EXP/logs
mkdir -p "$LOGS" "$OUT"

PILOT_TRIALS=${PILOT_TRIALS:-150}
BATCH=${BATCH:-4}                      # measured optimum on M5; larger is slower
POOL=50

stamp() { date "+%Y-%m-%d %H:%M:%S"; }
say()   { echo "[$(stamp)] $*" | tee -a "$LOGS/orchestrator.log"; }

run_cell () {
  local model="$1" cell="$2" short="$3"
  local nk="${cell%,*}" nu="${cell#*,}"
  local run="${short}_${nk}k_${nu}u"
  local pilot_dir="$OUT/${run}_pilot"
  local full_dir="$OUT/$run"
  local tmap="$OUT/${run}_T.json"

  say "=== $model  cell ($cell) ==="

  if [[ -f "$tmap" ]]; then
    say "  pilot: T map exists, skipping ($tmap)"
  else
    say "  pilot: $PILOT_TRIALS trials/condition, with generate for the audit"
    $PY $EXP/probe50_collect.py \
        --base_model "$model" --cell "$cell" --pool_size $POOL \
        --trials "$PILOT_TRIALS" --pilot --batch_size "$BATCH" \
        --emit_T_json "$tmap" --tag _pilot \
        >> "$LOGS/${run}_pilot.log" 2>&1
    if [[ $? -ne 0 || ! -f "$tmap" ]]; then
      say "  PILOT FAILED -- see $LOGS/${run}_pilot.log"; return 1
    fi
    say "  pilot done: $(tr -d '\n ' < "$tmap")"
  fi

  if [[ -f "$full_dir/manifest.json" && -f "$full_dir/reps_CVQ.npy" ]]; then
    say "  collect: outputs exist, skipping ($full_dir)"
  else
    say "  collect: full run at pilot-derived T"
    $PY $EXP/probe50_collect.py \
        --base_model "$model" --cell "$cell" --pool_size $POOL \
        --T_json "$tmap" --batch_size "$BATCH" \
        >> "$LOGS/${run}_full.log" 2>&1
    if [[ $? -ne 0 ]]; then
      say "  COLLECT FAILED -- see $LOGS/${run}_full.log"; return 1
    fi
    say "  collect done: $(du -sh "$full_dir" | cut -f1) of activations"
  fi

  if [[ -f "$full_dir/probe_fits.json" ]]; then
    say "  fit: probe_fits.json exists, skipping"
  else
    say "  fit: C grid x 4 label sets x 3 subsets x all layers"
    $PY $EXP/probe50_fit.py "$full_dir" --n_jobs 8 \
        >> "$LOGS/${run}_fit.log" 2>&1
    if [[ $? -ne 0 ]]; then
      say "  FIT FAILED -- see $LOGS/${run}_fit.log"; return 1
    fi
    say "  fit done -> $full_dir/probe_fits.json"
  fi
  say "=== $model ($cell) complete ==="
}

say "###### run_probe50_all.sh start ######"
say "pilot_trials=$PILOT_TRIALS batch=$BATCH pool=$POOL"

run_cell "Qwen/Qwen2.5-3B-Instruct" "5,10"  "Qwen2.5-3B-Instruct" || say "!! Qwen (5,10) failed, continuing"
run_cell "Qwen/Qwen2.5-3B-Instruct" "10,5"  "Qwen2.5-3B-Instruct" || say "!! Qwen (10,5) failed, continuing"
run_cell "google/gemma-3-4b-it"     "5,10"  "gemma-3-4b-it"       || say "!! gemma (5,10) failed, continuing"
run_cell "google/gemma-3-4b-it"     "10,5"  "gemma-3-4b-it"       || say "!! gemma (10,5) failed, continuing"

say "###### run_probe50_all.sh done ######"
