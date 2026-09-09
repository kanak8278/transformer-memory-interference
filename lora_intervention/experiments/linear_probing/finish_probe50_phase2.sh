#!/usr/bin/env bash
# Phase 2: Qwen (10,5) top-ups, then every remaining fit.
# Waits for phase 1 (finish_probe50.sh) to exit. Collections and fits are
# serialized: measured, they interfere badly -- a collect runs at 283 ms/trial
# alone and ~1000 ms/trial alongside a fit, and killing the fit did not recover
# it, so the two are never overlapped.
set -uo pipefail
cd "$(dirname "$0")/../../.."
PY=.venv-probe50/bin/python
EXP=lora_intervention/experiments/linear_probing
OUT=$EXP/results_probe50; LOGS=$EXP/logs
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/phase2.log"; }

while pgrep -f finish_probe50.sh >/dev/null; do sleep 60; done
while pgrep -f probe50_collect  >/dev/null; do sleep 60; done
say "###### phase2 start (phase1 exited, GPU free) ######"

d="$OUT/Qwen2.5-3B-Instruct_10k_5u"
if [[ -f "$d/.topup_done" ]]; then
  say "Qwen (10,5) top-up: already done"
else
  say "Qwen (10,5) top-up: CVQ +1958, FVQ +2665, k1 +2271"
  $PY $EXP/probe50_collect.py --base_model Qwen/Qwen2.5-3B-Instruct --cell 10,5 \
      --pool_size 50 --T_json "$OUT/topup_qwen_10k5u_T.json" \
      --conditions CVQ FVQ k1 --t_start 80000 --batch_size 4 --generate_subset 40 \
      >> "$LOGS/qwen_10k5u_topup.log" 2>&1 \
    && touch "$d/.topup_done" && say "Qwen (10,5) top-up done" \
    || say "Qwen (10,5) top-up FAILED"
fi

while pgrep -f probe50_collect >/dev/null; do sleep 30; done

# Targeted fits only. Whole-run refits were wasteful: Qwen (5,10) is already
# complete (wrong 12/12, ref 12/12), Qwen (10,5) k2-k5 are valid pre-top-up
# (only CVQ/FVQ/k1 gain trials), and gemma (5,10) needs just the three
# conditions lost when an earlier fit was killed at 28/40 calls.
fit () {   # run  outfile  conditions...
  local run="$1" out="$2"; shift 2
  local d="$OUT/$run"
  [[ -f "$d/manifest.json" ]] || { say "fit $run: no data, skip"; return; }
  [[ -f "$d/$out" ]] && { say "fit $run -> $out: exists, skip"; return; }
  say "fit $run -> $out: conditions $*"
  $PY $EXP/probe50_fit.py "$d" --C_grid 0.1 --subsets wrong \
      --label_sets expected shuffled cv_first cv_last --conditions "$@" \
      --n_jobs 8 --out "$d/$out" >> "$LOGS/${run}_fit_$(basename "$out" .json).log" 2>&1 \
    && say "fit $run -> $out: done" || say "fit $run -> $out: FAILED"
}

# gemma (5,10): the three conditions missing from probe_fits_wrong.json.
fit gemma-3-4b-it_5k_10u probe_fits_wrong_missing.json k9 FVQ k1

# Qwen (10,5): only the conditions the top-up grew.
fit Qwen2.5-3B-Instruct_10k_5u probe_fits_wrong_topup.json CVQ FVQ k1

# gemma (10,5): full run, unless phase 1 already produced it.
d="$OUT/gemma-3-4b-it_10k_5u"
if [[ -f "$d/probe_fits_wrong.json" ]]; then
  say "fit gemma (10,5) wrong: exists, skip"
else
  say "fit gemma (10,5): wrong x 4 label sets, all conditions"
  $PY $EXP/probe50_fit.py "$d" --C_grid 0.1 --subsets wrong \
      --label_sets expected shuffled cv_first cv_last --n_jobs 8 \
      --out "$d/probe_fits_wrong.json" >> "$LOGS/gemma-3-4b-it_10k_5u_fit_wrong.log" 2>&1 \
    && say "fit gemma (10,5) wrong: done" || say "fit gemma (10,5) wrong: FAILED"
fi

# correct + all subsets, expected label only, wherever still missing.
for run in gemma-3-4b-it_10k_5u Qwen2.5-3B-Instruct_10k_5u; do
  d="$OUT/$run"
  [[ -f "$d/manifest.json" ]] || continue
  [[ -f "$d/probe_fits_ref.json" ]] && { say "fit $run ref: exists, skip"; continue; }
  say "fit $run: correct + all subsets"
  $PY $EXP/probe50_fit.py "$d" --C_grid 0.1 --subsets correct all \
      --label_sets expected --n_jobs 8 --out "$d/probe_fits_ref.json" \
      >> "$LOGS/${run}_fit_ref.log" 2>&1 \
    && say "fit $run ref: done" || say "fit $run ref: FAILED"
done

say "###### phase2 done ######"
