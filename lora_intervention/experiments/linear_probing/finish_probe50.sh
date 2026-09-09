#!/usr/bin/env bash
# Finish the closed-pool probing run: remaining collections, then all fits.
#
# Collections are GPU-bound and strictly serialized. Fits are CPU-bound and run
# only when no collection is in flight -- this box is 24 GB with swap near full
# under desktop load and the collect stage was OOM-killed once already.
#
# Every stage is restartable: collections skip conditions that already have
# activations, fits skip runs that already have their output JSON.
set -uo pipefail
cd "$(dirname "$0")/../../.."

PY=.venv-probe50/bin/python
EXP=lora_intervention/experiments/linear_probing
OUT=$EXP/results_probe50
LOGS=$EXP/logs
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/finish.log"; }
gpu_free() { while pgrep -f probe50_collect >/dev/null; do sleep 30; done; }

QWEN=Qwen/Qwen2.5-3B-Instruct
GEMMA=google/gemma-3-4b-it

say "###### finish_probe50 start ######"

# 1. wait out the in-flight Qwen (10,5) collection
gpu_free; say "GPU free"

# 2. gemma (5,10) top-up: CVQ and k10 only.
#    FVQ/k1 are excluded deliberately -- gemma answers slot 1 at 0.991/0.992
#    accuracy, leaving 22 and 20 clean wrong trials. Reaching the 1250 target
#    would need ~139k and ~154k extra trials (~45 GPU-h each). Those two report
#    the `all` subset only.
if [[ ! -f "$OUT/gemma-3-4b-it_5k_10u/.topup_done" ]]; then
  say "gemma (5,10) top-up: CVQ +4248, k10 +61"
  $PY $EXP/probe50_collect.py --base_model "$GEMMA" --cell 5,10 --pool_size 50 \
      --T_json "$OUT/topup_gemma_5k10u_T.json" --conditions CVQ k10 \
      --t_start 60000 --batch_size 4 --generate_subset 40 \
      >> "$LOGS/gemma_5k10u_topup.log" 2>&1 \
    && touch "$OUT/gemma-3-4b-it_5k_10u/.topup_done" && say "gemma top-up done" \
    || say "gemma top-up FAILED (see $LOGS/gemma_5k10u_topup.log)"
else
  say "gemma (5,10) top-up: already done, skip"
fi

# 3. gemma (10,5)
gpu_free
if [[ ! -f "$OUT/gemma-3-4b-it_10k_5u/manifest.json" ]] \
   || [[ ! -f "$OUT/gemma-3-4b-it_10k_5u/reps_k1.npy" ]]; then
  say "gemma (10,5) collect: 19,200 trials"
  $PY $EXP/probe50_collect.py --base_model "$GEMMA" --cell 10,5 --pool_size 50 \
      --T_json "$OUT/gemma-3-4b-it_10k_5u_T.json" \
      --batch_size 4 --generate_subset 40 \
      >> "$LOGS/gemma_10k5u_collect.log" 2>&1 \
    && say "gemma (10,5) collect done" \
    || say "gemma (10,5) collect FAILED (see $LOGS/gemma_10k5u_collect.log)"
else
  say "gemma (10,5): already collected, skip"
fi

# 3b. gemma (5,10) CVQ/k10 fits, deferred until their top-up landed. The other
#     ten conditions were fitted while Qwen (10,5) was still collecting -- the
#     fit holds ~0.23 GB after the mmap/per-layer fixes, so it does not contend.
d="$OUT/gemma-3-4b-it_5k_10u"
if [[ -f "$d/probe_fits_wrong_topup.json" ]]; then
  say "fit gemma (5,10) CVQ/k10: exists, skip"
else
  say "fit gemma (5,10) CVQ/k10 (post top-up)"
  $PY $EXP/probe50_fit.py "$d" --C_grid 0.1 --subsets wrong \
      --label_sets expected shuffled cv_first cv_last --conditions CVQ k10 \
      --n_jobs 4 --out "$d/probe_fits_wrong_topup.json" \
      >> "$LOGS/gemma_5k10u_fit_wrong_topup.log" 2>&1 \
    && say "fit gemma (5,10) CVQ/k10: done" || say "fit gemma (5,10) CVQ/k10: FAILED"
fi

# 4. all fits at the frozen C=0.1.
#    C was selected once on Qwen (5,10) CVQ/wrong/expected (best mean top-1 over
#    layers) and is frozen everywhere, so a layer curve never mixes
#    representation quality with regularization strength.
#    No gpu_free here: measured concurrently, the fit is ~0.23 GB and the
#    collect ~0.36 GB, so they do not contend.
for run in Qwen2.5-3B-Instruct_5k_10u gemma-3-4b-it_5k_10u \
           Qwen2.5-3B-Instruct_10k_5u gemma-3-4b-it_10k_5u; do
  d="$OUT/$run"
  [[ -f "$d/manifest.json" ]] || { say "fit $run: no activations, skip"; continue; }

  if [[ -f "$d/probe_fits_wrong.json" ]]; then
    say "fit $run wrong: exists, skip"
  else
    say "fit $run: wrong subset x 4 label sets"
    $PY $EXP/probe50_fit.py "$d" --C_grid 0.1 --subsets wrong \
        --label_sets expected shuffled cv_first cv_last --n_jobs 4 \
        --out "$d/probe_fits_wrong.json" >> "$LOGS/${run}_fit_wrong.log" 2>&1 \
      && say "fit $run wrong: done" || say "fit $run wrong: FAILED"
  fi

  if [[ -f "$d/probe_fits_ref.json" ]]; then
    say "fit $run correct/all: exists, skip"
  else
    say "fit $run: correct + all subsets, expected label only"
    $PY $EXP/probe50_fit.py "$d" --C_grid 0.1 --subsets correct all \
        --label_sets expected --n_jobs 4 \
        --out "$d/probe_fits_ref.json" >> "$LOGS/${run}_fit_ref.log" 2>&1 \
      && say "fit $run correct/all: done" || say "fit $run correct/all: FAILED"
  fi
done

say "###### finish_probe50 done ######"
for d in "$OUT"/*/; do
  [[ -f "$d/manifest.json" ]] || continue
  w=$([[ -f "$d/probe_fits_wrong.json" ]] && echo wrong || echo -)
  r=$([[ -f "$d/probe_fits_ref.json" ]] && echo ref || echo -)
  say "  $(basename "$d"): $(du -sh "$d" | cut -f1)  fits[$w $r]"
done
