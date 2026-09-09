#!/usr/bin/env bash
# Phase 3, deliberately last: gemma (10,5) CVQ top-up, then re-fit that one
# condition.
#
# gemma answers "last" at slot 5 with 0.918 accuracy, so 3000 trials yielded
# only 245 clean wrong -- 4.9 per class, adequate for rank/MRR but too thin for
# 50-way top-1. CVQ carries the paired "last" vs "5th" contrast that is the
# headline comparison on this cell, so it is worth the ~12k extra trials, but
# only after every other fit is banked.
set -uo pipefail
cd "$(dirname "$0")/../../.."
PY=.venv-probe50/bin/python
EXP=lora_intervention/experiments/linear_probing
OUT=$EXP/results_probe50; LOGS=$EXP/logs
D="$OUT/gemma-3-4b-it_10k_5u"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/phase3.log"; }

while pgrep -f "finish_probe50_phase2.sh" >/dev/null; do sleep 120; done
while pgrep -f "probe50_(collect|fit)" >/dev/null; do sleep 60; done
say "###### phase3 start (all fits banked, GPU+CPU free) ######"

ADD=$($PY - <<'PY'
import json, math
D="lora_intervention/experiments/linear_probing/results_probe50/gemma-3-4b-it_10k_5u"
m=json.load(open(f"{D}/manifest.json")); e=m["conditions"]["CVQ"]
pool={v.lower() for v in m["pool"]}
cw=sum(1 for r in e["records"] if not r["correct_single_token"]
       and (r["pred_single_token"] or "").strip().lower() in pool)
print(max(0, math.ceil(1250/(cw/e["n"])) - e["n"]) if cw else 0)
PY
)
say "gemma (10,5) CVQ needs +$ADD trials for a 1250 clean-wrong subset"
if [[ "$ADD" -le 0 ]]; then say "nothing to add; done"; exit 0; fi

echo "{\"CVQ\": $ADD}" > "$OUT/topup_gemma_10k5u_T.json"
$PY $EXP/probe50_collect.py --base_model google/gemma-3-4b-it --cell 10,5 \
    --pool_size 50 --T_json "$OUT/topup_gemma_10k5u_T.json" --conditions CVQ \
    --t_start 100000 --batch_size 4 --generate_subset 40 \
    >> "$LOGS/gemma_10k5u_topup.log" 2>&1 \
  && say "CVQ top-up collected" || { say "CVQ top-up FAILED"; exit 1; }

say "re-fitting gemma (10,5) CVQ at frozen C=0.1"
$PY $EXP/probe50_fit.py "$D" --C_grid 0.1 --subsets wrong \
    --label_sets expected shuffled cv_first cv_last --conditions CVQ --n_jobs 8 \
    --out "$D/probe_fits_wrong_topup.json" \
    >> "$LOGS/gemma_10k5u_fit_topup.log" 2>&1 \
  && say "CVQ re-fit done" || say "CVQ re-fit FAILED"
say "###### phase3 done ######"
