#!/bin/bash
# Reproducible Entropy-Lens suite. Waits for the linear-probing grid to free the
# GPUs, analyses it, then runs base+lora (both GPUs) and scratch, then analyses.
# All seeds deterministic via PYTHONHASHSEED=0.
set -u
cd /rnd_ai_datasets1/projects/raka6003/transformer-memory-interference

export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
export HF_HOME=/rnd_ai_datasets1/projects/raka6003/.hf_home
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
export PYTHONHASHSEED=0

PY=.venv/bin/python
EL=lora_intervention/experiments/entropy_lens
LP=lora_intervention/experiments/linear_probing
RES=$EL/results
mkdir -p "$RES"

echo "[suite] $(date) waiting for linear probing to finish..."
if [ -f /tmp/v3grid31_pids.txt ]; then
  read BASE_PID LORA_PID < /tmp/v3grid31_pids.txt
  while ps -p "$BASE_PID" >/dev/null 2>&1 || ps -p "$LORA_PID" >/dev/null 2>&1; do sleep 30; done
fi
echo "[suite] $(date) linear probing done."

echo "[suite] analysing linear-probing 31-cell grid..."
$PY $LP/analyze_isoaccuracy.py > "$RES/../lp_grid31_analysis.log" 2>&1 || echo "[suite] LP analysis error (continuing)"

echo "[suite] HF smoke gate (base, trials=3)..."
$PY $EL/run_entropy_lens.py --model base --device cuda:0 --trials 3 --out_name base_smoke > "$RES/base_smoke.log" 2>&1
ACC=$($PY -c "import json;print(json.load(open('$RES/entropy_base_smoke.json'))['cells']['K2_N4']['FVQ']['behavioral_accuracy'])" 2>/dev/null || echo 0)
echo "[suite] smoke K2_N4 FVQ acc=$ACC"
if ! $PY -c "import sys; sys.exit(0 if float('$ACC')>0.5 else 1)"; then
  echo "[suite] SMOKE FAILED (acc=$ACC) -- HF logit-lens path suspect; NOT running full suite."
  exit 1
fi
rm -f "$RES/entropy_base_smoke.json"
echo "[suite] smoke OK."

echo "[suite] $(date) running base (cuda:0) + lora (cuda:1) in parallel..."
nohup $PY $EL/run_entropy_lens.py --model base --device cuda:0 > "$RES/base.log" 2>&1 &
B=$!
nohup $PY $EL/run_entropy_lens.py --model lora --device cuda:1 > "$RES/lora.log" 2>&1 &
L=$!
wait $B; echo "[suite] base exit $?"
wait $L; echo "[suite] lora exit $?"

echo "[suite] $(date) running scratch (cuda:0)..."
$PY $EL/run_entropy_lens.py --model scratch --device cuda:0 --scratch_source hf > "$RES/scratch.log" 2>&1
echo "[suite] scratch exit $?"

echo "[suite] $(date) analysing entropy results..."
$PY $EL/analyze_entropy.py > "$RES/analysis.log" 2>&1
echo "[suite] $(date) SUITE COMPLETE"
echo "[suite] outputs: $(ls $RES/entropy_*.json 2>/dev/null | tr '\n' ' ')"