#!/bin/bash
# Supervisor: runs the E1 scan-C runner for Qwen, retrying on crash.
# Safe to retry — the runner skips (model,scan,K,N,condition) tuples already
# in results.jsonl, so a crash never loses completed work.
set -uo pipefail
cd "$(dirname "$0")/../../.."

set -a
[ -f lora_intervention/.env ] && source lora_intervention/.env
set +a
export HUGGING_FACE_HUB_TOKEN="${HF_API_KEY:-}"
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
export CURL_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
export CUDA_VISIBLE_DEVICES=1

OUT=lora_intervention/experiments/bigcell_ivq/out/qwen
mkdir -p "$OUT"
SUP_LOG="$OUT/supervisor.log"

MAX_ATTEMPTS=5
ATTEMPT=0
while [ "$ATTEMPT" -lt "$MAX_ATTEMPTS" ]; do
  ATTEMPT=$((ATTEMPT + 1))
  echo "[supervisor] attempt $ATTEMPT/$MAX_ATTEMPTS at $(date -Iseconds)" >> "$SUP_LOG"
  .venv/bin/python lora_intervention/experiments/e1_extrapolation_frontier.py \
    --out-dir "$OUT" --scans C --only both \
    --base-model Qwen/Qwen2.5-3B-Instruct \
    --adapter lora_intervention/checkpoints/adapter \
    --backend hf --micro-batch 32 >> "$OUT/stdout.log" 2>&1
  code=$?
  echo "[supervisor] exited with code $code at $(date -Iseconds)" >> "$SUP_LOG"
  if [ "$code" -eq 0 ]; then
    echo "[supervisor] DONE" >> "$SUP_LOG"
    exit 0
  fi
  sleep 15
done
echo "[supervisor] FAILED after $MAX_ATTEMPTS attempts" >> "$SUP_LOG"
exit 1
