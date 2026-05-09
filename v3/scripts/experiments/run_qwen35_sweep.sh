#!/bin/bash
# Run Stage 1 behavioral sweep for Qwen3.5 models (0.8B, 4B, 9B).
# Saves results to v3/results_vllm/arbitrary_single/ and semantic_multi/
# in the same structure as existing experiments.
#
# Requires: setup_vllm_qwen35.sh to have been run first
#   (transformers >= 5.8.0.dev0, vLLM 0.20.1+)
#
# Usage:
#   export HF_TOKEN=<your_token>
#   nohup bash run_qwen35_sweep.sh > v3/results_vllm/qwen35_run.log 2>&1 &

set -e
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
export HF_TOKEN=${HF_TOKEN}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODELS=(
    "Qwen/Qwen3.5-0.8B"
    "Qwen/Qwen3.5-4B"
    "Qwen/Qwen3.5-9B"
)
TOTAL=${#MODELS[@]}

echo "============================================================"
echo "  Qwen3.5 Stage 1 Sweep (ARBITRARY_SINGLE + SEMANTIC_MULTI)"
echo "  Models: $TOTAL (0.8B, 4B, 9B)"
echo "  $(date)"
echo "============================================================"

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    IDX=$((i + 1))

    echo ""
    echo "[$IDX/$TOTAL] $MODEL — $(date)"
    python "$SCRIPT_DIR/stage1_sweep_vllm.py" \
        --model "$MODEL" \
        --dataset "ARBITRARY_SINGLE,SEMANTIC_MULTI" \
        --trials 100 \
        --gpu 0

    echo "[$IDX/$TOTAL] $MODEL done."

    # Clean HF cache between large models to avoid disk fill
    FREE_GB=$(df -BG /home/sagemaker-user | tail -1 | awk '{print $4}' | tr -d 'G')
    if [ "${FREE_GB:-99}" -lt 15 ] 2>/dev/null; then
        echo "  Disk low (${FREE_GB}GB), cleaning Qwen3.5 cache..."
        rm -rf "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-${MODEL##*/Qwen3.5-}"
    fi
done

echo ""
echo "============================================================"
echo "  ALL DONE — $(date)"
echo "============================================================"
