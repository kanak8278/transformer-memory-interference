#!/bin/bash
# Training dynamics for SmolLM3-3B: 40 checkpoints (pretraining → SFT → alignment)
# Each checkpoint as a separate process for clean GPU.
#
# Usage:
#   nohup bash run_training_dynamics_smollm3.sh > v3/results_vllm/training_dynamics_smollm3_run.log 2>&1 &

set -e
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
export HF_TOKEN=${HF_TOKEN}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/../../results_vllm/training_dynamics_smollm3"
SWEEP_SCRIPT="${SCRIPT_DIR}/training_dynamics_smollm3.py"

mkdir -p "$RESULTS_DIR"

CHECKPOINTS=(
    # Stage 1 pretraining (25)
    stage1-step-40000 stage1-step-160000 stage1-step-320000
    stage1-step-440000 stage1-step-600000 stage1-step-720000
    stage1-step-880000 stage1-step-1000000 stage1-step-1160000
    stage1-step-1280000 stage1-step-1440000 stage1-step-1560000
    stage1-step-1720000 stage1-step-1880000 stage1-step-2000000
    stage1-step-2160000 stage1-step-2280000 stage1-step-2440000
    stage1-step-2560000 stage1-step-2720000 stage1-step-2840000
    stage1-step-3000000 stage1-step-3120000 stage1-step-3280000
    stage1-step-3440000
    # Stage 2 (5)
    stage2-step-3480000 stage2-step-3640000 stage2-step-3840000
    stage2-step-4000000 stage2-step-4200000
    # Stage 3 (5)
    stage3-step-4240000 stage3-step-4360000 stage3-step-4480000
    stage3-step-4600000 stage3-step-4720000
    # Instruction tuning (4)
    it-SFT it-mid-training it-soup-APO it-LC-expert
    # Final
    final
)

TOTAL=${#CHECKPOINTS[@]}
echo "============================================================"
echo "  Training Dynamics: SmolLM3-3B ($TOTAL checkpoints)"
echo "  Grid: keys=[2,3,5,7] x updates=[2,3,5,7,10,15,20,30]"
echo "  Results: $RESULTS_DIR"
echo "============================================================"

for i in "${!CHECKPOINTS[@]}"; do
    CKPT="${CHECKPOINTS[$i]}"
    IDX=$((i + 1))
    RESULT_FILE="${RESULTS_DIR}/${CKPT}.json"

    if [ -f "$RESULT_FILE" ]; then
        if python -c "import json; d=json.load(open('$RESULT_FILE')); assert 'cells' in d and len(d['cells']) > 0" 2>/dev/null; then
            echo "[$IDX/$TOTAL] SKIP $CKPT (already done)"
            continue
        else
            rm -f "$RESULT_FILE" "${RESULT_FILE%.json}_trials.json"
        fi
    fi

    echo ""
    echo "[$IDX/$TOTAL] Running $CKPT... $(date)"
    python "$SWEEP_SCRIPT" --gpu 0 --trials 100 --single-checkpoint "$CKPT" 2>&1 | tail -5

    if [ -f "$RESULT_FILE" ]; then
        SUMMARY=$(python -c "import json; d=json.load(open('$RESULT_FILE')); s=d.get('summary',{}); print(f'RI={s.get(\"mean_ri\",0):.1%} PI={s.get(\"mean_pi\",0):.1%} gap={s.get(\"mean_gap\",0):+.1%}')" 2>/dev/null)
        echo "  OK: $SUMMARY"
    else
        echo "  FAILED"
    fi

    # Clean HF cache only when disk is critically low (<10GB free)
    FREE_GB=$(df -BG /home/sagemaker-user | tail -1 | awk '{print $4}' | tr -d 'G')
    if [ "$FREE_GB" -lt 10 ] 2>/dev/null; then
        CACHE_DIR="$HOME/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM3-3B-checkpoints"
        if [ -d "$CACHE_DIR" ]; then
            echo "  Disk low (${FREE_GB}GB), cleaning checkpoint cache..."
            rm -rf "$CACHE_DIR"
        fi
    fi
done

echo ""
echo "============================================================"
echo "  ALL DONE — $(date)"
echo "============================================================"
