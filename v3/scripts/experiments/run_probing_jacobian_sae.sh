#!/bin/bash
# Run Probing Classifiers, Jacobian at Init, and SAE Features
# for all Qwen and Gemma series models.
#
# Usage:
#   nohup bash run_probing_jacobian_sae.sh > v3/results_vllm/probing_jacobian_sae.log 2>&1 &

set -e
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
export HF_TOKEN=${HF_TOKEN}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_PROBING="${SCRIPT_DIR}/../../results_vllm/probing"
RESULTS_JACOBIAN="${SCRIPT_DIR}/../../results_vllm/jacobian"
RESULTS_SAE="${SCRIPT_DIR}/../../results_vllm/sae"

mkdir -p "$RESULTS_PROBING" "$RESULTS_JACOBIAN" "$RESULTS_SAE"

echo "============================================================"
echo "  Probing + Jacobian + SAE Experiments"
echo "  $(date)"
echo "============================================================"

# ── PROBING CLASSIFIERS ──────────────────────────────────────────
# TransformerLens models only (Qwen, Gemma 1b, Pythia)
echo ""
echo "=== PROBING CLASSIFIERS ==="

PROBING_MODELS=(
    "Qwen/Qwen2.5-0.5B-Instruct"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-3B"
    "google/gemma-3-1b-it"
    "EleutherAI/pythia-410m"
)

for MODEL in "${PROBING_MODELS[@]}"; do
    SHORT=$(python -c "print('$MODEL'.split('/')[-1])")
    RESULT="$RESULTS_PROBING/probing_${SHORT}_2k_5u.json"

    if [ -f "$RESULT" ]; then
        echo "[$SHORT] SKIP probing (already done)"
        continue
    fi

    echo ""
    echo "[$SHORT] Running probing classifier... $(date)"
    cd "$SCRIPT_DIR"
    python probing_classifier.py --model "$MODEL" --point "2,5" --trials 200 --gpu 0 2>&1 | tail -15

    # Move result to vLLM results dir
    OLD_RESULT="$SCRIPT_DIR/results/probing/probing_${SHORT}_2k_5u.json"
    if [ -f "$OLD_RESULT" ]; then
        cp "$OLD_RESULT" "$RESULT"
        echo "  Saved: $RESULT"
    fi
    cd /home/sagemaker-user/transformer-memory-interference
done

# ── JACOBIAN AT INIT ─────────────────────────────────────────────
# HuggingFace models (all Qwen + Gemma)
echo ""
echo "=== JACOBIAN AT INITIALIZATION ==="

JACOBIAN_MODELS=(
    "Qwen/Qwen2.5-0.5B-Instruct"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-3B"
    "google/gemma-3-270m-it"
    "google/gemma-3-1b-it"
    "google/gemma-3-4b-it"
    "EleutherAI/pythia-410m"
)

for MODEL in "${JACOBIAN_MODELS[@]}"; do
    SHORT=$(python -c "print('$MODEL'.split('/')[-1])")

    for MODE in "untrained" "pretrained"; do
        RESULT="$RESULTS_JACOBIAN/jacobian_${SHORT}_${MODE}.json"
        if [ -f "$RESULT" ]; then
            echo "[$SHORT $MODE] SKIP jacobian (already done)"
            continue
        fi

        echo ""
        echo "[$SHORT $MODE] Running jacobian... $(date)"
        cd "$SCRIPT_DIR"
        FLAG=""
        if [ "$MODE" = "untrained" ]; then
            FLAG="--untrained"
        fi
        python jacobian_at_init.py --model "$MODEL" --seq-len 50 --n-samples 20 $FLAG 2>&1 | tail -15

        # Move result
        OLD_RESULT="$SCRIPT_DIR/results/jacobian/jacobian_${SHORT}_${MODE}.json"
        if [ -f "$OLD_RESULT" ]; then
            cp "$OLD_RESULT" "$RESULT"
            echo "  Saved: $RESULT"
        fi
        cd /home/sagemaker-user/transformer-memory-interference
    done
done

# ── SAE FEATURES ─────────────────────────────────────────────────
# Gemma models with Gemma Scope SAEs
echo ""
echo "=== SAE FEATURES ==="

# SAE available for gemma-3-1b-it (main), can also try gemma-3-270m-it and 4b-it
SAE_MODELS=(
    "google/gemma-3-270m-it"
    "google/gemma-3-1b-it"
    "google/gemma-3-4b-it"
)

for MODEL in "${SAE_MODELS[@]}"; do
    SHORT=$(python -c "print('$MODEL'.split('/')[-1])")
    RESULT="$RESULTS_SAE/sae_${SHORT}.json"

    if [ -f "$RESULT" ]; then
        echo "[$SHORT] SKIP SAE (already done)"
        continue
    fi

    echo ""
    echo "[$SHORT] Running SAE analysis... $(date)"
    cd "$SCRIPT_DIR"
    # The SAE script is hardcoded to gemma-3-1b-it — run as-is for 1b,
    # for others we'd need to modify the script. Run 1b first.
    if [ "$MODEL" = "google/gemma-3-1b-it" ]; then
        python gemma_scope_sae.py 2>&1 | tail -20
        OLD_RESULT="$SCRIPT_DIR/results/sae/gemma_scope_sae_analysis.json"
        if [ -f "$OLD_RESULT" ]; then
            cp "$OLD_RESULT" "$RESULT"
            echo "  Saved: $RESULT"
        fi
    else
        echo "  NOTE: SAE script hardcoded to gemma-3-1b-it, skipping $MODEL (needs script modification)"
    fi
    cd /home/sagemaker-user/transformer-memory-interference
done

echo ""
echo "============================================================"
echo "  ALL DONE — $(date)"
echo "============================================================"
