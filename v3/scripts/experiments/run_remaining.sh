#!/bin/bash
# Run all remaining experiments:
# 1. Gemma 270m + 4b: Stage 3 Causal (need float32 Stage 2 first)
# 2. Gemma 4b: Jacobian (untrained + pretrained)
# 3. SmolLM2-1.7B: remaining 26 training dynamics checkpoints
#
# Each model loaded and unloaded individually. Cache cleaned between runs.

set -e
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
export HF_TOKEN=${HF_TOKEN}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  Remaining Experiments"
echo "  $(date)"
echo "============================================================"

# ── 1. GEMMA 4b JACOBIAN ─────────────────────────────────────────
echo ""
echo "=== JACOBIAN: Gemma 4b ==="

for MODE in untrained pretrained; do
    RESULT="$REPO_ROOT/v3/results_vllm/jacobian/jacobian_gemma-3-4b-it_${MODE}.json"
    if [ -f "$RESULT" ]; then
        echo "[gemma-4b $MODE] SKIP (done)"
        continue
    fi
    echo "[gemma-4b $MODE] Running... $(date)"
    FLAG=""
    [ "$MODE" = "untrained" ] && FLAG="--untrained"
    python jacobian_at_init.py --model google/gemma-3-4b-it --seq-len 50 --n-samples 20 $FLAG 2>&1 | tail -10
    # Copy to vLLM results
    SRC="$SCRIPT_DIR/results/jacobian/jacobian_gemma-3-4b-it_${MODE}.json"
    [ -f "$SRC" ] && cp "$SRC" "$RESULT" && echo "  Saved: $RESULT"
done

# ── 2. GEMMA 270m + 4b STAGE 2 (float32) + STAGE 3 ──────────────
echo ""
echo "=== STAGE 2+3: Gemma 270m (float32) ==="

python -c "
import sys, os, torch, gc
sys.path.insert(0, '$REPO_ROOT')
sys.path.insert(0, '$(dirname $SCRIPT_DIR)')
sys.path.insert(0, '$SCRIPT_DIR')
from pathlib import Path

model_name = 'google/gemma-3-270m-it'
from mechanistic_probing_v2.core.model_loader import model_short_name
m_short = model_short_name(model_name)

s2_dir = Path('$REPO_ROOT/v3/results_vllm/logit_lens') / m_short
s3_dir = Path('$REPO_ROOT/v3/results_vllm/causal') / m_short
s3_dir.mkdir(parents=True, exist_ok=True)

# Check if Stage 3 already done
if list(s3_dir.glob('stage3_causal_*.json')):
    print('SKIP: Stage 3 already done')
else:
    # Need clean Stage 2 first
    from mechanistic_probing_v2.core.model_loader import load_model
    print('Loading gemma-3-270m-it in float32...')
    model, tokenizer, info = load_model(model_name, gpu_idx=0, dtype=torch.float32)

    # Run Stage 2
    import stage2_logit_lens
    stage2_logit_lens.get_save_dir = lambda mn: s2_dir
    config = {'model': model_name, 'points': [(2,5), (2,7), (2,10)], 'trials': 100, 'gpu': 0}
    stage2_logit_lens.run_stage2(config, model=model, tokenizer=tokenizer, info=info)

    # Run Stage 3
    s2_files = sorted(s2_dir.glob('stage2_logit_lens_*.json'))
    if s2_files:
        import stage3_causal
        stage3_causal.get_save_dir = lambda mn: s3_dir
        config3 = {'model': model_name, 'stage2_path': str(s2_files[-1]),
                    'trials': 50, 'top_k': 20, 'gpu': 0, 'experiments': ['3A', '3B', '3C']}
        stage3_causal.run_stage3(config3, model=model, tokenizer=tokenizer, info=info)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
" 2>&1 | tail -15

echo ""
echo "=== STAGE 2+3: Gemma 4b (float32) ==="

python -c "
import sys, os, torch, gc
sys.path.insert(0, '$REPO_ROOT')
sys.path.insert(0, '$(dirname $SCRIPT_DIR)')
sys.path.insert(0, '$SCRIPT_DIR')
from pathlib import Path

model_name = 'google/gemma-3-4b-it'
from mechanistic_probing_v2.core.model_loader import model_short_name
m_short = model_short_name(model_name)

s2_dir = Path('$REPO_ROOT/v3/results_vllm/logit_lens') / m_short
s3_dir = Path('$REPO_ROOT/v3/results_vllm/causal') / m_short
s3_dir.mkdir(parents=True, exist_ok=True)

if list(s3_dir.glob('stage3_causal_*.json')):
    print('SKIP: Stage 3 already done')
else:
    from mechanistic_probing_v2.core.model_loader import load_model
    print('Loading gemma-3-4b-it in float32 (~16GB VRAM)...')
    model, tokenizer, info = load_model(model_name, gpu_idx=0, dtype=torch.float32)

    import stage2_logit_lens
    stage2_logit_lens.get_save_dir = lambda mn: s2_dir
    config = {'model': model_name, 'points': [(2,5), (2,10), (2,50)], 'trials': 100, 'gpu': 0}
    stage2_logit_lens.run_stage2(config, model=model, tokenizer=tokenizer, info=info)

    s2_files = sorted(s2_dir.glob('stage2_logit_lens_*.json'))
    if s2_files:
        import stage3_causal
        stage3_causal.get_save_dir = lambda mn: s3_dir
        config3 = {'model': model_name, 'stage2_path': str(s2_files[-1]),
                    'trials': 50, 'top_k': 20, 'gpu': 0, 'experiments': ['3A', '3B', '3C']}
        stage3_causal.run_stage3(config3, model=model, tokenizer=tokenizer, info=info)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
" 2>&1 | tail -15

# ── 3. SmolLM2-1.7B REMAINING CHECKPOINTS ────────────────────────
echo ""
echo "=== SmolLM2-1.7B: Remaining Training Dynamics ==="

# All 41 checkpoints — script auto-skips completed ones
ALL_STEPS=(
    step-125000 step-250000 step-375000 step-500000
    step-625000 step-750000 step-875000 step-1000000
    step-1125000 step-1250000 step-1375000 step-1500000
    step-1625000 step-1750000 step-1875000 step-2000000
    step-2125000 step-2250000 step-2375000 step-2500000
    step-2625000 step-2750000 step-2875000 step-3000000
    step-3125000 step-3250000 step-3375000 step-3500000
    step-3625000 step-3750000 step-3875000 step-4000000
    step-4125000 step-4250000 step-4375000 step-4500000
    step-4625000 step-4750000 step-4875000 step-5000000
    step-5125000
)

RESULTS_DIR="$REPO_ROOT/v3/results_vllm/training_dynamics"
TOTAL=${#ALL_STEPS[@]}

for i in "${!ALL_STEPS[@]}"; do
    CKPT="${ALL_STEPS[$i]}"
    IDX=$((i + 1))
    RESULT_FILE="${RESULTS_DIR}/${CKPT}.json"

    # Skip if good result exists
    if [ -f "$RESULT_FILE" ]; then
        if python -c "import json; d=json.load(open('$RESULT_FILE')); assert 'cells' in d and len(d['cells'])>0" 2>/dev/null; then
            echo "[$IDX/$TOTAL] SKIP $CKPT (done)"
            continue
        else
            rm -f "$RESULT_FILE" "${RESULT_FILE%.json}_trials.json"
        fi
    fi

    echo ""
    echo "[$IDX/$TOTAL] Running $CKPT... $(date)"
    python training_dynamics_sweep.py --gpu 0 --trials 100 --single-checkpoint "$CKPT" 2>&1 | tail -5

    if [ -f "$RESULT_FILE" ]; then
        SUMMARY=$(python -c "import json; d=json.load(open('$RESULT_FILE')); s=d.get('summary',{}); print(f'RI={s.get(\"mean_ri\",0):.1%} PI={s.get(\"mean_pi\",0):.1%} gap={s.get(\"mean_gap\",0):+.1%}')" 2>/dev/null)
        echo "  OK: $SUMMARY"
    else
        echo "  FAILED"
    fi

    # Clean HF cache when disk < 10GB free
    FREE_GB=$(df -BG /home/sagemaker-user | tail -1 | awk '{print $4}' | tr -d 'G')
    if [ "$FREE_GB" -lt 10 ] 2>/dev/null; then
        echo "  Cleaning cache (${FREE_GB}GB free)..."
        rm -rf "$HOME/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-1.7B-intermediate-checkpoints"
    fi
done

echo ""
echo "============================================================"
echo "  ALL DONE — $(date)"
echo "============================================================"
