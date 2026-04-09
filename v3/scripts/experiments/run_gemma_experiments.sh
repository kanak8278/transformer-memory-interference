#!/bin/bash
# Run Probing + Stage 2 + Stage 3 for all Gemma models
#
# - Gemma 270m: Probing (TL), Stage 2/3 retry in float32
# - Gemma 1b: Already done (probing + S2 + S3) — skip
# - Gemma 4b: Probing (TL), Stage 2/3 retry in float32

set -e
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
export HF_TOKEN=${HF_TOKEN}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

RESULTS_PROBING="$REPO_ROOT/v3/results_vllm/probing"
RESULTS_LOGIT="$REPO_ROOT/v3/results_vllm/logit_lens"
RESULTS_CAUSAL="$REPO_ROOT/v3/results_vllm/causal"

mkdir -p "$RESULTS_PROBING" "$RESULTS_LOGIT" "$RESULTS_CAUSAL"

echo "============================================================"
echo "  Gemma Experiments: Probing + Stage 2 + Stage 3"
echo "  $(date)"
echo "============================================================"

# ── GEMMA 270m ────────────────────────────────────────────────────
echo ""
echo "=== google/gemma-3-270m-it ==="

# Probing (if not done)
if [ ! -f "$RESULTS_PROBING/probing_gemma-3-270m-it_2k_5u.json" ]; then
    echo "[270m] Running probing..."
    python probing_classifier.py --model google/gemma-3-270m-it --point "2,5" --trials 200 --gpu 0 2>&1 | tail -10
    cp results/probing/probing_gemma-3-270m-it_2k_5u.json "$RESULTS_PROBING/" 2>/dev/null || true
else
    echo "[270m] SKIP probing (done)"
fi

# Stage 2 in float32 (redo — previous was NaN in float16)
echo "[270m] Running Stage 2 (float32)..."
python -c "
import sys, os
sys.path.insert(0, '$(dirname $SCRIPT_DIR)')
sys.path.insert(0, '$REPO_ROOT')
os.environ['LD_LIBRARY_PATH'] = '/opt/conda/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

from stage2_logit_lens import run_stage2, get_save_dir
from mechanistic_probing_v2.core.model_loader import load_model, model_short_name
from pathlib import Path
import torch

model_name = 'google/gemma-3-270m-it'
save_dir = Path('$RESULTS_LOGIT') / model_short_name(model_name)

# Check if clean result exists
existing = list(save_dir.glob('stage2_logit_lens_*_f32.json')) if save_dir.exists() else []
if existing:
    print('SKIP: float32 result exists')
else:
    print('Loading in float32...')
    model, tokenizer, info = load_model(model_name, gpu_idx=0, dtype=torch.float32)
    config = {'model': model_name, 'points': [(2,5), (2,7), (2,10)], 'trials': 100, 'gpu': 0}

    import stage2_logit_lens
    stage2_logit_lens.get_save_dir = lambda mn: save_dir
    results = run_stage2(config, model=model, tokenizer=tokenizer, info=info)
" 2>&1 | tail -10

# Stage 3
echo "[270m] Running Stage 3..."
python -c "
import sys, os, json, glob
sys.path.insert(0, '$(dirname $SCRIPT_DIR)')
sys.path.insert(0, '$REPO_ROOT')

from stage3_causal import run_stage3
from mechanistic_probing_v2.core.model_loader import load_model, model_short_name
from pathlib import Path
import torch

model_name = 'google/gemma-3-270m-it'
save_dir = Path('$RESULTS_CAUSAL') / model_short_name(model_name)

existing = list(save_dir.glob('stage3_causal_*.json')) if save_dir.exists() else []
if existing:
    print('SKIP: Stage 3 exists')
else:
    s2_dir = Path('$RESULTS_LOGIT') / model_short_name(model_name)
    s2_files = sorted(s2_dir.glob('stage2_logit_lens_*.json'))
    if not s2_files:
        print('SKIP: no Stage 2 results')
    else:
        model, tokenizer, info = load_model(model_name, gpu_idx=0, dtype=torch.float32)
        config = {'model': model_name, 'stage2_path': str(s2_files[-1]), 'trials': 50, 'top_k': 20, 'gpu': 0, 'experiments': ['3A', '3B', '3C']}
        import stage3_causal
        stage3_causal.get_save_dir = lambda mn: save_dir
        run_stage3(config, model=model, tokenizer=tokenizer, info=info)
" 2>&1 | tail -10

# ── GEMMA 4b ─────────────────────────────────────────────────────
echo ""
echo "=== google/gemma-3-4b-it ==="

# Probing
if [ ! -f "$RESULTS_PROBING/probing_gemma-3-4b-it_2k_5u.json" ]; then
    echo "[4b] Running probing..."
    python probing_classifier.py --model google/gemma-3-4b-it --point "2,5" --trials 200 --gpu 0 2>&1 | tail -10
    cp results/probing/probing_gemma-3-4b-it_2k_5u.json "$RESULTS_PROBING/" 2>/dev/null || true
else
    echo "[4b] SKIP probing (done)"
fi

# Stage 2 in float32 (redo — previous was 82% NaN in float16)
echo "[4b] Running Stage 2 (float32)..."
python -c "
import sys, os
sys.path.insert(0, '$(dirname $SCRIPT_DIR)')
sys.path.insert(0, '$REPO_ROOT')

from stage2_logit_lens import run_stage2
from mechanistic_probing_v2.core.model_loader import load_model, model_short_name
from pathlib import Path
import torch

model_name = 'google/gemma-3-4b-it'
save_dir = Path('$RESULTS_LOGIT') / model_short_name(model_name)

existing = list(save_dir.glob('stage2_logit_lens_*_f32.json')) if save_dir.exists() else []
if existing:
    print('SKIP: float32 result exists')
else:
    print('Loading in float32 (16GB VRAM)...')
    model, tokenizer, info = load_model(model_name, gpu_idx=0, dtype=torch.float32)
    config = {'model': model_name, 'points': [(2,5), (2,10), (2,50)], 'trials': 100, 'gpu': 0}

    import stage2_logit_lens
    stage2_logit_lens.get_save_dir = lambda mn: save_dir
    results = run_stage2(config, model=model, tokenizer=tokenizer, info=info)
" 2>&1 | tail -10

# Stage 3
echo "[4b] Running Stage 3..."
python -c "
import sys, os, json
sys.path.insert(0, '$(dirname $SCRIPT_DIR)')
sys.path.insert(0, '$REPO_ROOT')

from stage3_causal import run_stage3
from mechanistic_probing_v2.core.model_loader import load_model, model_short_name
from pathlib import Path
import torch

model_name = 'google/gemma-3-4b-it'
save_dir = Path('$RESULTS_CAUSAL') / model_short_name(model_name)

existing = list(save_dir.glob('stage3_causal_*.json')) if save_dir.exists() else []
if existing:
    print('SKIP: Stage 3 exists')
else:
    s2_dir = Path('$RESULTS_LOGIT') / model_short_name(model_name)
    s2_files = sorted(s2_dir.glob('stage2_logit_lens_*.json'))
    if not s2_files:
        print('SKIP: no Stage 2 results')
    else:
        model, tokenizer, info = load_model(model_name, gpu_idx=0, dtype=torch.float32)
        config = {'model': model_name, 'stage2_path': str(s2_files[-1]), 'trials': 50, 'top_k': 20, 'gpu': 0, 'experiments': ['3A', '3B', '3C']}
        import stage3_causal
        stage3_causal.get_save_dir = lambda mn: save_dir
        run_stage3(config, model=model, tokenizer=tokenizer, info=info)
" 2>&1 | tail -10

echo ""
echo "============================================================"
echo "  ALL DONE — $(date)"
echo "============================================================"
