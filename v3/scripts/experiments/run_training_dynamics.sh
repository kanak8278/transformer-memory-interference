#!/bin/bash
# Run training dynamics sweep — each checkpoint as a SEPARATE process
# to avoid vLLM GPU memory leak between model loads.
#
# Usage:
#   bash run_training_dynamics.sh
#   nohup bash run_training_dynamics.sh > v3/results_vllm/training_dynamics_run.log 2>&1 &

set -e
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
export HF_TOKEN=${HF_TOKEN}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/../../results_vllm/training_dynamics"
SWEEP_SCRIPT="${SCRIPT_DIR}/training_dynamics_sweep.py"

mkdir -p "$RESULTS_DIR"

# SmolLM2-1.7B checkpoints (sorted by step)
CHECKPOINTS=(
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
    final
)

TOTAL=${#CHECKPOINTS[@]}
echo "============================================================"
echo "  Training Dynamics: SmolLM2-1.7B ($TOTAL checkpoints)"
echo "  Each checkpoint runs as separate process (clean GPU)"
echo "  Results: $RESULTS_DIR"
echo "============================================================"

for i in "${!CHECKPOINTS[@]}"; do
    CKPT="${CHECKPOINTS[$i]}"
    IDX=$((i + 1))
    RESULT_FILE="${RESULTS_DIR}/${CKPT}.json"

    # Skip if already done
    if [ -f "$RESULT_FILE" ]; then
        # Check if it's a real result (has "cells" key) or an error
        if python -c "import json; d=json.load(open('$RESULT_FILE')); assert 'cells' in d" 2>/dev/null; then
            echo "[$IDX/$TOTAL] SKIP $CKPT (already done)"
            continue
        else
            rm -f "$RESULT_FILE" "${RESULT_FILE%.json}_trials.json"
        fi
    fi

    echo ""
    echo "[$IDX/$TOTAL] Running $CKPT..."

    # Run as separate process — GPU fully released when process exits
    if [ "$CKPT" = "final" ]; then
        python "$SWEEP_SCRIPT" --gpu 0 --trials 100 --single-checkpoint "$CKPT" 2>&1 | tail -5
    else
        python "$SWEEP_SCRIPT" --gpu 0 --trials 100 --single-checkpoint "$CKPT" 2>&1 | tail -5
    fi

    # Check result
    if [ -f "$RESULT_FILE" ]; then
        SUMMARY=$(python -c "import json; d=json.load(open('$RESULT_FILE')); s=d.get('summary',{}); print(f'RI={s.get(\"mean_ri\",0):.1%} PI={s.get(\"mean_pi\",0):.1%} gap={s.get(\"mean_gap\",0):+.1%}')" 2>/dev/null)
        echo "  OK: $SUMMARY"
    else
        echo "  FAILED"
    fi

    echo "  $(date)"
done

echo ""
echo "============================================================"
echo "  ALL DONE — $(date)"
echo "============================================================"

# Generate combined summary
python -c "
import json, os
from pathlib import Path

results_dir = Path('$RESULTS_DIR')
summary = {'checkpoints': {}}

for f in sorted(results_dir.glob('step-*.json')) + list(results_dir.glob('final.json')):
    name = f.stem
    d = json.load(open(f))
    if 'summary' in d:
        summary['checkpoints'][name] = d['summary']
        summary['checkpoints'][name]['step'] = d.get('step')

with open(results_dir / 'training_dynamics_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print('Training trajectory:')
for name in sorted(summary['checkpoints'], key=lambda x: int(x.split('-')[1]) if x.startswith('step-') else 99999999):
    s = summary['checkpoints'][name]
    print(f'  {name:>15} RI={s[\"mean_ri\"]:.1%} PI={s[\"mean_pi\"]:.1%} gap={s[\"mean_gap\"]:+.1%}')
"
