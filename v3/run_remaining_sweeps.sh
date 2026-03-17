#!/bin/bash
# Run 200-trial sweeps for models that still need them
# Execute from v3/ directory
# Each model uses MPS so they must run sequentially

PYTHON="/Users/kanak.raj/workspace/hobby/research_work_ri/.venv/bin/python"
export PYTHONUNBUFFERED=1

echo "=== Starting 200-trial sweep pipeline ==="
echo "Time: $(date)"

# TinyLlama
echo ""
echo "=== TinyLlama-1.1B-Chat ==="
$PYTHON stage1_sweep.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --key-levels 2 3 \
    --update-levels 5 7 10 \
    --trials 200

# StableLM
echo ""
echo "=== StableLM-2-1.6B-Chat ==="
$PYTHON stage1_sweep.py \
    --model stabilityai/stablelm-2-1_6b-chat \
    --key-levels 2 3 \
    --update-levels 5 7 10 \
    --trials 200

echo ""
echo "=== All sweeps complete ==="
echo "Time: $(date)"
