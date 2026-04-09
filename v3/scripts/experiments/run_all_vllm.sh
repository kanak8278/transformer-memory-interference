#!/bin/bash
# Run all 11 models on the unified grid using vLLM.
#
# Usage:
#   bash run_all_vllm.sh                   # defaults: gpu=0, 100 trials
#   bash run_all_vllm.sh --gpu 1           # use GPU 1
#   bash run_all_vllm.sh --trials 50       # fewer trials for testing
#   nohup bash run_all_vllm.sh &           # run in background

set -e

# Fix GLIBCXX_3.4.31 not found
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python "${SCRIPT_DIR}/stage1_sweep_vllm.py" --all "$@"
