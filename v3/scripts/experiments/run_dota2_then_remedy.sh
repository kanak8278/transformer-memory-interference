#!/bin/bash
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
export HF_TOKEN=${HF_TOKEN}  # Set before running

echo "Waiting for Dota2 narrative to finish..."
while ps aux | grep -v grep | grep -q "narrative_dota2_vllm"; do
    DONE=$(find v3/results_vllm/narrative_dota2 -name "narrative_dota2_*.json" -not -name "*trials*" 2>/dev/null | wc -l)
    echo "  $(date +%H:%M): $DONE/7 models done"
    sleep 120
done

echo ""
echo "Dota2 done! Starting Remedy experiment..."
python v3/scripts/experiments/remedy_vllm.py --all --trials 100 --gpu 0
