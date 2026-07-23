#!/usr/bin/env bash
# On-demand training status. Run anytime:  bash synthetic_scratch_training/status.sh
# Reads only the on-disk logs/results (source of truth), so it works even if the
# background watcher died. Never depends on any running helper.
ROOT="/rnd_ai_datasets1/projects/raka6003/transformer-memory-interference/synthetic_scratch_training"

echo "===================== GPUs ====================="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null

echo "=============== training processes ============="
pgrep -af "train.py --gpu" | grep -oE "gpu [0-9] --schedule [a-z]+ --tag [a-z0-9_]+" || echo "  (none running)"

for d in "$ROOT"/05_gpt2_scratch_h100 "$ROOT"/06_first_last_only "$ROOT"/07_gpt2_pretrained_first_last; do
  for log in "$d"/logs_*.log; do
    [ -f "$log" ] || continue
    name=$(basename "$log" .log | sed 's/^logs_//')
    echo "----------------- $name -----------------"
    grep -E "step [0-9]+/" "$log" | tail -1 | sed 's/^/  /'
    echo "  evals:"; grep -oE "val@[0-9]+ acc=[0-9.]+" "$log" | tail -8 | sed 's/^/    /'
    grep -q "ALL DONE" "$log" && grep -E "FINAL|ZERO-SHOT" "$log" | sed 's/^/  /'
    grep -iE "error|traceback|out of memory" "$log" >/dev/null 2>&1 && echo "  !! errors present in log — inspect $log"
  done
done
