"""
Post-LoRA attention routing — run v3/attention_routing.py on the merged
(Qwen2.5-3B-Instruct + LoRA) model.

Baseline: v3/results_vllm/attention_routing/Qwen2.5-3B-Instruct__normal.json
Saves to: v3/results_vllm/attention_routing/Qwen2.5-3B-Instruct-LoRA__normal.json

Strategy: register a new entry in OPERATING_POINTS at import time pointing
hf_id at the merged dir, then call run_one_config unchanged.

Usage:
    .venv/bin/python lora_intervention/run_attention_routing_lora.py \
        --merged_path lora_intervention/checkpoints/merged --trials 50
"""
import argparse
import importlib.metadata as _imeta
import sys
from pathlib import Path

_orig = _imeta.version
def _safe(pkg):
    try:
        v = _orig(pkg); return v if v is not None else "0.0.0"
    except Exception:
        return "0.0.0"
_imeta.version = _safe

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "v3"))

import attention_routing as ar  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--merged_path", required=True)
    p.add_argument("--mode", default="normal", choices=["normal", "reversal"])
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--out_name", default="Qwen2.5-3B-Instruct-LoRA")
    p.add_argument("--no_per_head", action="store_true")
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    merged = str(Path(args.merged_path).resolve())

    base_cfg = ar.OPERATING_POINTS["Qwen2.5-3B-Instruct"]
    ar.OPERATING_POINTS[args.out_name] = {
        "hf_id": merged,
        "modes": base_cfg["modes"],  # reuse same K/N points as baseline
    }
    print(f"Registered {args.out_name} -> {merged}")
    print(f"Modes: {list(base_cfg['modes'].keys())}  (using '{args.mode}')")

    out_dir = Path(args.out_dir) if args.out_dir else ar.RESULTS_ROOT

    ar.run_one_config(
        model_short=args.out_name,
        mode=args.mode,
        n_trials=args.trials,
        out_dir=out_dir,
        save_per_head=not args.no_per_head,
    )


if __name__ == "__main__":
    main()
