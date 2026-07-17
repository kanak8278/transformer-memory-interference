"""Standalone final evaluation: run a checkpoint on iid_test + heldout_test, aggregate
by cell / step-role / step-index / duplicate-flag, and report the held-out generalization
gap (accuracy on held-out (n,step) cells vs. accuracy on the same step values when they
were NOT held out, i.e. trained elsewhere in the grid).

Adapted from experiment 01's evaluate.py — only model loading differs (fixed pretrained
GPT-2 architecture, no architecture args to reconstruct from the checkpoint) and eval
uses a small chunk_size (attention memory blows up on MPS at exp01's default 512 —
see eval_utils.evaluate_split's docstring).
"""

import argparse
import json
import os
import sys

import torch

SRC_DIR = os.path.dirname(__file__)
EXP01_SRC = os.path.join(SRC_DIR, "..", "..", "01_baseline_step_ood", "src")
sys.path.insert(0, os.path.abspath(EXP01_SRC))
sys.path.insert(0, SRC_DIR)

import eval_utils
from grid import select_held_out_cells
from model_gpt2_pretrained import GPT2Pretrained

EXP01_DATA_DIR = os.path.join(EXP01_SRC, "..", "data")


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GPT2Pretrained().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--eval_chunk_size", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model, ckpt = load_model(args.ckpt, device)
    print(f"loaded {args.ckpt} (trained to step {ckpt['step']}, train-time val_acc={ckpt['val_acc']:.4f})")

    iid_test = torch.load(os.path.join(EXP01_DATA_DIR, "iid_test.pt"), weights_only=False)
    heldout_test = torch.load(os.path.join(EXP01_DATA_DIR, "heldout_test.pt"), weights_only=False)

    iid_results = eval_utils.evaluate_split(model, iid_test, device, chunk_size=args.eval_chunk_size)
    heldout_results = eval_utils.evaluate_split(model, heldout_test, device, chunk_size=args.eval_chunk_size)

    iid_agg = eval_utils.aggregate_results(iid_results)
    heldout_agg = eval_utils.aggregate_results(heldout_results)

    held_out_cells = select_held_out_cells(seed=42)
    steps_held_out = sorted({s for (_, s) in held_out_cells})
    gap_report = {}
    for s in steps_held_out:
        gap_report[str(s)] = {
            "trained_elsewhere_acc": iid_agg["by_step"].get(str(s), {}).get("acc"),
            "held_out_acc": heldout_agg["by_step"].get(str(s), {}).get("acc"),
        }

    report = {
        "ckpt": args.ckpt,
        "trained_to_step": ckpt["step"],
        "iid_test": iid_agg,
        "heldout_test": heldout_agg,
        "held_out_steps": steps_held_out,
        "generalization_gap_by_step": gap_report,
    }

    out_path = args.out or os.path.join(os.path.dirname(args.ckpt), "..", "..", "results", "final_eval.json")
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"iid_test overall: {iid_agg['overall']:.4f} (n={iid_agg['n']})")
    print(f"heldout_test overall: {heldout_agg['overall']:.4f} (n={heldout_agg['n']})")
    print(f"by_role (iid_test): { {k: v['acc'] for k, v in iid_agg['by_role'].items()} }")
    print(f"by_duplicate (iid_test): { {k: v['acc'] for k, v in iid_agg['by_duplicate'].items()} }")
    print(f"by_duplicate (heldout_test): { {k: v['acc'] for k, v in heldout_agg['by_duplicate'].items()} }")
    print(f"generalization gap by held-out step: {json.dumps(gap_report, indent=2)}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
