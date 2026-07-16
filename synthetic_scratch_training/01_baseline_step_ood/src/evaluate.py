"""Standalone final evaluation: run a checkpoint on iid_test + heldout_test, aggregate
by cell / step-role / step-index / duplicate-flag, and report the held-out generalization
gap (accuracy on held-out (n,step) cells vs. accuracy on the same step values when they
were NOT held out, i.e. trained elsewhere in the grid).
"""

import argparse
import json
import os

import torch

import eval_utils
from grid import select_held_out_cells
from model import TinyTransformer

SRC_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(SRC_DIR, "..", "data")


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = TinyTransformer(n_layers=a["n_layers"], d_model=a["d_model"], n_heads=a["n_heads"], d_ff=a["d_ff"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model, ckpt = load_model(args.ckpt, device)
    print(f"loaded {args.ckpt} (trained to step {ckpt['step']}, train-time val_acc={ckpt['val_acc']:.4f})")

    iid_test = torch.load(os.path.join(DATA_DIR, "iid_test.pt"), weights_only=False)
    heldout_test = torch.load(os.path.join(DATA_DIR, "heldout_test.pt"), weights_only=False)

    iid_results = eval_utils.evaluate_split(model, iid_test, device)
    heldout_results = eval_utils.evaluate_split(model, heldout_test, device)

    iid_agg = eval_utils.aggregate_results(iid_results)
    heldout_agg = eval_utils.aggregate_results(heldout_results)

    # generalization gap: for each held-out step value s, compare heldout accuracy at
    # step=s against the trained (iid) accuracy at that same step value in other N's.
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
    print(f"generalization gap by held-out step: {json.dumps(gap_report, indent=2)}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
