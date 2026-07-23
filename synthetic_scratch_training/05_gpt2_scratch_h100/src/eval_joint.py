"""Re-evaluate a checkpoint and save the JOINT (k, n, step) accuracy that the
aggregated final_eval.json marginalizes away. Needed to show how accuracy varies
with step position across the different (K, N) configs.

Usage: python eval_joint.py <ckpt.pt> <out.json> [--gpu N]
"""

import argparse
import json

import torch

from data_gen import build_frozen_eval_set, pack_split
from eval_utils import evaluate_split
from grid import all_triples, select_held_out_cells
from model_gpt2 import GPT2FromScratch

SEED = 42


def joint(results):
    """flat per-example results -> {'k,n,step': {'acc':..,'n':..,'held':bool}}."""
    buckets = {}
    for r in results:
        key = f"{r['k']},{r['n']},{r['query_step']}"
        b = buckets.setdefault(key, {"correct": 0, "n": 0})
        b["correct"] += int(r["correct"])
        b["n"] += 1
    return {k: {"acc": v["correct"] / v["n"], "n": v["n"]} for k, v in buckets.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=256)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    held_out_cells = select_held_out_cells(seed=SEED)
    allowed_triples = all_triples(held_out_cells, only_held_out=False)
    held_triples = all_triples(held_out_cells, only_held_out=True)

    model = GPT2FromScratch(dropout=0.0).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    print(f"loaded {args.ckpt} (step {ck.get('step')}, val {ck.get('val_acc')})", flush=True)

    iid = pack_split(build_frozen_eval_set(SEED + 2, allowed_triples, 200))
    heldout = pack_split(build_frozen_eval_set(SEED + 3, held_triples, 500))
    iid_res = evaluate_split(model, iid, device, args.chunk)
    held_res = evaluate_split(model, heldout, device, args.chunk)

    out = {
        "ckpt_step": ck.get("step"),
        "held_out_cells": sorted(list(held_out_cells)),
        "iid": joint(iid_res),
        "heldout": joint(held_res),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
