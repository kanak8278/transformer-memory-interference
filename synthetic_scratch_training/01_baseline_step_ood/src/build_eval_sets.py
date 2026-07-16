"""Generate + save the frozen eval sets: IID val, IID test, held-out-step test.

Run once (deterministic given SEED). Output: ../data/{iid_val,iid_test,heldout_test}.pt
Each file: {"input_ids": LongTensor[num_examples] of variable-length lists (saved as
list of tensors, grouped by (k,n) length), "meta": list of dicts}.
Since examples within a (k,n) share length exactly, we group by (k,n) for storage.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import data_gen
from grid import all_triples, select_held_out_cells

SEED = 42
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

IID_VAL_PER_TRIPLE = 50
IID_TEST_PER_TRIPLE = 200
HELDOUT_TEST_PER_TRIPLE = 500


def save_split(name, examples):
    """Group by (k,n) so each group is a fixed-length tensor; save as a dict of groups."""
    groups = {}
    for full_ids, meta in examples:
        key = (meta["k"], meta["n"])
        groups.setdefault(key, {"input_ids": [], "meta": []})
        groups[key]["input_ids"].append(full_ids)
        groups[key]["meta"].append(meta)

    packed = {}
    for (k, n), g in groups.items():
        packed[(k, n)] = {
            "input_ids": torch.tensor(g["input_ids"], dtype=torch.long),
            "meta": g["meta"],
        }

    path = os.path.join(DATA_DIR, f"{name}.pt")
    torch.save(packed, path)
    total = sum(v["input_ids"].shape[0] for v in packed.values())
    print(f"{name}: {total} examples across {len(packed)} (k,n) groups -> {path}")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    held_out_cells = select_held_out_cells(seed=SEED)
    print(f"held-out (n,step) cells ({len(held_out_cells)}): {sorted(held_out_cells)}")

    allowed_triples = all_triples(held_out_cells, only_held_out=False)
    held_out_triples = all_triples(held_out_cells, only_held_out=True)
    print(f"allowed triples: {len(allowed_triples)}, held-out triples: {len(held_out_triples)}")

    iid_val = data_gen.build_frozen_eval_set(SEED + 1, allowed_triples, IID_VAL_PER_TRIPLE)
    save_split("iid_val", iid_val)

    iid_test = data_gen.build_frozen_eval_set(SEED + 2, allowed_triples, IID_TEST_PER_TRIPLE)
    save_split("iid_test", iid_test)

    heldout_test = data_gen.build_frozen_eval_set(SEED + 3, held_out_triples, HELDOUT_TEST_PER_TRIPLE)
    save_split("heldout_test", heldout_test)

    # also persist the held-out cell set itself, needed by train.py to exclude them from streaming
    torch.save({"held_out_cells": held_out_cells, "seed": SEED}, os.path.join(DATA_DIR, "held_out_cells.pt"))


if __name__ == "__main__":
    main()
