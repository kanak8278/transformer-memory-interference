"""
Generate training / validation / test data for the LoRA intervention experiment.

Outputs (all in lora_intervention/data/):
  train.jsonl       18,000 examples — 35 train categories, training cells
  val.jsonl          2,000 examples — 35 train categories, training cells (different seeds)
  test_id.jsonl     held-out categories, test cells, ARBITRARY_SINGLE
  test_ood.jsonl    SEMANTIC_MULTI, all categories, test cells

Query mix:
  40% FVQ  — "What was the first value of X?"
  40% CVQ  — "What was the last value of X?"
  20% IVQ  — "What was the k-th value of X?"  (k uniform from 2 to N-2)

Cell splits:
  Train cells  : K in {2,3,5,10} x N in {5,10,15,20}   (16 cells)
  Test cells   : K in {7,10,15,20,25,30}, N in {30,50,75}
                 plus K in {15,20,25,30}, N in {15,20}
                 (no overlap with train cells by construction)

Usage:
    cd /home/sagemaker-user/transformer-memory-interference
    python lora_intervention/data_gen.py
    python lora_intervention/data_gen.py --smoke   # quick test, 50 examples
"""

import sys
import json
import math
import random
import hashlib
import argparse
from pathlib import Path
from collections import defaultdict


def stable_seed(*parts) -> int:
    """Deterministic seed independent of PYTHONHASHSEED."""
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from mechanistic_probing_v2.core.dataset_configs import (
    get_eligible_categories,
    generate_values_for_trial,
)

SYSTEM_PROMPT = (
    "You are a precise data extraction tool. "
    "Output ONLY a single word - the exact value requested. "
    "No other text, no explanation, no punctuation."
)

TRAIN_CATEGORIES = [
    "visual art", "tools", "landform", "musical instrument", "gemstone",
    "fabric", "tree species", "cheese variety", "architectural style",
    "cloud formation", "bird species", "culinary herb", "flower species",
    "wine variety", "dance style", "pasta shape", "literary genre",
    "cooking method", "mathematical concept", "weather phenomenon",
    "ocean current", "mineral type", "coffee variety", "telescope type",
    "martial art", "sea creature", "psychology term", "chemical element",
    "dinosaur genus", "programming language", "ancient civilization",
    "bridge type", "photography technique", "boat type", "cartoon character",
]

HELD_OUT_CATEGORIES = [
    "stadium name", "surgical procedure", "constellation", "spice blend",
    "guitar type", "hat style", "painting medium", "volcano name",
    "fruit variety", "sword type", "board game",
]

TRAIN_KEYS    = [2, 3, 5, 10]
TRAIN_UPDATES = [5, 10, 15, 20]

# Test cells: K=7/10 need N>=30; K=15-30 all N including 15,20
TEST_GRID = {
    7:  [30, 50, 75],
    10: [30, 50, 75],
    15: [15, 20, 30, 50, 75],
    20: [15, 20, 30, 50, 75],
    25: [10, 15, 20, 30, 50, 75],
    30: [10, 15, 20, 30, 50, 75],
}

TRAIN_TOTAL = 20_000  # 18k train + 2k val (split at generation time)
VAL_FRAC    = 0.10

# ─── ordinal helper ───────────────────────────────────────────────────────────
_ORDINALS = {
    2: "2nd", 3: "3rd", 4: "4th", 5: "5th", 6: "6th", 7: "7th",
    8: "8th", 9: "9th", 10: "10th",
}

def ordinal(k: int) -> str:
    if k in _ORDINALS:
        return _ORDINALS[k]
    if 11 <= k % 100 <= 13:
        return f"{k}th"
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(k % 10, "th")
    return f"{k}{suffix}"


# ─── stream + prompt builders ─────────────────────────────────────────────────

def shuffle_no_consecutive(items: list, rng: random.Random) -> list:
    for _ in range(100):
        c = items.copy()
        rng.shuffle(c)
        if all(c[i]["category"] != c[i - 1]["category"] for i in range(1, len(c))):
            return c
    # greedy fallback
    result, remaining = [], items.copy()
    rng.shuffle(remaining)
    last_cat = None
    while remaining:
        valid = [i for i, x in enumerate(remaining) if x["category"] != last_cat]
        if not valid:
            result.extend(remaining)
            break
        idx = rng.choice(valid)
        result.append(remaining.pop(idx))
        last_cat = result[-1]["category"]
    return result


def build_example(
    dataset_type: str,
    categories: list,
    num_updates: int,
    condition: str,           # "FVQ", "CVQ", "IVQ"
    seed: int,
) -> dict | None:
    """
    Build one JSONL record.

    Returns None if the trial can't be constructed (e.g. pool too small).
    """
    rng = random.Random(seed)

    try:
        values_per_cat = generate_values_for_trial(
            dataset_type, categories, num_updates, rng
        )
    except ValueError:
        return None

    test_category = categories[seed % len(categories)]

    items = []
    for cat in categories:
        for val in values_per_cat[cat]:
            items.append({"category": cat, "value": val})
    items = shuffle_no_consecutive(items, rng)

    stream = "\n".join(f"{it['category']}: {it['value']}" for it in items)
    cat_values = [it["value"] for it in items if it["category"] == test_category]
    n_vals = len(cat_values)

    if condition == "FVQ":
        query_word = "first"
        label = cat_values[0]
        k_pos = 0
    elif condition == "CVQ":
        query_word = "last"
        label = cat_values[-1]
        k_pos = n_vals - 1
    else:  # IVQ
        if n_vals < 3:
            return None  # not enough values for intermediate
        k = rng.randint(2, n_vals - 1)   # 1-indexed, excludes first and last
        query_word = ordinal(k)
        label = cat_values[k - 1]        # convert to 0-indexed
        k_pos = k - 1

    user_text = (
        f"Read the following key-value stream. Each key gets updated multiple times.\n\n"
        f"{stream}\n\n"
        f"What was the {query_word} value of {test_category}?"
    )

    # Conversational format for TRL 1.x assistant_only_loss
    # "prompt" = system + user turns, "completion" = assistant turn
    prompt_msgs = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": user_text},
    ]
    completion_msgs = [
        {"role": "assistant", "content": label},
    ]

    return {
        "prompt":            prompt_msgs,
        "completion":        completion_msgs,
        "label":             label,
        "condition":         condition,
        "num_keys":          len(categories),
        "num_updates":       num_updates,
        "query_position_idx": k_pos,
        "seed":              seed,
        "test_category":     test_category,
        "dataset_type":      dataset_type,
    }


# ─── dataset generators ───────────────────────────────────────────────────────

def generate_split(
    dataset_type: str,
    categories_pool: list,
    key_levels: list,
    update_levels: list,
    total_examples: int,
    seed_offset: int,
    condition_fracs: dict,   # {"FVQ": 0.4, "CVQ": 0.4, "IVQ": 0.2}
    smoke: bool = False,
) -> list[dict]:
    """Generate total_examples records distributed over the (K,N) grid."""
    if smoke:
        total_examples = min(total_examples, 50)

    cells = [(nk, nu) for nk in key_levels for nu in update_levels]
    per_cell = max(1, total_examples // len(cells))

    # Condition counts per cell
    cond_counts = {}
    for cond, frac in condition_fracs.items():
        cond_counts[cond] = max(1, round(per_cell * frac))
    # Adjust so total matches per_cell
    total_per_cell = sum(cond_counts.values())
    diff = per_cell - total_per_cell
    if diff > 0:
        cond_counts["CVQ"] += diff  # add remainder to CVQ
    elif diff < 0:
        cond_counts["IVQ"] = max(0, cond_counts["IVQ"] + diff)

    records = []
    global_idx = seed_offset

    for nk, nu in cells:
        eligible = get_eligible_categories(dataset_type, min_values=nu)
        # Filter to categories_pool
        available = [c for c in eligible if c in categories_pool]
        if len(available) < nk:
            print(f"  SKIP K={nk}, N={nu}: only {len(available)} eligible cats (need {nk})")
            continue

        for cond, count in cond_counts.items():
            for i in range(count):
                seed = stable_seed(dataset_type, nk, nu, cond, global_idx)
                global_idx += 1

                rng = random.Random(seed)
                cats = rng.sample(available, nk)

                rec = build_example(dataset_type, cats, nu, cond, seed)
                if rec is not None:
                    records.append(rec)

    return records


def generate_test_split(
    dataset_type: str,
    categories_pool: list | None,     # None = use all eligible
    test_grid: dict,                  # {K: [N, ...]}
    trials_per_cell_per_cond: int,
    seed_offset: int,
    smoke: bool = False,
) -> list[dict]:
    """Generate test records: only FVQ and CVQ (matches evaluation conditions)."""
    if smoke:
        trials_per_cell_per_cond = 3

    records = []
    global_idx = seed_offset

    for nk, nu_list in test_grid.items():
        for nu in nu_list:
            eligible = get_eligible_categories(dataset_type, min_values=nu)
            if categories_pool:
                available = [c for c in eligible if c in categories_pool]
            else:
                available = eligible

            if len(available) < nk:
                continue

            for cond in ["FVQ", "CVQ"]:
                for i in range(trials_per_cell_per_cond):
                    seed = stable_seed(dataset_type, "test", nk, nu, cond, global_idx)
                    global_idx += 1

                    rng = random.Random(seed)
                    cats = rng.sample(available, nk)

                    rec = build_example(dataset_type, cats, nu, cond, seed)
                    if rec is not None:
                        records.append(rec)

    return records


def save_jsonl(records: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"  Saved {len(records):>6,} records → {path}")


def print_stats(records: list, label: str):
    from collections import Counter
    conds = Counter(r["condition"] for r in records)
    cells = Counter(f"K={r['num_keys']},N={r['num_updates']}" for r in records)
    print(f"\n  {label}: {len(records):,} total")
    print(f"    Conditions: {dict(conds)}")
    n_cells = len(set((r['num_keys'], r['num_updates']) for r in records))
    print(f"    Cells: {n_cells} unique")


# ─── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="Quick test, ~50 examples per split")
    p.add_argument("--total", type=int, default=TRAIN_TOTAL, help=f"Total train+val examples (default {TRAIN_TOTAL})")
    p.add_argument("--test-trials", type=int, default=50, help="Trials per cell per condition in test sets (default 50)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(__file__).parent / "data"

    print("=" * 60)
    print("LoRA Intervention — Data Generation")
    print(f"  Train total:  {args.total:,}")
    print(f"  Val fraction: {VAL_FRAC:.0%}")
    print(f"  Test trials:  {args.test_trials} per cell per condition")
    print(f"  Smoke test:   {args.smoke}")
    print("=" * 60)

    cond_fracs = {"FVQ": 0.40, "CVQ": 0.40, "IVQ": 0.20}

    # ── Training + val set ────────────────────────────────────────────────────
    print("\nGenerating train+val (ARBITRARY_SINGLE, train categories, train cells)...")
    all_train = generate_split(
        dataset_type     = "ARBITRARY_SINGLE",
        categories_pool  = TRAIN_CATEGORIES,
        key_levels       = TRAIN_KEYS,
        update_levels    = TRAIN_UPDATES,
        total_examples   = args.total,
        seed_offset      = 0,
        condition_fracs  = cond_fracs,
        smoke            = args.smoke,
    )
    random.Random(stable_seed("train_val_split")).shuffle(all_train)

    val_n   = max(1, int(len(all_train) * VAL_FRAC))
    val_set = all_train[:val_n]
    trn_set = all_train[val_n:]
    print_stats(trn_set, "Train")
    print_stats(val_set, "Val")
    save_jsonl(trn_set, out_dir / "train.jsonl")
    save_jsonl(val_set,  out_dir / "val.jsonl")

    # ── In-distribution test (held-out categories, test cells) ────────────────
    print("\nGenerating test_id (ARBITRARY_SINGLE, held-out categories, test cells)...")
    test_id = generate_test_split(
        dataset_type            = "ARBITRARY_SINGLE",
        categories_pool         = HELD_OUT_CATEGORIES,
        test_grid               = TEST_GRID,
        trials_per_cell_per_cond= args.test_trials,
        seed_offset             = 1_000_000,
        smoke                   = args.smoke,
    )
    print_stats(test_id, "Test-ID")
    save_jsonl(test_id, out_dir / "test_id.jsonl")

    # ── OOD test (SEMANTIC_MULTI, all categories, test cells) ─────────────────
    print("\nGenerating test_ood (SEMANTIC_MULTI, all categories, test cells)...")
    test_ood = generate_test_split(
        dataset_type            = "SEMANTIC_MULTI",
        categories_pool         = None,  # all eligible
        test_grid               = TEST_GRID,
        trials_per_cell_per_cond= args.test_trials,
        seed_offset             = 2_000_000,
        smoke                   = args.smoke,
    )
    print_stats(test_ood, "Test-OOD")
    save_jsonl(test_ood, out_dir / "test_ood.jsonl")

    print("\n" + "=" * 60)
    print("Data generation complete.")
    print(f"  Files: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
