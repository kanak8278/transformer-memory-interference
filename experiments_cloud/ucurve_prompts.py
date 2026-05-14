#!/usr/bin/env python3
"""
Prompt generation + verification for U-curve experiment.
Run this to visually inspect all 4 formats and verify:
  - Deterministic regeneration from seed
  - Different seeds → different keys, values, shuffle order
  - Same seed → identical output every time
  - Update index tracking is correct
  - Query positions are right
"""
import sys, random, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mechanistic_probing_v2.core.dataset_configs import (
    get_eligible_categories,
    generate_values_for_trial,
)

SYSTEM_PROMPT = (
    "Output ONLY a single word — the exact value requested. "
    "No explanation, no punctuation, no extra words."
)

# ─── query positions ──────────────────────────────────────────────────────────
def query_positions(nu: int, n_points: int = 11) -> list[int]:
    """n_points evenly spaced positions from 1 to nu, always including 1 and nu."""
    if nu <= n_points:
        return list(range(1, nu + 1))
    positions = set()
    for i in range(n_points):
        pos = round(1 + i * (nu - 1) / (n_points - 1))
        positions.add(max(1, min(nu, pos)))
    return sorted(positions)


# ─── stream generation ────────────────────────────────────────────────────────
def shuffle_no_consecutive(items: list, rng: random.Random) -> list:
    """Shuffle items so no two adjacent entries have the same category."""
    items = items[:]
    for attempt in range(200):
        rng.shuffle(items)
        ok = all(items[i]["category"] != items[i+1]["category"]
                 for i in range(len(items)-1))
        if ok:
            return items
    return items  # best effort


def generate_stream(nk: int, nu: int, seed: int):
    """
    Returns:
        categories      — list of nk category names
        test_category   — which category will be queried
        values_per_cat  — {cat: [val_1, ..., val_nu]}  (ordered 1→nu)
        flat_items      — shuffled list of {category, value, update_idx}
        block_items     — list of nu blocks, each block is shuffled list of {category, value, update_idx}
    """
    rng = random.Random(seed)

    eligible = get_eligible_categories("ARBITRARY_SINGLE", min_values=nu)
    categories = rng.sample(eligible, min(nk, len(eligible)))
    test_category = categories[seed % nk]

    values_per_cat = generate_values_for_trial("ARBITRARY_SINGLE", categories, nu, rng)

    # Build flat items with update_idx
    flat_items = []
    for cat in categories:
        for k, val in enumerate(values_per_cat[cat], start=1):
            flat_items.append({"category": cat, "value": val, "update_idx": k})

    flat_items = shuffle_no_consecutive(flat_items, rng)

    # Build block items (separate rng advances per block for key order within block)
    block_items = []
    for k in range(1, nu + 1):
        block = [{"category": cat, "value": values_per_cat[cat][k-1], "update_idx": k}
                 for cat in categories]
        rng.shuffle(block)          # shuffle key order within this block
        block_items.append(block)

    return categories, test_category, values_per_cat, flat_items, block_items


# ─── 4 prompt formats ─────────────────────────────────────────────────────────
QUERY_LABELED = "What was the value of {category} in Update {k}?"

def ordinal(n: int) -> str:
    """1 → '1st', 2 → '2nd', 3 → '3rd', 4 → '4th', ..."""
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"

def prompt_block(block_items: list, test_category: str, k: int) -> str:
    """Format 1: [Update K] block, all keys listed together (shuffled within block)."""
    lines = []
    for block in block_items:
        lines.append(f"[Update {block[0]['update_idx']}]")
        for item in block:
            lines.append(f"  {item['category']}: {item['value']}")
    stream = "\n".join(lines)
    return (
        f"Read the following key-value stream. "
        f"Each key is updated multiple times, grouped by update round.\n\n"
        f"{stream}\n\n"
        f"{QUERY_LABELED.format(category=test_category, k=k)}\n"
        f"Answer with ONLY the exact value. No explanation."
    )


def prompt_flat_short(flat_items: list, test_category: str, k: int) -> str:
    """Format 2: flat interleaved + [U3] notation + preamble."""
    stream = "\n".join(
        f"{item['category']} [U{item['update_idx']}]: {item['value']}"
        for item in flat_items
    )
    return (
        f"Read the following key-value stream. Each key is updated multiple times. "
        f"The number in brackets after each key name indicates which update of that key it is "
        f"— for example, \"coffee variety [U3]: rural\" means the 3rd time "
        f"coffee variety was updated, its value was \"rural\".\n\n"
        f"{stream}\n\n"
        f"{QUERY_LABELED.format(category=test_category, k=k)}\n"
        f"Answer with ONLY the exact value. No explanation."
    )


def prompt_flat_verbose(flat_items: list, test_category: str, k: int) -> str:
    """Format 3: flat interleaved + (update 3) verbose, self-explanatory."""
    stream = "\n".join(
        f"{item['category']} (update {item['update_idx']}): {item['value']}"
        for item in flat_items
    )
    return (
        f"Read the following key-value stream. Each key is updated multiple times.\n\n"
        f"{stream}\n\n"
        f"{QUERY_LABELED.format(category=test_category, k=k)}\n"
        f"Answer with ONLY the exact value. No explanation."
    )


def prompt_flat_nolabel(flat_items: list, test_category: str, k: int) -> str:
    """Format 4: flat interleaved, no labels — model must count occurrences."""
    stream = "\n".join(
        f"{item['category']}: {item['value']}"
        for item in flat_items
    )
    return (
        f"Read the following key-value stream. Each key appears multiple times "
        f"as it gets updated. Count each occurrence of a key as one update.\n\n"
        f"{stream}\n\n"
        f"What was the {ordinal(k)} value of {test_category}?\n"
        f"Answer with ONLY the exact value. No explanation."
    )


def prompt_original(flat_items: list, test_category: str, k: int) -> str:
    """Format 5: original first/last prompt — no labels, no ordinal.
    k=1 asks 'first', k=nu asks 'last'. Middle positions not meaningful here
    but included for completeness; query always uses first/last anchors.
    Only positions 1 and nu produce valid comparisons with existing results.
    """
    stream = "\n".join(
        f"{item['category']}: {item['value']}"
        for item in flat_items
    )
    # Use the same query logic as the original sweep_arbitrary.py
    query_word = "first" if k == 1 else "last"
    return (
        f"Read the following key-value stream. "
        f"Each key gets updated multiple times.\n\n"
        f"{stream}\n\n"
        f"What was the {query_word} value of {test_category}?\n"
        f"Answer with ONLY the exact value. No explanation."
    )


FORMATS = {
    "block":        prompt_block,
    "flat_short":   prompt_flat_short,
    "flat_verbose": prompt_flat_verbose,
    "flat_nolabel": prompt_flat_nolabel,
    "original":     prompt_original,
}


# ─── verification helpers ─────────────────────────────────────────────────────
def make_seed(nk: int, nu: int, trial_idx: int, version: str = "ucurve_v1") -> int:
    return hash((nk, nu, trial_idx, version)) % (2**31)


def verify_update_tracking(flat_items, values_per_cat, test_category):
    """Verify that update_idx on each item matches values_per_cat order."""
    errors = []
    for cat in set(i["category"] for i in flat_items):
        cat_items = sorted(
            [i for i in flat_items if i["category"] == cat],
            key=lambda x: x["update_idx"]
        )
        expected = values_per_cat[cat]
        for i, item in enumerate(cat_items):
            if item["value"] != expected[i]:
                errors.append(
                    f"  MISMATCH: {cat} update {item['update_idx']}: "
                    f"got {item['value']!r}, expected {expected[i]!r}"
                )
            if item["update_idx"] != i + 1:
                errors.append(
                    f"  BAD INDEX: {cat} got update_idx={item['update_idx']}, expected {i+1}"
                )
    return errors


# ─── MAIN: print examples and run verifications ────────────────────────────────
if __name__ == "__main__":
    NK, NU = 5, 10    # small so examples are readable

    print("=" * 80)
    print("VERIFICATION 1: Same seed → identical output every time")
    print("=" * 80)
    seed = make_seed(NK, NU, trial_idx=0)
    cats_a, test_a, vals_a, flat_a, block_a = generate_stream(NK, NU, seed)
    cats_b, test_b, vals_b, flat_b, block_b = generate_stream(NK, NU, seed)
    assert cats_a == cats_b, "FAIL: categories differ"
    assert test_a == test_b, "FAIL: test_category differs"
    assert flat_a == flat_b, "FAIL: flat_items differ"
    assert block_a == block_b, "FAIL: block_items differ"
    print(f"  seed={seed}  categories={cats_a}")
    print(f"  test_category={test_a!r}")
    print("  ✓ Identical regeneration confirmed")

    print()
    print("=" * 80)
    print("VERIFICATION 2: Different seeds → different categories, keys, shuffle")
    print("=" * 80)
    for trial_idx in range(4):
        seed = make_seed(NK, NU, trial_idx)
        cats, test, vals, flat, block = generate_stream(NK, NU, seed)
        first_4 = [(i["category"][:12], i["value"]) for i in flat[:4]]
        print(f"  trial={trial_idx}  seed={seed}  test_cat={test!r:<25}  "
              f"first_4_entries={first_4}")

    print()
    print("=" * 80)
    print("VERIFICATION 3: Update index tracking is correct")
    print("=" * 80)
    seed = make_seed(NK, NU, trial_idx=0)
    cats, test, vals, flat, block = generate_stream(NK, NU, seed)
    errors = verify_update_tracking(flat, vals, test)
    if errors:
        for e in errors: print(e)
    else:
        print("  ✓ All update_idx values match values_per_cat order")
    # Show one category's update sequence in stream order vs expected order
    sample_cat = cats[0]
    in_stream = [i for i in flat if i["category"] == sample_cat]
    print(f"\n  {sample_cat!r} in stream order (shuffled):")
    for i in in_stream:
        marker = " ← [U{k}] = expected value".format(k=i['update_idx'])
        print(f"    stream_pos={flat.index(i)+1:2d}  update_idx={i['update_idx']}  "
              f"value={i['value']!r}{marker}")
    print(f"\n  {sample_cat!r} in update order (expected):")
    for k, v in enumerate(vals[sample_cat], 1):
        print(f"    update {k}: {v!r}")

    print()
    print("=" * 80)
    print("VERIFICATION 4: Query positions for each nu value")
    print("=" * 80)
    for nu in [10, 30, 50, 75, 100]:
        positions = query_positions(nu)
        pct = [f"{round(100*(p-1)/(nu-1))}%" for p in positions]
        print(f"  nu={nu:3d}: {len(positions)} positions → {positions}")
        print(f"         percentiles → {pct}")

    print()
    print("=" * 80)
    print("EXAMPLES: All 4 formats for nk=5, nu=10, trial_idx=0, querying Update 5")
    print("=" * 80)
    seed = make_seed(NK, NU, trial_idx=7)
    cats, test, vals, flat, block = generate_stream(NK, NU, seed)
    k = 5   # query middle position

    correct_answer = vals[test][k - 1]
    print(f"\n  test_category = {test!r}")
    print(f"  querying Update {k}")
    print(f"  CORRECT ANSWER = {correct_answer!r}  (vals_per_cat[test][{k-1}])")
    print(f"  all values for {test!r}: {vals[test]}")

    for fmt_name, fmt_fn in FORMATS.items():
        print(f"\n{'─'*80}")
        print(f"  FORMAT: {fmt_name}")
        print(f"{'─'*80}")
        if fmt_name == "block":
            prompt = fmt_fn(block, test, k)
        else:
            prompt = fmt_fn(flat, test, k)
        print(prompt)

    print()
    print("=" * 80)
    print("EXAMPLE: block format with nk=3, nu=5 (very small — easiest to read)")
    print("=" * 80)
    seed = make_seed(3, 5, trial_idx=2)
    cats, test, vals, flat, block = generate_stream(3, 5, seed)
    positions = query_positions(5)
    print(f"\n  categories={cats}")
    print(f"  test_category={test!r}")
    print(f"  query_positions for nu=5: {positions}")
    print(f"  values for {test!r}: {vals[test]}")
    print()
    print("  [block format, querying Update 3]")
    print()
    print(prompt_block(block, test, 3))

    print()
    print("=" * 80)
    print("SANITY: Verify block and flat agree on the same underlying values")
    print("=" * 80)
    seed = make_seed(NK, NU, trial_idx=0)
    cats, test, vals, flat, block = generate_stream(NK, NU, seed)
    for k in query_positions(NU):
        # From flat: find item with category==test and update_idx==k
        flat_val = next(i["value"] for i in flat
                        if i["category"] == test and i["update_idx"] == k)
        # From block: find block k, find test category in it
        block_val = next(i["value"] for i in block[k-1]
                         if i["category"] == test)
        # From ground truth
        gt_val = vals[test][k - 1]
        status = "✓" if flat_val == block_val == gt_val else "✗ MISMATCH"
        print(f"  update {k:3d}: flat={flat_val!r:<12} block={block_val!r:<12} "
              f"gt={gt_val!r:<12} {status}")

    print()
    print("All verifications complete.")
