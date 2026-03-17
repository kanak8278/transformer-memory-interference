"""
Narrative Interference Experiment: Test PI > RI on Dota 2 narratives.

Tests whether the PI > RI pattern from key-value streams also appears
in naturalistic narrative data (Dota 2 match commentaries).

Usage:
    cd v3
    python narrative_experiment.py --model Qwen/Qwen2.5-1.5B-Instruct --trials 50
    python narrative_experiment.py --model Qwen/Qwen2.5-3B-Instruct --trials 50
"""

import sys
import json
import time
import argparse
import torch
from pathlib import Path
from collections import defaultdict

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mechanistic_probing_v2.core.model_loader import load_model_hf, model_short_name

NARRATIVE_DATA = _PROJECT_ROOT / "data" / "narrative_interference" / "dota2" / "dota2_gold_same_100t_20260227_112740.json"

SYSTEM_PROMPT = (
    "You are a precise data extraction tool. "
    "Read the match narrative carefully and answer the question. "
    "Output ONLY the exact number. No other text, no units, no explanation."
)


def parse_args():
    p = argparse.ArgumentParser(description="Narrative PI/RI experiment")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--trials", type=int, default=50, help="Max trials per cell")
    p.add_argument("--max-new-tokens", type=int, default=20)
    p.add_argument("--key-levels", type=int, nargs="+", default=[2, 3, 5])
    p.add_argument("--update-levels", type=int, nargs="+", default=[3, 5, 10, 20])
    return p.parse_args()


def extract_answer(raw_output, expected):
    """Try to extract a numeric answer from model output."""
    raw = raw_output.strip()
    # Try exact match first
    if raw == str(expected):
        return True, raw
    # Try extracting first number
    import re
    numbers = re.findall(r'\d+', raw)
    if numbers:
        # Check if any number matches expected
        for n in numbers:
            if n == str(expected):
                return True, n
        return False, numbers[0] if numbers else raw
    return False, raw


def run_experiment(model, tokenizer, trials_data, args):
    device = next(model.parameters()).device
    results = defaultdict(lambda: {"RI": [], "PI": []})

    # Group trials by (num_keys, num_updates)
    trial_groups = defaultdict(list)
    for t in trials_data:
        nk, nu = t["num_keys"], t["num_updates"]
        if nk in args.key_levels and nu in args.update_levels:
            trial_groups[(nk, nu)].append(t)

    total_cells = len(trial_groups)
    cell_idx = 0

    for (nk, nu), cell_trials in sorted(trial_groups.items()):
        cell_idx += 1
        cell_key = f"{nk}k_{nu}u"
        print(f"\n[{cell_idx}/{total_cells}] {cell_key} ({len(cell_trials)} available trials)")

        # Take up to args.trials trials
        selected = cell_trials[:args.trials]

        for t_idx, trial in enumerate(selected):
            narrative = trial["narrative"]

            for condition in ["RI", "PI"]:
                q_data = trial["questions"][condition]
                question = q_data["question"]
                expected = str(q_data["expected_answer"])

                # Build prompt
                user_msg = f"{narrative}\n\n{question}"
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ]

                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                # Check length
                n_tokens = len(tokenizer.encode(prompt))
                if n_tokens > 8000:
                    continue  # Skip too-long narratives

                inputs = tokenizer(prompt, return_tensors="pt").to(device)

                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                    )

                generated = out[0][inputs["input_ids"].shape[1]:]
                raw_output = tokenizer.decode(generated, skip_special_tokens=True).strip()

                correct, extracted = extract_answer(raw_output, expected)

                results[cell_key][condition].append({
                    "correct": correct,
                    "expected": expected,
                    "predicted": raw_output[:80],
                    "extracted": extracted,
                    "n_tokens": n_tokens,
                    "trial_id": trial.get("id", t_idx),
                })

            if (t_idx + 1) % 10 == 0:
                ri_acc = sum(r["correct"] for r in results[cell_key]["RI"]) / max(len(results[cell_key]["RI"]), 1)
                pi_acc = sum(r["correct"] for r in results[cell_key]["PI"]) / max(len(results[cell_key]["PI"]), 1)
                print(f"  [{t_idx+1}/{len(selected)}] RI={ri_acc:.0%} PI={pi_acc:.0%}")

        # Cell summary
        for cond in ["RI", "PI"]:
            trials_cond = results[cell_key][cond]
            if trials_cond:
                acc = sum(r["correct"] for r in trials_cond) / len(trials_cond)
                n = len(trials_cond)
                print(f"  {cond}: {acc:.0%} ({n} trials)")

    return dict(results)


def main():
    args = parse_args()

    print(f"Loading narrative data from {NARRATIVE_DATA}")
    data = json.load(open(NARRATIVE_DATA))
    trials = data["trials"]
    print(f"  {len(trials)} trials available")

    print(f"\nLoading {args.model}...")
    model, tokenizer, info = load_model_hf(args.model, device="mps")
    device = next(model.parameters()).device
    print(f"  Device: {device}")

    print(f"\nRunning narrative PI/RI experiment...")
    print(f"  Key levels: {args.key_levels}")
    print(f"  Update levels: {args.update_levels}")
    print(f"  Trials per cell: {args.trials}")

    t0 = time.time()
    results = run_experiment(model, tokenizer, trials, args)
    elapsed = time.time() - t0

    # Summary
    print(f"\n{'='*60}")
    print(f"NARRATIVE PI/RI RESULTS — {args.model}")
    print(f"{'='*60}")

    all_ri, all_pi = [], []
    for cell_key in sorted(results.keys()):
        ri_trials = results[cell_key]["RI"]
        pi_trials = results[cell_key]["PI"]
        ri_acc = sum(r["correct"] for r in ri_trials) / max(len(ri_trials), 1)
        pi_acc = sum(r["correct"] for r in pi_trials) / max(len(pi_trials), 1)
        gap = ri_acc - pi_acc
        print(f"  {cell_key}: RI={ri_acc:.0%} ({len(ri_trials)}t) | PI={pi_acc:.0%} ({len(pi_trials)}t) | gap={gap:+.0%}")
        all_ri.append(ri_acc)
        all_pi.append(pi_acc)

    if all_ri:
        mean_ri = sum(all_ri) / len(all_ri)
        mean_pi = sum(all_pi) / len(all_pi)
        print(f"\n  Mean RI={mean_ri:.0%}, Mean PI={mean_pi:.0%}, Gap={mean_ri - mean_pi:+.0%}")

    print(f"\n  Elapsed: {elapsed:.1f}s")

    # Save
    m_short = model_short_name(args.model)
    save_dir = _SCRIPT_DIR / "results" / "narrative"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"narrative_{m_short}.json"

    output = {
        "model": args.model,
        "dataset": "dota2_gold_same",
        "config": {
            "key_levels": args.key_levels,
            "update_levels": args.update_levels,
            "trials_per_cell": args.trials,
        },
        "results": results,
        "elapsed_sec": elapsed,
    }
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    main()
