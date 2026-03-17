"""
Narrative PI/RI experiment on API models (Claude, GPT).
Uses existing Dota 2 narrative trials.

Usage:
    cd v3
    python narrative_api_experiment.py --model claude-haiku --trials 20
    python narrative_api_experiment.py --model gpt-4.1-mini --trials 20
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load .env
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.model_factory import create_model

NARRATIVE_DATA = _PROJECT_ROOT / "data" / "narrative_interference" / "dota2" / "dota2_gold_same_100t_20260227_112740.json"

SYSTEM_PROMPT = (
    "You are a precise data extraction tool. "
    "Read the match narrative carefully and answer the question. "
    "Output ONLY the exact number. No other text, no units, no explanation."
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="claude-haiku")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--key-levels", type=int, nargs="+", default=[2, 3, 5, 7])
    p.add_argument("--update-levels", type=int, nargs="+", default=[3, 5, 10, 20])
    p.add_argument("--workers", type=int, default=10)
    return p.parse_args()


def extract_answer(raw_output, expected):
    import re
    raw = raw_output.strip()
    if raw == str(expected):
        return True, raw
    numbers = re.findall(r'\d+', raw)
    if numbers:
        for n in numbers:
            if n == str(expected):
                return True, n
        return False, numbers[0] if numbers else raw
    return False, raw


def run_single_trial(model_wrapper, trial, condition):
    """Run a single narrative trial through the API model."""
    q_data = trial["questions"][condition]
    question = q_data["question"]
    expected = str(q_data["expected_answer"])
    narrative = trial["narrative"]

    prompt = f"{SYSTEM_PROMPT}\n\n{narrative}\n\n{question}\n\nAnswer with ONLY the number:"

    try:
        response = model_wrapper.generate(prompt=prompt)
        correct, extracted = extract_answer(response, expected)
        return {
            "correct": correct,
            "expected": expected,
            "predicted": response[:80],
            "extracted": extracted,
            "trial_id": trial.get("id", ""),
            "condition": condition,
            "cell": f"{trial['num_keys']}k_{trial['num_updates']}u",
        }
    except Exception as e:
        return {
            "correct": False,
            "expected": expected,
            "predicted": f"ERROR: {str(e)[:60]}",
            "extracted": "",
            "trial_id": trial.get("id", ""),
            "condition": condition,
            "cell": f"{trial['num_keys']}k_{trial['num_updates']}u",
            "error": True,
        }


def main():
    args = parse_args()

    print(f"Loading narrative data...")
    data = json.load(open(NARRATIVE_DATA))
    trials = data["trials"]
    print(f"  {len(trials)} trials available")

    print(f"\nCreating model: {args.model}")
    model_wrapper = create_model(args.model)
    print(f"  Model initialized")

    # Group trials
    trial_groups = defaultdict(list)
    for t in trials:
        nk, nu = t["num_keys"], t["num_updates"]
        if nk in args.key_levels and nu in args.update_levels:
            trial_groups[(nk, nu)].append(t)

    # Build work items
    work_items = []
    for (nk, nu), cell_trials in sorted(trial_groups.items()):
        selected = cell_trials[:args.trials]
        for trial in selected:
            for condition in ["RI", "PI"]:
                work_items.append((trial, condition))

    print(f"\n  Total API calls: {len(work_items)}")
    print(f"  Workers: {args.workers}")

    # Run with thread pool
    results = defaultdict(lambda: {"RI": [], "PI": []})
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_single_trial, model_wrapper, trial, cond): (trial, cond)
            for trial, cond in work_items
        }

        done = 0
        for future in as_completed(futures):
            result = future.result()
            cell = result["cell"]
            cond = result["condition"]
            results[cell][cond].append(result)
            done += 1
            if done % 20 == 0:
                print(f"  [{done}/{len(work_items)}]")

    elapsed = time.time() - t0

    # Summary
    print(f"\n{'='*60}")
    print(f"NARRATIVE PI/RI — {args.model}")
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

    print(f"  Elapsed: {elapsed:.1f}s")

    # Save
    save_dir = _SCRIPT_DIR / "results" / "narrative"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"narrative_{args.model}.json"
    with open(save_path, "w") as f:
        json.dump({"model": args.model, "results": dict(results), "elapsed_sec": elapsed}, f, indent=2)
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    main()
