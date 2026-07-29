"""
Narrative Interference Experiment — Wildlife, ICU, ATC domains via API models.

Informed by Dota 2 baseline analysis:
- Frontier models (Haiku, GPT-4.1-mini) need N≥10 to show interference
  (at N=3-5, PI≈100% — ceiling effect)
- Strong signal at 10-20 updates; calibration at 5 updates
- 20 trials gives ±22% CI — need 50+ for meaningful results

Targeted grid (8 cells, covers the discriminating regime):
  Strong interference: 2k_10u, 2k_20u, 3k_10u, 3k_20u, 5k_10u, 5k_20u
  Calibration:        3k_5u, 5k_5u
  Skip: 2k_3u, 2k_5u, 3k_3u, 5k_3u (ceiling effect on frontier models)

Usage:
    cd /Users/kanak.raj/workspace/hobby/research_work_ri
    .venv/bin/python v3/scripts/experiments/narrative_api_new_domains.py \\
        --model claude-haiku --domain all --trials 50
    .venv/bin/python v3/scripts/experiments/narrative_api_new_domains.py \\
        --model gpt-4.1-mini --domain all --trials 50
    .venv/bin/python v3/scripts/experiments/narrative_api_new_domains.py \\
        --model claude-haiku --domain wildlife --trials 50 --workers 20
"""

import sys
import json
import time
import re
import argparse
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from dotenv import load_dotenv

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.model_factory import create_model

# ── Operating point grid ────────────────────────────────────────────────────
# Data-driven: from Dota2 Qwen1.5B (12/12 cells PI>RI) and Haiku results.
# Haiku shows ceiling effect (PI≈100%) at N≤5 with 2-3 keys.
# Interference appears at N≥10 for frontier models.

TARGETED_GRID = [
    (2, 10), (2, 20),   # 2 entities, deep: strong signal
    (3, 5),              # 3 entities, moderate: calibration
    (3, 10), (3, 20),   # 3 entities, deep: strongest zone
    (5, 5),              # 5 entities, moderate: calibration
    (5, 10), (5, 20),   # 5 entities, deep: hardest regime
]

FULL_GRID = [
    (2, 3), (2, 5), (2, 10), (2, 20),
    (3, 3), (3, 5), (3, 10), (3, 20),
    (5, 3), (5, 5), (5, 10), (5, 20),
]

# Hard regime: 10+ entities, 15+ updates.
# Designed for frontier models (Haiku, GPT-4.1-mini) that resist interference
# at moderate N. 10 keys creates 10-entity tracking; 15-50 updates fills
# the narrative with enough interleaved mentions to create genuine interference.
# Generator caps: wildlife=15, icu=12, atc=20.
HARD_GRID = [
    (10, 15), (10, 20), (10, 30), (10, 40), (10, 50),
    (12, 15), (12, 20), (12, 30), (12, 40), (12, 50),
]

# System prompt — domain-agnostic, mirrors Dota2 API experiment
SYSTEM_PROMPT = (
    "You are a precise data extraction tool. "
    "Read the provided text carefully and answer the question exactly. "
    "Output ONLY the exact value as it appears in the text — no explanation, no extra words."
)


# ── Generators ──────────────────────────────────────────────────────────────

def get_generator(domain: str):
    if domain == "wildlife":
        from narrative_generator.wildlife.generator import WildlifeTrialGenerator
        return WildlifeTrialGenerator()
    elif domain == "icu":
        from narrative_generator.icu.generator import ICUTrialGenerator
        return ICUTrialGenerator()
    elif domain == "atc":
        from narrative_generator.atc.generator import ATCTrialGenerator
        return ATCTrialGenerator()
    raise ValueError(f"Unknown domain: {domain}")


# ── Answer extraction ────────────────────────────────────────────────────────

def extract_answer(raw_output: str, expected: str) -> tuple:
    """Extract and check answer. Handles numeric+unit, FL-altitude, categorical."""
    raw = raw_output.strip()

    if raw == expected:
        return True, raw

    # Expected anywhere in raw
    if expected in raw:
        return True, expected

    # Strip common units from expected then check
    stripped = re.sub(
        r'\s*(kg|km|m|%|bpm|mmHg|°C|C|mmol/L|mg/dL|U/L|g/dL|K/uL|'
        r'mcg/kg/min|mEq/L|cmH2O|mL/hr|pg/mL|nmol/L|ng/mL|IU/mL|knots?)\s*$',
        '', expected, flags=re.IGNORECASE
    ).strip()
    if stripped and stripped in raw:
        return True, stripped

    # FL-format: FL240 → accept "240" or "flight level 240" or "two four zero"
    fl_match = re.match(r'^FL(\d+)$', expected)
    if fl_match:
        num = fl_match.group(1)
        if num in raw or f"flight level {num}" in raw.lower():
            return True, expected

    # Percentage: "92%" → accept "92"
    pct_match = re.match(r'^(\d+(?:\.\d+)?)%$', expected)
    if pct_match:
        if pct_match.group(1) in raw:
            return True, expected

    # Decimal with unit: "38.2 kg" → accept "38.2"
    dec_match = re.match(r'^(\d+\.\d+)', expected)
    if dec_match and dec_match.group(1) in raw:
        return True, dec_match.group(1)

    # Case-insensitive categorical
    if expected.lower() in raw.lower():
        return True, expected

    # First token as extracted
    tokens = raw.split()
    return False, (tokens[0] if tokens else raw[:50])


# ── Single trial runner ──────────────────────────────────────────────────────

def run_single_trial(model_wrapper, narrative: str, question: str,
                     expected: str, condition: str,
                     trial_id: str, cell_key: str, tracked_attr: str) -> dict:
    prompt = f"{SYSTEM_PROMPT}\n\n{narrative}\n\n{question}"

    try:
        response = model_wrapper.generate(prompt=prompt)
        correct, extracted = extract_answer(response, expected)
        return {
            "correct": correct,
            "expected": expected,
            "predicted": response[:100],
            "extracted": extracted,
            "trial_id": trial_id,
            "condition": condition,
            "cell": cell_key,
            "tracked_attr": tracked_attr,
        }
    except Exception as e:
        return {
            "correct": False,
            "expected": expected,
            "predicted": f"ERROR: {str(e)[:80]}",
            "extracted": "",
            "trial_id": trial_id,
            "condition": condition,
            "cell": cell_key,
            "tracked_attr": tracked_attr,
            "error": True,
        }


# ── Domain experiment ─────────────────────────────────────────────────────────

def run_domain(domain: str, model_wrapper, model_name: str,
               grid: list, n_trials: int, n_workers: int,
               results_dir: Path, resume: bool = True) -> dict:
    model_short = model_name.replace("/", "-").replace(":", "-")
    results_file = results_dir / f"narrative_api_{domain}_{model_short}.json"

    if resume and results_file.exists():
        with open(results_file) as f:
            saved = json.load(f)
        all_results = saved.get("results", {})
        existing_total = sum(len(v.get("RI", [])) for v in all_results.values())
        print(f"  Resuming: {len(all_results)} cells, {existing_total} existing RI trials")
    else:
        all_results = {}

    gen = get_generator(domain)

    for nk, nu in grid:
        cell_key = f"{nk}k_{nu}u"
        existing_n = len(all_results.get(cell_key, {}).get("RI", []))
        if existing_n >= n_trials:
            print(f"  {cell_key}: skip (have {existing_n}/{n_trials})")
            continue

        print(f"\n  {domain.upper()} {cell_key} — generating {n_trials} trials, {n_workers} workers")

        if cell_key not in all_results:
            all_results[cell_key] = {"RI": [], "PI": []}

        # Generate all trials for this cell
        trials_data = []
        for trial_idx in range(n_trials):
            seed = trial_idx * 1000 + nk * 100 + nu
            try:
                trial = gen.generate_trial(
                    num_keys=nk, num_updates=nu, condition="both", seed=seed
                )
                trials_data.append(trial)
            except Exception as e:
                print(f"    gen error trial {trial_idx}: {e}")

        # Build work items (only new ones)
        work_items = []
        for t_idx, trial in enumerate(trials_data[existing_n:], start=existing_n):
            for condition in ["RI", "PI"]:
                q = trial["questions"][condition]
                work_items.append((
                    trial["narrative"],
                    q["question"],
                    str(q["expected_answer"]),
                    condition,
                    trial.get("id", f"{domain}_{cell_key}_{t_idx}"),
                    cell_key,
                    trial["questions"]["RI"]["target_attribute"],
                ))

        if not work_items:
            continue

        print(f"    {len(work_items)} API calls queued")
        t0 = time.time()

        # Run in parallel
        temp_results = defaultdict(list)
        done = 0

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(run_single_trial, model_wrapper, *item): item
                for item in work_items
            }
            for future in as_completed(futures):
                r = future.result()
                temp_results[r["condition"]].append(r)
                done += 1
                if done % 20 == 0:
                    elapsed = time.time() - t0
                    ri_done = [x for x in temp_results["RI"]]
                    pi_done = [x for x in temp_results["PI"]]
                    ri_acc = sum(x["correct"] for x in ri_done) / max(len(ri_done), 1)
                    pi_acc = sum(x["correct"] for x in pi_done) / max(len(pi_done), 1)
                    print(f"    [{done}/{len(work_items)}] {elapsed:.0f}s | RI={ri_acc:.0%} PI={pi_acc:.0%}")

        # Merge
        for cond in ["RI", "PI"]:
            all_results[cell_key][cond].extend(temp_results[cond])

        # Cell summary
        ri = all_results[cell_key]["RI"]
        pi = all_results[cell_key]["PI"]
        ri_acc = sum(r["correct"] for r in ri) / len(ri)
        pi_acc = sum(r["correct"] for r in pi) / len(pi)
        gap = ri_acc - pi_acc
        elapsed = time.time() - t0
        print(f"    DONE {cell_key}: RI={ri_acc:.0%} PI={pi_acc:.0%} gap={gap:+.0%} "
              f"n={len(ri)} ({elapsed:.0f}s)")

        # Save checkpoint
        _save(all_results, domain, model_name, results_file)

    _save(all_results, domain, model_name, results_file)
    return all_results


def _save(all_results: dict, domain: str, model_name: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = _compute_summary(all_results)
    out = {
        "domain": domain,
        "model": model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": all_results,
        "summary": summary,
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


def _compute_summary(all_results: dict) -> dict:
    summary = {}
    for cell_key, cell_data in all_results.items():
        ri = cell_data.get("RI", [])
        pi = cell_data.get("PI", [])
        if not ri:
            continue
        ri_acc = sum(r["correct"] for r in ri) / len(ri)
        pi_acc = sum(r["correct"] for r in pi) / len(pi)
        summary[cell_key] = {
            "ri_accuracy": round(ri_acc, 3),
            "pi_accuracy": round(pi_acc, 3),
            "gap": round(ri_acc - pi_acc, 3),
            "pi_gt_ri": pi_acc < ri_acc,
            "n_trials": len(ri),
        }
    cells = [v for v in summary.values()]
    if cells:
        pi_gt = sum(1 for v in cells if v["pi_gt_ri"])
        summary["_overall"] = {
            "cells_with_pi_gt_ri": pi_gt,
            "total_cells": len(cells),
            "pct_pi_gt_ri": round(pi_gt / len(cells), 3),
            "mean_gap": round(sum(v["gap"] for v in cells) / len(cells), 3),
        }
    return summary


def print_table(all_results: dict, domain: str, model_name: str):
    summary = _compute_summary(all_results)
    print(f"\n{'─'*65}")
    print(f"  {domain.upper()} | {model_name}")
    print(f"  {'Cell':<12} {'RI':>8} {'PI':>8} {'Gap':>9} {'PI>RI?':>8} {'N':>5}")
    print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*9} {'─'*8} {'─'*5}")
    for cell_key in sorted(k for k in summary if not k.startswith('_')):
        s = summary[cell_key]
        m = "YES" if s["pi_gt_ri"] else "no"
        print(f"  {cell_key:<12} {s['ri_accuracy']:>8.1%} {s['pi_accuracy']:>8.1%} "
              f"{s['gap']:>+9.1%} {m:>8} {s['n_trials']:>5}")
    ov = summary.get("_overall", {})
    if ov:
        print(f"\n  PI>RI: {ov['cells_with_pi_gt_ri']}/{ov['total_cells']} cells "
              f"({ov['pct_pi_gt_ri']:.0%}) | mean gap: {ov['mean_gap']:+.1%}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="API narrative interference experiment — new domains")
    p.add_argument("--model", default="claude-haiku",
                   help="Model name: claude-haiku, gpt-4.1-mini, claude-sonnet-4-5, etc.")
    p.add_argument("--domain", choices=["wildlife", "icu", "atc", "all"],
                   default="all")
    p.add_argument("--trials", type=int, default=50,
                   help="Trials per cell (50 gives ±14% CI, 100 gives ±10% CI)")
    p.add_argument("--workers", type=int, default=20,
                   help="Parallel API workers")
    p.add_argument("--full-grid", action="store_true",
                   help="Use full 12-cell grid instead of targeted 8-cell grid")
    p.add_argument("--hard-grid", action="store_true",
                   help="Use hard regime (10-12 keys × 15-50 updates) for frontier models")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--no-resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    results_dir = Path(args.output_dir) if args.output_dir else \
        _PROJECT_ROOT / "v3" / "results" / "narrative"
    results_dir.mkdir(parents=True, exist_ok=True)

    domains = ["wildlife", "icu", "atc"] if args.domain == "all" else [args.domain]
    if args.hard_grid:
        grid = HARD_GRID
    elif args.full_grid:
        grid = FULL_GRID
    else:
        grid = TARGETED_GRID

    print(f"Model: {args.model}")
    print(f"Domains: {domains}")
    grid_label = "hard" if args.hard_grid else ("full" if args.full_grid else "targeted")
    print(f"Grid: {len(grid)} cells ({grid_label}): {grid}")
    print(f"Trials/cell: {args.trials}  Workers: {args.workers}")
    print(f"Total API calls: ~{len(grid) * args.trials * 2 * len(domains)}")
    print()

    model_wrapper = create_model(args.model)
    print(f"Model initialized: {args.model}\n")

    t_start = time.time()

    for domain in domains:
        print(f"\n{'='*65}")
        print(f"DOMAIN: {domain.upper()}")
        print(f"{'='*65}")
        try:
            all_results = run_domain(
                domain=domain,
                model_wrapper=model_wrapper,
                model_name=args.model,
                grid=grid,
                n_trials=args.trials,
                n_workers=args.workers,
                results_dir=results_dir,
                resume=not args.no_resume,
            )
            print_table(all_results, domain, args.model)
        except Exception as e:
            import traceback
            print(f"  FAILED: {e}")
            traceback.print_exc()

    elapsed = time.time() - t_start
    print(f"\nTotal: {elapsed/60:.1f} min")
    print(f"Results: {results_dir}")


if __name__ == "__main__":
    main()
