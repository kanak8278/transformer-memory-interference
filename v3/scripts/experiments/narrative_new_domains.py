"""
Narrative Interference Experiment — Wildlife, ICU, and ATC domains.

Tests whether the PI > RI asymmetry from synthetic KV-pair data and Dota 2
narratives transfers to three new naturalistic narrative domains:
  - wildlife: GPS-collar field studies (ecological data)
  - icu:      ICU patient monitoring (clinical notes)
  - atc:      Air traffic control sector logs (ATC phraseology)

The script:
1. Generates trials on-the-fly using the three new generators
2. Queries the model for first (RI) and last (PI) attribute values
3. Records accuracy per cell (num_keys × num_updates)
4. Saves results compatible with the existing v3 narrative results format

Usage:
    cd /Users/kanak.raj/workspace/hobby/research_work_ri
    .venv/bin/python v3/scripts/experiments/narrative_new_domains.py \\
        --model Qwen/Qwen2.5-3B-Instruct \\
        --domain all \\
        --trials 30

    .venv/bin/python v3/scripts/experiments/narrative_new_domains.py \\
        --model Qwen/Qwen2.5-1.5B-Instruct \\
        --domain wildlife \\
        --trials 50
"""

import sys
import json
import time
import argparse
import re
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

# ── Path setup ─────────────────────────────────────────────────────────────

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Imports ─────────────────────────────────────────────────────────────────

try:
    from mechanistic_probing_v2.core.model_loader import load_model_hf
    HAS_LOADER = True
except ImportError:
    HAS_LOADER = False
    print("Warning: model_loader not found, using direct transformers loading")

# Operating point grid — mirrors the v3 behavioral sweep grid
OPERATING_POINTS = [
    (2, 3), (2, 5), (2, 7),
    (3, 3), (3, 5), (3, 7), (3, 10),
    (5, 3), (5, 5), (5, 7), (5, 10),
    (7, 5), (7, 10),
    (10, 5), (10, 10),
]

SYSTEM_PROMPT = (
    "You are a precise data extraction tool. "
    "Read the provided text carefully and answer the question exactly. "
    "Output ONLY the exact value — no explanations, no units unless they appear in the text."
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
    else:
        raise ValueError(f"Unknown domain: {domain}")


# ── Answer extraction ────────────────────────────────────────────────────────

def extract_answer(raw_output: str, expected: str) -> tuple:
    """Try to extract the expected answer from model output.

    For narrative domains, expected answers can be:
    - Numeric: "38.2 kg", "125", "FL240", "92%"
    - Categorical: "Lamar Valley", "sepsis", "heading 270"

    Returns (correct: bool, extracted: str)
    """
    raw = raw_output.strip()

    # Direct match
    if raw == expected:
        return True, raw

    # Check if expected string appears anywhere in raw output
    if expected in raw:
        return True, expected

    # For numeric values, try extracting just the number
    # e.g., expected="38.2 kg", output might say "38.2" or "38.2 kg"
    # Strip units from expected and check
    expected_stripped = re.sub(r'\s*(kg|km|m|%|bpm|mmHg|C|mmol/L|mg/dL|U/L|g/dL|K/uL|mcg/kg/min|mEq/L|cmH2O|mL/hr|pg/mL)\s*$', '', expected).strip()
    if expected_stripped and expected_stripped in raw:
        return True, expected_stripped

    # For FL-format altitudes: "FL240" might appear as "flight level 240" or "240"
    fl_match = re.match(r'^FL(\d+)$', expected)
    if fl_match:
        fl_num = fl_match.group(1)
        if fl_num in raw or f"flight level {fl_num}" in raw.lower():
            return True, expected

    # For categorical values, check case-insensitive
    if expected.lower() in raw.lower():
        return True, expected

    # Extract first plausible token (number, word, or FL-format)
    tokens = raw.split()
    if tokens:
        return False, tokens[0]

    return False, raw[:50]


# ── Model loading (fallback if model_loader not available) ──────────────────

def load_model_simple(model_name: str):
    """Direct transformers loading as fallback."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device != "cpu" else torch.float32

    print(f"  Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device
    )
    model.eval()
    return model, tokenizer, device


# ── Main experiment ──────────────────────────────────────────────────────────

def run_domain_experiment(domain: str, model, tokenizer, device,
                          n_trials: int, max_new_tokens: int,
                          results_dir: Path, model_short: str,
                          resume: bool = True) -> dict:
    """Run RI/PI sweep for one domain."""
    import torch

    gen = get_generator(domain)

    results_file = results_dir / f"narrative_{domain}_{model_short}.json"

    # Resume from checkpoint
    if resume and results_file.exists():
        with open(results_file) as f:
            saved = json.load(f)
        all_results = saved.get("results", {})
        print(f"  Resuming: {len(all_results)} cells already complete")
    else:
        all_results = {}

    total_cells = len(OPERATING_POINTS)

    for cell_idx, (nk, nu) in enumerate(OPERATING_POINTS):
        cell_key = f"{nk}k_{nu}u"

        # Skip if already have enough trials
        existing = all_results.get(cell_key, {})
        n_ri = len(existing.get("RI", []))
        if n_ri >= n_trials:
            print(f"  [{cell_idx+1}/{total_cells}] {cell_key}: skip (already {n_ri} trials)")
            continue

        print(f"\n  [{cell_idx+1}/{total_cells}] {domain} {cell_key}")

        if cell_key not in all_results:
            all_results[cell_key] = {"RI": [], "PI": []}

        start_from = len(all_results[cell_key]["RI"])

        for trial_idx in range(start_from, n_trials):
            seed = trial_idx * 1000 + nk * 100 + nu

            try:
                trial = gen.generate_trial(
                    num_keys=nk, num_updates=nu, condition="both", seed=seed
                )
            except Exception as e:
                print(f"    Trial {trial_idx} generation failed: {e}")
                continue

            narrative = trial["narrative"]

            for condition in ["RI", "PI"]:
                q_data = trial["questions"][condition]
                question = q_data["question"]
                expected = str(q_data["expected_answer"])

                user_msg = f"{narrative}\n\nQuestion: {question}"
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ]

                try:
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    # Fallback: no chat template
                    prompt = f"{narrative}\n\nQuestion: {question}\nAnswer:"

                n_tokens = len(tokenizer.encode(prompt))
                if n_tokens > 10000:
                    # Skip very long trials (rare with num_updates <= 10)
                    break

                inputs = tokenizer(prompt, return_tensors="pt").to(device)

                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                generated = out[0][inputs["input_ids"].shape[1]:]
                raw_output = tokenizer.decode(generated, skip_special_tokens=True).strip()

                correct, extracted = extract_answer(raw_output, expected)

                all_results[cell_key][condition].append({
                    "correct": correct,
                    "expected": expected,
                    "predicted": raw_output[:100],
                    "extracted": extracted,
                    "n_tokens": n_tokens,
                    "trial_id": trial.get("id", f"{domain}_{cell_key}_{trial_idx}"),
                    "tracked_attribute": trial["questions"]["RI"]["target_attribute"],
                    "config": trial.get("config", {}),
                })

            # Progress update every 10 trials
            if (trial_idx + 1) % 10 == 0:
                ri_n = len(all_results[cell_key]["RI"])
                pi_n = len(all_results[cell_key]["PI"])
                if ri_n > 0:
                    ri_acc = sum(r["correct"] for r in all_results[cell_key]["RI"]) / ri_n
                    pi_acc = sum(r["correct"] for r in all_results[cell_key]["PI"]) / pi_n
                    print(f"    [{trial_idx+1}/{n_trials}] RI={ri_acc:.0%} PI={pi_acc:.0%}")

                # Checkpoint
                _save_results(all_results, domain, model_short, results_dir, results_file)

        # Cell summary
        ri_trials = all_results[cell_key]["RI"]
        pi_trials = all_results[cell_key]["PI"]
        if ri_trials:
            ri_acc = sum(r["correct"] for r in ri_trials) / len(ri_trials)
            pi_acc = sum(r["correct"] for r in pi_trials) / len(pi_trials)
            gap = ri_acc - pi_acc
            pi_gt_ri = "YES (PI>RI)" if pi_acc < ri_acc else "no"
            print(f"  {cell_key}: RI={ri_acc:.1%} PI={pi_acc:.1%} gap={gap:+.1%} PI>RI:{pi_gt_ri}")

    _save_results(all_results, domain, model_short, results_dir, results_file)
    return all_results


def _save_results(all_results, domain, model_short, results_dir, results_file):
    """Save results to checkpoint file."""
    results_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "domain": domain,
        "model": model_short,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": all_results,
        "summary": _compute_summary(all_results),
    }
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)


def _compute_summary(all_results) -> dict:
    """Compute RI/PI accuracy summary across all cells."""
    summary = {}
    cells_with_pi_gt_ri = 0
    total_cells = 0

    for cell_key, cell_data in all_results.items():
        ri = cell_data.get("RI", [])
        pi = cell_data.get("PI", [])
        if not ri or not pi:
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
        if pi_acc < ri_acc:
            cells_with_pi_gt_ri += 1
        total_cells += 1

    if total_cells > 0:
        summary["_overall"] = {
            "cells_with_pi_gt_ri": cells_with_pi_gt_ri,
            "total_cells": total_cells,
            "pct_pi_gt_ri": round(cells_with_pi_gt_ri / total_cells, 3),
        }

    return summary


def print_summary_table(all_results, domain):
    """Print a formatted summary table."""
    print(f"\n{'─'*60}")
    print(f"DOMAIN: {domain.upper()}")
    print(f"{'Cell':<12} {'RI':>8} {'PI':>8} {'Gap':>8} {'PI>RI?':>8} {'N':>5}")
    print(f"{'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*5}")

    for cell_key, cell_data in sorted(all_results.items()):
        ri = cell_data.get("RI", [])
        pi = cell_data.get("PI", [])
        if not ri:
            continue
        ri_acc = sum(r["correct"] for r in ri) / len(ri)
        pi_acc = sum(r["correct"] for r in pi) / len(pi)
        gap = ri_acc - pi_acc
        marker = "YES" if pi_acc < ri_acc else "no"
        print(f"{cell_key:<12} {ri_acc:>8.1%} {pi_acc:>8.1%} {gap:>+8.1%} {marker:>8} {len(ri):>5}")

    print(f"{'─'*60}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Narrative interference experiment — new domains")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct",
                   help="HuggingFace model ID")
    p.add_argument("--domain", choices=["wildlife", "icu", "atc", "all"],
                   default="all", help="Domain(s) to run")
    p.add_argument("--trials", type=int, default=30,
                   help="Trials per operating point cell")
    p.add_argument("--max-new-tokens", type=int, default=30,
                   help="Max tokens to generate per answer")
    p.add_argument("--gpu", type=int, default=0,
                   help="GPU index to use (if CUDA available)")
    p.add_argument("--output-dir", type=str,
                   default=None,
                   help="Output directory (default: v3/results/narrative/)")
    p.add_argument("--no-resume", action="store_true",
                   help="Start fresh, don't resume from checkpoint")
    return p.parse_args()


def main():
    args = parse_args()

    # Set output dir
    if args.output_dir:
        results_dir = Path(args.output_dir)
    else:
        results_dir = _PROJECT_ROOT / "v3" / "results" / "narrative"
    results_dir.mkdir(parents=True, exist_ok=True)

    domains = ["wildlife", "icu", "atc"] if args.domain == "all" else [args.domain]

    # Load model
    print(f"Loading model: {args.model}")
    start_load = time.time()

    if HAS_LOADER:
        try:
            model, tokenizer, info = load_model_hf(args.model, gpu_idx=args.gpu)
            device = info.device
            model_short = args.model.split("/")[-1]
        except Exception as e:
            print(f"  load_model_hf failed ({e}), using direct loading")
            model, tokenizer, device = load_model_simple(args.model)
            model_short = args.model.split("/")[-1]
    else:
        model, tokenizer, device = load_model_simple(args.model)
        model_short = args.model.split("/")[-1]

    print(f"  Loaded in {time.time()-start_load:.1f}s on {device}")
    print(f"  Model short name: {model_short}")

    # Run experiments
    start = time.time()

    for domain in domains:
        print(f"\n{'='*60}")
        print(f"DOMAIN: {domain.upper()} | Model: {model_short} | Trials/cell: {args.trials}")
        print(f"{'='*60}")

        try:
            all_results = run_domain_experiment(
                domain=domain,
                model=model,
                tokenizer=tokenizer,
                device=device,
                n_trials=args.trials,
                max_new_tokens=args.max_new_tokens,
                results_dir=results_dir,
                model_short=model_short,
                resume=not args.no_resume,
            )
            print_summary_table(all_results, domain)
        except Exception as e:
            import traceback
            print(f"  Domain {domain} FAILED: {e}")
            traceback.print_exc()

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
