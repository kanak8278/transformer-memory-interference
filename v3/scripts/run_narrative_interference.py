#!/usr/bin/env python3
"""
Run RI/PI interference experiments for narrative domains (wildlife, ICU, ATC).

Mirrors the Dota 2 narrative experiment design. For each domain:
- Generates trials at multiple (num_keys, num_updates) operating points
- Evaluates RI and PI accuracy using target LLMs
- Computes accuracy, confidence intervals, effect sizes
- Saves results to v3/results/narrative_{domain}/

Usage:
    .venv/bin/python v3/scripts/run_narrative_interference.py \
        --domain wildlife \
        --model qwen2.5-3b \
        --trials 50

Requires: generators implemented and tested (run test_narrative_generators.py first)
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Operating point grid (mirrors v3 behavioral sweep)
# ---------------------------------------------------------------------------

OPERATING_POINTS = [
    (2, 3), (2, 5), (2, 7),
    (3, 3), (3, 5), (3, 7), (3, 10),
    (5, 3), (5, 5), (5, 7), (5, 10),
    (7, 5), (7, 10),
    (10, 5), (10, 10),
]

DEFAULT_TRIALS = 30  # per cell (use 50+ for final results)
DEFAULT_DOMAIN = "wildlife"

# ---------------------------------------------------------------------------
# Generator registry
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------

def evaluate_trial_local(trial: dict, model_name: str) -> tuple:
    """
    Evaluate a trial using a local HuggingFace model.
    Returns (ri_correct: bool, pi_correct: bool, ri_pred: str, pi_pred: str)
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    narrative = trial["narrative"]
    ri_q = trial["questions"]["RI"]["question"]
    ri_ans = trial["questions"]["RI"]["expected_answer"]
    pi_q = trial["questions"]["PI"]["question"]
    pi_ans = trial["questions"]["PI"]["expected_answer"]

    MODEL_MAP = {
        "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
        "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
        "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
        "smollm2-135m": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "smollm2-360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    }

    if not hasattr(evaluate_trial_local, "_model_cache"):
        evaluate_trial_local._model_cache = {}

    if model_name not in evaluate_trial_local._model_cache:
        model_id = MODEL_MAP.get(model_name, model_name)
        print(f"  Loading model {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16,
        )
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        evaluate_trial_local._model_cache[model_name] = (tokenizer, model, device)

    tokenizer, model, device = evaluate_trial_local._model_cache[model_name]

    def run_query(question: str) -> str:
        prompt = f"{narrative}\n\nQuestion: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=20, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return response.split("\n")[0].strip()

    ri_pred = run_query(ri_q)
    pi_pred = run_query(pi_q)

    ri_correct = str(ri_ans).strip() in ri_pred
    pi_correct = str(pi_ans).strip() in pi_pred

    return ri_correct, pi_correct, ri_pred, pi_pred


# ---------------------------------------------------------------------------
# Wilson confidence interval
# ---------------------------------------------------------------------------

def wilson_ci(n_correct: int, n_total: int, z: float = 1.96) -> tuple:
    if n_total == 0:
        return 0.0, 0.0, 0.0
    p = n_correct / n_total
    denominator = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denominator
    margin = (z * (p * (1 - p) / n_total + z**2 / (4 * n_total**2))**0.5) / denominator
    return p, max(0, center - margin), min(1, center + margin)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(domain: str, model_name: str, n_trials: int, output_dir: Path,
              resume: bool = True):
    gen = get_generator(domain)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / f"{domain}_{model_name.replace('/', '-')}_results.json"

    # Resume from existing results
    if resume and results_file.exists():
        with open(results_file) as f:
            all_results = json.load(f)
        print(f"  Resuming from {len(all_results)} existing results")
    else:
        all_results = {}

    for num_keys, num_updates in OPERATING_POINTS:
        cell_key = f"{num_keys}k_{num_updates}u"
        if cell_key in all_results and len(all_results[cell_key]["trials"]) >= n_trials:
            print(f"  Skipping {cell_key} (already have {len(all_results[cell_key]['trials'])} trials)")
            continue

        print(f"\n  {domain} {cell_key} | model={model_name} | n={n_trials}")

        if cell_key not in all_results:
            all_results[cell_key] = {"trials": [], "num_keys": num_keys, "num_updates": num_updates}

        existing = len(all_results[cell_key]["trials"])

        for trial_idx in range(existing, n_trials):
            seed = trial_idx * 1000 + num_keys * 100 + num_updates
            try:
                trial = gen.generate_trial(
                    num_keys=num_keys, num_updates=num_updates,
                    condition="both", seed=seed
                )
                ri_correct, pi_correct, ri_pred, pi_pred = evaluate_trial_local(trial, model_name)

                all_results[cell_key]["trials"].append({
                    "trial_id": f"{domain}_{cell_key}_{trial_idx}",
                    "seed": seed,
                    "ri_correct": ri_correct,
                    "pi_correct": pi_correct,
                    "ri_pred": ri_pred,
                    "pi_pred": pi_pred,
                    "ri_expected": trial["questions"]["RI"]["expected_answer"],
                    "pi_expected": trial["questions"]["PI"]["expected_answer"],
                    "target_entity": trial["questions"]["RI"]["target_entity"],
                    "tracked_attribute": trial["questions"]["RI"]["target_attribute"],
                    "config": trial.get("config", {}),
                })

                if (trial_idx + 1) % 10 == 0:
                    # Save checkpoint
                    with open(results_file, "w") as f:
                        json.dump(all_results, f, indent=2)
                    ri_acc = sum(t["ri_correct"] for t in all_results[cell_key]["trials"]) / len(all_results[cell_key]["trials"])
                    pi_acc = sum(t["pi_correct"] for t in all_results[cell_key]["trials"]) / len(all_results[cell_key]["trials"])
                    print(f"    [{trial_idx+1}/{n_trials}] RI={ri_acc:.1%} PI={pi_acc:.1%} gap={ri_acc-pi_acc:+.1%}")

            except Exception as e:
                print(f"    Trial {trial_idx} failed: {e}")
                continue

        # Compute final stats for this cell
        trials = all_results[cell_key]["trials"]
        if trials:
            ri_acc = sum(t["ri_correct"] for t in trials) / len(trials)
            pi_acc = sum(t["pi_correct"] for t in trials) / len(trials)
            ri_p, ri_lo, ri_hi = wilson_ci(sum(t["ri_correct"] for t in trials), len(trials))
            pi_p, pi_lo, pi_hi = wilson_ci(sum(t["pi_correct"] for t in trials), len(trials))
            all_results[cell_key]["summary"] = {
                "n_trials": len(trials),
                "ri_accuracy": ri_acc,
                "pi_accuracy": pi_acc,
                "gap": ri_acc - pi_acc,
                "ri_ci": [ri_lo, ri_hi],
                "pi_ci": [pi_lo, pi_hi],
                "pi_gt_ri": pi_acc < ri_acc,
            }
            print(f"  {cell_key}: RI={ri_acc:.1%} [{ri_lo:.1%},{ri_hi:.1%}] | PI={pi_acc:.1%} [{pi_lo:.1%},{pi_hi:.1%}] | gap={ri_acc-pi_acc:+.1%}")

        # Save after each cell
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\n  Results saved to {results_file}")

    # Print summary table
    print(f"\n  {'Cell':<12} {'RI':>8} {'PI':>8} {'Gap':>8} {'PI>RI?':>8}")
    print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for cell_key in sorted(all_results.keys()):
        s = all_results[cell_key].get("summary", {})
        if s:
            ri = s.get("ri_accuracy", 0)
            pi = s.get("pi_accuracy", 0)
            gap = s.get("gap", 0)
            pi_gt_ri = "NO" if s.get("pi_gt_ri") else "YES"
            print(f"  {cell_key:<12} {ri:>8.1%} {pi:>8.1%} {gap:>+8.1%} {pi_gt_ri:>8}")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run narrative interference experiments")
    parser.add_argument("--domain", choices=["wildlife", "icu", "atc", "all"],
                        default=DEFAULT_DOMAIN, help="Domain to run")
    parser.add_argument("--model", default="qwen2.5-3b",
                        help="Model name (qwen2.5-3b, smollm2-360m, etc.)")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS,
                        help="Number of trials per operating point cell")
    parser.add_argument("--output-dir", type=str,
                        default=str(ROOT / "v3" / "results" / "narrative"),
                        help="Output directory for results")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh, don't resume from existing results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    domains_to_run = ["wildlife", "icu", "atc"] if args.domain == "all" else [args.domain]

    start = datetime.now()
    print(f"Starting narrative interference experiments")
    print(f"  Domains: {domains_to_run}")
    print(f"  Model: {args.model}")
    print(f"  Trials/cell: {args.trials}")
    print(f"  Output: {output_dir}")
    print(f"  Start: {start.isoformat()}")
    print()

    for domain in domains_to_run:
        print(f"\n{'='*70}")
        print(f"DOMAIN: {domain.upper()}")
        print(f"{'='*70}")
        try:
            run_sweep(
                domain=domain,
                model_name=args.model,
                n_trials=args.trials,
                output_dir=output_dir / domain,
                resume=not args.no_resume
            )
        except Exception as e:
            print(f"  DOMAIN FAILED: {e}")
            import traceback
            traceback.print_exc()

    elapsed = (datetime.now() - start).total_seconds()
    print(f"\nCompleted in {elapsed:.0f}s")
