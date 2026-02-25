"""
Runner: Load model ONCE, run all mechanistic experiments across operating points.

Avoids re-loading the model for each experiment (saves ~15-20s per call).
Uses monkey-patching on core.model_loader.load_model so each experiment's
main() thinks it's loading fresh, but gets the cached model instead.

Usage:
    cd mechanistic_probing_v2
    uv run python experiments/run_all.py \
        --model google/gemma-3-1b-it \
        --points "2,1 2,2 2,5 5,2" \
        --trials 100

    # Run only specific experiments:
    uv run python experiments/run_all.py \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --points "1,5 2,5" \
        --trials 100 \
        --exps "15 22 23"

    # Skip already-completed experiments:
    uv run python experiments/run_all.py \
        --model google/gemma-3-1b-it \
        --points "2,2" \
        --trials 100 \
        --skip-existing
"""

import sys
import time
import argparse
import importlib
import traceback
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model_loader import load_model
from core.single_token_values import verify_single_token
from core.output import get_output_path


# ── Experiment registry ──────────────────────────────────────────────────
# Maps experiment ID to module filename (without .py)
EXPERIMENT_MODULES = {
    "12": "12_phase2_trial",
    "13": "13_positional_gradient",
    "14": "14_pi_mass_distribution",
    "15": "15_activation_patching",
    "16": "16_head_identification",
    "17": "17_instruction_sensitivity",
    "18": "18_forced_attention",
    "19": "19_positional_bias_sweep",
    "19b": "19b_bias_attention_proof",
    "20": "20_minority_override_analysis",
    "22": "22_query_patching_granular",
    "23": "23_ablation_patching_interaction",
    "25a": "25a_per_head_knockout",
    "25b": "25b_attribution_patching_heads",
    "25c": "25c_observational_head_metrics",
}

# Maps experiment ID to its output JSON name (for --skip-existing)
EXPERIMENT_OUTPUT_NAMES = {
    "12": "logit_lens",
    "13": "positional_gradient",
    "14": "pi_mass_distribution",
    "15": "activation_patching",
    "16": "head_identification",
    "17": "instruction_sensitivity",
    "18": "forced_attention",
    "19": "positional_bias_sweep_blind",
    "19b": "bias_attention_proof",
    "20": "minority_override",
    "22": "query_patching_granular",
    "23": "ablation_patching_interaction",
    "25a": "per_head_knockout",
    "25b": "attribution_patching_heads",
    "25c": "observational_head_metrics",
}

DEFAULT_EXP_ORDER = ["12", "13", "14", "15", "16", "17", "18", "19", "19b", "20", "22", "23", "25c", "25b", "25a"]


def patch_model_loader(cached_model, cached_tokenizer, cached_info):
    """Monkey-patch core.model_loader.load_model to return cached model.

    Every experiment calls load_model() in its main(). This patch makes
    that call return instantly with the pre-loaded model.
    """
    import core.model_loader as ml

    original_load = ml.load_model

    def load_model_cached(model_name, device=None, n_ctx=8192):
        print(f"  [runner] Using cached model (skipping reload of {model_name})")
        return cached_model, cached_tokenizer, cached_info

    ml.load_model = load_model_cached
    return original_load


def restore_model_loader(original_load):
    """Restore original load_model."""
    import core.model_loader as ml
    ml.load_model = original_load


def run_experiment(exp_id, model_name, keys, updates, trials, n_ctx):
    """Run a single experiment by importing its module and calling main().

    Temporarily overrides sys.argv so argparse in the experiment's main()
    sees the right arguments.
    """
    module_name = EXPERIMENT_MODULES[exp_id]

    # Override sys.argv for the experiment's argparse
    saved_argv = sys.argv
    sys.argv = [
        f"experiments/{module_name}.py",
        "--model", model_name,
        "--keys", str(keys),
        "--updates", str(updates),
        "--trials", str(trials),
        "--n-ctx", str(n_ctx),
    ]

    try:
        # Import or reload the module
        full_module = f"experiments.{module_name}"
        if full_module in sys.modules:
            mod = importlib.reload(sys.modules[full_module])
        else:
            mod = importlib.import_module(full_module)

        mod.main()
        return True

    except SystemExit:
        # Some scripts call sys.exit() — catch and continue
        return True
    except Exception:
        traceback.print_exc()
        return False
    finally:
        sys.argv = saved_argv


def parse_points(points_str):
    """Parse '2,1 2,2 2,5 5,2' into [(2,1), (2,2), (2,5), (5,2)]."""
    points = []
    for p in points_str.strip().split():
        parts = p.split(",")
        points.append((int(parts[0]), int(parts[1])))
    return points


def main():
    parser = argparse.ArgumentParser(description="Run all mechanistic experiments with shared model")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--points", required=True,
                        help="Operating points as 'keys,updates' pairs, space-separated. "
                             "E.g., '2,1 2,2 2,5 5,2'")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--exps", default=None,
                        help="Space-separated experiment IDs to run. "
                             "E.g., '15 22 23'. Default: all 12 experiments.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip experiments that already have output JSON files")
    args = parser.parse_args()

    points = parse_points(args.points)
    exp_ids = args.exps.split() if args.exps else DEFAULT_EXP_ORDER

    # Validate experiment IDs
    for eid in exp_ids:
        if eid not in EXPERIMENT_MODULES:
            print(f"ERROR: Unknown experiment ID '{eid}'. Valid: {list(EXPERIMENT_MODULES.keys())}")
            sys.exit(1)

    total_runs = len(points) * len(exp_ids)

    print("=" * 70)
    print("EXPERIMENT RUNNER — Single model load, multiple experiments")
    print("=" * 70)
    print(f"  Model:    {args.model}")
    print(f"  Points:   {points}")
    print(f"  Exps:     {exp_ids}")
    print(f"  Trials:   {args.trials}")
    print(f"  Total:    {total_runs} runs")
    print(f"  Skip existing: {args.skip_existing}")
    print("=" * 70)

    # ── Load model ONCE ──
    t0 = time.time()
    model, tokenizer, info = load_model(args.model, n_ctx=args.n_ctx)

    # Pre-verify single token values (most experiments do this)
    value_to_tid = verify_single_token(tokenizer)
    print(f"Value pool: {len(value_to_tid)} single-token words")

    load_time = time.time() - t0
    print(f"\nModel loaded in {load_time:.1f}s (this cost is now paid ONCE)\n")

    # ── Patch model loader ──
    original_load = patch_model_loader(model, tokenizer, info)

    # ── Run experiments ──
    results_log = []
    run_start = time.time()
    completed = 0

    try:
        for keys, updates in points:
            print(f"\n{'#' * 70}")
            print(f"# OPERATING POINT: {keys}k, {updates}u")
            print(f"{'#' * 70}")

            for exp_id in exp_ids:
                completed += 1
                exp_name = EXPERIMENT_MODULES[exp_id]

                # Check if output already exists
                if args.skip_existing:
                    out_name = EXPERIMENT_OUTPUT_NAMES.get(exp_id, exp_name)
                    out_path = get_output_path(args.model, keys, updates, out_name)
                    if out_path.exists():
                        print(f"\n  [{completed}/{total_runs}] exp {exp_id} at {keys}k,{updates}u — SKIPPED (exists: {out_path.name})")
                        results_log.append((exp_id, keys, updates, "skipped"))
                        continue

                print(f"\n{'=' * 70}")
                print(f"  [{completed}/{total_runs}] exp {exp_id} ({exp_name}) at {keys}k,{updates}u")
                print(f"{'=' * 70}")

                t_exp = time.time()
                success = run_experiment(exp_id, args.model, keys, updates, args.trials, args.n_ctx)
                elapsed = time.time() - t_exp

                status = "OK" if success else "FAILED"
                results_log.append((exp_id, keys, updates, status, elapsed))

                elapsed_total = time.time() - run_start
                remaining = total_runs - completed
                avg_per_run = elapsed_total / completed if completed > 0 else 0
                eta = avg_per_run * remaining

                print(f"\n  [{status}] {elapsed:.0f}s | Total: {elapsed_total/60:.0f}min | ETA: {eta/60:.0f}min")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    finally:
        restore_model_loader(original_load)

    # ── Summary ──
    total_time = time.time() - run_start
    print(f"\n{'=' * 70}")
    print(f"RUNNER COMPLETE — {total_time/60:.1f} minutes (model load: {load_time:.1f}s)")
    print(f"{'=' * 70}")
    print(f"\n  {'Exp':>5} {'Point':>8} {'Status':>8} {'Time':>8}")
    print(f"  {'-' * 35}")
    for entry in results_log:
        exp_id, keys, updates, status = entry[0], entry[1], entry[2], entry[3]
        elapsed = f"{entry[4]:.0f}s" if len(entry) > 4 else "-"
        print(f"  {exp_id:>5} {keys}k,{updates}u {status:>8} {elapsed:>8}")

    n_ok = sum(1 for e in results_log if e[3] == "OK")
    n_fail = sum(1 for e in results_log if e[3] == "FAILED")
    n_skip = sum(1 for e in results_log if e[3] == "skipped")
    print(f"\n  {n_ok} OK, {n_fail} FAILED, {n_skip} skipped")


if __name__ == "__main__":
    main()
