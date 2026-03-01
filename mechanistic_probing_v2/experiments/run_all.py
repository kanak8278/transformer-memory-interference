"""
Runner: Load model ONCE, run all mechanistic experiments across operating points.

Two-phase pipeline:
  Phase 1 (no heads needed):   12, 13, 14, 15, 22, 25a, 25b, 25c
  Phase 2 (heads from 25a):    17b, 18b, 19, 19b, 20, 21a, 23, 25d, 30

run_all.py handles both phases automatically:
  1. Runs phase 1 experiments
  2. Reads 25a output to extract top primacy heads
  3. Runs phase 2 experiments with those heads

Usage:
    cd mechanistic_probing_v2

    # Full pipeline (both phases):
    python experiments/run_all.py \\
        --model Qwen/Qwen2.5-1.5B-Instruct \\
        --points "1,3 2,5" \\
        --trials 100

    # Phase 1 only (no heads needed):
    python experiments/run_all.py \\
        --model Qwen/Qwen2.5-1.5B-Instruct \\
        --points "1,3" --trials 100 --phase 1

    # Phase 2 only (provide heads manually):
    python experiments/run_all.py \\
        --model Qwen/Qwen2.5-1.5B-Instruct \\
        --points "1,3" --trials 100 --phase 2 \\
        --heads "8,3 0,7 0,3"

    # Run only specific experiments:
    python experiments/run_all.py \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --points "2,3" --trials 50 --exps "12 15 25a"

    # Skip already-completed experiments:
    python experiments/run_all.py \\
        --model Qwen/Qwen2.5-1.5B-Instruct \\
        --points "1,3 2,5" --trials 100 --skip-existing
"""

import sys
import json
import time
import argparse
import importlib
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model_loader import load_model, verify_single_token
from core.output import get_output_path, get_output_dir


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT REGISTRY
# ═══════════════════════════════════════════════════════════════════════════

# Phase 1: don't need head identification
PHASE1_EXPS = ["12", "13", "14", "15", "22", "25b", "25c", "25a"]

# Phase 2: need --heads from 25a output
PHASE2_EXPS = ["21a", "17b", "18b", "19", "19b", "20", "23", "25d", "30"]

EXPERIMENT_MODULES = {
    "12":  "12_phase2_trial",
    "13":  "13_positional_gradient",
    "14":  "14_pi_mass_distribution",
    "15":  "15_activation_patching",
    "17b": "17b_instruction_sensitivity_causal",
    "18b": "18b_forced_attention_causal",
    "19":  "19_positional_bias_sweep",
    "19b": "19b_bias_attention_proof",
    "20":  "20_minority_override_analysis",
    "21a": "21a_logit_lens_under_ablation",
    "22":  "22_query_patching_granular",
    "23":  "23_ablation_patching_interaction",
    "25a": "25a_per_head_knockout",
    "25b": "25b_attribution_patching_heads",
    "25c": "25c_observational_head_metrics",
    "25d": "25d_knockout_validation",
    "30":  "30_copy_suppression_test",
}

EXPERIMENT_OUTPUT_NAMES = {
    "12":  "logit_lens",
    "13":  "positional_gradient",
    "14":  "pi_mass_distribution",
    "15":  "activation_patching",
    "17b": "instruction_sensitivity_causal",
    "18b": "forced_attention_causal",
    "19":  "positional_bias_sweep_blind",
    "19b": "bias_attention_proof",
    "20":  "minority_override",
    "21a": "logit_lens_under_ablation",
    "22":  "query_patching_granular",
    "23":  "ablation_patching_interaction",
    "25a": "per_head_knockout",
    "25b": "attribution_patching_heads",
    "25c": "observational_head_metrics",
    "25d": "knockout_validation",
    "30":  "copy_suppression_test",
}

# Experiments that require --heads argument (phase 2)
NEEDS_HEADS = {"17b", "18b", "19", "19b", "21a", "23", "30"}

# Experiments that require --primacy-heads AND --recency-heads
NEEDS_PRIMACY_RECENCY = {"20"}

# Experiments that require --target-heads (25d uses different param name)
NEEDS_TARGET_HEADS = {"25d"}

# Number of top heads to pass from 25a
N_TOP_HEADS = 5


# ═══════════════════════════════════════════════════════════════════════════
# HEAD EXTRACTION FROM 25a
# ═══════════════════════════════════════════════════════════════════════════

def load_heads_from_25a(model_name, keys, updates, n_top=N_TOP_HEADS):
    """Load top primacy and recency heads from 25a results.

    Returns (primacy_heads_str, recency_heads_str) — space-separated "layer,head" strings.
    E.g., ("8,3 0,7 0,3", "12,5 16,2 22,1")
    """
    out_dir = get_output_dir(model_name, keys, updates)

    # Find latest per_head_knockout file
    knockout_files = sorted(out_dir.glob("per_head_knockout*.json"))
    if not knockout_files:
        return None, None

    latest = knockout_files[-1]
    with open(latest) as f:
        data = json.load(f)

    primacy = data.get("top_primacy_heads", [])[:n_top]
    recency = data.get("top_retrieval_heads", [])[:n_top]

    if not primacy:
        return None, None

    primacy_str = " ".join(f"{h['layer']},{h['head']}" for h in primacy)
    recency_str = " ".join(f"{h['layer']},{h['head']}" for h in recency)
    causal_str = " ".join(f"{h['causal_effect']:.3f}" for h in primacy)

    return primacy_str, recency_str, causal_str


# ═══════════════════════════════════════════════════════════════════════════
# MODEL LOADER PATCHING
# ═══════════════════════════════════════════════════════════════════════════

def patch_model_loader(cached_model, cached_tokenizer, cached_info):
    """Monkey-patch load_model to return cached model — avoids reloading."""
    import core.model_loader as ml
    original_load = ml.load_model

    def load_model_cached(model_name, device=None, n_ctx=8192, **kwargs):
        print(f"  [runner] Using cached model")
        return cached_model, cached_tokenizer, cached_info

    ml.load_model = load_model_cached
    return original_load


def restore_model_loader(original_load):
    import core.model_loader as ml
    ml.load_model = original_load


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_experiment(exp_id, model_name, keys, updates, trials, n_ctx,
                   heads=None, primacy_heads=None, recency_heads=None,
                   causal_effects=None):
    """Run a single experiment by importing its module and calling main()."""
    module_name = EXPERIMENT_MODULES[exp_id]

    argv = [
        f"experiments/{module_name}.py",
        "--model", model_name,
        "--keys", str(keys),
        "--updates", str(updates),
        "--trials", str(trials),
        "--n-ctx", str(n_ctx),
    ]

    # Add head arguments based on experiment type
    if exp_id in NEEDS_HEADS and heads:
        argv += ["--heads", heads]
    elif exp_id in NEEDS_PRIMACY_RECENCY and primacy_heads and recency_heads:
        argv += ["--primacy-heads", primacy_heads,
                 "--recency-heads", recency_heads]
        if causal_effects:
            argv += ["--causal-effects", causal_effects]
    elif exp_id in NEEDS_TARGET_HEADS and heads:
        # 25d uses --target-heads and needs --primary-point and --secondary-point
        argv = [
            f"experiments/{module_name}.py",
            "--model", model_name,
            "--primary-point", f"{keys},{updates}",
            "--secondary-point", f"{keys},{min(updates * 2, 20)}",
            "--target-heads", heads,
            "--sweep-trials", str(min(trials, 50)),
            "--targeted-trials", str(trials),
            "--n-ctx", str(n_ctx),
        ]

    saved_argv = sys.argv
    sys.argv = argv

    try:
        full_module = f"experiments.{module_name}"
        if full_module in sys.modules:
            mod = importlib.reload(sys.modules[full_module])
        else:
            mod = importlib.import_module(full_module)
        mod.main()
        return True
    except SystemExit:
        return True
    except Exception:
        traceback.print_exc()
        return False
    finally:
        sys.argv = saved_argv


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def parse_points(points_str):
    """Parse '2,1 2,2' into [(2,1), (2,2)]."""
    return [(int(p.split(",")[0]), int(p.split(",")[1]))
            for p in points_str.strip().split()]


def main():
    parser = argparse.ArgumentParser(description="Run all mechanistic experiments with shared model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--points", required=True,
                        help="Operating points as 'keys,updates' pairs. E.g., '2,3 1,5'")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--phase", type=int, default=0, choices=[0, 1, 2],
                        help="0=both phases, 1=phase1 only, 2=phase2 only")
    parser.add_argument("--heads", default=None,
                        help="Manual heads for phase 2 (overrides 25a auto-extraction). "
                             "E.g., '8,3 0,7 0,3'")
    parser.add_argument("--n-top-heads", type=int, default=N_TOP_HEADS,
                        help=f"Number of top heads to use from 25a (default: {N_TOP_HEADS})")
    parser.add_argument("--exps", default=None,
                        help="Space-separated experiment IDs to run. Overrides --phase.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip experiments that already have output JSON files")
    args = parser.parse_args()

    points = parse_points(args.points)

    # Determine which experiments to run
    if args.exps:
        exp_ids = args.exps.split()
    elif args.phase == 1:
        exp_ids = PHASE1_EXPS
    elif args.phase == 2:
        exp_ids = PHASE2_EXPS
    else:
        exp_ids = PHASE1_EXPS + PHASE2_EXPS  # both phases

    # Validate
    for eid in exp_ids:
        if eid not in EXPERIMENT_MODULES:
            print(f"ERROR: Unknown experiment ID '{eid}'. Valid: {sorted(EXPERIMENT_MODULES.keys())}")
            sys.exit(1)

    # Phase 2 without heads — check we can auto-extract from 25a
    phase2_exps = [e for e in exp_ids if e in PHASE2_EXPS]
    if phase2_exps and not args.heads and "25a" not in exp_ids:
        print(f"  NOTE: Phase 2 experiments {phase2_exps} need heads from 25a.")
        print(f"  Will attempt to auto-extract from existing 25a results.")

    total_runs = len(points) * len(exp_ids)

    print("=" * 70)
    print("EXPERIMENT RUNNER")
    print("=" * 70)
    print(f"  Model:   {args.model}")
    print(f"  Points:  {points}")
    print(f"  Exps:    {exp_ids}")
    print(f"  Trials:  {args.trials}")
    print(f"  Total:   {total_runs} runs")
    if args.heads:
        print(f"  Heads:   {args.heads} (manual)")
    print("=" * 70)

    # Load model once
    t0 = time.time()
    model, tokenizer, info = load_model(args.model, n_ctx=args.n_ctx)
    candidate_pool_size = len(verify_single_token(tokenizer))
    print(f"  {candidate_pool_size} single-token values verified")
    load_time = time.time() - t0
    print(f"  Model loaded in {load_time:.1f}s (paid once)\n")

    original_load = patch_model_loader(model, tokenizer, info)

    results_log = []
    run_start = time.time()
    completed = 0

    try:
        for keys, updates in points:
            print(f"\n{'#' * 70}")
            print(f"# OPERATING POINT: {keys}k, {updates}u")
            print(f"{'#' * 70}")

            # Per-point head state (filled after 25a runs)
            point_heads = args.heads  # manual override
            point_primacy = None
            point_recency = None
            point_causal = None

            for exp_id in exp_ids:
                completed += 1
                exp_name = EXPERIMENT_MODULES[exp_id]

                # Auto-extract heads after 25a completes
                if exp_id != "25a" and exp_id in (PHASE2_EXPS) and point_heads is None:
                    result = load_heads_from_25a(args.model, keys, updates, args.n_top_heads)
                    if result and result[0]:
                        point_heads, point_recency, point_causal = result
                        point_primacy = point_heads
                        print(f"\n  [runner] Auto-extracted from 25a:")
                        print(f"    Primacy heads: {point_heads}")
                        print(f"    Recency heads: {point_recency}")
                    else:
                        print(f"\n  [runner] WARNING: No 25a results found for {keys}k,{updates}u.")
                        print(f"    Phase 2 experiments will be skipped.")

                # Skip if no heads available for head-dependent experiments
                needs_heads_exp = (exp_id in NEEDS_HEADS or
                                   exp_id in NEEDS_PRIMACY_RECENCY or
                                   exp_id in NEEDS_TARGET_HEADS)
                if needs_heads_exp and point_heads is None:
                    print(f"\n  [{completed}/{total_runs}] exp {exp_id} — SKIPPED (no heads available)")
                    results_log.append((exp_id, keys, updates, "skipped_no_heads"))
                    continue

                # Skip existing
                if args.skip_existing:
                    out_name = EXPERIMENT_OUTPUT_NAMES.get(exp_id, exp_name)
                    out_path = get_output_path(args.model, keys, updates, out_name)
                    existing = list(get_output_dir(args.model, keys, updates).glob(
                        f"{out_name}_*.json"))
                    if existing:
                        print(f"\n  [{completed}/{total_runs}] exp {exp_id} — SKIPPED (exists)")
                        results_log.append((exp_id, keys, updates, "skipped"))
                        continue

                print(f"\n{'=' * 70}")
                print(f"  [{completed}/{total_runs}] exp {exp_id} ({exp_name}) at {keys}k,{updates}u")
                print(f"{'=' * 70}")

                t_exp = time.time()
                success = run_experiment(
                    exp_id, args.model, keys, updates, args.trials, args.n_ctx,
                    heads=point_heads,
                    primacy_heads=point_primacy,
                    recency_heads=point_recency,
                    causal_effects=point_causal,
                )
                elapsed = time.time() - t_exp
                status = "OK" if success else "FAILED"
                results_log.append((exp_id, keys, updates, status, elapsed))

                elapsed_total = time.time() - run_start
                remaining = total_runs - completed
                avg = elapsed_total / completed
                print(f"\n  [{status}] {elapsed:.0f}s | Total: {elapsed_total/60:.1f}min | "
                      f"ETA: {avg * remaining / 60:.1f}min")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        restore_model_loader(original_load)

    # Summary
    total_time = time.time() - run_start
    print(f"\n{'=' * 70}")
    print(f"DONE — {total_time/60:.1f} min (model load: {load_time:.1f}s)")
    print(f"{'=' * 70}")
    print(f"\n  {'Exp':>5} {'Point':>8} {'Status':>15} {'Time':>8}")
    print(f"  {'-' * 45}")
    for entry in results_log:
        exp_id, k, u, status = entry[0], entry[1], entry[2], entry[3]
        t = f"{entry[4]:.0f}s" if len(entry) > 4 else "-"
        print(f"  {exp_id:>5} {k}k,{u}u {status:>15} {t:>8}")

    n_ok = sum(1 for e in results_log if e[3] == "OK")
    n_fail = sum(1 for e in results_log if e[3] == "FAILED")
    n_skip = sum(1 for e in results_log if "skip" in e[3])
    print(f"\n  {n_ok} OK, {n_fail} FAILED, {n_skip} skipped")


if __name__ == "__main__":
    main()
