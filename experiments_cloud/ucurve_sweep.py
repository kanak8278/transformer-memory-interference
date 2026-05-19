#!/usr/bin/env python3
"""
U-curve / lost-in-the-middle sweep — Claude Haiku.

Tests recall accuracy at 11 positions across the update sequence under 4 prompt formats.
Paired approach: same stream queried at all positions per trial.
Wilson CI early stopping on positions 1 and nu (first/last anchors).

Usage examples:
  # Run all 4 formats, full grid
  python ucurve_sweep.py

  # Run only 2 formats
  python ucurve_sweep.py --formats block flat_nolabel

  # Run a single format on a subset of cells
  python ucurve_sweep.py --formats flat_short --nk 5 10 --nu 50 100

  # Resume from checkpoint
  python ucurve_sweep.py --resume results/ucurve/checkpoint.json

  # Run just one cell for quick testing
  python ucurve_sweep.py --nk 5 --nu 10 --trials 5 --formats block
"""
import sys, os, json, math, time, threading, argparse
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mechanistic_probing_v2.core.evaluation import bootstrap_ci
from models.model_factory import create_model

from experiments_cloud.ucurve_prompts import (
    generate_stream, make_seed, query_positions,
    prompt_block, prompt_flat_short, prompt_flat_verbose,
    prompt_flat_nolabel, prompt_original,
    LASTQUERY_BUILDERS,
    SYSTEM_PROMPT,
)

# ─── constants ────────────────────────────────────────────────────────────────
ALL_FORMATS  = [
    "block", "flat_short", "flat_verbose", "flat_nolabel", "original",
    "block_last", "flat_short_last", "flat_verbose_last", "flat_nolabel_last",
]
# _last formats: run same numbered queries as base format + one extra "last" query
# Results for the extra query stored under key "last" in positions dict
_LAST_FMTS         = {"block_last", "flat_short_last", "flat_verbose_last", "flat_nolabel_last"}
_STREAM_COUNT_FMTS = {"flat_nolabel", "flat_nolabel_last"}
DEFAULT_NK   = [5, 7, 10, 12, 15]
DEFAULT_NU   = [10, 30, 50, 75, 100]
MAX_TRIALS   = 200
CI_THRESHOLD = 0.05
MIN_TRIALS   = 50    # first convergence check after this many trials
DEFAULT_WORKERS = 20
SAVE_DIR     = "experiments_cloud/results/ucurve"

PROMPT_BUILDERS = {
    "block":             lambda flat, block, test, k: prompt_block(block, test, k),
    "flat_short":        lambda flat, block, test, k: prompt_flat_short(flat, test, k),
    "flat_verbose":      lambda flat, block, test, k: prompt_flat_verbose(flat, test, k),
    "flat_nolabel":      lambda flat, block, test, k: prompt_flat_nolabel(flat, test, k),
    "original":          lambda flat, block, test, k: prompt_original(flat, test, k),
    # _last formats reuse the base format's numbered queries for all positions,
    # then fire one extra "last" query per trial (see LASTQUERY_BUILDERS).
    "block_last":        lambda flat, block, test, k: prompt_block(block, test, k),
    "flat_short_last":   lambda flat, block, test, k: prompt_flat_short(flat, test, k),
    "flat_verbose_last": lambda flat, block, test, k: prompt_flat_verbose(flat, test, k),
    "flat_nolabel_last": lambda flat, block, test, k: prompt_flat_nolabel(flat, test, k),
}

# ─── thread-local model ───────────────────────────────────────────────────────
_thread_local = threading.local()

def get_model(model_name: str):
    if not hasattr(_thread_local, "model"):
        _thread_local.model = create_model(model_name, {"verbose": False})
    return _thread_local.model

# ─── CLI ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="U-curve sweep — Haiku",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model",   default="claude-haiku",
                   help="Model alias (default: claude-haiku)")
    p.add_argument("--formats", nargs="+", default=ALL_FORMATS,
                   choices=ALL_FORMATS, metavar="FORMAT",
                   help=f"Formats to run (default: all 4). "
                        f"Choices: {ALL_FORMATS}")
    p.add_argument("--nk",      nargs="+", type=int, default=DEFAULT_NK,
                   help=f"Key levels (default: {DEFAULT_NK})")
    p.add_argument("--nu",      nargs="+", type=int, default=DEFAULT_NU,
                   help=f"Update levels (default: {DEFAULT_NU})")
    p.add_argument("--trials",  type=int, default=MAX_TRIALS,
                   help=f"Max trials per cell (default: {MAX_TRIALS})")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                   help=f"Concurrent API calls (default: {DEFAULT_WORKERS})")
    p.add_argument("--ci",      type=float, default=CI_THRESHOLD,
                   help=f"Wilson CI threshold for early stopping (default: {CI_THRESHOLD})")
    p.add_argument("--save-dir", default=SAVE_DIR,
                   help=f"Output directory (default: {SAVE_DIR})")
    p.add_argument("--resume",  default=None,
                   help="Path to checkpoint.json to resume from")
    p.add_argument("--no-early-stop", action="store_true",
                   help="Disable Wilson early stopping — run full trial budget")
    p.add_argument("--n-positions", type=int, default=21,
                   help="Number of evenly spaced query positions per cell (default: 21)")
    p.add_argument("--dataset", default="ARBITRARY_SINGLE",
                   choices=["ARBITRARY_SINGLE", "SEMANTIC_MULTI"],
                   help="Dataset to use (default: ARBITRARY_SINGLE)")
    return p.parse_args()

# ─── pre-run verification ─────────────────────────────────────────────────────
def verify_expected_values(nk: int, nu: int, dataset: str = "ARBITRARY_SINGLE",
                           n_trials: int = 5) -> list[str]:
    """
    For each format, reconstruct trials and verify that expected values
    are semantically correct given what the prompt actually shows.
    Returns a list of error strings (empty = all good).
    """
    from experiments_cloud.ucurve_prompts import (
        prompt_block, prompt_flat_short, prompt_flat_verbose,
        prompt_flat_nolabel, prompt_original,
        LASTQUERY_BUILDERS,
    )
    errors = []
    positions = query_positions(nu)

    for trial_idx in range(n_trials):
        seed = make_seed(nk, nu, trial_idx)
        cats, test, vals, flat, block = generate_stream(nk, nu, seed, dataset=dataset)
        all_values  = vals[test]
        stream_vals = [i["value"] for i in flat if i["category"] == test]

        for k in positions:
            # Labeled formats: expected = chronological update k
            for fmt in ("block", "flat_short", "flat_verbose"):
                exp = all_values[k - 1]
                if fmt == "block":
                    entry_val = next(
                        i["value"] for i in block[k-1] if i["category"] == test
                    )
                else:
                    entry_val = next(
                        i["value"] for i in flat
                        if i["category"] == test and i["update_idx"] == k
                    )
                if exp != entry_val:
                    errors.append(
                        f"[{fmt}] trial={trial_idx} pos={k}: "
                        f"expected={exp!r} but prompt shows {entry_val!r}"
                    )

            # flat_nolabel: kth stream occurrence
            exp_nolabel = stream_vals[k - 1]
            if exp_nolabel != stream_vals[k - 1]:
                errors.append(
                    f"[flat_nolabel] trial={trial_idx} pos={k}: indexing error"
                )

            # original: k=1 → first stream occurrence, else last
            exp_orig = stream_vals[0] if k == 1 else stream_vals[-1]
            if exp_orig != (stream_vals[0] if k == 1 else stream_vals[-1]):
                errors.append(
                    f"[original] trial={trial_idx} pos={k}: expected {exp_orig!r}"
                )

        # _last formats: extra "last" query expected = last chronological value
        for fmt_last, ref_is_stream in [
            ("block_last",        False),
            ("flat_short_last",   False),
            ("flat_verbose_last", False),
            ("flat_nolabel_last", True),
        ]:
            exp_last = stream_vals[-1] if ref_is_stream else all_values[-1]
            # Confirm ground truth is consistent
            actual_last = stream_vals[-1] if ref_is_stream else all_values[-1]
            if exp_last != actual_last:
                errors.append(
                    f"[{fmt_last}] trial={trial_idx}: last-query expected mismatch"
                )

    return errors


# ─── Wilson CI ───────────────────────────────────────────────────────────────
def wilson_hw(n: int, k: int, z: float = 1.96) -> float:
    if n == 0: return 1.0
    n_t = n + z**2
    p_t = (k + z**2 / 2) / n_t
    return z * math.sqrt(p_t * (1 - p_t) / n_t)

def converged(correct_by_pos: dict[int, list[bool]], threshold: float) -> bool:
    """True when ALL positions have CI half-width ≤ threshold.
    This ensures middle positions (where the U-curve dip appears) also have
    tight enough CIs before we stop — not just the first/last anchors.
    """
    for results in correct_by_pos.values():
        n = len(results)
        k = sum(results)
        if wilson_hw(n, k) > threshold:
            return False
    return True

# ─── error classification ─────────────────────────────────────────────────────
def classify(predicted: str, expected: str, all_values: list[str]) -> dict:
    """
    Returns:
        error_type:           correct | in_sequence | garbage
        predicted_idx:        0-based index in all_values, None if garbage
        predicted_relative_pos: 0.0=first, 1.0=last, None if garbage
    """
    pred = predicted.lower().strip()
    exp  = expected.lower().strip()

    if pred == exp or exp in pred or pred.startswith(exp):
        return {"error_type": "correct",
                "predicted_idx": all_values.index(expected),
                "predicted_relative_pos": round(
                    all_values.index(expected) / max(len(all_values)-1, 1), 4)}

    for idx, val in enumerate(all_values):
        v = val.lower()
        if v == pred or v in pred or pred.startswith(v):
            return {"error_type": "in_sequence",
                    "predicted_idx": idx,
                    "predicted_relative_pos": round(idx / max(len(all_values)-1, 1), 4)}

    return {"error_type": "garbage",
            "predicted_idx": None,
            "predicted_relative_pos": None}

# ─── single API call ──────────────────────────────────────────────────────────
def call_api(model_name: str, prompt: str) -> tuple[str, dict]:
    """Returns (raw_output, token_usage)."""
    model = get_model(model_name)
    try:
        raw = model.generate(prompt)
        usage = {
            "input_tokens":  getattr(model, "last_input_tokens",  None),
            "output_tokens": getattr(model, "last_output_tokens", None),
        }
        return raw.strip(), usage
    except Exception as e:
        return f"[ERROR: {e}]", {}

# ─── run one trial (paired: all positions, one stream) ────────────────────────
def run_trial(model_name: str, fmt_name: str, nk: int, nu: int,
              trial_idx: int, positions: list[int],
              workers: int, dataset: str = "ARBITRARY_SINGLE") -> dict:
    """
    Generates the stream, builds prompts for all positions, fires them concurrently.
    Returns a dict keyed by position index.
    """
    seed = make_seed(nk, nu, trial_idx)
    cats, test, vals, flat, block = generate_stream(nk, nu, seed, dataset=dataset)
    all_values = vals[test]   # ordered list: all_values[k-1] = k-th update

    # For 'original' format: expected answer is stream-position order, not
    # chronological order. "first" = first occurrence in the shuffled stream,
    # "last" = last occurrence. Middle positions all ask "last".
    stream_vals = [i["value"] for i in flat if i["category"] == test]

    builder = PROMPT_BUILDERS[fmt_name]
    prompts = {k: builder(flat, block, test, k) for k in positions}

    # _last formats: also fire one extra "last" semantic query per trial.
    # Stored under key "last" (string) alongside the integer position results.
    if fmt_name in _LAST_FMTS:
        lq_builder = LASTQUERY_BUILDERS[fmt_name]
        items = block if "block" in fmt_name else flat
        prompts["last"] = lq_builder(items, test)

    # Fire all position queries (+ optional "last") concurrently
    results = {}
    with ThreadPoolExecutor(max_workers=min(workers, len(prompts))) as ex:
        futures = {ex.submit(call_api, model_name, prompt): k
                   for k, prompt in prompts.items()}
        for future in as_completed(futures):
            k = futures[future]
            raw, usage = future.result()
            if k == "last":
                # Semantic "last" query: expected = last chronological value
                if fmt_name in _STREAM_COUNT_FMTS:
                    expected = stream_vals[-1]
                    ref_list = stream_vals
                else:
                    expected = all_values[-1]
                    ref_list = all_values
            elif fmt_name == "original":
                # k=1 asks "first" → first stream occurrence; else "last" → last
                expected = stream_vals[0] if k == 1 else stream_vals[-1]
                ref_list = stream_vals
            elif fmt_name in _STREAM_COUNT_FMTS:
                # flat_nolabel / flat_nolabel_last: kth stream occurrence
                expected = stream_vals[k - 1]
                ref_list = stream_vals
            else:
                # Labeled formats + _last variants: chronological update k
                expected = all_values[k - 1]
                ref_list = all_values
            cls        = classify(raw, expected, ref_list)
            results[k] = {
                "expected":              expected,
                "predicted":             raw.strip(),
                "output_raw":            raw,
                "correct":               cls["error_type"] == "correct",
                "error_type":            cls["error_type"],
                "predicted_idx":         cls["predicted_idx"],
                "predicted_relative_pos":cls["predicted_relative_pos"],
                "token_usage":           usage,
            }

    return {
        "trial_idx":     trial_idx,
        "seed":          seed,
        "test_category": test,
        "categories":    cats,
        "num_keys":      nk,
        "num_updates":   nu,
        "stream_vals":   stream_vals if fmt_name == "original" else None,
        "positions":     results,
    }

# ─── cell runner (one nk × nu × format) ──────────────────────────────────────
def run_cell(model_name: str, fmt_name: str, nk: int, nu: int,
             max_trials: int, positions: list[int], workers: int,
             ci_threshold: float, early_stop: bool,
             dataset: str = "ARBITRARY_SINGLE") -> tuple[dict, list]:
    """
    Runs up to max_trials trials for one (nk, nu, format) cell.
    Early stops when positions[0] and positions[-1] both converge.
    Returns (cell_summary, trial_details_list).
    """
    pos_first = positions[0]   # position 1  (RI anchor)
    pos_last  = positions[-1]  # position nu (PI anchor)

    trial_details = []
    # Track correctness per position for Wilson.
    # _last formats also track the extra semantic "last" query under key "last".
    all_pos_keys = list(positions) + (["last"] if fmt_name in _LAST_FMTS else [])
    correct_by_pos = {k: [] for k in all_pos_keys}
    stopped_early = False

    for trial_idx in range(max_trials):
        trial = run_trial(model_name, fmt_name, nk, nu,
                          trial_idx, positions, workers, dataset=dataset)
        trial_details.append(trial)

        for k in all_pos_keys:
            if k in trial["positions"]:
                correct_by_pos[k].append(trial["positions"][k]["correct"])

        n_done = trial_idx + 1
        if (early_stop
                and n_done >= MIN_TRIALS
                and converged(correct_by_pos, ci_threshold)):
            stopped_early = True
            break

    # Build per-position stats (integer positions + "last" for _last formats)
    position_stats = {}
    for k in all_pos_keys:
        corrects = correct_by_pos[k]
        n = len(corrects)
        acc = sum(corrects) / n if n else 0.0
        hw  = wilson_hw(n, sum(corrects))
        _, ci_lo, ci_hi = bootstrap_ci(corrects)
        position_stats[str(k)] = {
            "accuracy":  round(acc, 4),
            "ci_lower":  round(ci_lo, 4),
            "ci_upper":  round(ci_hi, 4),
            "wilson_hw": round(hw, 4),
            "n":         n,
        }

    cell_summary = {
        "num_keys":     nk,
        "num_updates":  nu,
        "format":       fmt_name,
        "n_trials":     len(trial_details),
        "max_trials":   max_trials,
        "stopped_early": stopped_early,
        "positions":    position_stats,
        # Convenience: first/last accuracy for quick comparison with existing RI/PI
        "first_acc":    position_stats[str(pos_first)]["accuracy"],
        "last_acc":     position_stats[str(pos_last)]["accuracy"],
        "gap":          round(
            position_stats[str(pos_first)]["accuracy"] -
            position_stats[str(pos_last)]["accuracy"], 4),
    }
    return cell_summary, trial_details

# ─── checkpoint helpers ───────────────────────────────────────────────────────
def load_checkpoint(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

def save_checkpoint(data: dict, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)

def cell_done(results: dict, fmt: str, nk: int, nu: int) -> bool:
    return (fmt in results.get("cells", {}).get(f"{nk}_{nu}", {}))

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Validate formats
    bad = [f for f in args.formats if f not in ALL_FORMATS]
    if bad:
        print(f"Unknown formats: {bad}. Valid: {ALL_FORMATS}")
        sys.exit(1)

    # Per-model subdirectory so multiple models don't share a checkpoint file
    model_save_dir = Path(args.save_dir) / args.model
    ckpt_path      = model_save_dir / "checkpoint.json"

    print(f"\nU-curve sweep")
    print(f"  model   : {args.model}")
    print(f"  dataset : {args.dataset}")
    print(f"  formats : {args.formats}")
    print(f"  nk grid : {args.nk}")
    print(f"  nu grid : {args.nu}")
    print(f"  trials  : up to {args.trials} per cell per format")
    print(f"  workers : {args.workers}")
    print(f"  CI stop : {'disabled' if args.no_early_stop else f'Wilson {args.ci}'}")
    print(f"  save to : {model_save_dir}")
    print()

    # Verify API connection first so we can capture the real model_id
    print("Verifying API connection...", end=" ", flush=True)
    test_model = create_model(args.model, {"verbose": False})
    real_model_id = test_model.model_id
    print(f"✓  ({real_model_id})\n")

    # Load or init results
    if args.resume and Path(args.resume).exists():
        results = load_checkpoint(args.resume)
        print(f"Resuming from {args.resume}")
    else:
        results = {
            "model":      args.model,
            "model_id":   real_model_id,
            "dataset":    args.dataset,
            "config": {
                "formats":      args.formats,
                "nk_levels":    args.nk,
                "nu_levels":    args.nu,
                "max_trials":   args.trials,
                "ci_threshold": args.ci,
                "n_positions":  args.n_positions,
                "early_stop":   not args.no_early_stop,
            },
            "start_time":   datetime.now(timezone.utc).isoformat(),
            "cells":        {},
            "trial_details":{},
        }

    # Verify expected-value logic
    print("Verifying expected-value correctness...", end=" ", flush=True)
    sample_nk = args.nk[0]
    sample_nu = args.nu[0]
    errs = verify_expected_values(sample_nk, sample_nu, dataset=args.dataset)
    if errs:
        print(f"\n  FAILED — {len(errs)} error(s):")
        for e in errs[:5]: print(f"    {e}")
        sys.exit(1)
    print(f"✓  (nk={sample_nk}, nu={sample_nu}, 5 trials checked)")

    total_cells = len(args.nk) * len(args.nu) * len(args.formats)
    done_cells  = 0
    skipped     = 0

    for nk in args.nk:
        for nu in args.nu:
            cell_key = f"{nk}_{nu}"
            positions = query_positions(nu, n_points=args.n_positions)

            for fmt in args.formats:
                if cell_done(results, fmt, nk, nu):
                    skipped += 1
                    done_cells += 1
                    continue

                done_cells += 1
                print(f"[{done_cells}/{total_cells}] nk={nk} nu={nu} fmt={fmt}  "
                      f"positions={positions}", flush=True)

                t0 = time.time()
                cell_sum, trial_det = run_cell(
                    model_name  = args.model,
                    fmt_name    = fmt,
                    nk          = nk,
                    nu          = nu,
                    max_trials  = args.trials,
                    positions   = positions,
                    workers     = args.workers,
                    ci_threshold= args.ci,
                    early_stop  = not args.no_early_stop,
                    dataset     = args.dataset,
                )
                elapsed = time.time() - t0

                # Store in results
                if cell_key not in results["cells"]:
                    results["cells"][cell_key] = {}
                results["cells"][cell_key][fmt] = cell_sum

                if cell_key not in results["trial_details"]:
                    results["trial_details"][cell_key] = {}
                results["trial_details"][cell_key][fmt] = trial_det

                early_tag = " [early stop]" if cell_sum["stopped_early"] else ""
                print(f"  → n={cell_sum['n_trials']} trials{early_tag}  "
                      f"first={cell_sum['first_acc']:.3f}  "
                      f"last={cell_sum['last_acc']:.3f}  "
                      f"gap={cell_sum['gap']:+.3f}  "
                      f"({elapsed:.0f}s)", flush=True)

                # Save checkpoint after every cell
                results["last_updated"] = datetime.now(timezone.utc).isoformat()
                save_checkpoint(results, ckpt_path)

    results["end_time"] = datetime.now(timezone.utc).isoformat()
    save_checkpoint(results, ckpt_path)

    # Also save a clean final file
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = model_save_dir / f"ucurve_{ts}.json"
    save_checkpoint(results, final_path)

    print(f"\n✓ Done. Skipped {skipped} already-complete cells.")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Final file : {final_path}")


if __name__ == "__main__":
    main()
