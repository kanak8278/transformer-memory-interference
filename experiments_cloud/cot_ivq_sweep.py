"""CoT vs non-CoT interior-value-query sweep on Claude models.

Design
------
One stimulus, one prompt, two arms. The prompt is byte-identical in both; the
only manipulated variable is whether extended thinking is enabled on the API
call. Temperature is 1 in BOTH arms — extended thinking requires it, so the
non-CoT arm matches rather than leaving temperature confounded with reasoning.
Seeds are blake2b, so a given (cell, trial) yields the same stream in both arms
and the comparison is paired trial-by-trial.

Answers come back inside <answer>...</answer> and are compared by equality. The
containment rule in ucurve_sweep.classify is not used: with reasoning in play it
credits any response that merely mentions the target value.

Early stopping
--------------
PER POSITION, not per cell. The existing sweep stops a cell only when every
position has converged, so one mid-accuracy position (Wilson half-width ~0.069
at n=200, p=0.5) holds the entire cell at the cap — which is why the previous run
shows n=200 everywhere. Here each position drops out as soon as its own CI is
tight enough, and later trials simply stop querying it. Floor and ceiling
positions converge around n~50, so the saving is large on exactly the cells where
most positions sit at 0.

Usage
-----
  # eyeball the prompt, no API calls
  uv run python experiments_cloud/cot_ivq_sweep.py --dry-run

  # pilot: hardest cell, both arms, small n — sets thinking_budget/max_tokens
  uv run python experiments_cloud/cot_ivq_sweep.py --arm cot   --pilot
  uv run python experiments_cloud/cot_ivq_sweep.py --arm nocot --pilot

  # full run, one model+arm per process
  uv run python experiments_cloud/cot_ivq_sweep.py \
      --model claude-haiku-4-5-20251001 --arm cot
"""
import argparse
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from experiments_cloud.cot_ivq_prompts import (  # noqa: E402
    ANSWER_INSTRUCTION, SEED_VERSION, build_trial, query_positions,
)
from experiments_cloud.cot_ivq_score import score_response  # noqa: E402
from models.model_factory import create_model  # noqa: E402

# Dated IDs, not aliases. Aliases resolve at call time, so a pinned run must name
# the snapshot. It also sidesteps claude_model.CLAUDE_REASONING_MODELS, which
# auto-enables thinking for the alias 'claude-4.5-opus' but not for 'claude-opus'
# — same weights, opposite behaviour on the variable under test.
MODELS = {
    "haiku":  "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-5-20250929",
    "opus":   "claude-opus-4-5-20251101",
}

CELLS = [(5, 10), (10, 10), (5, 20), (10, 20), (5, 50), (10, 50)]
PILOT_CELLS = [(10, 50)]           # hardest cell: longest stream, widest interior

DATASET = "SEMANTIC_MULTI"
N_POSITIONS = 16
MIN_TRIALS = 30
MAX_TRIALS = 200
CI_THRESHOLD = 0.05

_tls = threading.local()


def wilson_hw(n: int, k: int, z: float = 1.96) -> float:
    """Half-width of the Wilson score interval. Same formula as ucurve_sweep."""
    if n == 0:
        return 1.0
    n_t = n + z ** 2
    p_t = (k + z ** 2 / 2) / n_t
    return z * math.sqrt(p_t * (1 - p_t) / n_t)


ARMS = ("nocot", "cot_thinking")
PREFILL = "<answer>"


def model_config(arm: str, max_tokens: int, thinking_budget: int) -> dict:
    """Per-arm request configuration.

    Two arms. Note that `thinking=False` alone does NOT give a no-reasoning
    control: with nothing stopping them, all three models enumerate the stream in
    the visible text block, "No explanation" notwithstanding. Suppressing
    reasoning takes the prefill AND the stop sequence.

      nocot         thinking off + assistant prefill '<answer>'. The value is
                    the first token generated, so there is nowhere for reasoning
                    to go — no thinking block AND no visible reasoning. The only
                    genuine no-CoT condition.
      cot_thinking  thinking on. Reasoning happens in a dedicated channel and is
                    saved verbatim; the answer cannot be contaminated by it.
                    This is the CoT arm.

    Temperature is 1.0 everywhere: extended thinking mandates it, so the control
    arms match rather than leaving temperature confounded with reasoning.
    """
    cfg = {
        "verbose": False,
        "temperature": 1.0,
        "max_tokens": max_tokens,
    }
    if arm == "cot_thinking":
        cfg["thinking"] = True
        cfg["thinking_budget"] = thinking_budget
    else:
        cfg["thinking"] = False
        if arm == "nocot":
            cfg["prefill"] = PREFILL
            # Halt at the closing tag. Prefill alone is not enough: models were
            # observed closing the tag and then reasoning onward to a corrected
            # answer, which the last-match extraction rule would have scored.
            cfg["stop_sequences"] = [PREFILL.replace("<", "</")]
    return cfg


def _short(model_id: str) -> str:
    """'claude-haiku-4-5-20251001' -> 'haiku' for progress-bar labels."""
    for name, full in MODELS.items():
        if full == model_id:
            return name
    return model_id


def get_model(model_id: str, cfg: dict):
    """Thread-local client — one per worker, no shared state."""
    if not hasattr(_tls, "model"):
        _tls.model = create_model(model_id, cfg)
    return _tls.model


def call_one(model_id: str, cfg: dict, prompt: str) -> dict:
    """One API call. Returns the text block plus everything needed to audit it."""
    model = get_model(model_id, cfg)
    t0 = time.time()
    try:
        text = model.generate(prompt)
        err = getattr(model, "last_error", None)
    except Exception as e:                                    # noqa: BLE001
        return {"text": None, "error": str(e), "thinking_blocks": [],
                "stop_reason": None, "truncated": None,
                "input_tokens": None, "output_tokens": None,
                "latency_s": round(time.time() - t0, 2)}

    # The wrapper returns "" both for an empty text block and for a failure
    # string; distinguish via the side channel rather than by parsing the text.
    present = getattr(model, "last_text_block_present", None)
    if err or (isinstance(text, str) and text.startswith(("Error:", "Mock response for"))):
        return {"text": None, "error": err or text, "thinking_blocks": [],
                "stop_reason": getattr(model, "last_stop_reason", None),
                "truncated": getattr(model, "last_was_truncated", None),
                "input_tokens": getattr(model, "last_input_tokens", None),
                "output_tokens": getattr(model, "last_output_tokens", None),
                "latency_s": round(time.time() - t0, 2)}

    return {
        "text": text if present else None,
        "error": None,
        "thinking_blocks": list(getattr(model, "last_thinking_blocks", []) or []),
        "stop_reason": getattr(model, "last_stop_reason", None),
        "truncated": getattr(model, "last_was_truncated", None),
        "input_tokens": getattr(model, "last_input_tokens", None),
        "output_tokens": getattr(model, "last_output_tokens", None),
        "latency_s": round(time.time() - t0, 2),
    }


def run_cell(model_id, arm, cfg, nk, nu, args, jsonl, log, outer=None):
    """One (model, arm, cell). Returns the per-position summary.

    Trials are fired in batches so that positions still active can be queried
    concurrently; convergence is re-checked after each batch. A batch may
    overshoot the stopping point slightly, which costs a few calls and buys a
    large amount of wall-clock.

    The progress bar's total is the worst case (max_trials x positions). Early
    stopping means the real number is unknowable in advance, so the bar is
    retotalled to the actual count when the cell finishes rather than left
    sitting at a fraction.
    """
    if args.positions:
        positions = [k for k in args.positions if k <= nu]
    else:
        positions = query_positions(nu, args.n_positions)
    pos_keys = [str(k) for k in positions] + ["last"]

    obs = {k: [] for k in pos_keys}        # well-formed observations only
    malformed = {k: 0 for k in pos_keys}   # tier != "tagged"
    active = set(pos_keys)
    usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0, "errors": 0}
    trials_run = 0
    t0 = time.time()

    bar = tqdm(
        total=args.max_trials * len(pos_keys),
        desc=f"{_short(model_id)} {arm} K{nk}N{nu}",
        unit="call", leave=False, dynamic_ncols=True,
    )

    while trials_run < args.max_trials and active:
        batch = range(trials_run, min(trials_run + args.trial_batch, args.max_trials))
        tasks = []
        for trial_idx in batch:
            trial = build_trial(nk, nu, trial_idx, positions, dataset=args.dataset)
            for pk, (prompt, expected) in trial["queries"].items():
                if pk in active:
                    tasks.append((trial_idx, pk, prompt, expected, trial))

        if not tasks:
            break

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {
                ex.submit(call_one, model_id, cfg, prompt): (trial_idx, pk, expected, trial)
                for trial_idx, pk, prompt, expected, trial in tasks
            }
            for fut in as_completed(futs):
                trial_idx, pk, expected, trial = futs[fut]
                r = fut.result()
                sc = score_response(
                    r["text"], expected, trial["stream_vals"],
                    trial["values_by_category"], trial["test_category"],
                )

                usage["calls"] += 1
                usage["input_tokens"] += r["input_tokens"] or 0
                usage["output_tokens"] += r["output_tokens"] or 0
                if r["error"]:
                    usage["errors"] += 1

                bar.update(1)

                if sc["well_formed"]:
                    obs[pk].append(sc["correct"])
                else:
                    malformed[pk] += 1

                # Everything raw is persisted: scoring is derived at analysis
                # time, so the run can be re-scored without re-querying.
                jsonl.write(json.dumps({
                    "model_id": model_id, "arm": arm,
                    "num_keys": nk, "num_updates": nu, "position": pk,
                    "trial_idx": trial_idx, "seed": trial["seed"],
                    "seed_version": SEED_VERSION,
                    "test_category": trial["test_category"],
                    "expected": expected,
                    "text_block": r["text"],
                    "thinking_blocks": r["thinking_blocks"],
                    "answer_extracted": sc["answer"],
                    "answer_raw": sc.get("answer_raw"),
                    "extraction_tier": sc["tier"],
                    "n_tag_matches": sc["n_matches"],
                    "placeholder_echoed": sc["placeholder"],
                    "outcome": sc["outcome"],
                    "correct": sc["correct"],
                    "well_formed": sc["well_formed"],
                    "predicted_idx": sc["predicted_idx"],
                    "predicted_relative_pos": sc["predicted_relative_pos"],
                    "predicted_category": sc["predicted_category"],
                    "stop_reason": r["stop_reason"], "truncated": r["truncated"],
                    "error": r["error"],
                    "input_tokens": r["input_tokens"],
                    "output_tokens": r["output_tokens"],
                    "latency_s": r["latency_s"],
                    "temperature": cfg["temperature"],
                    "thinking_enabled": cfg.get("thinking"),
                    "thinking_budget": cfg.get("thinking_budget"),
                    "max_tokens": cfg["max_tokens"],
                }) + "\n")
        jsonl.flush()

        trials_run = max(batch) + 1

        # Per-position convergence. Each position retires on its own n.
        for pk in sorted(active):
            n = len(obs[pk])
            if n >= args.min_trials and wilson_hw(n, sum(obs[pk])) <= args.ci:
                active.discard(pk)

        seen = sum(len(v) for v in obs.values())
        acc = sum(sum(v) for v in obs.values()) / seen if seen else float("nan")
        bar.set_postfix_str(
            f"trial {trials_run}/{args.max_trials} "
            f"pos {len(active)}/{len(pos_keys)} active "
            f"acc {acc:.2f} err {usage['errors']}",
            refresh=True,
        )
        log(f"  {nk}k{nu}u trials={trials_run} active={len(active)}/{len(pos_keys)} "
            f"calls={usage['calls']} out_tok={usage['output_tokens']}")

    # Retotal so a bar that early-stopped reads 100%, not "204/5780".
    bar.total = bar.n
    bar.refresh()
    bar.close()
    if outer is not None:
        outer.update(1)

    stats = {}
    for pk in pos_keys:
        n = len(obs[pk])
        c = sum(obs[pk])
        hw = wilson_hw(n, c)
        acc = c / n if n else None
        stats[pk] = {
            "accuracy": round(acc, 4) if acc is not None else None,
            "n": n, "n_correct": c,
            "wilson_hw": round(hw, 4),
            "ci_lower": round(max(0.0, (acc or 0) - hw), 4) if acc is not None else None,
            "ci_upper": round(min(1.0, (acc or 0) + hw), 4) if acc is not None else None,
            "n_malformed": malformed[pk],
            "converged": n >= args.min_trials and hw <= args.ci,
        }

    return {
        "num_keys": nk, "num_updates": nu, "dataset": args.dataset,
        "trials_run": trials_run, "max_trials": args.max_trials,
        "all_converged": not active,
        "unconverged_positions": sorted(active),
        "positions": stats, "usage": usage,
        "elapsed_sec": round(time.time() - t0, 1),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=MODELS["haiku"],
                   help=f"Dated model ID. Shorthands: {list(MODELS)}")
    p.add_argument("--arm", choices=list(ARMS), default="cot_thinking")
    p.add_argument("--out-dir", default="experiments_cloud/results/cot_ivq")
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--n-positions", type=int, default=N_POSITIONS)
    p.add_argument("--min-trials", type=int, default=MIN_TRIALS)
    p.add_argument("--max-trials", type=int, default=MAX_TRIALS)
    p.add_argument("--ci", type=float, default=CI_THRESHOLD,
                   help="Retire a position once its Wilson half-width is <= this")
    p.add_argument("--trial-batch", type=int, default=4,
                   help="Trials fired per convergence check")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--thinking-budget", type=int, default=8000)
    p.add_argument("--max-tokens", type=int, default=None,
                   help="Default: budget+2048 (cot_thinking), 64 (nocot, prefilled)")
    p.add_argument("--positions", type=lambda s: [int(x) for x in s.split(",")],
                   default=None, help="Explicit position subset, e.g. 1,25,50")
    p.add_argument("--smoke", action="store_true",
                   help="2 trials x 4 positions on K10N50 — inspect before spending")
    p.add_argument("--pilot", action="store_true",
                   help=f"Only {PILOT_CELLS}, n=20 — calibrates budget/max_tokens")
    p.add_argument("--dry-run", action="store_true",
                   help="Print one prompt of each type and exit. No API calls.")
    return p.parse_args()


def main():
    args = parse_args()
    args.model = MODELS.get(args.model, args.model)

    if args.dry_run:
        trial = build_trial(10, 50, 0, query_positions(50, args.n_positions),
                            dataset=args.dataset)
        qs = trial["queries"]
        mid = [k for k in qs if k.isdigit()][len(qs) // 2]
        print("=" * 78)
        print(f"ANSWER CONTRACT (identical in both arms):\n{ANSWER_INSTRUCTION}")
        print("=" * 78)
        for label, key in (("ORDINAL (interior)", mid), ("SEMANTIC LAST", "last")):
            prompt, expected = qs[key]
            head, tail = prompt.split("\n\n", 1)
            stream, question = tail.rsplit("\n\n", 1)
            lines = stream.split("\n")
            print(f"\n--- {label}  position={key}  expected={expected!r} ---")
            print(head)
            print(f"\n{lines[0]}\n{lines[1]}\n  ... [{len(lines)} stream lines] ...\n{lines[-1]}\n")
            print(question)
        return

    load_dotenv(_ROOT / ".env")
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set (expected in .env)")

    cells = PILOT_CELLS if (args.pilot or args.smoke) else CELLS
    if args.pilot:
        args.max_trials = min(args.max_trials, 20)
    if args.smoke:
        args.max_trials = 2
        args.trial_batch = 2
        # first / shallow / deep-interior / final: enough to see every regime
        args.positions = args.positions or [1, 5, 25, 50]
        args.min_trials = 10 ** 6   # never retire a position during a smoke run

    if args.max_tokens is None:
        # nocot only needs room for the value: the stop sequence fires at the
        # closing tag (max observed 7 tokens). Never set a small ceiling on an
        # arm that can reason — truncation lands on deep positions and never on
        # shallow ones, which manufactures the very U-shape under test.
        args.max_tokens = {
            "cot_thinking": args.thinking_budget + 2048,
            "nocot": 64,
        }[args.arm]
    cfg = model_config(args.arm, args.max_tokens, args.thinking_budget)

    suffix = "__smoke" if args.smoke else ("__pilot" if args.pilot else "")
    tag = f"{args.model}__{args.arm}{suffix}"
    out_dir = Path(args.out_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "checkpoint.json"
    jsonl_path = out_dir / "trials.jsonl"
    log_path = out_dir / "run.log"

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        # tqdm.write, not print: a bare print lands on top of the live bar and
        # leaves the terminal littered with half-drawn progress lines.
        tqdm.write(line)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    import anthropic  # recorded as provenance, not used directly here

    ckpt = json.loads(ckpt_path.read_text()) if ckpt_path.exists() else {
        "model_id": args.model, "arm": args.arm, "dataset": args.dataset,
        # The SDK version is part of the run's identity: 1.x moved temperature
        # into extra_body, so the same code on 0.x sends a different request.
        "anthropic_sdk_version": anthropic.__version__,
        "config": {k: v for k, v in cfg.items() if k != "verbose"},
        "seed_version": SEED_VERSION,
        "min_trials": args.min_trials, "max_trials": args.max_trials, "ci": args.ci,
        "cells": {},
    }

    log(f"model={args.model} arm={args.arm} cells={cells} "
        f"thinking={cfg.get('thinking')} budget={cfg.get('thinking_budget')} "
        f"max_tokens={cfg['max_tokens']} temp={cfg['temperature']}")

    todo = [c for c in cells if f"{c[0]}_{c[1]}" not in ckpt["cells"]]
    outer = tqdm(total=len(todo), desc=f"{_short(args.model)} {args.arm} cells",
                 unit="cell", dynamic_ncols=True)

    # Append: a resumed run must not clobber the trials already on disk.
    with open(jsonl_path, "a") as jsonl:
        for nk, nu in cells:
            key = f"{nk}_{nu}"
            if key in ckpt["cells"]:
                log(f"skip {key} (done)")
                continue
            log(f"--- cell {nk}k{nu}u ---")
            summary = run_cell(args.model, args.arm, cfg, nk, nu, args, jsonl, log,
                               outer=outer)
            ckpt["cells"][key] = summary
            ckpt_path.write_text(json.dumps(ckpt, indent=1))

            tot = sum(c["usage"]["calls"] for c in ckpt["cells"].values())
            mal = sum(p["n_malformed"] for p in summary["positions"].values())
            log(f"{key}: {summary['trials_run']} trials, "
                f"{summary['usage']['calls']} calls, malformed={mal}, "
                f"converged={summary['all_converged']}, "
                f"{summary['elapsed_sec']}s (run total calls={tot})")

    outer.close()

    io_tok = sum(c["usage"]["input_tokens"] for c in ckpt["cells"].values())
    o_tok = sum(c["usage"]["output_tokens"] for c in ckpt["cells"].values())
    log(f"DONE  input_tokens={io_tok:,}  output_tokens={o_tok:,}  -> {out_dir}")


if __name__ == "__main__":
    main()
