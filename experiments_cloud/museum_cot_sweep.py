"""CoT vs non-CoT position sweep on the museum M0 naturalistic narrative.

The naturalistic counterpart of `cot_ivq_sweep.py`. Same two arms, same
per-position Wilson early stopping, same answer contract, same on-disk layout —
only the stimulus changes, from a `key: value` stream to museum-visit prose.

The arm definitions (`MODELS`, `ARMS`, `PREFILL`, `model_config`, `call_one`,
`wilson_hw`) are IMPORTED from `cot_ivq_sweep`, not copied. The CoT-vs-noCoT
contrast is therefore defined in exactly one place in the repo, and the two
sweeps cannot drift apart on the variable under test.

Design
------
One narrative, one prompt, two arms. Byte-identical prompt in both; the only
manipulated variable is whether extended thinking is enabled. Temperature is 1.0
in BOTH arms — extended thinking requires it, so the control matches rather than
leaving temperature confounded with reasoning. Seeds are blake2b, so a given
(cell, trial) yields the same narrative in both arms and the comparison is
paired trial-by-trial.

Note `thinking=False` alone is not a no-reasoning control: models enumerate in
the visible text block regardless of instructions. The `nocot` arm prefills
`<answer>` and stops at `</answer>`, so the value is the first token generated
and there is nowhere for reasoning to go.

Relationship to the July 2026 museum runs
-----------------------------------------
This is a re-measurement, not a reproduction. It differs from
`results/museum_endpoint/` and `results/museum_ivq/` on three axes at once:
scoring is exact-equality on an <answer> span rather than substring containment,
CoT suppression is prefill+stop rather than an instruction, and temperature is
1.0 rather than 0. The July numbers remain valid as the temp-0/containment
measurement; these cannot be pooled with them.

Usage
-----
  # inspect the prompt, no API calls
  uv run python experiments_cloud/museum_cot_sweep.py --dry-run

  # 2 trials x 4 positions on the hardest cell, to eyeball before spending
  uv run python experiments_cloud/museum_cot_sweep.py --arm cot_thinking --smoke

  # full run, one arm per process
  uv run python experiments_cloud/museum_cot_sweep.py \
      --model claude-haiku-4-5-20251001 --arm nocot
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from experiments_cloud.cot_ivq_sweep import (  # noqa: E402
    ARMS, MODELS, call_one, model_config, wilson_hw,
)
from experiments_cloud.museum_cot_prompts import (  # noqa: E402
    ANSWER_INSTRUCTION, POOL_MODE, RENDER_MODE, SEED_VERSION, build_trial,
    position_keys, query_positions,
)
from experiments_cloud.museum_cot_score import score_response  # noqa: E402

# Same six cells as cot_ivq_sweep.CELLS, which are also exactly the cells the
# July museum IVQ run used. Slice-for-slice comparable to both.
CELLS = [(5, 10), (10, 10), (5, 20), (10, 20), (5, 50), (10, 50)]
PILOT_CELLS = [(10, 50)]        # hardest: longest narrative, widest interior

N_POSITIONS = 16
MIN_TRIALS = 30
MAX_TRIALS = 200
CI_THRESHOLD = 0.05

# Matched to `run_cot_ivq_full.sh` so museum and plain share one budget, which
# is the point: a fixed budget across both stimulus families keeps the CoT arms
# comparable.
#
# The cost of that choice is measured, not guessed. A 28-call probe at budget
# 16000 on the deep positions of K10/N50 (the hardest cell) found:
#
#     output tokens by position   first 305-3,981   43 1,571-6,038
#                                 49 5,048-7,794    50 6,689-9,764
#     calls exceeding 3000        14/28 (50%)
#
# So reasoning WILL be cut off at depth here, on roughly half of deep-position
# calls, and essentially never on shallow ones. CoT accuracy at depth is
# therefore a lower bound — the same artifact `run_cot_ivq_full.sh` documents
# for the plain sweep at 7.4%, larger here because museum prompts are ~1.4x
# longer at matched cells.
#
# Budget truncation is SILENT: the model stops thinking and answers, so
# stop_reason stays 'end_turn', truncated stays False, and the response parses
# normally. `thinking_budget_hit` (below) is recorded per call so the artifact
# is visible in the data rather than only in this comment.
THINKING_BUDGET = 3000

# output_tokens counts thinking + visible text. The visible block is ~5 tokens
# in the nocot arm and ~5 in the cot arm too (the prefill/stop design holds, and
# the saved thinking_blocks are Anthropic's *summary* — full reasoning is what
# gets billed). So output_tokens at or above the budget means reasoning was
# capped. 0.95 leaves room for the answer tokens riding along.
BUDGET_HIT_FRACTION = 0.95


def _short(model_id: str) -> str:
    for name, full in MODELS.items():
        if full == model_id:
            return name
    return model_id


def _budget_hit(output_tokens, thinking_budget) -> bool | None:
    """Whether this call's reasoning was capped by the thinking budget.

    None in the nocot arm (no budget) or when usage is missing, so the field
    distinguishes "not applicable" from "did not hit".
    """
    if not thinking_budget or output_tokens is None:
        return None
    return output_tokens >= BUDGET_HIT_FRACTION * thinking_budget


def _run_wave(wave, model_id, cfg, args, obs, malformed, budget_hits, usage, bar,
              jsonl, arm, nk, nu):
    """Execute one wave of (trial, position) tasks and record every result."""
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(call_one, model_id, cfg, prompt): (trial_idx, pk, expected, trial)
            for trial_idx, pk, prompt, expected, trial in wave
        }
        for fut in as_completed(futs):
            trial_idx, pk, expected, trial = futs[fut]
            r = fut.result()
            sc = score_response(
                r["text"], expected, trial["stream_vals"],
                trial["values_by_category"], trial["test_category"],
                trial["narrative_records"],
            )

            usage["calls"] += 1
            usage["input_tokens"] += r["input_tokens"] or 0
            usage["output_tokens"] += r["output_tokens"] or 0
            usage["cache_write_tokens"] += r.get("cache_write_tokens") or 0
            usage["cache_read_tokens"] += r.get("cache_read_tokens") or 0
            if r["error"]:
                usage["errors"] += 1

            bar.update(1)

            if sc["well_formed"]:
                obs[pk].append(sc["correct"])
            else:
                malformed[pk] += 1

            if _budget_hit(r["output_tokens"], cfg.get("thinking_budget")):
                budget_hits[pk] += 1

            # Everything raw is persisted: scoring is derived at analysis time,
            # so the run can be re-scored without re-querying.
            jsonl.write(json.dumps({
                "model_id": model_id, "arm": arm,
                "dataset": "museum_m0",
                "render_mode": trial["render_mode"],
                "pool_mode": trial["pool_mode"],
                "num_keys": nk, "num_updates": nu, "position": pk,
                # 'ordinal' for a numbered query, 'semantic' for first/last.
                # These differ by up to 47x on the same target in the July data,
                # so the two must never be pooled.
                "query_style": "semantic" if pk in ("first", "last") else "ordinal",
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
                # Museum-specific: where the answer sits in the narrative.
                "narrative_idx": sc["narrative_idx"],
                "dist_from_narrative_end": sc["dist_from_narrative_end"],
                "narrative_entity": sc["narrative_entity"],
                "took_narrative_final": sc["took_narrative_final"],
                "stop_reason": r["stop_reason"], "truncated": r["truncated"],
                # Reasoning was capped by the thinking budget. Distinct from
                # `truncated` (max_tokens), which the API reports: budget
                # exhaustion is silent, so this has to be derived. Expect ~50%
                # at position 50 and ~0% at shallow positions on budget 3000.
                "thinking_budget_hit": _budget_hit(
                    r["output_tokens"], cfg.get("thinking_budget")),
                "error": r["error"],
                "input_tokens": r["input_tokens"],
                "output_tokens": r["output_tokens"],
                "cache_write_tokens": r.get("cache_write_tokens"),
                "cache_read_tokens": r.get("cache_read_tokens"),
                "latency_s": r["latency_s"],
                "temperature": cfg["temperature"],
                "thinking_enabled": cfg.get("thinking"),
                "thinking_budget": cfg.get("thinking_budget"),
                "max_tokens": cfg["max_tokens"],
            }) + "\n")


def run_cell(model_id, arm, cfg, nk, nu, args, jsonl, log, outer=None):
    """One (model, arm, cell). Returns the per-position summary.

    Early stopping is PER POSITION, not per cell: each position retires as soon
    as its own Wilson half-width is tight enough and later trials stop querying
    it. Floor positions converge near n=30-60, so the saving is largest on
    exactly the cells where most positions sit at zero.

    The consequence to remember at analysis time is that n differs by position
    within a cell, so positions must not be pooled into a cell-level mean.
    """
    if args.positions:
        positions = [k for k in args.positions if k <= nu]
        pos_keys = [str(k) for k in positions] + ["first", "last"]
    else:
        positions = query_positions(nu, args.n_positions)
        pos_keys = position_keys(nu, args.n_positions)

    obs = {k: [] for k in pos_keys}        # well-formed observations only
    malformed = {k: 0 for k in pos_keys}   # tier != "tagged"
    budget_hits = {k: 0 for k in pos_keys}  # reasoning capped by the budget
    active = set(pos_keys)
    usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0, "errors": 0,
             "cache_write_tokens": 0, "cache_read_tokens": 0}
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
            trial = build_trial(nk, nu, trial_idx, positions, cache=args.cache,
                                render_mode=args.render_mode,
                                pool_mode=args.pool_mode)
            for pk, (prompt, expected) in trial["queries"].items():
                if pk in active:
                    tasks.append((trial_idx, pk, prompt, expected, trial))

        if not tasks:
            break

        # With caching on, fire ONE call per trial first and let it land before
        # the rest. All queries in a trial share the narrative prefix, so firing
        # 18 at once against a cold cache means 18 simultaneous misses, each
        # billed at the 1.25x write rate — strictly worse than not caching. The
        # warm-up wave pays one write per trial and makes the rest 0.1x reads.
        if args.cache:
            first_seen = set()
            warm, rest = [], []
            for t in tasks:
                (warm if t[0] not in first_seen else rest).append(t)
                first_seen.add(t[0])
            waves = [warm, rest]
        else:
            waves = [tasks]

        for wave in waves:
            if not wave:
                continue
            _run_wave(wave, model_id, cfg, args, obs, malformed, budget_hits,
                      usage, bar, jsonl, arm, nk, nu)
        jsonl.flush()

        trials_run = max(batch) + 1

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
            # Calls whose reasoning was capped. Read this BEFORE quoting a CoT
            # accuracy: a position with a high rate is measuring truncated
            # reasoning, not the model's ceiling.
            "n_thinking_budget_hit": budget_hits[pk],
            "query_style": "semantic" if pk in ("first", "last") else "ordinal",
            "converged": n >= args.min_trials and hw <= args.ci,
        }

    return {
        "num_keys": nk, "num_updates": nu,
        "dataset": "museum_m0",
        "render_mode": args.render_mode, "pool_mode": args.pool_mode,
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
    p.add_argument("--out-dir", default="experiments_cloud/results/museum_cot")
    p.add_argument("--render-mode", default=RENDER_MODE,
                   help="M0 atemporal (default), M1 timestamped, M2 indexed")
    p.add_argument("--pool-mode", default=POOL_MODE)
    # Caching is ON by default here, unlike cot_ivq_sweep. All 12-18 queries in a
    # trial share the narrative, which at K10/N50 is ~4.7k tokens, so the
    # breakpoint is a ~4x input-cost cut. It must be set the same way for both
    # arms: the block seam can shift tokenization by a token or two, which is
    # harmless when held constant and a confound when not.
    p.add_argument("--no-cache", dest="cache", action="store_false",
                   help="Disable the prompt-cache breakpoint after the narrative")
    p.set_defaults(cache=True)
    p.add_argument("--n-positions", type=int, default=N_POSITIONS)
    p.add_argument("--min-trials", type=int, default=MIN_TRIALS)
    p.add_argument("--max-trials", type=int, default=MAX_TRIALS)
    p.add_argument("--ci", type=float, default=CI_THRESHOLD,
                   help="Retire a position once its Wilson half-width is <= this")
    p.add_argument("--trial-batch", type=int, default=8,
                   help="Trials fired per convergence check")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--thinking-budget", type=int, default=THINKING_BUDGET)
    p.add_argument("--max-tokens", type=int, default=None,
                   help="Default: budget+2048 (cot_thinking), 64 (nocot, prefilled)")
    p.add_argument("--positions", type=lambda s: [int(x) for x in s.split(",")],
                   default=None, help="Explicit ordinal subset, e.g. 1,25,50")
    p.add_argument("--smoke", action="store_true",
                   help="2 trials x a few positions on K10N50 — inspect before spending")
    p.add_argument("--pilot", action="store_true",
                   help=f"Only {PILOT_CELLS}, n=20 — calibrates budget/max_tokens")
    p.add_argument("--cells", type=lambda s: [tuple(int(x) for x in c.split("_"))
                                              for c in s.split(",")],
                   help="Override the cell list, e.g. 5_10,10_50. Format K_N.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print one prompt of each type and exit. No API calls.")
    return p.parse_args()


def main():
    args = parse_args()
    args.model = MODELS.get(args.model, args.model)

    if args.dry_run:
        trial = build_trial(10, 50, 0, query_positions(50, args.n_positions),
                            render_mode=args.render_mode, pool_mode=args.pool_mode)
        qs = trial["queries"]
        ordinals = [k for k in qs if k.isdigit()]
        mid = ordinals[len(ordinals) // 2]
        print("=" * 78)
        print(f"ANSWER CONTRACT (identical in both arms):\n{ANSWER_INSTRUCTION}")
        print(f"queried visitor: {trial['test_category']}   "
              f"roster: {trial['categories']}")
        print(f"narrative records: {len(trial['narrative_records'])}   "
              f"queried visitor's values: {len(trial['stream_vals'])}")
        print("=" * 78)
        for label, key in (("ORDINAL (interior)", mid),
                           ("SEMANTIC FIRST", "first"),
                           ("SEMANTIC LAST", "last")):
            prompt, expected = qs[key]
            narrative, question = prompt.rsplit("\n\n", 1)
            sents = narrative.split(". ")
            print(f"\n--- {label}  position={key}  expected={expected!r} ---")
            print(f"{sents[0]}. {sents[1]}.")
            print(f"  ... [{len(sents)} narrative sentences] ...")
            print(f"{sents[-1]}\n")
            print(question)
        return

    load_dotenv(_ROOT / ".env")
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set (expected in .env)")

    cells = args.cells or (PILOT_CELLS if (args.pilot or args.smoke) else CELLS)
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
        # closing tag. Never set a small ceiling on an arm that can reason —
        # truncation lands on deep positions and never on shallow ones, which
        # manufactures the very U-shape under test.
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
        # tqdm.write, not print: a bare print lands on top of the live bar.
        tqdm.write(line)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    import anthropic  # recorded as provenance, not used directly here

    ckpt = json.loads(ckpt_path.read_text()) if ckpt_path.exists() else {
        "model_id": args.model, "arm": args.arm,
        "dataset": "museum_m0",
        "render_mode": args.render_mode, "pool_mode": args.pool_mode,
        # The SDK version is part of the run's identity: 1.x moved temperature
        # into extra_body, so the same code on 0.x sends a different request.
        "anthropic_sdk_version": anthropic.__version__,
        "config": {k: v for k, v in cfg.items() if k != "verbose"},
        "cache": args.cache,
        "seed_version": SEED_VERSION,
        "min_trials": args.min_trials, "max_trials": args.max_trials, "ci": args.ci,
        "cells": {},
    }

    log(f"model={args.model} arm={args.arm} cells={cells} "
        f"render={args.render_mode} pool={args.pool_mode} cache={args.cache} "
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
    cw = sum(c["usage"].get("cache_write_tokens", 0) for c in ckpt["cells"].values())
    cr = sum(c["usage"].get("cache_read_tokens", 0) for c in ckpt["cells"].values())
    log(f"DONE  input_tokens={io_tok:,}  output_tokens={o_tok:,}  "
        f"cache_write={cw:,}  cache_read={cr:,}  -> {out_dir}")
    if cw or cr:
        # Reads are billed at 0.1x and writes at 1.25x, so this is the honest
        # comparison against having sent every prefix at full price.
        eff = io_tok + cw * 1.25 + cr * 0.10
        raw = io_tok + cw + cr
        log(f"      billable input equivalent {eff:,.0f} vs {raw:,.0f} uncached "
            f"({raw / eff:.1f}x saving)")


if __name__ == "__main__":
    main()
