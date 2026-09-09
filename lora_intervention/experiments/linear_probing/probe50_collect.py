"""
Closed-pool value-identity probing: activation collection.

Design (differs deliberately from run_probing.py, which probes correctness):

  * The value pool is closed and small, sized so that K*N == len(pool). Because
    generate_values_for_trial draws `rng.sample(pool, K*N)` without replacement,
    every value in the pool appears exactly once in every stream. The probe
    therefore cannot score above chance by detecting which values are present --
    presence carries zero information by construction.

  * The label is the ground-truth answer value (which of the `len(pool)` values
    sits at the queried slot), taken from the trial definition, not from the
    model's output. So it stays defined at cells where the model is at 0% or
    100% accuracy, where a correctness probe degenerates.

  * Conditions are indexed by the queried slot k directly (k = 1 .. N), not by
    depth fraction. run_probing.py's `k = min(max(round(depth*N),2),N-1)` maps
    the five depth fractions onto only {2,2,5,8,9} at N=10, i.e. IVQ_d10 and
    IVQ_d25 collide on the same slot. Indexing k removes the collision.

  * All conditions share one stream generator and one shuffle. run_probing.py
    routes FVQ/CVQ through probing_classifier.generate_trial (plain
    rng.shuffle) and IVQ through gen_ivq_trial (anti-consecutive shuffle_nc),
    which confounds any FVQ-vs-IVQ comparison with the shuffle.

  * Query wording is a measured variable, not a nuisance. "first"/"last" name a
    slot without counting; "5th" requires counting -- that is the FVQ/CVQ vs IVQ
    distinction and it is deliberate. To size the wording effect rather than
    assume it, both endpoints run twice at the same slot: CVQ ("last") against
    k{N} ("Nth"), and FVQ ("first") against k1 ("1st"). Same slot, same label,
    different phrasing.

  * shuffle_strict replaces shuffle_nc, which silently fails its own
    anti-consecutive guarantee in 20.4% of (5,10) trials, always at stream
    position 48-49 -- precisely where CVQ reads and where cv_last lives.

This module only collects activations and behaviour. Probe fitting is separate
(probe50_fit.py) and runs on CPU off the saved arrays.
"""
import argparse
import importlib.metadata as _imeta
import json
import os
import random
import sys
import time
import zlib
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_orig_version = _imeta.version
def _safe_version(pkg):
    try:
        v = _orig_version(pkg)
        return v if v is not None else "0.0.0"
    except Exception:
        return "0.0.0"
_imeta.version = _safe_version

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_ROOT / "v3" / "scripts"))
sys.path.insert(0, str(_ROOT / "lora_intervention"))
sys.path.insert(0, str(_ROOT))

from evaluate_ivq import ordinal, shuffle_nc, is_correct  # noqa: E402
from mechanistic_probing_v2.core.dataset_configs import (  # noqa: E402
    get_value_pool, get_eligible_categories, generate_values_for_trial,
)
import mechanistic_probing_v2.core.dataset_configs as _dataset_configs  # noqa: E402

DATASET_TYPE = "ARBITRARY_SINGLE"

# Two system prompts exist in this repo. "local" is used by every local-model
# script -- evaluate_ivq.py:50, evaluate.py:56, probing_classifier.py:45,
# quick_probe.py:43, run_probing.py:105 -- and, critically, by data_gen.py:50,
# which generated the LoRA's training data. "cloud" is the API-sweep variant
# (experiments_cloud/ucurve_prompts.py:20, remedy_sweep.py:151); note the em
# dash and the absence of the role framing.
#
# Default is "local": the v3 wrong_frac numbers used to size T came from
# evaluate_ivq.py, and the LoRA is locked to this string, so a base-vs-LoRA
# comparison later requires it.
SYSTEM_PROMPTS = {
    "local": (
        "You are a precise data extraction tool. "
        "Output ONLY a single word - the exact value requested. "
        "No other text, no explanation, no punctuation."
    ),
    "cloud": (
        "Output ONLY a single word \u2014 the exact value requested. "
        "No explanation, no punctuation, no extra words."
    ),
}
SYSTEM_PROMPT = SYSTEM_PROMPTS["local"]


# ─── closed value pool ───────────────────────────────────────────────────────

def select_pool(tokenizer, size, seed):
    """Pick `size` values that are safe for this tokenizer and unambiguous to score.

    Three filters:
      1. single token WITH a leading space (" wonder"), how values appear
         mid-stream after "category: ";
      2. single token WITHOUT a leading space ("wonder"), how the model's answer
         begins at a fresh generation position -- so the argmax of one forward
         pass is the whole answer and `scoring=single_token` is exact;
      3. no selected value is a substring of another. evaluate_ivq.is_correct
         accepts `exp in pred`, so with a closed pool of short words a pair like
         ("ace", "access") would silently score a wrong answer as correct.

    Deterministic given `seed`. Returns the pool, sorted.
    """
    raw = get_value_pool(DATASET_TYPE)
    safe = [v for v in raw
            if len(tokenizer.encode(f" {v}", add_special_tokens=False)) == 1
            and len(tokenizer.encode(v, add_special_tokens=False)) == 1]

    rng = random.Random(seed)
    order = safe[:]
    rng.shuffle(order)
    chosen = []
    for v in order:
        vl = v.lower()
        if any(vl in c.lower() or c.lower() in vl for c in chosen):
            continue
        chosen.append(v)
        if len(chosen) == size:
            break
    if len(chosen) < size:
        raise ValueError(
            f"only {len(chosen)} substring-free single-token values available "
            f"for this tokenizer, need {size}")
    return sorted(chosen), len(safe), len(raw)


def install_pool(pool):
    """Point every downstream trial generator at the closed pool."""
    _dataset_configs._cached_pools["arbitrary_single"] = list(pool)


# ─── trial generation ────────────────────────────────────────────────────────

def ordinal_numeral(k):
    """Numeral ordinal: 1 -> '1st', 2 -> '2nd', 10 -> '10th'.

    Matches evaluate_ivq.ordinal for k >= 2. It differs only at k=1, where
    evaluate_ivq special-cases "first" before its lookup table, so the numeral
    form was previously unreachable.
    """
    if 11 <= k % 100 <= 13:
        return f"{k}th"
    return f"{k}{['th', 'st', 'nd', 'rd', 'th'][min(k % 10, 4)]}"


def condition_spec(condition, nu):
    """(slot k, query wording) for a condition name.

    The word/numeral split is deliberate, not a confound: "first" and "last"
    name a slot without counting, while "5th" requires counting. That is the
    FVQ/CVQ-vs-IVQ distinction.

    To measure the wording effect rather than assume it, the endpoints are run
    twice at the same slot with the two phrasings:

        CVQ  -> slot N, "last"        k{N} -> slot N, "Nth"
        FVQ  -> slot 1, "first"       k1   -> slot 1, "1st"

    Same stream, same label (cv[N-1] or cv[0]), only the wording differs, so
    the accuracy and probe gaps isolate phrasing from position.
    """
    if condition == "FVQ":
        return 1, "first"
    if condition == "CVQ":
        return nu, "last"
    if condition.startswith("k"):
        k = int(condition[1:])
        return k, ordinal_numeral(k)
    raise ValueError(f"unknown condition: {condition}")


def shuffle_strict(items, rng):
    """Anti-consecutive shuffle that always succeeds when one is possible.

    `evaluate_ivq.shuffle_nc` retries a plain shuffle 100 times, then falls back
    to a greedy loop that dead-ends once only one category's items remain and
    dumps the remainder verbatim. Measured at (5,10): 20.4% of trials end up
    with an adjacent same-category pair, 100% of them at stream position 48-49
    -- exactly where CVQ reads and where the cv_last recency control lives.

    Greedy by largest remaining count is the standard construction and succeeds
    whenever max(per-category count) <= ceil(n/2), which holds for every cell
    here (10 <= 25 at (5,10)).
    """
    buckets = {}
    for it in items:
        buckets.setdefault(it["category"], []).append(it)
    for v in buckets.values():
        rng.shuffle(v)
    counts = {c: len(v) for c, v in buckets.items()}
    n = len(items)
    if max(counts.values()) > (n + 1) // 2:
        raise ValueError(
            f"no anti-consecutive arrangement exists: max count "
            f"{max(counts.values())} > ceil({n}/2)")

    out, last = [], None
    for _ in range(n):
        cands = [c for c, m in counts.items() if m > 0 and c != last]
        top = max(counts[c] for c in cands)
        pick = rng.choice(sorted(c for c in cands if counts[c] == top))
        out.append(buckets[pick].pop())
        counts[pick] -= 1
        last = pick
    return out


def gen_trial(nk, nu, k, word, seed, tokenizer, shuffle="strict",
              system_prompt=None):
    """One trial querying slot k (1-indexed) of the test category, phrased `word`.

    Mirrors evaluate_ivq.make_prompt's stream construction (same category
    sampling, same prompt text), then adds the bookkeeping the probe needs: the
    test category's values in stream order, plus the first and last of them for
    the recency controls.

    `shuffle="strict"` fixes shuffle_nc's tail artifact; "legacy" reproduces it.
    """
    rng = random.Random(seed)
    elig = get_eligible_categories(DATASET_TYPE, min_values=nu)
    cats = rng.sample(elig, nk)
    try:
        vals = generate_values_for_trial(DATASET_TYPE, cats, nu, rng)
    except ValueError:
        return None

    test_cat = cats[seed % nk]
    items = [{"category": c, "value": v} for c in cats for v in vals[c]]
    items = (shuffle_strict if shuffle == "strict" else shuffle_nc)(items, rng)
    cv = [it["value"] for it in items if it["category"] == test_cat]
    if k > len(cv):
        return None

    stream = "\n".join(f"{it['category']}: {it['value']}" for it in items)
    user_txt = (
        f"Read the following key-value stream. Each key gets updated multiple times.\n\n"
        f"{stream}\n\nWhat was the {word} value of {test_cat}?"
    )
    msgs = [{"role": "system", "content": system_prompt or SYSTEM_PROMPT},
            {"role": "user", "content": user_txt}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    return {
        "prompt": prompt,
        "expected": cv[k - 1],
        "cv": cv,                       # test category's values, stream order
        "cv_first": cv[0],              # recency control label
        "cv_last": cv[-1],              # recency control label
        "test_category": test_cat,
        "k": k,
        "query_word": word,
        "seed": seed,
    }


def conditions_for(nu):
    """Run order: CVQ, then numeral slots 2..N, then FVQ, then numeral slot 1.

    `k{N}` pairs with CVQ and `k1` pairs with FVQ -- same slot, same label,
    numeral vs word phrasing (see condition_spec). That turns the wording
    difference into a measurement instead of an assumption.

    FVQ and k1 run last on purpose: slot 1 is the least interesting position and
    the most expensive (highest accuracy -> smallest wrong subset per trial), so
    they can be cut without losing the rest of the run.
    """
    return ["CVQ"] + [f"k{k}" for k in range(2, nu + 1)] + ["FVQ", "k1"]


def trial_seed(nk, nu, k, t_idx):
    """Stable across processes, unlike `hash()`.

    run_probing.py:200 uses `hash((nk, nu, condition, t_idx, "linear_probing"))`,
    but CPython salts string hashing per process (PYTHONHASHSEED), so the same
    t_idx yields different trials on every invocation. crc32 over a formatted
    byte string is deterministic forever.

    Keyed on the slot `k`, not the condition name, so conditions that share a
    slot share streams: CVQ and k{N} both get slot N, FVQ and k1 both get slot 1.
    The wording contrast is then *paired* -- byte-identical streams and labels,
    only the final query sentence differs -- which removes stream-to-stream
    variance from the comparison. The two probes are still independent fits on
    different activations, so pairing leaks nothing between them.
    """
    return zlib.crc32(f"probe50|{nk}|{nu}|{k}|{t_idx}".encode()) % (2 ** 31)


def build_trials(nk, nu, condition, n_trials, tokenizer, t_start=0,
                 shuffle="strict", system_prompt=None):
    """Draw `n_trials` valid trials, skipping seeds that produce degenerate streams.

    Seeded on the condition *name*, not on k, so CVQ and k{N} -- which share a
    slot -- draw different streams. Reusing streams across the pair would make
    their probes share activations and correlate their CV folds.
    """
    k, word = condition_spec(condition, nu)
    out, t_idx = [], t_start
    while len(out) < n_trials:
        seed = trial_seed(nk, nu, k, t_idx)
        tr = gen_trial(nk, nu, k, word, seed, tokenizer, shuffle=shuffle,
                       system_prompt=system_prompt)
        t_idx += 1
        if tr is not None:
            out.append(tr)
        if t_idx - t_start > 50 * n_trials + 1000:
            raise RuntimeError(
                f"cannot generate {n_trials} trials for {condition} at ({nk},{nu})")
    return out, t_idx


# ─── batched activation collection ───────────────────────────────────────────

@torch.no_grad()
def collect(model, tokenizer, trials, device, batch_size=16, generate_subset=0,
            max_new_tokens=8):
    """Residual stream at the final position, every layer, for each trial.

    Trials are bucketed by exact token length so batches need no padding at all.
    That sidesteps position_ids: a raw `model(**batch)` call derives positions
    from an arange over the sequence, which is wrong for left-padded rows. With
    uniform-length buckets there is no padding to get wrong, and `[:, -1, :]` is
    the true final position for every row in the batch.

    `single_token` correctness reads the argmax of the same forward pass, so
    there is one forward pass per trial rather than a forward plus a generate.

    `generate_subset=N` additionally runs greedy generation on the first N
    trials, for the instruction-following audit and to verify the two scorings
    agree. Generation costs 3.6x a forward pass on MPS (929 vs 255 ms/trial
    measured), so it is deliberately not run on every trial: the smoke test
    showed single_token/generate agreement of 1.000 on all 12 conditions, so a
    subset is enough to keep that check honest.
    """
    gen_ids = {id(t) for t in trials[:generate_subset]} if generate_subset else set()
    # Store n_layers+1 rows: index 0 is the embedding output (hidden_states[0]),
    # indices 1..n_layers are the transformer blocks. The design's layer-0 anchor
    # control needs index 0 -- at the final prompt position that is the query
    # token's embedding, which carries nothing about the answer, so the probe
    # there must sit at chance. Dropping it would make that control unrunnable.
    cfg = model.config
    if getattr(cfg, "num_hidden_layers", None) is None:
        cfg = cfg.text_config          # gemma-3 CG nests the text params
    n_layers = cfg.num_hidden_layers + 1
    d_model = cfg.hidden_size

    enc = [tokenizer(t["prompt"], return_tensors="pt")["input_ids"][0] for t in trials]
    order = sorted(range(len(trials)), key=lambda i: len(enc[i]))

    reps = np.zeros((len(trials), n_layers, d_model), dtype=np.float16)
    pred_tid = [None] * len(trials)
    gen_text = [None] * len(trials)

    buckets = {}
    for i in order:
        buckets.setdefault(len(enc[i]), []).append(i)

    t0 = time.time()
    done = 0
    for L, idxs in sorted(buckets.items()):
        for s in range(0, len(idxs), batch_size):
            chunk = idxs[s:s + batch_size]
            ids = torch.stack([enc[i] for i in chunk]).to(device)
            out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states          # tuple len n_layers+1, [0] = embeddings
            stack = torch.stack([h[:, -1, :] for h in hs], dim=1)  # (B, n_layers+1, d)
            reps[chunk] = stack.float().cpu().numpy().astype(np.float16)
            am = out.logits[:, -1, :].argmax(-1).cpu().tolist()
            for j, i in enumerate(chunk):
                pred_tid[i] = int(am[j])
            del out, hs, stack

            gen_rows = [j for j, i in enumerate(chunk) if id(trials[i]) in gen_ids]
            if gen_rows:
                g = model.generate(input_ids=ids[gen_rows],
                                   max_new_tokens=max_new_tokens,
                                   do_sample=False,
                                   pad_token_id=tokenizer.pad_token_id
                                   or tokenizer.eos_token_id)
                for row, j in enumerate(gen_rows):
                    gen_text[chunk[j]] = tokenizer.decode(
                        g[row, ids.shape[1]:], skip_special_tokens=True).strip()
                del g

            done += len(chunk)
            if device == "mps":
                torch.mps.synchronize()
                # The MPS caching allocator holds freed blocks in a pool and
                # does not return them to the OS. With 37 hidden-state tensors
                # per forward (~204 MB at bs=4, seq 337) across ~14 distinct
                # sequence-length buckets, the pool fragments upward over
                # thousands of iterations until the OS kills the process.
                # Draining it periodically keeps the footprint flat; the cost is
                # a few re-allocations.
                if done % (batch_size * 25) < batch_size:
                    torch.mps.empty_cache()
            if done % (batch_size * 10) < batch_size:
                el = time.time() - t0
                print(f"      {done}/{len(trials)}  {el/done*1000:.0f} ms/trial",
                      flush=True)

    elapsed = time.time() - t0
    records = []
    for i, t in enumerate(trials):
        pt = tokenizer.decode([pred_tid[i]]).strip()
        rec = {
            "expected": t["expected"],
            "cv": t["cv"],
            "cv_first": t["cv_first"],
            "cv_last": t["cv_last"],
            "test_category": t["test_category"],
            "k": t["k"],
            "seed": t["seed"],
            "pred_single_token": pt,
            "correct_single_token": pt.lower() == t["expected"].lower(),
        }
        if gen_text[i] is not None:
            rec["pred_generate"] = gen_text[i]
            rec["correct_generate"] = bool(is_correct(gen_text[i], t["expected"]))
        records.append(rec)
    return reps, records, elapsed


# ─── instruction-following audit ─────────────────────────────────────────────

def audit(records, pool, key="pred_generate"):
    """Classify what the model actually emitted.

    Every pool value is present in every stream (closed-pool permutation), so
    `off_pool` means the model produced a token that was nowhere in its context
    -- a genuine instruction violation rather than a retrieval error.
    """
    poolset = {v.lower() for v in pool}
    tally = {k: 0 for k in ("exact", "wrong_slot_same_key", "other_key_value",
                            "off_pool", "multi_word", "empty")}
    examples = {k: [] for k in tally}
    for r in records:
        raw = (r.get(key) or "")
        p = raw.strip().lower()
        cvs = {v.lower() for v in r["cv"]}
        if not p:
            bucket = "empty"
        elif len(raw.split()) > 1:
            bucket = "multi_word"
        elif p == r["expected"].lower():
            bucket = "exact"
        elif p in cvs:
            bucket = "wrong_slot_same_key"
        elif p in poolset:
            bucket = "other_key_value"
        else:
            bucket = "off_pool"
        tally[bucket] += 1
        if len(examples[bucket]) < 5:
            examples[bucket].append({"expected": r["expected"], "got": raw,
                                     "k": r["k"]})
    n = len(records) or 1
    return {"n": len(records),
            "counts": tally,
            "fractions": {k: v / n for k, v in tally.items()},
            "examples": examples}


# ─── model loading ───────────────────────────────────────────────────────────

def load(base_id, device, dtype):
    tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if "gemma-3" in base_id.lower():
        # gemma-3-4b-it ships as a multimodal Gemma3ForConditionalGeneration
        # with the LM under model.language_model.*. run_probing.py transplants
        # that text tower into a standalone Gemma3ForCausalLM so layer names
        # match `model.layers.N` -- necessary there ONLY because the LoRA
        # adapter was trained against those names, and attaching it to the
        # conditional-generation model silently binds it to the vision tower.
        #
        # This script is base-only, so there is no adapter and no reason to
        # transplant. Skipping it avoids holding two full copies of a 4B model
        # (~8.6 GB + ~8 GB) at peak, which on this 24 GB box with swap already
        # near full is the difference between running and being OOM-killed.
        # Reading hidden states straight off the CG model is equivalent: it
        # returns the text tower's hidden_states for a text-only input.
        from transformers import Gemma3ForConditionalGeneration
        model = Gemma3ForConditionalGeneration.from_pretrained(
            base_id, dtype=dtype, low_cpu_mem_usage=True,
            attn_implementation="sdpa").to(device).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_id, dtype=dtype, low_cpu_mem_usage=True).to(device).eval()
    return model, tokenizer


# ─── main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--cell", default="5,10", help="'keys,updates'; keys*updates must equal --pool_size")
    p.add_argument("--pool_size", type=int, default=50)
    p.add_argument("--pool_seed", type=int, default=1234)
    p.add_argument("--trials", type=int, default=200,
                   help="trials per condition when --T_json is not given")
    p.add_argument("--T_json", default=None,
                   help="JSON mapping condition -> trial count (see T_PRIORS)")
    p.add_argument("--generate_subset", type=int, default=40,
                   help="run greedy generate on the first N trials per condition, "
                        "for the instruction-following audit and the "
                        "single_token-vs-generate agreement check. Generation "
                        "costs 3.6x a forward pass on MPS, so it is not run on "
                        "every trial.")
    p.add_argument("--conditions", nargs="+", default=None,
                   help="restrict to these conditions. Required with --t_start, "
                        "which otherwise bypasses the already-collected skip for "
                        "every condition and would append to all of them.")
    p.add_argument("--force", action="store_true",
                   help="re-collect conditions that already have activations")
    p.add_argument("--t_start", type=int, default=0,
                   help="trial index offset; use to top up a condition that fell "
                        "short of the target wrong-subset size")
    p.add_argument("--shuffle", default="strict", choices=["strict", "legacy"],
                   help="strict: guaranteed anti-consecutive stream. legacy: "
                        "evaluate_ivq.shuffle_nc, which leaves an adjacent "
                        "same-category pair at stream position 48-49 in 20.4%% of "
                        "(5,10) trials -- exactly where CVQ and cv_last read.")
    p.add_argument("--T_cap", type=int, default=2500,
                   help="upper bound on T for --cap_conditions")
    p.add_argument("--cap_conditions", nargs="+", default=["FVQ", "k1"],
                   help="slot-1 conditions, capped at --T_cap by design")
    p.add_argument("--system_prompt", default="local", choices=["local", "cloud"],
                   help="local: the string used by evaluate_ivq.py and by "
                        "data_gen.py (the LoRA training data). cloud: the "
                        "API-sweep variant. Default local -- the LoRA is locked "
                        "to it and the T sizing came from it.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", default="mps")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--tag", default="")
    return p.parse_args()


def main():
    args = parse_args()
    nk, nu = (int(x) for x in args.cell.split(","))
    if nk * nu != args.pool_size:
        raise SystemExit(
            f"keys*updates = {nk*nu} != pool_size {args.pool_size}. The closed-pool "
            f"design requires equality so every stream is a permutation of the pool; "
            f"otherwise only {nk*nu} of {args.pool_size} values appear and the probe "
            f"can score above chance from presence alone.")

    dtype = torch.bfloat16 if args.device == "mps" else torch.float32
    print(f"Loading {args.base_model} on {args.device} ({dtype})...", flush=True)
    t = time.time()
    model, tokenizer = load(args.base_model, args.device, dtype)
    _c = model.config if getattr(model.config, "num_hidden_layers", None) is not None \
        else model.config.text_config
    n_layers_cfg, d_model_cfg = _c.num_hidden_layers, _c.hidden_size
    print(f"  loaded in {time.time()-t:.1f}s  layers={n_layers_cfg} "
          f"d_model={d_model_cfg}", flush=True)

    pool, n_safe, n_raw = select_pool(tokenizer, args.pool_size, args.pool_seed)
    install_pool(pool)
    print(f"Closed pool: {len(pool)} values (from {n_safe} generation-safe of "
          f"{n_raw} raw), seed={args.pool_seed}")
    print(f"  {pool}")

    short = args.base_model.split("/")[-1]
    out_dir = Path(args.out_dir) if args.out_dir else (
        _HERE / "results_probe50" / f"{short}_{nk}k_{nu}u{args.tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Always merge into an existing manifest rather than overwriting it, so an
    # interrupted run resumes instead of restarting. A condition that already
    # has activations is skipped unless --t_start > 0 (top-up) or --force.
    prev_manifest = {}
    if (out_dir / "manifest.json").exists():
        prev_manifest = json.load(open(out_dir / "manifest.json"))
        print(f"Resuming: {out_dir}/manifest.json has "
              f"{len(prev_manifest.get('conditions', {}))} conditions "
              f"({', '.join(prev_manifest.get('conditions', {}))})")

    T_map = json.load(open(args.T_json)) if args.T_json else {}
    conds = conditions_for(nu)
    if args.conditions:
        unknown = set(args.conditions) - set(conds)
        if unknown:
            raise SystemExit(f"unknown conditions {sorted(unknown)}; valid: {conds}")
        conds = [c for c in conds if c in args.conditions]
    print(f"Conditions ({len(conds)}): {conds}")

    manifest = {
        "base_model": args.base_model,
        "n_layers": n_layers_cfg,
        "d_model": d_model_cfg,
        "cell": {"keys": nk, "updates": nu},
        "pool": pool,
        "pool_seed": args.pool_seed,
        "device": args.device,
        "dtype": str(dtype),
        "generate_subset": args.generate_subset,
        "t_start": args.t_start,
        "shuffle": args.shuffle,
        "system_prompt_name": args.system_prompt,
        "system_prompt": SYSTEM_PROMPTS[args.system_prompt],
        "conditions": dict(prev_manifest.get("conditions", {})),
    }

    for cond in conds:
        already = (out_dir / f"reps_{cond}.npy").exists() and cond in manifest["conditions"]
        if already and args.t_start == 0 and not args.force:
            e = manifest["conditions"][cond]
            print(f"\n  [{cond}] already collected: n={e['n']} "
                  f"n_wrong={e.get('n_wrong')}, skipping", flush=True)
            continue
        k, word = condition_spec(cond, nu)
        n_trials = int(T_map.get(cond, args.trials))
        print(f"\n  [{cond}] slot k={k}, phrased \"{word}\", {n_trials} trials",
              flush=True)
        trials, _ = build_trials(nk, nu, cond, n_trials, tokenizer,
                                 t_start=args.t_start, shuffle=args.shuffle,
                                 system_prompt=SYSTEM_PROMPTS[args.system_prompt])
        reps, records, elapsed = collect(
            model, tokenizer, trials, args.device,
            batch_size=args.batch_size, generate_subset=args.generate_subset)

        acc_st = sum(r["correct_single_token"] for r in records) / len(records)
        entry = {
            "k": k, "query_word": word, "n": len(records),
            "elapsed_sec": elapsed, "ms_per_trial": elapsed / len(records) * 1000,
            "accuracy_single_token": acc_st,
            "wrong_frac_single_token": 1 - acc_st,
        }
        msg = (f"    acc(single_token)={acc_st:.3f} wrong_frac={1-acc_st:.3f} "
               f"{elapsed/len(records)*1000:.0f} ms/trial")

        gen_recs = [r for r in records if "pred_generate" in r]
        if gen_recs:
            acc_gen = sum(r["correct_generate"] for r in gen_recs) / len(gen_recs)
            agree = sum(r["correct_generate"] == r["correct_single_token"]
                        for r in gen_recs) / len(gen_recs)
            entry.update({
                "n_generate": len(gen_recs),
                "accuracy_generate": acc_gen,
                "scoring_agreement": agree,
                "audit_generate": audit(gen_recs, pool, "pred_generate"),
            })
            msg += f" | gen n={len(gen_recs)} acc={acc_gen:.3f} agree={agree:.3f}"
        entry["audit_single_token"] = audit(records, pool, "pred_single_token")
        n_wrong = sum(1 for r in records if not r["correct_single_token"])
        entry["n_wrong"] = n_wrong
        msg += f" | n_wrong={n_wrong}"
        print(msg, flush=True)

        # Top-up support: with --t_start > 0 the new trials are appended to any
        # existing arrays and records for this condition, rather than replacing
        # them. Trial seeds are a deterministic function of (nk, nu, k, t_idx),
        # so starting at t_start = previous n draws disjoint streams and the
        # concatenation contains no duplicates.
        npy = out_dir / f"reps_{cond}.npy"
        if args.t_start > 0 and npy.exists():
            prev_reps = np.load(npy)
            prev_recs = prev_manifest.get("conditions", {}).get(cond, {}).get("records", [])
            seen = {r["seed"] for r in prev_recs}
            keep = [i for i, r in enumerate(records) if r["seed"] not in seen]
            if len(keep) != len(records):
                print(f"    dropped {len(records)-len(keep)} duplicate seeds", flush=True)
            reps = np.concatenate([prev_reps, reps[keep]], axis=0)
            records = prev_recs + [records[i] for i in keep]
            entry["n"] = len(records)
            entry["n_wrong"] = sum(1 for r in records if not r["correct_single_token"])
            entry["topped_up_from"] = len(prev_recs)
            # Recompute the audits over the MERGED record set. They were
            # computed above on the increment only, so leaving them would make
            # every audit table after a top-up describe just the new trials.
            entry["audit_single_token"] = audit(records, pool, "pred_single_token")
            merged_gen = [r for r in records if "pred_generate" in r]
            if merged_gen:
                entry["n_generate"] = len(merged_gen)
                entry["accuracy_generate"] = sum(
                    r["correct_generate"] for r in merged_gen) / len(merged_gen)
                entry["scoring_agreement"] = sum(
                    r["correct_generate"] == r["correct_single_token"]
                    for r in merged_gen) / len(merged_gen)
                entry["audit_generate"] = audit(merged_gen, pool, "pred_generate")
            print(f"    topped up: {len(prev_recs)} + {len(keep)} = {len(records)} trials, "
                  f"n_wrong={entry['n_wrong']}", flush=True)
        np.save(npy, reps)
        entry["records"] = records
        del reps
        if args.device == "mps":
            torch.mps.empty_cache()
        manifest["conditions"][cond] = entry

        with open(out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    # Report which conditions fell short of the target wrong-subset size. No
    # separate pilot run: T comes from the v3 wrong_frac priors with headroom,
    # and any shortfall is topped up by re-running with --t_start (activations
    # append). Cross-condition figures use matched-n subsampling to the common
    # minimum wrong-subset size, which is exact -- strictly better than sizing T
    # from a noisy pilot estimate of wrong_frac.
    n_classes = len(pool)
    target = int(np.ceil(n_classes * 20 / 0.8))
    short = {}
    print(f"\nwrong-subset sizes (target {target} = {n_classes} classes x 20 / 0.8):")
    for cond, e in manifest["conditions"].items():
        nw = e["n_wrong"]
        wf = nw / e["n"] if e["n"] else 0.0
        gap = target - nw
        flag = ""
        if cond in args.cap_conditions:
            flag = "  [slot-1, capped by design]"
        elif gap > 0:
            short[cond] = int(np.ceil(gap / wf)) if wf > 0 else None
            flag = f"  SHORT by {gap} -> add ~{short[cond]} trials"
        print(f"  {cond:5} n={e['n']:5} wrong={nw:5} wrong_frac={wf:.3f}{flag}")
    manifest["wrong_subset"] = {
        "target": target,
        "common_min": min(e["n_wrong"] for e in manifest["conditions"].values()),
        "shortfall_extra_trials": short,
    }
    if short:
        print(f"\nTop up with: --t_start {max(e['n'] for e in manifest['conditions'].values())} "
              f"and a --T_json containing {json.dumps(short)}")
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
