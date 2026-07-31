"""
Linear probing — per-layer correctness decodability, base vs LoRA.

Fresh, self-contained pipeline (see PLAN.md in this folder). Reuses existing
trial-generation code verbatim (prompts are fixed, not redesigned):
  - FVQ/CVQ: v3/scripts/experiments/probing_classifier.generate_trial
  - IVQ:     lora_intervention/evaluate_ivq.make_prompt (+ local wrapper to
             also capture the candidate-value list / test category, which
             make_prompt doesn't return)

For each (state in {base, lora}) x (cell K,N) x (condition in
{FVQ, CVQ, IVQ_d10, IVQ_d25, IVQ_d50, IVQ_d75, IVQ_d90}):
  1. Run `trials` prompts through the model, collect the residual stream
     (HF hidden_states) at the last token for every layer, and the greedy
     one-token prediction.
  2. Label each trial correct/incorrect by exact string match against the
     (deterministically known) expected value.
  3. Train one logistic-regression probe per layer on (residual, correct)
     pairs, 5-fold CV, report accuracy alongside the majority-class baseline.

Usage (smoke test):
    .venv/bin/python lora_intervention/experiments/linear_probing/run_probing.py \\
        --state base --trials 20 --smoke
    .venv/bin/python lora_intervention/experiments/linear_probing/run_probing.py \\
        --state lora --trials 20 --smoke
"""
import argparse
import importlib.metadata as _imeta
import json
import random
import sys
import time
from pathlib import Path

# Match the lazy-import patches used elsewhere in this repo (some torch/peft
# versions choke on missing tensorflow/jax version strings).
import os
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]  # repo root (linear_probing -> experiments -> lora_intervention -> root)
sys.path.insert(0, str(_ROOT / "v3" / "scripts"))
sys.path.insert(0, str(_ROOT / "lora_intervention"))
sys.path.insert(0, str(_ROOT))

from experiments.probing_classifier import (  # noqa: E402
    generate_trial as _gen_fvq_cvq_trial,
)
from evaluate_ivq import ordinal, shuffle_nc, EVAL_CELLS, is_correct  # noqa: E402
from mechanistic_probing_v2.core.model_loader import verify_single_token  # noqa: E402
from mechanistic_probing_v2.core.dataset_configs import (  # noqa: E402
    get_value_pool, get_eligible_categories, generate_values_for_trial,
)
import mechanistic_probing_v2.core.dataset_configs as _dataset_configs  # noqa: E402


def restrict_pool_to_generation_safe_values(tokenizer, dataset_type="ARBITRARY_SINGLE"):
    """`verify_single_token` (existing repo helper) checks a value is one token
    only WITH a leading space (" wonder" -> 1 token) -- correct for how values
    appear mid-stream ("category: wonder"), but the model's greedy answer
    starts fresh with NO leading space, and many such words are 2+ tokens
    there (e.g. "wonder" -> "w"+"onder", "stolen" -> "st"+"olen"). Scoring
    only the first argmax token then misses the rest of the word for a large
    fraction of trials, silently undercounting correctness.

    Filters to values that are single-token in BOTH forms, then monkeypatches
    the dataset_configs pool cache so every downstream trial generator (FVQ,
    CVQ via probing_classifier.generate_trial, IVQ via
    generate_values_for_trial) draws only from genuinely generation-safe
    values, without changing any of their prompt/generation logic.
    """
    raw_pool = get_value_pool(dataset_type)
    safe = []
    for v in raw_pool:
        ids_sp = tokenizer.encode(f" {v}", add_special_tokens=False)
        ids_nosp = tokenizer.encode(v, add_special_tokens=False)
        if len(ids_sp) == 1 and len(ids_nosp) == 1:
            safe.append(v)
    print(f"Generation-safe single-token pool: {len(safe)}/{len(raw_pool)} "
          f"values survive the with-space AND no-space check")
    _dataset_configs._cached_pools["arbitrary_single"] = safe
    return safe

DATASET_TYPE = "ARBITRARY_SINGLE"
IVQ_SYSTEM_PROMPT = (
    "You are a precise data extraction tool. "
    "Output ONLY a single word - the exact value requested. "
    "No other text, no explanation, no punctuation."
)
IVQ_DEPTHS = [0.10, 0.25, 0.50, 0.75, 0.90]


def _model_class_for(model_id: str):
    if "gemma-3" in model_id.lower():
        try:
            from transformers import Gemma3ForCausalLM
            return Gemma3ForCausalLM
        except ImportError:
            pass
    return AutoModelForCausalLM


# ─── trial generation ───────────────────────────────────────────────────────

def gen_ivq_trial(nk, nu, depth, seed, tokenizer, model_name):
    """Mirror evaluate_ivq.make_prompt exactly (same stream/prompt text,
    same shuffle_nc anti-consecutive shuffle, same ordinal wording), but also
    return the candidate-value list + test category so we can log the same
    bookkeeping the FVQ/CVQ path logs. Returns None if depth position is
    invalid for the sampled N (e.g. degenerate stream)."""
    rng = random.Random(seed)
    elig = get_eligible_categories(DATASET_TYPE, min_values=nu)
    cats = rng.sample(elig, nk)
    try:
        vals = generate_values_for_trial(DATASET_TYPE, cats, nu, rng)
    except ValueError:
        return None

    test_cat = cats[seed % nk]
    items = [{"category": c, "value": v} for c in cats for v in vals[c]]
    items = shuffle_nc(items, rng)
    cv = [it["value"] for it in items if it["category"] == test_cat]
    n_values = len(cv)

    k = min(max(round(depth * n_values), 2), n_values - 1)
    if n_values < 3:
        return None  # no valid interior position

    expected = cv[k - 1]
    query_word = ordinal(k)

    stream = "\n".join(f"{it['category']}: {it['value']}" for it in items)
    user_txt = (
        f"Read the following key-value stream. Each key gets updated multiple times.\n\n"
        f"{stream}\n\nWhat was the {query_word} value of {test_cat}?"
    )
    msgs = [{"role": "system", "content": IVQ_SYSTEM_PROMPT},
            {"role": "user", "content": user_txt}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    return {
        "prompt": prompt, "expected": expected, "all_values": cv,
        "test_category": test_cat, "num_keys": nk, "num_updates": nu,
        "k": k, "n_values": n_values, "seed": seed,
    }


def gen_one_trial(condition, nk, nu, t_idx, tokenizer, model_name):
    """condition in {FVQ, CVQ, IVQ_d10, ..., IVQ_d90}. Returns one trial dict,
    or None if this index produced an invalid trial (e.g. IVQ depth position
    doesn't exist for the sampled N -- caller should just draw the next t_idx)."""
    seed = hash((nk, nu, condition, t_idx, "linear_probing")) % (2**31)
    if condition in ("FVQ", "CVQ"):
        rq_condition = "RI" if condition == "FVQ" else "PI"
        candidate_pool = get_value_pool(DATASET_TYPE)
        return _gen_fvq_cvq_trial(nk, nu, rq_condition, seed, candidate_pool,
                                   tokenizer, model_name)
    elif condition.startswith("IVQ_d"):
        depth = int(condition.split("_d")[1]) / 100.0
        return gen_ivq_trial(nk, nu, depth, seed, tokenizer, model_name)
    else:
        raise ValueError(f"Unknown condition: {condition}")


def build_trials(condition, nk, nu, n_trials, tokenizer, model_name, t_start=0):
    """Fixed-count trial batch (used by --smoke). See gen_one_trial for the
    per-index generator used by the adaptive (--full) path."""
    trials = []
    for t_idx in range(t_start, t_start + n_trials):
        trial = gen_one_trial(condition, nk, nu, t_idx, tokenizer, model_name)
        if trial is not None:
            trials.append(trial)
    return trials


def wilson_half_width(n_correct, n, z=1.96):
    """Half-width of the Wilson score interval for a binomial proportion."""
    if n == 0:
        return 1.0
    phat = n_correct / n
    denom = 1 + z * z / n
    half = z * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5)
    return half / denom


# ─── model + activation collection ──────────────────────────────────────────

def load_model(base_id, adapter_path, device, dtype):
    print(f"Loading base {base_id} (device={device}, dtype={dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    model_cls = _model_class_for(base_id)
    model = model_cls.from_pretrained(
        base_id, dtype=dtype, low_cpu_mem_usage=True,
        trust_remote_code=True, device_map=device,
    )
    if adapter_path:
        from peft import PeftModel
        print(f"Loading adapter {adapter_path} (no merge — PEFT-wrapped forward pass)...")
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model, tokenizer


@torch.no_grad()
def collect_activations(model, tokenizer, trials, device, scoring="generate", max_new_tokens=8):
    """Returns list of dicts: {reps [n_layers, d_model], correct, expected, pred_text}.

    Two independent halves per trial, deliberately decoupled:
      (1) Probe INPUT -- one forward pass over the prompt (teacher-forced, no
          generation), residual stream at the last prompt token, every layer.
          Identical regardless of `scoring`.
      (2) Correctness LABEL:
        - scoring="single_token": read the argmax of that same forward pass's
          final-position logits (1 token, no extra cost). Only reliable if
          every candidate value is single-token in a bare (no-leading-space)
          generation context -- see restrict_pool_to_generation_safe_values.
        - scoring="generate" (default): a separate short greedy generation
          (up to max_new_tokens), decoded and lenient-matched via
          evaluate_ivq.is_correct -- matches the paper's own behavioral-eval
          convention. Costs one extra (KV-cached, cheap) generation call per
          trial; does not change what's fed to the probe.
    """
    cfg = model.config  # PeftModel mirrors base_model.config onto .config too
    n_layers = cfg.num_hidden_layers
    d_model = cfg.hidden_size
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    out = []
    for trial in trials:
        inputs = tokenizer(trial["prompt"], return_tensors="pt").to(device)
        outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)

        hidden = outputs.hidden_states  # (n_layers+1) tensors, index 0 = embeddings
        reps = np.zeros((n_layers, d_model), dtype=np.float32)
        for L in range(n_layers):
            reps[L] = hidden[L + 1][0, -1, :].float().cpu().numpy()

        if scoring == "single_token":
            logits = outputs.logits[0, -1, :]
            pred_tid = int(logits.argmax().item())
            pred_text = tokenizer.decode([pred_tid]).strip().lower()
            correct = pred_text == trial["expected"].lower()
        elif scoring == "generate":
            gen_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=pad_id,
            )
            new_ids = gen_ids[0, inputs["input_ids"].shape[1]:]
            pred_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            correct = is_correct(pred_text, trial["expected"])
        else:
            raise ValueError(f"Unknown scoring mode: {scoring}")

        out.append({"reps": reps, "correct": correct, "expected": trial["expected"],
                    "pred_text": pred_text})

        del outputs, hidden, inputs

    return out, n_layers


def collect_activations_adaptive(model, tokenizer, condition, nk, nu, model_name, device,
                                  scoring="generate", max_new_tokens=8,
                                  min_trials=30, max_trials=200, hw_threshold=0.07,
                                  batch_size=10):
    """Wilson-CI early-stopping wrapper around collect_activations, matching the
    min30/cap200/HW<=0.07 convention used elsewhere in this repo (E1, E3).
    Draws trials in batches (skipping invalid IVQ indices) until either the
    correctness-rate Wilson interval half-width drops to hw_threshold or
    max_trials is reached. Returns (records, n_layers, stopped_early)."""
    records = []
    t_idx = 0
    n_layers = None
    guard = 0
    while len(records) < max_trials:
        batch, attempts = [], 0
        while len(batch) < batch_size and len(records) + len(batch) < max_trials and attempts < batch_size * 20:
            trial = gen_one_trial(condition, nk, nu, t_idx, tokenizer, model_name)
            t_idx += 1
            attempts += 1
            if trial is not None:
                batch.append(trial)
        guard += 1
        if not batch or guard > 5000:
            break  # condition/cell combo can't produce valid trials (e.g. IVQ depth doesn't exist)
        batch_records, n_layers = collect_activations(model, tokenizer, batch, device,
                                                        scoring=scoring, max_new_tokens=max_new_tokens)
        records.extend(batch_records)
        if len(records) >= min_trials:
            n_correct = sum(r["correct"] for r in records)
            hw = wilson_half_width(n_correct, len(records))
            if hw <= hw_threshold:
                return records, n_layers, True
    return records, n_layers, len(records) < max_trials


# ─── probe training ──────────────────────────────────────────────────────────

def train_layer_probes(records, n_layers):
    """One logistic-regression probe per layer on (residual, correct) pairs."""
    y = np.array([1 if r["correct"] else 0 for r in records])
    n = len(y)
    n_pos, n_neg = int(y.sum()), int(n - y.sum())
    majority_baseline = max(n_pos, n_neg) / n if n else 0.5

    layer_results = {}
    if n_pos == 0 or n_neg == 0:
        for L in range(n_layers):
            layer_results[str(L)] = {
                "degenerate": True, "cv_accuracy": None,
                "majority_baseline": majority_baseline,
                "n_pos": n_pos, "n_neg": n_neg,
            }
        return layer_results, majority_baseline, n_pos, n_neg

    n_splits = min(5, n_pos, n_neg)
    if n_splits < 2:
        for L in range(n_layers):
            layer_results[str(L)] = {
                "degenerate": True, "cv_accuracy": None,
                "majority_baseline": majority_baseline,
                "n_pos": n_pos, "n_neg": n_neg,
                "note": "too few minority-class samples for CV",
            }
        return layer_results, majority_baseline, n_pos, n_neg

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    for L in range(n_layers):
        X = np.stack([r["reps"][L] for r in records])
        clf = Pipeline([
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(C=0.1, max_iter=500, solver="lbfgs",
                                       class_weight="balanced")),
        ])
        try:
            scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
            cv_acc = float(scores.mean())
        except Exception as e:
            cv_acc = None
            layer_results[str(L)] = {
                "degenerate": True, "cv_accuracy": None,
                "majority_baseline": majority_baseline,
                "n_pos": n_pos, "n_neg": n_neg, "error": str(e),
            }
            continue
        layer_results[str(L)] = {
            "degenerate": False, "cv_accuracy": cv_acc,
            "majority_baseline": majority_baseline,
            "n_pos": n_pos, "n_neg": n_neg,
        }
    return layer_results, majority_baseline, n_pos, n_neg


# ─── orchestration ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--adapter", default="lora_intervention/checkpoints/adapter")
    p.add_argument("--state", required=True, choices=["base", "lora"])
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    p.add_argument("--smoke", action="store_true",
                   help="Smoke-test scope: 6 EVAL_CELLS x {FVQ,CVQ,IVQ_d50}, fixed --trials count")
    p.add_argument("--full", action="store_true",
                   help="Full-pilot scope: 6 EVAL_CELLS x {FVQ,CVQ,IVQ_d10..d90}, "
                        "Wilson-adaptive trial count (--min_trials/--max_trials/--hw_threshold)")
    p.add_argument("--scoring", default="generate", choices=["generate", "single_token"],
                   help="generate: multi-token greedy + lenient match (default, matches "
                        "the paper's behavioral-eval convention). single_token: read argmax "
                        "off the same forward pass's logits (needs restrict_pool_to_"
                        "generation_safe_values, and still misses hedged answers).")
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--min_trials", type=int, default=30)
    p.add_argument("--max_trials", type=int, default=200)
    p.add_argument("--hw_threshold", type=float, default=0.07)
    p.add_argument("--no_wilson", action="store_true",
                   help="Disable Wilson early-stopping for --full: every condition runs "
                        "the full --max_trials, fixed count, no matter how skewed the "
                        "running accuracy looks. Costs more but removes 'is this margin "
                        "real or just a small-n artifact' as a variable.")
    p.add_argument("--out_name", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    adapter_path = args.adapter if args.state == "lora" else None
    out_name = args.out_name or f"{Path(args.base).name}-{args.state}"

    model, tokenizer = load_model(args.base, adapter_path, args.device, dtype)

    if args.scoring == "single_token":
        # Only needed so the single-argmax-token label lines up with a
        # genuinely one-token value; "generate" scoring handles multi-token
        # values fine via lenient text match, so it uses the full pool.
        restrict_pool_to_generation_safe_values(tokenizer, DATASET_TYPE)
    candidate_pool = get_value_pool(DATASET_TYPE)
    print(f"Value pool: {len(candidate_pool)} values (scoring={args.scoring})")

    if args.full:
        cells = [c for split in EVAL_CELLS.values() for c in split]
        conditions = ["FVQ", "CVQ", "IVQ_d10", "IVQ_d25", "IVQ_d50", "IVQ_d75", "IVQ_d90"]
    elif args.smoke:
        cells = [c for split in EVAL_CELLS.values() for c in split]
        conditions = ["FVQ", "CVQ", "IVQ_d50"]
    else:
        raise NotImplementedError("Pass --smoke or --full")

    save_dir = _HERE / "results"
    save_dir.mkdir(parents=True, exist_ok=True)

    for nk, nu in cells:
        t0 = time.time()
        cell_out = {
            "state": args.state, "base_model": args.base,
            "adapter_path": str(adapter_path) if adapter_path else None,
            "cell": {"keys": nk, "updates": nu},
            "scoring": args.scoring, "mode": "full" if args.full else "smoke",
            "conditions": {},
            "device": args.device, "dtype": str(dtype),
        }
        print(f"\n{'='*60}\nCell K={nk} N={nu}  (state={args.state})\n{'='*60}")

        for condition in conditions:
            if args.full:
                records, n_layers, stopped_early = collect_activations_adaptive(
                    model, tokenizer, condition, nk, nu, args.base, args.device,
                    scoring=args.scoring, max_new_tokens=args.max_new_tokens,
                    min_trials=args.min_trials, max_trials=args.max_trials,
                    hw_threshold=(-1.0 if args.no_wilson else args.hw_threshold),
                )
            else:
                trials = build_trials(condition, nk, nu, args.trials, tokenizer, args.base)
                records, n_layers = (None, None) if not trials else collect_activations(
                    model, tokenizer, trials, args.device,
                    scoring=args.scoring, max_new_tokens=args.max_new_tokens)
                stopped_early = None

            if not records:
                print(f"  {condition}: no valid trials generated, skipping")
                continue

            behavioral_acc = sum(r["correct"] for r in records) / len(records)
            layer_results, majority_baseline, n_pos, n_neg = train_layer_probes(records, n_layers)

            n_degenerate = sum(1 for v in layer_results.values() if v["degenerate"])
            print(f"  {condition}: behavioral_acc={behavioral_acc:.0%} "
                  f"(n_pos={n_pos}, n_neg={n_neg}, n={len(records)}"
                  f"{', stopped_early' if stopped_early else ''}), "
                  f"majority_baseline={majority_baseline:.0%}, "
                  f"{n_degenerate}/{n_layers} layers degenerate")
            if n_degenerate < n_layers:
                last_layers = sorted(layer_results.keys(), key=int)[-5:]
                for L in last_layers:
                    r = layer_results[L]
                    if not r["degenerate"]:
                        print(f"    L{L:>3}: probe_acc={r['cv_accuracy']:.0%} "
                              f"(baseline {r['majority_baseline']:.0%})")

            cell_out["conditions"][condition] = {
                "behavioral_accuracy": behavioral_acc, "n": len(records),
                "stopped_early": stopped_early,
                "n_layers": n_layers, "layers": layer_results,
                # Raw generations (no `reps` -- those are large per-layer
                # float arrays; behavioral audit trail only).
                "trials": [{"expected": r["expected"], "pred_text": r["pred_text"],
                            "correct": r["correct"]} for r in records],
            }

        cell_out["elapsed_sec"] = time.time() - t0
        save_path = save_dir / f"probe_{out_name}_{nk}k_{nu}u.json"
        with open(save_path, "w") as f:
            json.dump(cell_out, f, indent=2)
        print(f"  Saved: {save_path}  (elapsed {cell_out['elapsed_sec']:.1f}s)")


if __name__ == "__main__":
    main()
