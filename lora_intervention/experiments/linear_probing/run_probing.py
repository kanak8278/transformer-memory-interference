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
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from sklearn.metrics import roc_auc_score
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

# Full 31-cell grid: 3 in-training [(2,5),(5,10),(10,20)] + the 28 canonical
# held-out cells (K in {7,10,15,20,25,30} x N in {10,15,20,30,50,75}, 28 of 36)
# used in lora_intervention/results/main_eval_20260523_055309.json, so probe
# behavioral_accuracy is cross-checkable against that table at every cell.
GRID_FULL31 = [
    (2, 5), (5, 10), (10, 20),
    (7, 30), (7, 50), (7, 75),
    (10, 30), (10, 50), (10, 75),
    (15, 15), (15, 20), (15, 30), (15, 50), (15, 75),
    (20, 15), (20, 20), (20, 30), (20, 50), (20, 75),
    (25, 10), (25, 15), (25, 20), (25, 30), (25, 50), (25, 75),
    (30, 10), (30, 15), (30, 20), (30, 30), (30, 50), (30, 75),
]


def _model_class_for(model_id: str):
    if "gemma-3" in model_id.lower():
        # gemma-3-4b-it is a multimodal checkpoint (keys under language_model.* +
        # a vision_tower); in transformers 5.14.1 Gemma3ForCausalLM matches none
        # of them (loads random weights). Gemma3ForConditionalGeneration maps the
        # checkpoint cleanly and forwards text-only input fine.
        try:
            from transformers import Gemma3ForConditionalGeneration
            return Gemma3ForConditionalGeneration
        except ImportError:
            pass
    return AutoModelForCausalLM


def _cfg_dims(cfg):
    """(num_hidden_layers, hidden_size), handling nested configs (gemma-3's
    Gemma3Config keeps them under .text_config)."""
    n = getattr(cfg, "num_hidden_layers", None)
    d = getattr(cfg, "hidden_size", None)
    if n is None or d is None:
        tc = getattr(cfg, "text_config", None)
        n = n or getattr(tc, "num_hidden_layers", None)
        d = d or getattr(tc, "hidden_size", None)
    return n, d


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

def _load_gemma_text_causal_lm(base_id, device, dtype):
    """gemma-3-4b-it ships as a MULTIMODAL Gemma3ForConditionalGeneration
    checkpoint (LM under model.language_model.*, plus a vision_tower). Loading it
    as Gemma3ForCausalLM in transformers 5.14.1 matches no keys (random weights);
    loading it as the conditional-generation model puts the LM at
    `model.language_model.layers.N`, which does NOT match the LoRA adapter (trained
    on a plain Gemma3ForCausalLM with `model.layers.N`) -- the adapter then silently
    attaches to the vision tower and leaves the LM at LoRA init (B=0, zero effect;
    base==lora bug). Fix: load the multimodal checkpoint correctly, then transplant
    the text tower into a standalone Gemma3ForCausalLM whose structure
    (`model.layers.N`) matches exactly how the adapter was trained."""
    from transformers import Gemma3ForConditionalGeneration, Gemma3ForCausalLM
    cg = Gemma3ForConditionalGeneration.from_pretrained(
        base_id, dtype=dtype, low_cpu_mem_usage=True, attn_implementation="sdpa")
    text_cfg = cg.config.text_config
    text_cfg._attn_implementation = "sdpa"
    text = Gemma3ForCausalLM(text_cfg)
    # cg.model.language_model is a Gemma3TextModel; so is text.model -> same keys.
    missing, unexpected = text.model.load_state_dict(
        cg.model.language_model.state_dict(), strict=False)
    assert not unexpected, f"unexpected keys transplanting gemma text tower: {unexpected[:5]}"
    text.tie_weights()  # gemma ties lm_head <- embed_tokens (loaded above)
    del cg
    return text.to(dtype).to(device).eval()


def load_model(base_id, adapter_path, device, dtype):
    print(f"Loading base {base_id} (device={device}, dtype={dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    if "gemma-3" in base_id.lower():
        model = _load_gemma_text_causal_lm(base_id, device, dtype)
    else:
        model = _model_class_for(base_id).from_pretrained(
            base_id, dtype=dtype, low_cpu_mem_usage=True,
            trust_remote_code=True, device_map=device)
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
    n_layers, d_model = _cfg_dims(cfg)
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
                    "pred_text": pred_text,
                    # in-context candidate values (stream order) for the retrieval
                    # probe -- the plausible answers actually present in context.
                    "candidates": list(trial.get("all_values", []))})

        del outputs, hidden, inputs

    return out, n_layers


def collect_activations_adaptive(model, tokenizer, condition, nk, nu, model_name, device,
                                  scoring="generate", max_new_tokens=8,
                                  min_trials=30, max_trials=200, hw_threshold=0.07,
                                  batch_size=10, stop_mode="balanced",
                                  min_minority=40, sat_check_at=60):
    """Adaptive trial collection. Two stopping regimes:

    stop_mode="balanced" (default, the probe-appropriate rule): keep drawing
      until BOTH classes have >= `min_minority` examples (probe power is set by
      minority-class count, not by the CI of the mean), or until `max_trials`.
      Bails early as "saturated" if after `sat_check_at` trials the minority
      class is still empty and the rate is extreme -- those cells are
      correctness-untestable no matter how many more we draw, so we don't waste
      GPU on them. This replaces the Wilson-on-accuracy rule, which stopped
      skewed cells at n~=30-60 with 1-4 minority samples (statistically dead).

    stop_mode="wilson" (legacy): stop when the correctness-rate Wilson interval
      half-width drops to `hw_threshold`. Kept for reproducing the pilot.

    Returns (records, n_layers, stop_reason)."""
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
        n = len(records)
        n_pos = sum(r["correct"] for r in records)
        n_neg = n - n_pos

        if stop_mode == "wilson":
            if n >= min_trials and wilson_half_width(n_pos, n) <= hw_threshold:
                return records, n_layers, "wilson"
        else:  # balanced
            if min(n_pos, n_neg) >= min_minority:
                return records, n_layers, "balanced"
            if n >= sat_check_at and min(n_pos, n_neg) == 0:
                rate = n_pos / n
                if rate >= 0.98 or rate <= 0.02:
                    return records, n_layers, "saturated"
    return records, n_layers, "max_trials"


# ─── probe training ──────────────────────────────────────────────────────────

_PROBE_SCORERS = {"acc": "accuracy", "auc": "roc_auc", "bal_acc": "balanced_accuracy"}


def train_layer_probes(records, n_layers):
    """One logistic-regression probe per layer on (residual, correct) pairs.

    Reports three metrics per layer:
      - cv_accuracy: plain accuracy (kept for continuity with the pilot). Its
        chance level is the majority_baseline, which under class skew a
        balanced-loss probe structurally can't beat even when it separates the
        classes -- so it under-reports signal at skewed cells.
      - cv_auc: ROC-AUC. Threshold-free, imbalance-robust, chance == 0.5
        regardless of skew. This is the primary metric: it makes base vs LoRA
        comparable at *every* cell with both classes present, not just the
        near-balanced ones.
      - cv_balanced_acc: balanced accuracy (mean per-class recall), chance 0.5.
    """
    y = np.array([1 if r["correct"] else 0 for r in records])
    n = len(y)
    n_pos, n_neg = int(y.sum()), int(n - y.sum())
    majority_baseline = max(n_pos, n_neg) / n if n else 0.5

    layer_results = {}

    def _all_degenerate(note=None):
        for L in range(n_layers):
            rec = {
                "degenerate": True, "cv_accuracy": None, "cv_auc": None,
                "cv_balanced_acc": None, "majority_baseline": majority_baseline,
                "n_pos": n_pos, "n_neg": n_neg,
            }
            if note:
                rec["note"] = note
            layer_results[str(L)] = rec
        return layer_results, majority_baseline, n_pos, n_neg

    if n_pos == 0 or n_neg == 0:
        return _all_degenerate()

    n_splits = min(5, n_pos, n_neg)
    if n_splits < 2:
        return _all_degenerate("too few minority-class samples for CV")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    for L in range(n_layers):
        X = np.stack([r["reps"][L] for r in records])
        clf = Pipeline([
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(C=0.1, max_iter=500, solver="lbfgs",
                                       class_weight="balanced")),
        ])
        try:
            cv = cross_validate(clf, X, y, cv=skf, scoring=_PROBE_SCORERS,
                                error_score=np.nan)
            def _mean(key):
                v = float(np.nanmean(cv[f"test_{key}"]))
                return None if np.isnan(v) else v
            cv_acc = _mean("acc")
            cv_auc = _mean("auc")
            cv_bal = _mean("bal_acc")
        except Exception as e:
            layer_results[str(L)] = {
                "degenerate": True, "cv_accuracy": None, "cv_auc": None,
                "cv_balanced_acc": None, "majority_baseline": majority_baseline,
                "n_pos": n_pos, "n_neg": n_neg, "error": str(e),
            }
            continue
        layer_results[str(L)] = {
            "degenerate": False, "cv_accuracy": cv_acc, "cv_auc": cv_auc,
            "cv_balanced_acc": cv_bal, "majority_baseline": majority_baseline,
            "n_pos": n_pos, "n_neg": n_neg,
        }
    return layer_results, majority_baseline, n_pos, n_neg


# ─── retrieval probe (correctness-INDEPENDENT) ───────────────────────────────

def _first_token_id(tokenizer, value):
    """Token id the model emits FIRST when generating `value` at a fresh
    (no-leading-space) answer position -- i.e. what a last-position residual is
    about to produce. Multi-token values contribute their leading subword."""
    ids = tokenizer.encode(value, add_special_tokens=False)
    return ids[0] if ids else None


def train_retrieval_probes(records, n_layers, unembed, tokenizer, min_pairs=20):
    """Correctness-INDEPENDENT probe: is the *ground-truth answer value* linearly
    decodable from the residual stream, discriminated from the other in-context
    candidate values? Label comes from the trial (which value is correct), NOT
    from the model's output -- so it is defined even where every trial is
    correct (LoRA saturated) or every trial is wrong (base IVQ floor).

    Per layer, per trial with >=1 usable distractor, we form:
        positive: feature(r_L, e(expected))   -> 1
        negative: feature(r_L, e(distractor)) -> 0
    where feature = r_L (elementwise*) e(v), e(v) = unembedding row of v's first
    token. LR on this diagonal bilinear form is a per-layer *trained* logit lens
    (chance AUC 0.5). Reported for three trial subsets:
      - all:     every trial (base vs LoRA "is the answer represented at all")
      - wrong:   only trials the model got WRONG -> the "tracked but suppressed"
                 test (is the correct value still decodable when not emitted?)
      - correct: only trials the model got right (sanity; should be high)
    """
    rng = random.Random(0)

    def _pairs_for(subset):
        """Yield (record, expected_tid, distractor_tid) for records in subset."""
        pairs = []
        for r in subset:
            exp = r["expected"]
            exp_tid = _first_token_id(tokenizer, exp)
            if exp_tid is None:
                continue
            distractors = [v for v in r["candidates"]
                           if v.lower() != exp.lower()
                           and _first_token_id(tokenizer, v) not in (None, exp_tid)]
            if not distractors:
                continue
            d = rng.choice(distractors)
            pairs.append((r, exp_tid, _first_token_id(tokenizer, d)))
        return pairs

    def _auc_by_layer(pairs):
        if len(pairs) < min_pairs:
            return None, None  # too few trials to say anything
        # y: 1 for the positive of each pair, 0 for the negative
        y = np.array([1, 0] * len(pairs))
        n_splits = min(5, len(pairs))
        if n_splits < 2:
            return None, None
        # group-aware: keep a trial's pos+neg in the same fold so the probe can't
        # cheat off the shared residual. KFold over trial (pair) index.
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        pair_idx = np.arange(len(pairs))
        layer_auc = {}
        best = (None, -1.0)
        for L in range(n_layers):
            X = np.empty((2 * len(pairs), unembed.shape[1]), dtype=np.float32)
            for i, (r, exp_tid, dis_tid) in enumerate(pairs):
                rL = r["reps"][L]
                X[2 * i] = rL * unembed[exp_tid]
                X[2 * i + 1] = rL * unembed[dis_tid]
            clf = Pipeline([
                ("scale", StandardScaler()),
                ("lr", LogisticRegression(C=0.1, max_iter=500, solver="lbfgs")),
            ])
            aucs = []
            for tr, te in kf.split(pair_idx):
                tr_rows = np.concatenate([[2 * i, 2 * i + 1] for i in tr])
                te_rows = np.concatenate([[2 * i, 2 * i + 1] for i in te])
                try:
                    clf.fit(X[tr_rows], y[tr_rows])
                    p = clf.predict_proba(X[te_rows])[:, 1]
                    aucs.append(roc_auc_score(y[te_rows], p))
                except Exception:
                    pass
            auc = float(np.mean(aucs)) if aucs else None
            layer_auc[str(L)] = {"retrieval_auc": auc, "n_pairs": len(pairs)}
            if auc is not None and auc > best[1]:
                best = (str(L), auc)
        return layer_auc, {"best_layer": best[0], "best_auc": best[1] if best[0] else None}

    subsets = {
        "all": records,
        "wrong": [r for r in records if not r["correct"]],
        "correct": [r for r in records if r["correct"]],
    }
    out = {}
    for name, subset in subsets.items():
        layer_auc, summ = _auc_by_layer(_pairs_for(subset))
        out[name] = {"n_trials": len(subset), "summary": summ, "layers": layer_auc}
    return out


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
    p.add_argument("--max_trials", type=int, default=400,
                   help="Hard cap on trials per (condition, cell). Raised from the "
                        "pilot's 200 because balanced stopping needs headroom to "
                        "accumulate the minority class at skewed cells.")
    p.add_argument("--hw_threshold", type=float, default=0.07)
    p.add_argument("--stop_mode", default="balanced", choices=["balanced", "wilson"],
                   help="balanced (default): draw until both classes have "
                        "--min_minority examples (probe-appropriate). wilson: legacy "
                        "accuracy-CI early-stop that reproduces the pilot.")
    p.add_argument("--min_minority", type=int, default=40,
                   help="balanced stop target: min examples required in the smaller "
                        "class before stopping.")
    p.add_argument("--no_wilson", action="store_true",
                   help="Force the full --max_trials regardless of stop_mode (fixed "
                        "count). Removes 'is this margin a small-n artifact' as a "
                        "variable, at extra GPU cost.")
    p.add_argument("--grid", default="eval6", choices=["eval6", "full31"],
                   help="Cell grid for --full. eval6: the 6 EVAL_CELLS (pilot). "
                        "full31: 3 in-training + 28 canonical held-out cells "
                        "(cross-checkable against main_eval_*.json).")
    p.add_argument("--no_retrieval", action="store_true",
                   help="Skip the correctness-independent retrieval probe (Fix 3). "
                        "By default it runs on the same collected residuals (CPU "
                        "only, no extra generation) and adds the wrong-answer "
                        "'tracked but suppressed' test.")
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

    # Unembedding matrix for the correctness-independent retrieval probe (Fix 3).
    # Pulled once to CPU float32; rows are indexed by value first-token id.
    unembed = None
    if not args.no_retrieval:
        W = model.get_output_embeddings().weight  # [vocab, d_model]
        unembed = W.detach().float().cpu().numpy()
        print(f"Unembedding for retrieval probe: {unembed.shape}")

    if args.full:
        if args.grid == "full31":
            cells = list(GRID_FULL31)
        else:
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
                # --no_wilson forces the full cap: min_minority beyond reach and
                # hw_threshold impossible, so neither stop rule fires early.
                records, n_layers, stop_reason = collect_activations_adaptive(
                    model, tokenizer, condition, nk, nu, args.base, args.device,
                    scoring=args.scoring, max_new_tokens=args.max_new_tokens,
                    min_trials=args.min_trials, max_trials=args.max_trials,
                    hw_threshold=(-1.0 if args.no_wilson else args.hw_threshold),
                    stop_mode=args.stop_mode,
                    min_minority=(10**9 if args.no_wilson else args.min_minority),
                )
            else:
                trials = build_trials(condition, nk, nu, args.trials, tokenizer, args.base)
                records, n_layers = (None, None) if not trials else collect_activations(
                    model, tokenizer, trials, args.device,
                    scoring=args.scoring, max_new_tokens=args.max_new_tokens)
                stop_reason = None

            if not records:
                print(f"  {condition}: no valid trials generated, skipping")
                continue

            behavioral_acc = sum(r["correct"] for r in records) / len(records)
            layer_results, majority_baseline, n_pos, n_neg = train_layer_probes(records, n_layers)

            n_degenerate = sum(1 for v in layer_results.values() if v["degenerate"])
            print(f"  {condition}: behavioral_acc={behavioral_acc:.0%} "
                  f"(n_pos={n_pos}, n_neg={n_neg}, n={len(records)}, stop={stop_reason}), "
                  f"majority_baseline={majority_baseline:.0%}, "
                  f"{n_degenerate}/{n_layers} layers degenerate")
            # Report the best-AUC layer (primary metric) instead of just the last
            # 5 layers' accuracy -- AUC's chance is 0.5 everywhere, so this reads
            # the same at balanced and skewed cells.
            testable = [v for v in layer_results.values()
                        if not v["degenerate"] and v.get("cv_auc") is not None]
            if testable:
                best = max(testable, key=lambda v: v["cv_auc"])
                best_L = next(L for L, v in layer_results.items() if v is best)
                print(f"    best AUC: L{best_L}={best['cv_auc']:.2f} "
                      f"(acc={best['cv_accuracy']:.0%} vs baseline "
                      f"{best['majority_baseline']:.0%}, bal_acc={best['cv_balanced_acc']:.0%})")

            retrieval = None
            if unembed is not None:
                retrieval = train_retrieval_probes(records, n_layers, unembed, tokenizer)
                for name in ("all", "wrong", "correct"):
                    s = retrieval[name]["summary"]
                    nt = retrieval[name]["n_trials"]
                    if s and s.get("best_auc") is not None:
                        tag = " <-- suppression test" if name == "wrong" else ""
                        print(f"    retrieval[{name:7}] n={nt:3}: best AUC "
                              f"L{s['best_layer']}={s['best_auc']:.2f}{tag}")
                    else:
                        print(f"    retrieval[{name:7}] n={nt:3}: (too few usable trials)")

            cell_out["conditions"][condition] = {
                "behavioral_accuracy": behavioral_acc, "n": len(records),
                "stop_reason": stop_reason,
                "retrieval": retrieval,
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
