"""
Entropy-Lens (Ruggeri/Irwin et al., arXiv:2502.16570) on the KV-interference
task, run comparably across three models:

  - base   : Qwen/Qwen2.5-3B-Instruct, unmodified
  - lora    : + main LoRA adapter (PEFT-wrapped, no merge)
  - scratch : the from-scratch GPT-2-small trained ONLY on the synthetic task
              (05_gpt2_scratch_h100/checkpoints/h100_cosine/best.pt)

For every trial we take a single forward pass, and at the ANSWER position apply
a logit lens at EACH layer -- unembed(final_norm(resid_post_L)) -> softmax over
the full vocab -> Shannon entropy (nats). This is the Entropy-Lens per-layer
scalar. We keep both the raw entropy (nats) and the entropy normalised by
log(vocab) so the huge-vocab Qwen (V=151936) and the tiny-vocab scratch GPT
(V=51) are on the same [0,1] scale, and we index layers by RELATIVE depth
(layer / n_layers) so 36-layer Qwen and 12-layer GPT-2 line up (the paper's
"invariant under depth rescaling").

Same TASK for all three: FVQ (first value), CVQ (last value), IVQ at relative
depths {0.25,0.50,0.75} (intermediate), on the scratch model's native
(K,N) grid so every cell is runnable by all three. Qwen sees the text
ARBITRARY_SINGLE prompts (reusing linear_probing's trial generators verbatim);
the scratch GPT sees its own 51-symbol integer prompts (reusing data_gen).
These are necessarily different token sequences -- the models have different
tokenizers -- but the same task, cell, query-type and relative depth, which is
what makes the entropy PROFILES comparable.

Correctness is scored for free from the same forward (single-token argmax at the
answer position; Qwen values restricted to the generation-safe single-token
pool), so entropy profiles can be split into correct vs wrong trials -- the
"confusion" view.

Usage:
    python run_entropy_lens.py --model base    --device cuda:0
    python run_entropy_lens.py --model lora    --device cuda:1
    python run_entropy_lens.py --model scratch --device cuda:0
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]  # entropy_lens -> experiments -> lora_intervention -> root
_LP = _ROOT / "lora_intervention" / "experiments" / "linear_probing"
_SCRATCH_SRC = _ROOT / "synthetic_scratch_training" / "05_gpt2_scratch_h100" / "src"
SCRATCH_CKPT = _ROOT / "synthetic_scratch_training" / "05_gpt2_scratch_h100" / "checkpoints" / "h100_cosine" / "best.pt"

sys.path.insert(0, str(_LP))

# ─── shared experiment config (identical across all three models) ─────────────
GRID = [(k, n) for k in (2, 4, 6, 8, 10, 12) for n in (4, 6, 8, 10, 12)]
IVQ_DEPTHS = {"IVQ_d25": 0.25, "IVQ_d50": 0.50, "IVQ_d75": 0.75}
CONDITIONS = ["FVQ", "CVQ"] + list(IVQ_DEPTHS)
TRIALS = 150


def ivq_step(depth, n):
    """Same interior-position rule as linear_probing.gen_ivq_trial."""
    return min(max(round(depth * n), 2), n - 1)


# ─── entropy core (shared) ────────────────────────────────────────────────────

@torch.no_grad()
def layer_entropies(hidden_states, pos, norm, head, log_vocab):
    """Per-layer logit-lens Shannon entropy (nats) + normalised entropy at token
    `pos`. hidden_states: tuple len n_layers+1 (index 0 = embeddings). Returns
    (raw[list], norm[list], final_pred_token_id)."""
    n_layers = len(hidden_states) - 1
    raw, normed = [], []
    pred_tid = None
    for L in range(1, n_layers + 1):
        h = hidden_states[L][:, pos, :]              # [1, d]
        logits = head(norm(h)).float()               # [1, vocab]
        logp = torch.log_softmax(logits, dim=-1)
        H = float(-(logp.exp() * logp).sum(-1).item())
        raw.append(H)
        normed.append(H / log_vocab)
        if L == n_layers:
            pred_tid = int(logits.argmax(-1).item())
    return raw, normed, pred_tid


# ─── HF models (base / lora) ──────────────────────────────────────────────────

def hf_norm_and_head(model):
    """Final norm + unembedding for the logit lens, unwrapping PEFT if present."""
    m = model
    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        m = m.base_model.model           # PeftModel -> underlying CausalLM
    return m.model.norm, m.get_output_embeddings()


@torch.no_grad()
def run_hf(args, device):
    from run_probing import (load_model, gen_one_trial, DATASET_TYPE,
                             restrict_pool_to_generation_safe_values)
    from mechanistic_probing_v2.core.dataset_configs import get_value_pool

    adapter = args.adapter if args.model == "lora" else None
    model, tokenizer = load_model(args.base, adapter, device,
                                  torch.bfloat16 if args.dtype == "bfloat16" else torch.float32)
    # restrict to generation-safe single-token values so the free single-token
    # argmax correctness label is accurate (same fix as linear_probing).
    restrict_pool_to_generation_safe_values(tokenizer, DATASET_TYPE)
    vocab_size = model.config.vocab_size
    log_vocab = math.log(vocab_size)
    norm, head = hf_norm_and_head(model)
    n_layers = model.config.num_hidden_layers

    def one_trial(condition, nk, nu, t_idx):
        trial = gen_one_trial(condition, nk, nu, t_idx, tokenizer, args.base)
        if trial is None:
            return None
        inputs = tokenizer(trial["prompt"], return_tensors="pt").to(device)
        out = model(**inputs, output_hidden_states=True, use_cache=False)
        raw, normed, pred_tid = layer_entropies(out.hidden_states, -1, norm, head, log_vocab)
        pred_text = tokenizer.decode([pred_tid]).strip().lower()
        correct = pred_text == trial["expected"].lower()
        return raw, normed, correct

    return collect(args, one_trial, n_layers, vocab_size, device, meta_extra={})


# ─── scratch GPT-2 ────────────────────────────────────────────────────────────

SCRATCH_HF_REPO = "kanak8278/gpt2-small-synthetic-kv-interference"


def _hf_token():
    env = _ROOT / ".env"
    if env.exists():
        import re
        for line in open(env):
            m = re.match(r"HF_API_KEY=(.+)", line.strip())
            if m:
                return m.group(1).strip().strip('"').strip("'")
    return None


def load_scratch_gpt(device, source="hf"):
    """Return (gpt2_model, source_str). Prefer the canonical HF-published weights
    (kanak8278/gpt2-small-synthetic-kv-interference, verified bit-identical to
    the local best.pt); fall back to the local checkpoint if HF is unavailable."""
    from transformers import GPT2LMHeadModel
    if source == "hf":
        try:
            gpt = GPT2LMHeadModel.from_pretrained(SCRATCH_HF_REPO, token=_hf_token())
            return gpt.to(device).eval(), f"hf:{SCRATCH_HF_REPO}"
        except Exception as e:
            print(f"HF load failed ({e}); falling back to local checkpoint")
    sys.path.insert(0, str(_SCRATCH_SRC))
    from model_gpt2 import GPT2FromScratch
    wrap = GPT2FromScratch(dropout=0.0)
    ck = torch.load(str(SCRATCH_CKPT), map_location="cpu", weights_only=False)
    wrap.load_state_dict(ck["model"])
    return wrap.gpt2.to(device).eval(), f"local:{SCRATCH_CKPT.name}(step{ck.get('step')})"


@torch.no_grad()
def run_scratch(args, device):
    sys.path.insert(0, str(_SCRATCH_SRC))
    import vocab as sv
    from data_gen import build_example

    gpt, source = load_scratch_gpt(device, source=args.scratch_source)
    print(f"Loaded scratch GPT-2 from {source}")
    norm, head = gpt.transformer.ln_f, gpt.lm_head
    vocab_size = sv.VOCAB_SIZE
    log_vocab = math.log(vocab_size)
    n_layers = gpt.config.n_layer

    def one_trial(condition, nk, nu, t_idx):
        # deterministic per (cell, condition, index), independent of the HF path
        rng = random.Random(hash((nk, nu, condition, t_idx, "entropy_scratch")) % (2**31))
        if condition == "FVQ":
            step = 1
        elif condition == "CVQ":
            step = nu
        else:
            step = ivq_step(IVQ_DEPTHS[condition], nu)
        full_ids, meta = build_example(rng, nk, nu, step)
        # eval convention: feed sequence without EOS; answer computed at the
        # step-token position (index -2 of `inp`), whose logits predict VALUE.
        inp = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
        out = gpt(input_ids=inp, output_hidden_states=True)
        raw, normed, pred_tid = layer_entropies(out.hidden_states, -2, norm, head, log_vocab)
        target_value_id = full_ids[-2]  # VALUE token id
        correct = pred_tid == target_value_id
        return raw, normed, correct

    return collect(args, one_trial, n_layers, vocab_size, device,
                   meta_extra={"scratch_source": source})


# ─── collection / aggregation (shared) ────────────────────────────────────────

def collect(args, one_trial, n_layers, vocab_size, device, meta_extra):
    out = {
        "model": args.model, "n_layers": n_layers, "vocab_size": vocab_size,
        "log_vocab": math.log(vocab_size), "trials_per_cond": args.trials,
        "grid": [list(c) for c in GRID], "conditions": CONDITIONS,
        "ivq_depths": IVQ_DEPTHS, "cells": {}, **meta_extra,
    }
    t0 = time.time()
    for nk, nu in GRID:
        cell_key = f"K{nk}_N{nu}"
        out["cells"][cell_key] = {}
        for condition in CONDITIONS:
            raws, normeds, corrects = [], [], []
            t_idx = 0
            attempts = 0
            while len(raws) < args.trials and attempts < args.trials * 5:
                r = one_trial(condition, nk, nu, t_idx)
                t_idx += 1
                attempts += 1
                if r is None:
                    continue
                raw, normed, correct = r
                raws.append(raw); normeds.append(normed); corrects.append(correct)
            if not raws:
                continue
            raws = np.array(raws); normeds = np.array(normeds)
            corr = np.array(corrects, dtype=bool)
            def prof(mask):
                if mask.sum() == 0:
                    return None
                return {"raw": raws[mask].mean(0).tolist(),
                        "norm": normeds[mask].mean(0).tolist(),
                        "n": int(mask.sum())}
            out["cells"][cell_key][condition] = {
                "n": len(raws), "behavioral_accuracy": float(corr.mean()),
                "all": prof(np.ones(len(raws), bool)),
                "correct": prof(corr), "wrong": prof(~corr),
            }
        acc = np.mean([out["cells"][cell_key][c]["behavioral_accuracy"]
                       for c in out["cells"][cell_key]])
        print(f"  {cell_key}: mean acc {acc:.2f}  ({time.time()-t0:.0f}s elapsed)")
    out["elapsed_sec"] = time.time() - t0
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["base", "lora", "scratch"])
    p.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--adapter", default="lora_intervention/checkpoints/adapter")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    p.add_argument("--trials", type=int, default=TRIALS)
    p.add_argument("--scratch_source", default="hf", choices=["hf", "local"],
                   help="Where to load the from-scratch GPT-2 weights: hf (the "
                        "canonical published repo, default) or local best.pt.")
    p.add_argument("--out_name", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"\n=== Entropy-Lens: model={args.model} device={args.device} "
          f"grid={len(GRID)} cells x {len(CONDITIONS)} conditions x {args.trials} trials ===")
    result = run_scratch(args, args.device) if args.model == "scratch" else run_hf(args, args.device)
    save_dir = _HERE / "results"
    save_dir.mkdir(parents=True, exist_ok=True)
    name = args.out_name or args.model
    path = save_dir / f"entropy_{name}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {path}  (elapsed {result['elapsed_sec']:.0f}s)")


if __name__ == "__main__":
    main()
