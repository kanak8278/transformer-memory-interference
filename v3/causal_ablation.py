"""
Causal ablation of the 11 query-discriminating heads in Qwen2.5-3B-Instruct.

Identified from `v3/extract_query_heads.py` cross-config consistency analysis:
heads that show FVQ-CVQ attention difference > 0.25 (gen-token → round 0) in
the normal config AND appear in all 4 Qwen consistency configs.

Heads to ablate (layer, head):
  L31 H15, L31 H7, L31 H8, L31 H12, L30 H11,
  L29 H1, L29 H4, L29 H5, L27 H1, L32 H3, L32 H7

Implementation: register a forward_pre_hook on each affected layer's o_proj.
The input to o_proj is the concatenation of all per-head outputs reshaped to
[batch, seq, num_heads * head_dim]. We reshape to [batch, seq, num_heads,
head_dim], zero out the target heads, and reshape back.

Compares baseline (no ablation) vs ablated FVQ + CVQ accuracy on the same trials.

Usage:
    .venv/bin/python v3/causal_ablation.py --trials 100 --K 2 --N 30
"""

import sys
import json
import random
import argparse
import time
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_SCRIPT_DIR = Path(__file__).resolve().parent
_V3_SCRIPTS = _SCRIPT_DIR / "scripts"
_PROJECT_ROOT = _SCRIPT_DIR.parent
for p in (_PROJECT_ROOT, _V3_SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from mechanistic_probing_v2.core.dataset_configs import get_eligible_categories  # noqa: E402
from mechanistic_probing_v2.core.model_loader import verify_single_token  # noqa: E402
from experiments.stage1_sweep import SYSTEM_PROMPT, shuffle_no_consecutive  # noqa: E402


# ═════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY — per-model configuration for ablation
#   - hf_id:           HuggingFace id to load
#   - num_heads:       number of query heads (for reshaping o_proj input)
#   - head_dim:        dim of each query head
#   - attn_input_dim:  in_features of o_proj  ==  num_heads * head_dim
#                      (may differ from hidden_size for some architectures)
#   - layers_path:     dotted path to ModuleList of decoder layers
#   - heads:           list of (layer, head_idx) — discriminating heads to ablate
# ═════════════════════════════════════════════════════════════════════════════

MODEL_CONFIGS = {
    # Qwen2.5-3B-Instruct: 36 layers × 16 query heads × 128 head_dim.
    # Heads identified from cross-config consistency analysis
    # (extract_query_heads.py top-15 across 4 Qwen configs)
    "Qwen2.5-3B-Instruct": {
        "hf_id": "Qwen/Qwen2.5-3B-Instruct",
        "num_heads": 16,
        "head_dim": 128,
        "attn_input_dim": 16 * 128,
        "layers_path": "model.layers",
        "heads": [
            (31, 15), (31, 7), (31, 8), (31, 12),
            (30, 11),
            (29, 1), (29, 4), (29, 5),
            (27, 1),
            (32, 3), (32, 7),
        ],
        "default_K": 2,
        "default_N": 30,
    },

    # gemma-3-4b-it: 34 layers × 8 query heads × 256 head_dim. hidden_size=2560.
    # o_proj weight is [2560, 2048] → in_features = 8 * 256 = 2048 (not hidden_size).
    # Heads identified from `gen → primacy_round0` FVQ-dominant with |FVQ-CVQ| > 0.1
    "gemma-3-4b-it": {
        "hf_id": "google/gemma-3-4b-it",
        "num_heads": 8,
        "head_dim": 256,
        "attn_input_dim": 8 * 256,
        "layers_path": "model.language_model.layers",
        "heads": [
            (23, 3), (23, 1), (23, 0), (23, 6), (23, 7),
            (17, 5), (17, 0), (17, 2),
            (29, 4), (29, 3),
        ],
        "default_K": 7,
        "default_N": 30,
    },
}

RESULTS_PATH = _SCRIPT_DIR / "results_vllm" / "causal_ablation"


# ═════════════════════════════════════════════════════════════════════════════
# ABLATION HOOK
# ═════════════════════════════════════════════════════════════════════════════

def make_ablation_hook(heads_to_zero: list[int], num_heads: int, head_dim: int, attn_input_dim: int):
    """Forward pre-hook for o_proj. Zeros out specific query heads' outputs
    before the output projection.

    Input to o_proj has shape [batch, seq, attn_input_dim = num_heads * head_dim].
    Note: attn_input_dim may differ from residual hidden_size (e.g., Gemma).
    """
    heads_to_zero_t = torch.tensor(heads_to_zero, dtype=torch.long)

    def hook(module, args):
        hidden = args[0]
        b, s, h = hidden.shape
        assert h == attn_input_dim, f"Expected o_proj in dim {attn_input_dim}, got {h}"
        reshaped = hidden.view(b, s, num_heads, head_dim).clone()
        reshaped[:, :, heads_to_zero_t, :] = 0
        return (reshaped.view(b, s, h),) + args[1:]
    return hook


def get_layers(model, layers_path: str):
    """Walk dotted path to the decoder layers ModuleList."""
    obj = model
    for part in layers_path.split("."):
        obj = getattr(obj, part)
    return obj


def install_ablation_hooks(model, heads_to_ablate, model_cfg):
    """Install forward_pre_hooks on relevant layers' o_proj. Returns handle list."""
    heads_by_layer = defaultdict(list)
    for L, H in heads_to_ablate:
        heads_by_layer[L].append(H)

    layers = get_layers(model, model_cfg["layers_path"])
    handles = []
    for L, heads in heads_by_layer.items():
        o_proj = layers[L].self_attn.o_proj
        h = o_proj.register_forward_pre_hook(
            make_ablation_hook(heads, model_cfg["num_heads"],
                               model_cfg["head_dim"], model_cfg["attn_input_dim"])
        )
        handles.append(h)
    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


# ═════════════════════════════════════════════════════════════════════════════
# TRIAL GENERATION (matches stage1_sweep convention)
# ═════════════════════════════════════════════════════════════════════════════

def build_trial(K: int, N: int, seed: int, filtered_values: list[str]):
    """Generate a trial. Returns dict with prompt-building inputs + expected answers."""
    rng = random.Random(seed)
    eligible = get_eligible_categories("ARBITRARY_SINGLE", min_values=N)
    cats = rng.sample(eligible, min(K, len(eligible)))
    test_category = cats[seed % len(cats)]

    total_needed = K * N
    sampled = rng.sample(filtered_values, total_needed)
    values_per_cat = {cat: sampled[i * N : (i + 1) * N] for i, cat in enumerate(cats)}

    items = []
    for cat in cats:
        for val in values_per_cat[cat]:
            items.append({"category": cat, "value": val})
    items = shuffle_no_consecutive(items, rng)

    cat_values = [it["value"] for it in items if it["category"] == test_category]
    return {
        "items": items,
        "test_category": test_category,
        "expected_FVQ": cat_values[0],
        "expected_CVQ": cat_values[-1],
        "all_values": cat_values,
    }


def render_prompt(trial: dict, query_word: str) -> str:
    """Return the full chat-template prompt for FVQ ('first') or CVQ ('last')."""
    stream = "\n".join(f"{it['category']}: {it['value']}" for it in trial["items"])
    raw_prompt = (
        "Read the following key-value stream. Each key gets updated multiple times.\n\n"
        f"{stream}\n\n"
        f"What was the {query_word} value of {trial['test_category']}?"
    )
    return raw_prompt


# ═════════════════════════════════════════════════════════════════════════════
# GENERATION + EVAL
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_answer(model, tokenizer, raw_prompt: str, device: str, max_new_tokens: int = 10) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": raw_prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(device)
    out = model.generate(
        inputs, max_new_tokens=max_new_tokens,
        do_sample=False, temperature=1.0, top_p=1.0,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    new_tokens = out[0, inputs.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def classify(prediction: str, expected: str, all_values: list[str]) -> dict:
    """Simple classification: did the answer match expected? where did it land?"""
    p = prediction.lower().strip()
    e = expected.lower()
    correct = (e in p) or p.startswith(e)
    # Which round did the prediction land on?
    matched_idx = None
    for i, v in enumerate(all_values):
        vlow = v.lower()
        if vlow in p or p.startswith(vlow):
            matched_idx = i
            break
    return {
        "correct": correct,
        "matched_round_idx": matched_idx,
        "relative_pos": matched_idx / max(len(all_values) - 1, 1) if matched_idx is not None else None,
    }


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_CONFIGS.keys()),
                    default="Qwen2.5-3B-Instruct")
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--K", type=int, default=None,
                    help="Override default K from model config")
    ap.add_argument("--N", type=int, default=None,
                    help="Override default N from model config")
    ap.add_argument("--out-dir", type=str, default=str(RESULTS_PATH))
    ap.add_argument("--control-random", action="store_true",
                    help="Ablate random non-discriminating heads instead of identified ones. "
                         "Matches original layer distribution.")
    ap.add_argument("--control-seed", type=int, default=42,
                    help="Seed for random head selection (control mode)")
    args = ap.parse_args()

    model_cfg = MODEL_CONFIGS[args.model]
    model_short = args.model
    hf_id = model_cfg["hf_id"]
    original_heads = model_cfg["heads"]
    K = args.K if args.K is not None else model_cfg["default_K"]
    N = args.N if args.N is not None else model_cfg["default_N"]

    if args.control_random:
        # Pick random non-identified heads. Try to match the original layer
        # distribution first; spill into nearest adjacent layers when a layer
        # has too few non-identified heads available (e.g., Gemma L23 has 5/8
        # identified, only 3 left to draw from).
        rng = random.Random(args.control_seed)
        per_layer_original = defaultdict(list)
        for L, H in original_heads:
            per_layer_original[L].append(H)

        identified_set = set(original_heads)
        used = set()
        heads_to_ablate = []
        num_heads = model_cfg["num_heads"]

        # Determine total layer count from the model config
        total_layers = sum(1 for _ in get_layers(model, model_cfg["layers_path"])) if False else None
        # Can't access model yet, so infer max layer from heads + buffer
        total_layers = 64  # over-estimate; we'll cap by skipping invalid layers later

        for L, original_H_list in per_layer_original.items():
            n_pick = len(original_H_list)
            # Same-layer first
            available_same = [h for h in range(num_heads)
                              if (L, h) not in identified_set and (L, h) not in used]
            n_from_same = min(n_pick, len(available_same))
            picked = rng.sample(available_same, n_from_same)
            heads_to_ablate.extend((L, h) for h in picked)
            used.update((L, h) for h in picked)
            remaining = n_pick - n_from_same
            # Spillover to adjacent layers
            for delta in [-1, 1, -2, 2, -3, 3, -4, 4, -5, 5]:
                if remaining == 0:
                    break
                L_alt = L + delta
                if L_alt < 0:
                    continue
                avail_alt = [h for h in range(num_heads)
                             if (L_alt, h) not in identified_set
                             and (L_alt, h) not in used]
                if not avail_alt:
                    continue
                take = min(remaining, len(avail_alt))
                picked = rng.sample(avail_alt, take)
                heads_to_ablate.extend((L_alt, h) for h in picked)
                used.update((L_alt, h) for h in picked)
                remaining -= take
            if remaining > 0:
                print(f"  WARN: layer {L} control short by {remaining} heads — exhausted nearby")

        print(f"  CONTROL MODE: ablating {len(heads_to_ablate)} RANDOM non-identified heads")
        print(f"  (seed={args.control_seed})")
        print(f"  picked: {sorted(heads_to_ablate)}")
    else:
        heads_to_ablate = original_heads

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device != "cpu" else torch.float32

    print(f"Loading {hf_id} on {device}…")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, dtype=dtype, device_map=device,
        trust_remote_code=True, attn_implementation="eager",
    )
    model.eval()
    print(f"  loaded ({time.time() - t0:.1f}s)")

    verified = verify_single_token(tokenizer)
    filtered_values = list(verified.keys())
    print(f"  single-token values: {len(filtered_values)}")
    print(f"  heads to ablate ({len(heads_to_ablate)}): {heads_to_ablate}")
    print(f"  K={K}, N={N}, trials={args.trials}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    t_loop = time.time()

    for trial_idx in range(args.trials):
        seed = 1000 + trial_idx
        trial = build_trial(K, N, seed, filtered_values)

        trial_record = {
            "seed": seed,
            "test_category": trial["test_category"],
            "expected_FVQ": trial["expected_FVQ"],
            "expected_CVQ": trial["expected_CVQ"],
            "all_values": trial["all_values"],
            "baseline": {},
            "ablated": {},
        }

        # ─── Baseline (no ablation) ───
        for qword, qlabel in [("first", "FVQ"), ("last", "CVQ")]:
            prompt = render_prompt(trial, qword)
            pred = generate_answer(model, tokenizer, prompt, device)
            cls = classify(pred, trial[f"expected_{qlabel}"], trial["all_values"])
            trial_record["baseline"][qlabel] = {"prediction": pred, **cls}

        # ─── With ablation ───
        handles = install_ablation_hooks(model, heads_to_ablate, model_cfg)
        try:
            for qword, qlabel in [("first", "FVQ"), ("last", "CVQ")]:
                prompt = render_prompt(trial, qword)
                pred = generate_answer(model, tokenizer, prompt, device)
                cls = classify(pred, trial[f"expected_{qlabel}"], trial["all_values"])
                trial_record["ablated"][qlabel] = {"prediction": pred, **cls}
        finally:
            remove_hooks(handles)

        results.append(trial_record)

        # Progress
        n_done = trial_idx + 1
        elapsed = time.time() - t_loop
        rate = n_done / elapsed
        eta = (args.trials - n_done) / max(rate, 1e-6)
        # Running aggregate
        baseline_FVQ = sum(r["baseline"]["FVQ"]["correct"] for r in results) / n_done
        baseline_CVQ = sum(r["baseline"]["CVQ"]["correct"] for r in results) / n_done
        ablated_FVQ = sum(r["ablated"]["FVQ"]["correct"] for r in results) / n_done
        ablated_CVQ = sum(r["ablated"]["CVQ"]["correct"] for r in results) / n_done

        print(
            f"  [{n_done}/{args.trials}] {elapsed:.0f}s, ETA {eta:.0f}s | "
            f"baseline: FVQ={baseline_FVQ:.2f} CVQ={baseline_CVQ:.2f} | "
            f"ablated: FVQ={ablated_FVQ:.2f} CVQ={ablated_CVQ:.2f}",
            flush=True,
        )

        # Save every 10 trials
        suffix = "__control" if args.control_random else ""
        if n_done % 10 == 0 or n_done == args.trials:
            with open(out_dir / f"ablation__{model_short}__K{K}N{N}{suffix}.json", "w") as f:
                json.dump({
                    "model": model_short,
                    "hf_id": hf_id,
                    "K": K,
                    "N": N,
                    "heads_ablated": heads_to_ablate,
                    "n_trials": n_done,
                    "device": device,
                    "dtype": str(dtype),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "trials": results,
                    "summary": {
                        "baseline_FVQ_acc": baseline_FVQ,
                        "baseline_CVQ_acc": baseline_CVQ,
                        "ablated_FVQ_acc": ablated_FVQ,
                        "ablated_CVQ_acc": ablated_CVQ,
                        "delta_FVQ": ablated_FVQ - baseline_FVQ,
                        "delta_CVQ": ablated_CVQ - baseline_CVQ,
                    },
                }, f)

    print(f"\n✓ Done. Saved → {out_dir / f'ablation__{model_short}__K{K}N{N}.json'}")
    print(f"\nFinal:")
    print(f"  baseline: FVQ={baseline_FVQ:.2f}  CVQ={baseline_CVQ:.2f}  gap=+{baseline_FVQ - baseline_CVQ:.2f}")
    print(f"  ablated:  FVQ={ablated_FVQ:.2f}  CVQ={ablated_CVQ:.2f}  gap=+{ablated_FVQ - ablated_CVQ:.2f}")
    print(f"  delta:    FVQ={ablated_FVQ - baseline_FVQ:+.2f}  CVQ={ablated_CVQ - baseline_CVQ:+.2f}")


if __name__ == "__main__":
    main()
