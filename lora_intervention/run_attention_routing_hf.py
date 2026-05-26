"""
HF-direct attention routing — replicates v3/attention_routing.py's normal mode
without transformer_lens (TL OOMs on Gemma at 23 GB).

Generates a "rounded" stream (K keys × N rounds; intra-round key order
shuffled), runs forward with output_attentions=True, then computes attention
from the generation position to each (round, key) cell per (layer, head).

Output JSON layout mirrors v3/results_vllm/attention_routing/<model>__normal.json
so compute_cis.py and the existing comparison code can read it.

Supports two prompt formats:
  --prompt-format plain   (default)  Interleaved stream, no markers
                                     [round][key]: [value]  shuffled
  --prompt-format block              Block-grouped with [Round j] headers
                                     Used by the Block ↔ L30-L33 locus probe

Usage examples:
  # Gemma replication for §7 (mechanism on Gemma+LoRA)
  python lora_intervention/run_attention_routing_hf.py \\
      --merged_path lora_intervention/checkpoints/gemma_merged \\
      --base google/gemma-3-4b-it \\
      --out_name gemma-3-4b-it-LoRA \\
      --K 2 --N 30 --trials 50

  # Baseline (no adapter)
  python lora_intervention/run_attention_routing_hf.py \\
      --merged_path google/gemma-3-4b-it \\
      --base google/gemma-3-4b-it \\
      --out_name gemma-3-4b-it-baseline-hf \\
      --K 2 --N 30 --trials 50

  # Block-locus probe (for §6 "two recoveries common locus" claim)
  python lora_intervention/run_attention_routing_hf.py \\
      --merged_path Qwen/Qwen2.5-3B-Instruct \\
      --base Qwen/Qwen2.5-3B-Instruct \\
      --out_name Qwen2.5-3B-Instruct-baseline-block \\
      --K 2 --N 30 --trials 50 \\
      --prompt-format block
"""
import argparse
import importlib.metadata as _imeta
import json
import os
import random
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_orig = _imeta.version
def _safe(pkg):
    try:
        v = _orig(pkg); return v if v is not None else "0.0.0"
    except Exception:
        return "0.0.0"
_imeta.version = _safe

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from mechanistic_probing_v2.core.dataset_configs import (
    get_eligible_categories, format_for_chat,
)
from mechanistic_probing_v2.core.model_loader import (
    verify_single_token, is_instruct_model,
)
from mechanistic_probing_v2.core.dataset_configs import get_value_pool

DATASET_TYPE = "ARBITRARY_SINGLE"
SYSTEM_PROMPT = (
    "You are a precise data extraction tool. "
    "Output ONLY a single word - the exact value requested. "
    "No other text, no explanation, no punctuation."
)
BASE_MODEL_DEFAULT = "Qwen/Qwen2.5-3B-Instruct"


def _model_class_for(model_id: str):
    if "gemma-3" in model_id.lower():
        try:
            from transformers import Gemma3ForCausalLM
            return Gemma3ForCausalLM
        except ImportError:
            pass
    return AutoModelForCausalLM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--merged_path", required=True,
                   help="HF id or local path to model (merged adapter or base)")
    p.add_argument("--base", default=BASE_MODEL_DEFAULT,
                   help="HF model id of the base architecture (for chat template detection)")
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--N", type=int, default=30)
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--seed-start", type=int, default=1000,
                   help="First seed (paired with existing Qwen attention_routing if 1000)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--prompt-format", default="plain", choices=["plain", "block"])
    p.add_argument("--out_name", required=True)
    p.add_argument("--out-dir", default=None,
                   help="Override v3/results_vllm/attention_routing/")
    return p.parse_args()


def load_model(merged_path, base_model, device, dtype):
    print(f"Loading model from {merged_path} ({device}, {dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(merged_path, trust_remote_code=True)
    model_cls = _model_class_for(base_model)
    model = model_cls.from_pretrained(
        merged_path, dtype=dtype, low_cpu_mem_usage=True,
        trust_remote_code=True, device_map=device,
        attn_implementation="eager",  # required for output_attentions
    )
    model.eval()
    return model, tokenizer


# ─── Stream construction ──────────────────────────────────────────────────────

def build_rounded_stream(K, N, value_pool, rng):
    """K keys, N rounds. Each round = one pass through K keys (shuffled).
    Returns:
        flat_items   — list of dicts {category, value, round} (length K*N)
        round_groups — list of N lists, each of K items in shuffled key order
        categories   — the K categories chosen
        values_per_cat — dict[cat] = [v_0, v_1, ..., v_{N-1}]
    """
    eligible = get_eligible_categories(DATASET_TYPE, min_values=N)
    if len(eligible) < K:
        raise ValueError(f"Need {K} eligible categories with ≥{N} values; got {len(eligible)}")
    categories = rng.sample(eligible, K)

    # Sample N distinct values per category from the value_pool
    values_per_cat = {}
    used = set()
    for cat in categories:
        chosen = []
        attempts = 0
        while len(chosen) < N and attempts < N * 20:
            attempts += 1
            v = rng.choice(value_pool)
            if v in used:
                continue
            used.add(v)
            chosen.append(v)
        if len(chosen) < N:
            raise ValueError(f"Couldn't sample {N} distinct values for {cat}")
        values_per_cat[cat] = chosen

    round_groups = []
    flat = []
    for r in range(N):
        round_items = [
            {"category": cat, "value": values_per_cat[cat][r], "round": r}
            for cat in categories
        ]
        rng.shuffle(round_items)
        round_groups.append(round_items)
        flat.extend(round_items)
    return flat, round_groups, categories, values_per_cat


def render_prompt_plain(flat_items, test_category, condition):
    """Plain interleaved stream — same format as the main FVQ/CVQ task."""
    stream = "\n".join(f"{it['category']}: {it['value']}" for it in flat_items)
    query_word = "first" if condition == "FVQ" else "last"
    return (
        f"Read the following key-value stream. Each key gets updated multiple times.\n\n"
        f"{stream}\n\n"
        f"What was the {query_word} value of {test_category}?"
    )


def render_prompt_block(round_groups, test_category, condition):
    """Block-grouped with [Round j] headers."""
    lines = []
    for r_idx, items in enumerate(round_groups):
        lines.append(f"[Round {r_idx + 1}]")
        for it in items:
            lines.append(f"  {it['category']}: {it['value']}")
    stream = "\n".join(lines)
    query_word = "first" if condition == "FVQ" else "last"
    return (
        f"Read the following key-value stream. Each key is updated multiple times, "
        f"grouped by update round.\n\n"
        f"{stream}\n\n"
        f"What was the {query_word} value of {test_category}?"
    )


# ─── Token-position → (round, key) mapping ────────────────────────────────────

def build_position_map(prompt_text, flat_items, round_groups, tokenizer, prompt_format):
    """Find which token positions correspond to each (round, key) value.

    Strategy: for each (round_idx, key_idx) → value, locate the value's tokens
    in the tokenized prompt by searching for its leading token id with the
    same surrounding context. We use a simple sequential search:
    - Walk through the items in order
    - For each item, find the next occurrence of its value's first token id
    - Record (token_idx_start, token_idx_end) for that item's value tokens
    """
    enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    token_ids = enc.input_ids[0].tolist()
    L = len(token_ids)

    # We'll iterate flat_items in *appearance order* in the prompt
    if prompt_format == "plain":
        items_in_order = flat_items
    else:  # block
        items_in_order = []
        for round_items in round_groups:
            for it in round_items:
                items_in_order.append(it)

    # For each item, get candidate token sequence (value with leading space)
    # and find its position in token_ids.
    pos = 0  # current scan position
    pos_map = {}  # token_idx → (round_idx, key_idx)
    # Map category → key_idx (stable across rounds)
    if prompt_format == "plain":
        # In plain, key index is per-category order of appearance
        cat_to_kidx = {}
        for it in items_in_order:
            cat = it["category"]
            if cat not in cat_to_kidx:
                cat_to_kidx[cat] = len(cat_to_kidx)
    else:
        # In block, key index is per-category-order within first round
        cat_to_kidx = {}
        for it in round_groups[0]:
            cat = it["category"]
            cat_to_kidx[cat] = len(cat_to_kidx)
        # any new category appearing in later rounds gets next index
        for items in round_groups[1:]:
            for it in items:
                if it["category"] not in cat_to_kidx:
                    cat_to_kidx[it["category"]] = len(cat_to_kidx)

    for it in items_in_order:
        value = it["value"]
        round_idx = it["round"]
        key_idx = cat_to_kidx[it["category"]]
        # try both " value" and "value" tokenizations to find the first token
        candidates = []
        for v_str in (" " + value, value):
            ids = tokenizer.encode(v_str, add_special_tokens=False)
            if ids:
                candidates.append(ids)
        # Find first hit at or after `pos`
        found_at = None
        found_len = None
        for ids in candidates:
            for i in range(pos, L - len(ids) + 1):
                if token_ids[i:i+len(ids)] == ids:
                    if found_at is None or i < found_at:
                        found_at = i
                        found_len = len(ids)
            if found_at is not None:
                break
        if found_at is None:
            # Fallback: just record nothing for this item — its attention will be 0
            continue
        for j in range(found_at, found_at + found_len):
            pos_map[j] = (round_idx, key_idx)
        pos = found_at + found_len

    return enc, pos_map


def detect_gen_position(prompt_text_with_assistant, tokenizer, base_model_name):
    """The last token of the formatted (chat-templated) prompt — the model
    will attend FROM this position when producing the answer. We extract
    attention at this index (i.e. the final token of the prompt).
    """
    enc = tokenizer(prompt_text_with_assistant, return_tensors="pt", add_special_tokens=True)
    return enc, enc.input_ids.shape[1] - 1


# ─── Per-trial attention extraction ───────────────────────────────────────────

@torch.no_grad()
def attn_at_gen_per_round(model, tokenizer, base_model_name, K, N, trial_seed,
                          prompt_format, value_pool, device):
    """Run one trial. Returns dict matching v3 attention_routing schema."""
    rng = random.Random(trial_seed)
    flat_items, round_groups, categories, values_per_cat = build_rounded_stream(
        K, N, value_pool, rng,
    )

    # FVQ and CVQ trials use the same stream, only the query word changes
    out_trial = {
        "seed": trial_seed,
        "test_category": None,
        "all_values": None,
        "expA_FVQ": None,
        "expA_CVQ": None,
    }

    # Pick a test category (same as v3 attention_routing: by seed index)
    test_cat_idx = trial_seed % K
    test_category = categories[test_cat_idx]
    out_trial["test_category"] = test_category
    out_trial["all_values"] = values_per_cat[test_category]

    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_heads = getattr(cfg, "num_attention_heads", cfg.num_attention_heads)

    for condition in ["FVQ", "CVQ"]:
        if prompt_format == "block":
            raw_user = render_prompt_block(round_groups, test_category, condition)
        else:
            raw_user = render_prompt_plain(flat_items, test_category, condition)

        use_chat = is_instruct_model(base_model_name) if base_model_name else True
        if use_chat:
            formatted = format_for_chat(raw_user, tokenizer, model_name=base_model_name)
        else:
            formatted = raw_user

        # Build the position map using the UNTEMPLATED stream — we need to know
        # where each (round, key) value lands in the chat-templated prompt.
        # Solution: tokenize the chat-templated prompt and find values by token id.
        enc = tokenizer(formatted, return_tensors="pt", add_special_tokens=True).to(device)
        token_ids = enc.input_ids[0].tolist()
        seq_len = enc.input_ids.shape[1]
        gen_pos = seq_len - 1  # last token of prompt

        # Build pos_map by searching for each value in the templated token stream.
        # Order through items in appearance order.
        if prompt_format == "plain":
            items_in_order = flat_items
        else:
            items_in_order = [it for grp in round_groups for it in grp]

        cat_to_kidx = {cat: idx for idx, cat in enumerate(categories)}

        pos_map = {}  # token_idx → (round_idx, key_idx)
        scan_from = 0
        for it in items_in_order:
            value = it["value"]
            round_idx = it["round"]
            key_idx = cat_to_kidx[it["category"]]
            found_at = None
            found_len = None
            for v_str in (" " + value, value):
                ids = tokenizer.encode(v_str, add_special_tokens=False)
                if not ids:
                    continue
                for i in range(scan_from, seq_len - len(ids) + 1):
                    if token_ids[i:i+len(ids)] == ids:
                        found_at = i
                        found_len = len(ids)
                        break
                if found_at is not None:
                    break
            if found_at is None:
                continue
            for j in range(found_at, found_at + found_len):
                pos_map[j] = (round_idx, key_idx)
            scan_from = found_at + found_len

        # Forward pass with output_attentions
        out = model(**enc, output_attentions=True, return_dict=True, use_cache=False)
        # out.attentions: tuple of n_layers, each [1, n_heads, T, T]
        attn_from_gen = np.zeros((n_layers, n_heads, N, K), dtype=np.float64)
        for L_idx, attn_L in enumerate(out.attentions):
            # row from generation position
            row = attn_L[0, :, gen_pos, :].float().cpu().numpy()  # [H, T]
            n_heads_actual = row.shape[0]
            for tok_idx, (r, k) in pos_map.items():
                attn_from_gen[L_idx, :n_heads_actual, r, k] += row[:, tok_idx]

        del out, enc
        if device == "cuda":
            torch.cuda.empty_cache()

        out_trial[f"expA_{condition}"] = {
            "n_layers": int(n_layers),
            "n_heads": int(n_heads),
            "n_rounds": int(N),
            "seq_len": int(seq_len),
            "attn_from_gen": attn_from_gen.tolist(),
        }

    return out_trial


def main():
    args = parse_args()
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}[args.dtype]

    print("=" * 60)
    print(f"Attention routing — {args.out_name}")
    print(f"  Base:        {args.base}")
    print(f"  Model path:  {args.merged_path}")
    print(f"  K={args.K}, N={args.N}, trials={args.trials}, format={args.prompt_format}")
    print("=" * 60)

    model, tokenizer = load_model(args.merged_path, args.base, args.device, dtype)
    n_layers = model.config.num_hidden_layers
    n_heads = getattr(model.config, "num_attention_heads", "?")
    print(f"  n_layers={n_layers}, n_heads={n_heads}")

    candidate_pool = get_value_pool(DATASET_TYPE)
    value_to_tid = verify_single_token(tokenizer, values=candidate_pool)
    value_pool = list(value_to_tid.keys())
    print(f"  value pool: {len(value_pool)} verified single-token values")

    trials = []
    t0 = time.time()
    for i in range(args.trials):
        seed = args.seed_start + i
        try:
            tr = attn_at_gen_per_round(
                model, tokenizer, args.base, args.K, args.N, seed,
                args.prompt_format, value_pool, args.device,
            )
            trials.append(tr)
        except Exception as e:
            print(f"  trial {i} (seed={seed}) FAILED: {e}")
            continue
        if (i + 1) % 10 == 0 or i + 1 == args.trials:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{args.trials}] elapsed={elapsed:.0f}s")

    out_dir = (Path(args.out_dir) if args.out_dir
               else _ROOT / "v3" / "results_vllm" / "attention_routing")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name_with_mode = f"{args.out_name}__{args.prompt_format}.json" if args.prompt_format == "block" else f"{args.out_name}__normal.json"
    out_path = out_dir / out_name_with_mode
    output = {
        "model": args.out_name,
        "hf_id": args.merged_path,
        "mode": "normal" if args.prompt_format == "plain" else "block",
        "prompt_format": args.prompt_format,
        "K": args.K, "N": args.N,
        "n_trials_completed": len(trials),
        "n_trials_target": args.trials,
        "save_per_head": True,
        "device": args.device,
        "dtype": str(dtype),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trials": trials,
    }
    with open(out_path, "w") as f:
        json.dump(output, f)
    print(f"\n✓ Saved {len(trials)} trials → {out_path}")


if __name__ == "__main__":
    main()
