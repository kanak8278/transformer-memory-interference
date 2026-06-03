"""
Attention routing experiments (Exp A / B / C).

Same forward passes drive all three experiments — we save raw per-head, per-layer
attention extracted at two source positions (generation token and query-key token)
so plot/analysis decisions can be made later without re-running.

Exp A — full sequence, all layers, both FVQ and CVQ.
       Source: generation token. Reveals at which depth the model routes to
       which round.

Exp B — incremental N (1..opN), final layer, CVQ only.
       Source: generation token. Tracks whether attention shifts toward the
       newest round as updates accumulate.

Exp C — incremental N (1..opN), final layer, both FVQ and CVQ.
       Source: query-key (test_category token in the question). Tracks the
       "composition" of the key as more values accumulate.

Output: one JSON per (model, mode) under v3/results_vllm/attention_routing/.

Usage:
    .venv/bin/python v3/attention_routing.py --model Qwen2.5-3B-Instruct --mode normal --trials 50
    .venv/bin/python v3/attention_routing.py --smoke                          # 2 trials, all configs
"""

import sys
import json
import time
import random
import argparse
from pathlib import Path
from datetime import datetime, timezone

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

_SCRIPT_DIR = Path(__file__).resolve().parent
_V3_SCRIPTS = _SCRIPT_DIR / "scripts"
_PROJECT_ROOT = _SCRIPT_DIR.parent
for p in (_PROJECT_ROOT, _V3_SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from mechanistic_probing_v2.core.dataset_configs import (
    get_eligible_categories,
    generate_values_for_trial,
)
from mechanistic_probing_v2.core.model_loader import verify_single_token

# Match stage1_sweep convention
from experiments.stage1_sweep import (  # noqa: E402
    SYSTEM_PROMPT,
    shuffle_no_consecutive,
)


# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════

OPERATING_POINTS = {
    "Qwen2.5-3B-Instruct": {
        "hf_id": "Qwen/Qwen2.5-3B-Instruct",
        "modes": {
            "normal":   {"K": 2,  "N": 30, "fvq": 0.86, "cvq": 0.41},
            "reversal": {"K": 15, "N": 20, "fvq": 0.32, "cvq": 0.58},
        },
    },
    "Qwen3.5-4B": {
        "hf_id": "Qwen/Qwen3.5-4B",
        "modes": {
            "normal":   {"K": 7,  "N": 10, "fvq": 0.90, "cvq": 0.60},
            "reversal": {"K": 10, "N": 50, "fvq": 0.43, "cvq": 0.65},
        },
    },
    "gemma-3-4b-it": {
        "hf_id": "google/gemma-3-4b-it",
        "modes": {
            "normal": {"K": 7, "N": 30, "fvq": 0.95, "cvq": 0.52},
        },
    },
}

DATASET_TYPE = "ARBITRARY_SINGLE"
DEFAULT_TRIALS = 50
RESULTS_ROOT = _SCRIPT_DIR / "results_vllm" / "attention_routing"


def checkpoint_steps(opN: int) -> list[int]:
    """Subsample of n_updates for Exp B/C — last is always opN."""
    base = [1, 2, 3, 5, 10, 20]
    base = [n for n in base if n < opN]
    return base + [opN]


# ═════════════════════════════════════════════════════════════════════════════
# PROMPT CONSTRUCTION WITH POSITION TRACKING
# ═════════════════════════════════════════════════════════════════════════════

def build_trial(
    *,
    K: int,
    N: int,
    seed: int,
    filtered_values: list[str],
    rng: random.Random,
) -> dict:
    """Generate items + test_category for a trial. No tokenization yet."""
    eligible = get_eligible_categories(DATASET_TYPE, min_values=N)
    cats_for_run = rng.sample(eligible, min(K, len(eligible)))
    test_category = cats_for_run[seed % len(cats_for_run)]

    # ARBITRARY_SINGLE pool is shared — but we need our model-filtered subset.
    # Re-implement generate_values_for_trial using filtered_values (deterministic on rng).
    total_needed = K * N
    if total_needed > len(filtered_values):
        raise ValueError(
            f"Need {total_needed} filtered single-token values, have {len(filtered_values)}"
        )
    sampled = rng.sample(filtered_values, total_needed)
    values_per_cat = {
        cat: sampled[i * N : (i + 1) * N]
        for i, cat in enumerate(cats_for_run)
    }

    items = []
    for cat in cats_for_run:
        for v in values_per_cat[cat]:
            items.append({"category": cat, "value": v})
    items = shuffle_no_consecutive(items, rng)

    return {
        "items": items,
        "test_category": test_category,
        "categories": cats_for_run,
    }


def render_prompt(
    trial: dict,
    n_updates_to_include: int | None,
    query_word: str,  # "first" or "last"
) -> tuple[str, list[dict], dict]:
    """Render the raw_prompt as a string while tracking char offsets of
    each test_category and value occurrence.

    n_updates_to_include: if not None, truncate so the test_category appears
    exactly n_updates_to_include times. Other categories included alongside,
    interleaved order preserved.

    Returns:
        raw_prompt: str
        stream_occurrences: list of dicts with char spans for each test_category
                            and its value, in stream order
        query_span: dict with char span for the test_category in the query
    """
    items = trial["items"]
    test_cat = trial["test_category"]

    if n_updates_to_include is not None:
        # Keep only first n_updates_to_include test_category occurrences,
        # plus all other-category items that appear before the n-th occurrence.
        keep_items = []
        seen_test = 0
        for it in items:
            if it["category"] == test_cat:
                if seen_test < n_updates_to_include:
                    keep_items.append(it)
                    seen_test += 1
                else:
                    # Stop — drop further test items and trailing others
                    break
            else:
                keep_items.append(it)
        items_active = keep_items
    else:
        items_active = items

    preamble = "Read the following key-value stream. Each key gets updated multiple times.\n\n"
    stream_start = len(preamble)

    raw_chunks = [preamble]
    stream_occurrences = []  # for test_category occurrences only
    cursor = stream_start

    for idx, it in enumerate(items_active):
        line = f"{it['category']}: {it['value']}"
        if it["category"] == test_cat:
            cat_start = cursor
            cat_end = cat_start + len(it["category"])
            val_start = cat_end + len(": ")
            val_end = val_start + len(it["value"])
            stream_occurrences.append({
                "value": it["value"],
                "cat_char_span": (cat_start, cat_end),
                "val_char_span": (val_start, val_end),
            })
        raw_chunks.append(line)
        cursor += len(line)
        if idx < len(items_active) - 1:
            raw_chunks.append("\n")
            cursor += 1

    raw_chunks.append("\n\n")
    cursor += 2

    query_prefix = f"What was the {query_word} value of "
    query_q_start = cursor + len(query_prefix)
    query_q_end = query_q_start + len(test_cat)
    query_span = {"cat_char_span": (query_q_start, query_q_end)}

    raw_chunks.append(f"{query_prefix}{test_cat}?")

    raw_prompt = "".join(raw_chunks)
    return raw_prompt, stream_occurrences, query_span


def chat_format(raw_prompt: str, tokenizer, model_name: str) -> tuple[str, int]:
    """Wrap with chat template. Returns (full_text, char_offset_of_raw_prompt)."""
    is_instruct = "instruct" in model_name.lower() or "-it" in model_name.lower()
    if not is_instruct or not hasattr(tokenizer, "apply_chat_template"):
        return raw_prompt, 0

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": raw_prompt},
    ]
    try:
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Some tokenizers reject system role — fall back to single user message
        messages = [{"role": "user", "content": SYSTEM_PROMPT + "\n\n" + raw_prompt}]
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    char_offset = full_text.find(raw_prompt)
    if char_offset < 0:
        # Chat template modified content (e.g., escaped chars). Try once more
        # by encoding without system message:
        messages = [{"role": "user", "content": raw_prompt}]
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        char_offset = full_text.find(raw_prompt)
        if char_offset < 0:
            raise RuntimeError(
                "Cannot locate raw_prompt inside chat-templated text — "
                "template is modifying content"
            )
    return full_text, char_offset


def char_span_to_token_span(
    char_start: int, char_end: int, offsets: list[tuple[int, int]]
) -> tuple[int, int]:
    """Map [char_start, char_end) → [tok_start, tok_end). Half-open both sides."""
    tok_start = None
    tok_end = None
    for i, (s, e) in enumerate(offsets):
        if s == 0 and e == 0:  # special token slot
            continue
        if tok_start is None and e > char_start:
            tok_start = i
        if s >= char_end and tok_end is None:
            tok_end = i
            break
    if tok_start is None:
        raise RuntimeError(f"Cannot map char_start={char_start}")
    if tok_end is None:
        tok_end = len(offsets)
    return tok_start, tok_end


# ═════════════════════════════════════════════════════════════════════════════
# FORWARD PASS + ATTENTION EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def forward_and_extract(
    model,
    tokenizer,
    full_text: str,
    raw_char_offset: int,
    stream_occurrences: list[dict],
    query_span: dict,
    device: str,
    keep_all_layers: bool,
) -> dict:
    """One forward pass, extract attention from gen_pos AND query_cat_pos.

    keep_all_layers: True for Exp A; False saves only the final layer.

    Returns:
        n_layers, n_heads, n_rounds, seq_len
        round_token_starts, round_token_ends (key spans), round_value_positions
        gen_pos, query_cat_pos (last subtoken of query test_category)
        attn_from_gen:   [n_layers_kept, n_heads, n_rounds, 2] — [:,:,:,0] = sum over key span, [:,:,:,1] = value pos attn
        attn_from_qcat:  [n_layers_kept, n_heads, n_rounds, 2] — same
    """
    enc = tokenizer(
        full_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        return_tensors="pt",
    )
    offsets = enc["offset_mapping"][0].tolist()
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)

    # Map char spans → token spans (shifted by raw_char_offset)
    round_key_spans = []
    round_value_pos = []
    for occ in stream_occurrences:
        cs, ce = occ["cat_char_span"]
        vs, ve = occ["val_char_span"]
        k_ts, k_te = char_span_to_token_span(cs + raw_char_offset, ce + raw_char_offset, offsets)
        v_ts, v_te = char_span_to_token_span(vs + raw_char_offset, ve + raw_char_offset, offsets)
        # Value is single-token by construction → use v_ts; assert v_te - v_ts == 1 in smoke
        round_key_spans.append((k_ts, k_te))
        round_value_pos.append(v_ts)

    qcs, qce = query_span["cat_char_span"]
    q_ts, q_te = char_span_to_token_span(qcs + raw_char_offset, qce + raw_char_offset, offsets)
    query_cat_last_pos = q_te - 1  # last subtoken of test_category in query

    gen_pos = input_ids.shape[1] - 1  # last prompt token

    out = model(
        input_ids=input_ids,
        attention_mask=attn_mask,
        output_attentions=True,
        use_cache=False,
    )
    attentions = out.attentions  # tuple of [batch, n_heads, q_len, k_len]
    n_layers = len(attentions)
    n_heads = attentions[0].shape[1]
    seq_len = input_ids.shape[1]
    n_rounds = len(stream_occurrences)

    layers_to_keep = range(n_layers) if keep_all_layers else [n_layers - 1]

    attn_from_gen = np.zeros((len(layers_to_keep), n_heads, n_rounds, 2), dtype=np.float32)
    attn_from_qcat = np.zeros((len(layers_to_keep), n_heads, n_rounds, 2), dtype=np.float32)

    for li, L in enumerate(layers_to_keep):
        # [n_heads, k_len] for query position = gen_pos
        a_gen = attentions[L][0, :, gen_pos, :].float().cpu().numpy()
        a_qcat = attentions[L][0, :, query_cat_last_pos, :].float().cpu().numpy()

        for k, ((ks, ke), vp) in enumerate(zip(round_key_spans, round_value_pos)):
            attn_from_gen[li, :, k, 0] = a_gen[:, ks:ke].sum(axis=-1)
            attn_from_gen[li, :, k, 1] = a_gen[:, vp]
            attn_from_qcat[li, :, k, 0] = a_qcat[:, ks:ke].sum(axis=-1)
            attn_from_qcat[li, :, k, 1] = a_qcat[:, vp]

    # Also save total attention going to anywhere outside the rounds (sanity check normalization)
    return {
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_rounds": n_rounds,
        "seq_len": seq_len,
        "gen_pos": gen_pos,
        "query_cat_pos": query_cat_last_pos,
        "round_key_spans": round_key_spans,
        "round_value_pos": round_value_pos,
        "attn_from_gen": attn_from_gen,
        "attn_from_qcat": attn_from_qcat,
    }


# ═════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═════════════════════════════════════════════════════════════════════════════

def device_pick() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_one_config(
    *,
    model_short: str,
    mode: str,
    n_trials: int,
    out_dir: Path,
    save_per_head: bool = True,
    override_K: int | None = None,
    override_N: int | None = None,
):
    cfg = OPERATING_POINTS[model_short]
    hf_id = cfg["hf_id"]
    if override_K is not None and override_N is not None:
        K, N = override_K, override_N
    else:
        K = cfg["modes"][mode]["K"]
        N = cfg["modes"][mode]["N"]

    print(f"\n══ {model_short} / {mode}  (K={K}, N={N}, trials={n_trials}) ══", flush=True)

    device = device_pick()
    print(f"  device: {device}", flush=True)

    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32
    print(f"  loading {hf_id}…", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        dtype=dtype,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager",  # output_attentions requires eager
    )
    model.eval()
    print(f"  loaded ({(time.time() - t0):.1f}s)", flush=True)

    # Filter values to single-token per this tokenizer
    verified = verify_single_token(tokenizer)
    filtered_values = list(verified.keys())
    print(f"  single-token values: {len(filtered_values)}", flush=True)

    checkpoints = checkpoint_steps(N)
    print(f"  Exp B/C checkpoints: {checkpoints}", flush=True)

    trial_results = []
    out_dir.mkdir(parents=True, exist_ok=True)
    if override_K is not None:
        out_path = out_dir / f"{model_short}__{mode}__K{K}N{N}.json"
    else:
        out_path = out_dir / f"{model_short}__{mode}.json"

    t_loop = time.time()
    for trial_idx in range(n_trials):
        seed = 1000 + trial_idx
        rng = random.Random(seed)
        trial = build_trial(K=K, N=N, seed=seed, filtered_values=filtered_values, rng=rng)

        trial_data = {
            "seed": seed,
            "test_category": trial["test_category"],
            "all_values": [it["value"] for it in trial["items"] if it["category"] == trial["test_category"]],
        }

        # Exp A: full sequence, all layers, both queries
        for qword, qlabel in [("first", "FVQ"), ("last", "CVQ")]:
            raw, stream_occ, qspan = render_prompt(trial, n_updates_to_include=None, query_word=qword)
            full_text, char_off = chat_format(raw, tokenizer, model_short)
            ex = forward_and_extract(
                model, tokenizer, full_text, char_off,
                stream_occ, qspan, device, keep_all_layers=True,
            )
            trial_data[f"expA_{qlabel}"] = {
                "n_layers": ex["n_layers"],
                "n_heads": ex["n_heads"],
                "n_rounds": ex["n_rounds"],
                "seq_len": ex["seq_len"],
                # head-averaged + per-head saved
                "attn_from_gen": ex["attn_from_gen"].tolist() if save_per_head else ex["attn_from_gen"].mean(axis=1).tolist(),
                "attn_from_qcat": ex["attn_from_qcat"].tolist() if save_per_head else ex["attn_from_qcat"].mean(axis=1).tolist(),
            }

        # Exp B + C: incremental, final layer only
        for qword, qlabel in [("first", "FVQ"), ("last", "CVQ")]:
            incr_data = []
            for n_steps in checkpoints:
                raw, stream_occ, qspan = render_prompt(trial, n_updates_to_include=n_steps, query_word=qword)
                if len(stream_occ) != n_steps:
                    # Not enough test_category occurrences for this n_steps (shouldn't happen)
                    continue
                full_text, char_off = chat_format(raw, tokenizer, model_short)
                ex = forward_and_extract(
                    model, tokenizer, full_text, char_off,
                    stream_occ, qspan, device, keep_all_layers=False,
                )
                incr_data.append({
                    "n_updates": n_steps,
                    "n_rounds": ex["n_rounds"],
                    "seq_len": ex["seq_len"],
                    "attn_from_gen": ex["attn_from_gen"].tolist() if save_per_head else ex["attn_from_gen"].mean(axis=1).tolist(),
                    "attn_from_qcat": ex["attn_from_qcat"].tolist() if save_per_head else ex["attn_from_qcat"].mean(axis=1).tolist(),
                })
            trial_data[f"incr_{qlabel}"] = incr_data

        trial_results.append(trial_data)
        elapsed = time.time() - t_loop
        rate = (trial_idx + 1) / elapsed
        print(
            f"  [{trial_idx + 1}/{n_trials}] {elapsed:.0f}s  "
            f"({rate:.2f} trials/s, ETA {((n_trials - trial_idx - 1) / max(rate, 1e-6)):.0f}s)",
            flush=True,
        )

        # Checkpoint every 5 trials
        if (trial_idx + 1) % 5 == 0 or trial_idx == n_trials - 1:
            with open(out_path, "w") as f:
                json.dump({
                    "model": model_short,
                    "hf_id": hf_id,
                    "mode": mode,
                    "K": K,
                    "N": N,
                    "n_trials_completed": trial_idx + 1,
                    "n_trials_target": n_trials,
                    "checkpoints": checkpoints,
                    "save_per_head": save_per_head,
                    "device": device,
                    "dtype": str(dtype),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "trials": trial_results,
                }, f)

    # Free
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

    print(f"  ✓ saved → {out_path}", flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(OPERATING_POINTS.keys()), help="Model short name")
    ap.add_argument("--mode", choices=["normal", "reversal"], help="Operating mode")
    ap.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    ap.add_argument("--smoke", action="store_true",
                    help="Run only 2 trials for one tiny config (smoke test)")
    ap.add_argument("--out-dir", type=str, default=str(RESULTS_ROOT))
    ap.add_argument("--no-per-head", action="store_true",
                    help="Save head-averaged only (smaller files)")
    ap.add_argument("--K", type=int, help="Override K (number of keys) — for consistency checks")
    ap.add_argument("--N", type=int, help="Override N (updates per key) — for consistency checks")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    if args.smoke:
        # Smoke: 2 trials on Qwen2.5-3B-Instruct normal
        run_one_config(
            model_short="Qwen2.5-3B-Instruct",
            mode="normal",
            n_trials=2,
            out_dir=out_dir / "smoke",
            save_per_head=not args.no_per_head,
        )
        return

    if args.model and args.mode:
        if args.mode not in OPERATING_POINTS[args.model]["modes"]:
            raise SystemExit(f"{args.model} has no '{args.mode}' mode")
        run_one_config(
            model_short=args.model,
            mode=args.mode,
            n_trials=args.trials,
            out_dir=out_dir,
            save_per_head=not args.no_per_head,
            override_K=args.K,
            override_N=args.N,
        )
    elif args.model:
        for mode in OPERATING_POINTS[args.model]["modes"]:
            run_one_config(
                model_short=args.model,
                mode=mode,
                n_trials=args.trials,
                out_dir=out_dir,
                save_per_head=not args.no_per_head,
            )
    else:
        for model_short, cfg in OPERATING_POINTS.items():
            for mode in cfg["modes"]:
                run_one_config(
                    model_short=model_short,
                    mode=mode,
                    n_trials=args.trials,
                    out_dir=out_dir,
                    save_per_head=not args.no_per_head,
                )


if __name__ == "__main__":
    main()
