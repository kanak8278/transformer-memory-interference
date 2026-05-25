"""Measure context-window usage of one Semantic-Multi prompt at each row's
representative (K, N) cell, for every model in Table 1.

Open-weight block: K=5, N=30 (the cell tab_main.tex evaluates them at).
Proprietary block: K=20, N=50 (the cell tab_main.tex evaluates them at).

Tokenisers:
- Open-weight: AutoTokenizer.from_pretrained per model (exact).
- GPT-4.1 family: tiktoken o200k_base (exact).
- Claude family: anthropic SDK messages.count_tokens (exact via API).
- Gemini family: google-genai SDK models.count_tokens (exact via API).
  If API call fails (no credentials), falls back to tiktoken o200k_base
  and the row is marked approx.

Outputs paper/figures/context_usage.csv with columns:
  model_label, K, N, tokens, ctx_window, pct, source

Run:
    .venv/bin/python paper/scripts/measure_context_usage.py
"""

import csv
import os
import random
import sys
from pathlib import Path

import tiktoken

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from mechanistic_probing_v2.core.dataset_configs import (
    generate_values_for_trial,
    build_interleaved_sequence,
    get_eligible_categories,
)

OUT_PATH = _ROOT / "paper" / "figures" / "context_usage.csv"
DATASET_TYPE = "SEMANTIC_MULTI"

# Same prompt builder as sweep_semantic.py:build_prompt (verbatim).
def build_prompt(sequence, test_category, condition):
    stream = "\n".join(f"{item['category']}: {item['value']}" for item in sequence)
    query_word = "first" if condition == "RI" else "last"
    return (
        f"Read the following key-value stream. "
        f"Each key gets updated multiple times.\n\n"
        f"{stream}\n\n"
        f"What was the {query_word} value of {test_category}?\n"
        f"Answer with ONLY the exact value. No explanation."
    )


def make_prompt(K, N, seed=0):
    rng = random.Random(seed)
    eligible = get_eligible_categories(DATASET_TYPE, min_values=N)
    categories = rng.sample(eligible, K)
    test_category = rng.choice(categories)
    values_per_cat = generate_values_for_trial(DATASET_TYPE, categories, N, rng)
    sequence = build_interleaved_sequence(categories, values_per_cat, rng)
    return build_prompt(sequence, test_category, "PI")  # PI = last-value (CVQ)


# (label, kind, identifier, ctx_window, K, N)
# kind ∈ {"hf", "tiktoken", "anthropic", "google"}
# All rows use tiktoken o200k_base for uniform comparability across providers.
# Context windows are verified:
#   - Open-weight: from HF config.json max_position_embeddings (text_config for
#     multi-modal Gemma-3-4b-it).
#   - Proprietary: from provider documentation as of 2026-Q2.
OPEN_WEIGHT = [
    # max_position_embeddings: 32768 (Qwen2.5-3B-Instruct/config.json)
    ("Qwen2.5-3B-Instruct", "tiktoken", "o200k_base",     32_768, 5, 30),
    # max_position_embeddings: 262144 (Qwen3.5-*/config.json text_config)
    ("Qwen3.5-2B",          "tiktoken", "o200k_base",    262_144, 5, 30),
    ("Qwen3.5-4B",          "tiktoken", "o200k_base",    262_144, 5, 30),
    ("Qwen3.5-9B",          "tiktoken", "o200k_base",    262_144, 5, 30),
    # max_position_embeddings: 32768 (gemma-3-1b-it/config.json)
    ("Gemma-3-1b-it",       "tiktoken", "o200k_base",     32_768, 5, 30),
    # Gemma-3-4b-it: published 128K (rope_scaling factor 8 × base = 128K).
    ("Gemma-3-4b-it",       "tiktoken", "o200k_base",    131_072, 5, 30),
]
PROPRIETARY = [
    # OpenAI announcement: GPT-4.1 / 4.1-mini both 1,000,000 tokens.
    ("GPT-4.1",          "tiktoken", "o200k_base", 1_000_000, 20, 50),
    ("GPT-4.1-mini",     "tiktoken", "o200k_base", 1_000_000, 20, 50),
    # Anthropic docs: Haiku 4.5 200K; Sonnet 4.5 1M (long-context option, now GA).
    ("Claude-4.5-Haiku", "tiktoken", "o200k_base",   200_000, 20, 50),
    ("Claude-4.5-Sonnet","tiktoken", "o200k_base", 1_000_000, 20, 50),
    # Google docs: Gemini-2.5-Flash 1,048,576; Gemini-2.5-Pro 1,000,000.
    ("Gemini-2.5-Flash", "tiktoken", "o200k_base", 1_048_576, 20, 50),
    ("Gemini-2.5-Pro",   "tiktoken", "o200k_base", 1_000_000, 20, 50),
]


def count_hf(prompt, model_id):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return len(tok.encode(prompt))


def count_tiktoken(prompt, encoding_name="o200k_base"):
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(prompt))


def count_anthropic(prompt, model_id):
    try:
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.count_tokens(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.input_tokens, "exact"
    except Exception as e:
        print(f"  anthropic.count_tokens failed ({e}); falling back to o200k_base")
        return count_tiktoken(prompt), "approx"


def count_google(prompt, model_id):
    try:
        from google import genai
        client = genai.Client()
        resp = client.models.count_tokens(model=model_id, contents=prompt)
        return resp.total_tokens, "exact"
    except Exception as e:
        print(f"  google.count_tokens failed ({e}); falling back to o200k_base")
        return count_tiktoken(prompt), "approx"


def main():
    rows = []
    # Cache prompts: only two cells across all rows.
    prompts = {
        (K, N): make_prompt(K, N, seed=0)
        for K, N in {(m[4], m[5]) for m in OPEN_WEIGHT + PROPRIETARY}
    }
    for K, N in sorted(prompts):
        chars = len(prompts[(K, N)])
        print(f"prompt @ K={K}, N={N}: {chars} chars")

    for label, kind, ident, ctx, K, N in OPEN_WEIGHT + PROPRIETARY:
        prompt = prompts[(K, N)]
        if kind == "hf":
            tokens = count_hf(prompt, ident); source = "exact"
        elif kind == "tiktoken":
            tokens = count_tiktoken(prompt, ident); source = "exact"
        elif kind == "anthropic":
            tokens, source = count_anthropic(prompt, ident)
        elif kind == "google":
            tokens, source = count_google(prompt, ident)
        else:
            raise ValueError(kind)
        pct = 100.0 * tokens / ctx
        print(f"  {label:24s} K={K:2d} N={N:2d}  tokens={tokens:6d}  "
              f"ctx={ctx:>9,}  pct={pct:.2f}%  ({source})")
        rows.append({
            "model_label": label, "K": K, "N": N,
            "tokens": tokens, "ctx_window": ctx,
            "pct": round(pct, 3), "source": source,
        })

    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
