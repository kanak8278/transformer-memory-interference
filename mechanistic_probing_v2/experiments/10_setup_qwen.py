"""
Step 1.1: Setup validation for Qwen2.5-0.5B-Instruct via TransformerLens.

Verifies:
1. Model loads on MPS
2. Tokenization works
3. Forward pass works
4. Generation works with instruction-following
5. Cache access works (needed for Phase 2)
6. Pre-flight context check works
"""

import sys
from pathlib import Path

# Add parent to path for core imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model_loader import load_model, get_context_limit
from core.dataset_configs import (
    generate_trial, preflight_context_check, compute_feasible_grid,
    KEY_LEVELS, UPDATE_LEVELS,
)


def main():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    # ── 1. Load model ────────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Loading model")
    print("=" * 60)
    model, tokenizer, info = load_model(model_name)
    print(f"  Backend: {info.backend}")
    print(f"  Device: {info.device}")

    # ── 2. Tokenization ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Tokenization check")
    print("=" * 60)
    test_text = "The first value of color was red"
    tokens = model.to_tokens(test_text)
    print(f"  '{test_text}' -> {tokens.shape[1]} tokens")
    decoded = model.to_string(tokens[0])
    print(f"  Round-trip decode: '{decoded}'")

    # ── 3. Forward pass ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Forward pass")
    print("=" * 60)
    logits = model(tokens)
    print(f"  Logits shape: {logits.shape}")
    pred_token = logits[0, -1].argmax().item()
    pred_text = tokenizer.decode([pred_token])
    print(f"  Next token prediction after '{test_text}': '{pred_text}'")

    # ── 4. Generation with interference prompt ──────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Generation test (simple interference)")
    print("=" * 60)
    trial = generate_trial("ARBITRARY_MULTI", num_keys=3, num_updates=3, condition="RI", seed=42)
    print(f"  Categories: {trial.categories}")
    print(f"  Test: {trial.test_category}")
    print(f"  Expected: {trial.expected_answer}")
    print(f"  Prompt length: {len(trial.prompt)} chars")
    print(f"\n  --- Prompt ---")
    print(f"  {trial.prompt[:500]}")
    print(f"  ...")

    gen_tokens = model.generate(
        trial.prompt, max_new_tokens=20, temperature=0, verbose=False
    )
    gen_text = model.to_string(gen_tokens[0]) if hasattr(gen_tokens, 'shape') else gen_tokens
    # Extract just the generated part
    answer_part = gen_text[len(trial.prompt):].strip().split("\n")[0]
    print(f"\n  Generated: '{answer_part}'")
    print(f"  Expected: '{trial.expected_answer}'")
    correct = trial.expected_answer.lower() in answer_part.lower()
    print(f"  Match: {correct}")

    # ── 5. Cache access (needed for Phase 2) ─────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: Cache access test")
    print("=" * 60)
    short_tokens = model.to_tokens("Hello world")
    logits, cache = model.run_with_cache(short_tokens)
    print(f"  Cache keys: {len(cache)} entries")
    # Check key cache entries exist
    for key in ["resid_post", "attn_out", "mlp_out", "pattern"]:
        matching = [k for k in cache.keys() if key in k]
        print(f"  '{key}' entries: {len(matching)}")
    # Check shapes
    resid = cache["resid_post", 0]
    print(f"  resid_post[0] shape: {resid.shape}")
    pattern = cache["pattern", 0]
    print(f"  pattern[0] shape: {pattern.shape} (batch, heads, seq, seq)")
    del cache  # free memory

    # ── 6. Pre-flight context check ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6: Pre-flight context grid")
    print("=" * 60)
    ctx_limit = get_context_limit(model_name, info)
    print(f"  Context limit: {ctx_limit}")

    feasible = compute_feasible_grid(KEY_LEVELS, UPDATE_LEVELS, tokenizer, ctx_limit)
    print(f"  Feasible cells: {len(feasible)} / {len(KEY_LEVELS) * len(UPDATE_LEVELS)}")

    # Print grid
    print(f"\n  {'Keys':>4} | Max feasible updates | Est. tokens at max")
    print(f"  {'-'*4}-+-{'-'*21}-+-{'-'*19}")
    for nk in KEY_LEVELS:
        max_nu = max((nu for (k, nu) in feasible if k == nk), default=0)
        if max_nu > 0:
            est = feasible[(nk, max_nu)]
            print(f"  {nk:>4} | {max_nu:>21} | {est:>19,}")
        else:
            print(f"  {nk:>4} | {'NONE':>21} | {'N/A':>19}")

    # ── 7. Quick RI vs PI test ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 7: Quick RI vs PI sanity check (keys=3, updates=3)")
    print("=" * 60)
    ri_correct = 0
    pi_correct = 0
    n_quick = 10
    for seed in range(n_quick):
        for cond in ["RI", "PI"]:
            trial = generate_trial("ARBITRARY_MULTI", num_keys=3, num_updates=3, condition=cond, seed=seed + 100)
            gen = model.generate(trial.prompt, max_new_tokens=15, temperature=0, verbose=False)
            gen_text = model.to_string(gen[0]) if hasattr(gen, 'shape') else gen
            answer = gen_text[len(trial.prompt):].strip().split("\n")[0].strip()
            correct = trial.expected_answer.lower() in answer.lower()
            if cond == "RI":
                ri_correct += correct
            else:
                pi_correct += correct

    print(f"  RI: {ri_correct}/{n_quick} = {ri_correct/n_quick:.0%}")
    print(f"  PI: {pi_correct}/{n_quick} = {pi_correct/n_quick:.0%}")

    if ri_correct > pi_correct:
        print("  -> PI > RI pattern confirmed (RI better than PI)")
    elif ri_correct == pi_correct:
        print("  -> No asymmetry at this level (try higher interference)")
    else:
        print("  -> Unexpected: PI > RI (model recalls last better than first)")

    print("\n" + "=" * 60)
    print("SETUP VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
