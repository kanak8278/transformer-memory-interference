# Plan — Attention Routing on Gemma+LoRA (§7 mechanism replication)

**Goal**: Replicate the Qwen attention-routing analysis from §5.1 on
Gemma-3-4b-it+LoRA so the §7 mechanism story is supported by 3 of 4
methods on both architectures (currently: probing ✓, logit lens ✓,
attention routing ✗, stage 3 ablation ✗).

**Why it matters for the paper**: the abstract claims "mechanism analyses
localise to the same attention-head clusters." With Gemma showing only
probing + logit lens, that claim rests on Qwen alone. Adding attention
routing closes the gap that reviewer would catch.

---

## 1. Reference: what was done for Qwen

**File**: `v3/results_vllm/attention_routing/Qwen2.5-3B-Instruct__normal.json`
(baseline) and `…-LoRA__normal.json` (post-LoRA).

**Protocol**:
- Operating point: **K=2, N=30**, mode `"normal"` (matches §5.1)
- **50 trials per condition** (FVQ and CVQ)
- Data structured into **explicit rounds**: each round = one pass through
  K keys. With K=2, N=30 → 60 stream items grouped as 30 rounds × 2 keys.
  This differs from the shuffled-no-consecutive format used in main
  LoRA training; rounds are needed so we can attribute attention to
  specific rounds.
- Forward pass with `output_attentions=True`; capture attention from the
  generation position (last token of the prompt before answer)
- Per (layer, head), compute attention to each (round, key) cell →
  shape `(L, H, R, K)`
- For CVQ ("recall last value"), the **answer = round N-1's value for
  the queried key**. So the focal metric is
  `P(attend to round N-1, queried_key)` per head, or equivalently after
  summing over keys: `P(attend to last round)` per head.

**Qwen result** (reproduced from the saved data, top 15 promoter heads by
`Δ = LoRA − baseline` on the CVQ-attention-to-last-round metric):

| Head | base | LoRA | Δ |
|---|---|---|---|
| L32H3  | 0.107 | 0.784 | +0.677 |
| L31H12 | 0.156 | 0.813 | +0.656 |
| L32H7  | 0.124 | 0.775 | +0.651 |
| L31H15 | 0.226 | 0.863 | +0.637 |
| L30H11 | 0.194 | 0.777 | +0.584 |
| L31H8  | 0.115 | 0.692 | +0.577 |
| L32H10 | 0.061 | 0.631 | +0.569 |
| L32H0  | 0.083 | 0.648 | +0.564 |
| L33H8  | 0.073 | 0.633 | +0.560 |
| L32H14 | 0.073 | 0.616 | +0.544 |
| L32H9  | 0.054 | 0.573 | +0.519 |
| L32H12 | 0.067 | 0.584 | +0.517 |
| L33H11 | 0.073 | 0.590 | +0.517 |
| L31H1  | 0.065 | 0.581 | +0.516 |
| L33H14 | 0.079 | 0.574 | +0.495 |

All 15 in **L30–L33** (the post-LoRA promoter cluster). Δ range
+0.49 to +0.68.

---

## 2. Saved data structure (target format for Gemma run)

Top-level JSON keys (replicate this layout):
```
{
  "model": "<short name>",            # e.g. "gemma-3-4b-it" or "gemma-3-4b-it-LoRA"
  "hf_id": "<full id or merged path>",
  "mode": "normal",
  "K": 2, "N": 30,
  "n_trials_completed": 50,
  "n_trials_target": 50,
  "device": "cuda",
  "dtype": "bfloat16",
  "timestamp": "...",
  "trials": [
    {
      "seed": 1000,
      "test_category": "...",
      "all_values": [...],            # all N=30 values for test_category, in order
      "expA_FVQ": {                   # FVQ trial: query first value
        "n_layers": 34,
        "n_heads": 8,
        "n_rounds": 30,
        "seq_len": int,
        "attn_from_gen": [...]        # shape [L, H, R, K] = (34, 8, 30, 2)
      },
      "expA_CVQ": { ... },            # same shape, CVQ query (last value)
      # Optional incremental (skip for v1):
      # "incr_FVQ": [...],
      # "incr_CVQ": [...]
    },
    ...
  ]
}
```

The `incr_*` arrays in Qwen's data appear to be incremental-attention
snapshots at varying prefix lengths — useful for the "where does the
v_last signal emerge" sub-analysis but not core to the head-comparison
result. Skip for v1; can add later.

---

## 3. Implementation: `run_attention_routing_hf.py`

### Approach
- Pure HF + `output_attentions=True`, no transformer_lens (TL OOMs on
  Gemma at 23 GB)
- Build prompt with explicit round structure
- Forward pass, slice attention tensor at generation position
- Aggregate per (round, key) by averaging attention weights over the
  token positions belonging to each (round, key) cell

### Pseudocode

```python
def build_round_prompt(K, N, test_category, seed):
    """K keys × N rounds. Stream emits round-by-round:
        round 0: key_0: v_0_0, key_1: v_1_0
        round 1: key_0: v_0_1, key_1: v_1_1
        ...
    Returns prompt + per-position metadata to attribute each token to (round, key).
    """
    rng = random.Random(seed)
    categories = rng.sample(eligible, K)
    # value pool: K × N distinct single-token values
    values = {c: rng.sample(value_pool, N) for c in categories}
    stream_items = []
    for r in range(N):
        # shuffle keys within the round so order doesn't always match
        ks = rng.sample(categories, K)
        for k in ks:
            stream_items.append((r, k, values[k][r]))
    # Encode prompt with per-token (round, key) labels
    ...
    return prompt, gen_position, position_to_round_key_map, expected_first, expected_last

def attn_at_gen_per_round(model, tokenizer, prompt, gen_pos, pos_map, K, N):
    """Forward pass; for each (L, H), compute sum of attention from
    gen_pos to positions belonging to each (round, key) cell.
    Returns shape (L, H, N, K).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(**inputs, output_attentions=True, return_dict=True, use_cache=False)
    # out.attentions: tuple of (L,) tensors, each [1, H, T, T]
    attn_from_gen = np.zeros((n_layers, n_heads, N, K))
    for L_idx, attn_L in enumerate(out.attentions):
        # query position = gen_pos; row of attention matrix
        # shape: [H, T]
        row = attn_L[0, :, gen_pos, :].float().cpu().numpy()
        for tok_idx, (r, k_idx) in pos_map.items():
            attn_from_gen[L_idx, :, r, k_idx] += row[:, tok_idx]
    return attn_from_gen

def run_trial(model, tokenizer, K, N, seed):
    out = {}
    for cond in ["FVQ", "CVQ"]:
        prompt, gen_pos, pos_map, exp_first, exp_last = build_round_prompt(...)
        attn = attn_at_gen_per_round(...)
        out[f"expA_{cond}"] = {
            "n_layers": n_layers, "n_heads": n_heads,
            "n_rounds": N, "seq_len": seq_len,
            "attn_from_gen": attn.tolist(),
        }
    return out
```

### CLI
```
python lora_intervention/run_attention_routing_hf.py \
    --merged_path lora_intervention/checkpoints/gemma_merged \
    --base google/gemma-3-4b-it \
    --out_name gemma-3-4b-it-LoRA \
    --K 2 --N 30 --trials 50 \
    --device cuda --dtype bfloat16

python lora_intervention/run_attention_routing_hf.py \
    --merged_path google/gemma-3-4b-it \
    --base google/gemma-3-4b-it \
    --out_name gemma-3-4b-it-baseline-hf \
    --K 2 --N 30 --trials 50
```

Output: `v3/results_vllm/attention_routing/<out_name>__normal.json`
matching the Qwen file layout.

---

## 4. Comparison script: `compare_attention_routing_gemma.py`

After both runs land, produce the head ranking + 95% CIs (analogous
to `compute_cis.py`'s `attention_routing_cis()` function).

```python
def per_trial_p_last_per_head(rec, condition="expA_CVQ"):
    """Returns array shape (n_trials, L, H)."""
    out = []
    for t in rec["trials"]:
        a = np.asarray(t[condition]["attn_from_gen"])  # (L, H, R, K)
        a_sum = a.sum(axis=-1)                          # (L, H, R) — sum over keys
        norm = a_sum / (a_sum.sum(axis=-1, keepdims=True) + 1e-12)
        out.append(norm[..., -1])                       # P(attend to last round) per head
    return np.stack(out)

def top_promoter_heads(base, lora, k=15):
    delta = lora.mean(axis=0) - base.mean(axis=0)
    flat = sorted(
        ((L, H, delta[L, H]) for L in range(delta.shape[0]) for H in range(delta.shape[1])),
        key=lambda x: -x[2]
    )
    return flat[:k]
```

Then bootstrap CIs on each of the top 15 heads (n_boot=2000, paired
resampling not possible since seeds aren't paired — same as Qwen).
Output: `lora_intervention/results/attention_routing_comparison_gemma.txt`.

---

## 5. Acceptance criteria

1. **Format parity**: output JSON matches Qwen's structure (passes
   `assert set(qwen_keys) ⊆ set(gemma_keys)` for top-level and trial-level).
2. **Numerical sanity**:
   - `attn_from_gen[L, H, :, :].sum()` ≤ 1.0 for each (L, H) — attention
     mass shouldn't exceed unity for one query row
   - Across all heads at the gen position, mean total attention to
     stream tokens > 0.5 (most attention is to content, not BOS/system)
3. **Smoke test before full run**: 1 trial, K=2/N=30 — verify
   `attn_from_gen.shape == (34, 8, 30, 2)` and values are in [0, 1].
4. **Run timing**: ~3 min for 50 trials on L4 (forward passes are
   short — N=30 stream is ~700 tokens, 50 forwards × ~0.05s ≈ 3s of GPU
   time, dominated by attention extraction and Python loops). If
   >10 min, profile.

---

## 6. Expected outcome on Gemma

Two scenarios:

**A. LoRA reuses an existing late-layer cluster.** We see a Δ-cluster
similar to Qwen's L30-L33, but possibly at different layer indices
(Gemma is 34L vs Qwen 36L; baseline suppressor cluster is at L17-L19+L30H6,
so promoters might localise just above or alongside L17-L19, or near
L30-L33).
→ §7 claim *fully replicated*: "different baseline circuits → different
LoRA promoter locations → same functional role."

**B. LoRA promotion is diffuse or in different layers.** Top-15 heads
spread across many layers, no clean cluster.
→ Honest report: "Gemma's promoter pathway is more distributed; the
clean cluster seen on Qwen is architecture-specific, but the functional
behavior (P(attend to v_last) ≈ 1.0 at the gen position) is replicated."

Either is publishable; (A) is cleaner.

---

## 7. Sequencing & dependencies

1. **Prerequisite**: `gemma_merged/` standalone model on disk
   (already exists from §7 work, 14.8 GB at
   `lora_intervention/checkpoints/gemma_merged/`).
2. **Implementation**: write `run_attention_routing_hf.py` (~3-4 h).
3. **Smoke test** on 1 trial (~5 min).
4. **Run baseline + LoRA** (50 trials each, ~10 min total).
5. **Comparison script + bootstrap CIs** (~30 min).
6. **Document + commit** (~30 min).

Total: ~5-6 hours when done end-to-end.

---

## 8. Equivalent that's already done for Qwen — checklist

This section documents what to mirror exactly so the runs are
apples-to-apples.

| Spec                    | Qwen (committed) | Gemma (target) |
|-------------------------|------------------|----------------|
| Operating point         | K=2, N=30        | K=2, N=30      |
| Mode                    | normal           | normal         |
| n_trials                | 50               | 50             |
| Trial seed range        | 1000-1049        | **same** (so any seed-paired analysis can be added later) |
| Conditions captured     | FVQ + CVQ        | FVQ + CVQ      |
| Stream structure        | Explicit rounds: N×K=60 items grouped as 30 rounds × 2 keys, intra-round key order shuffled per round | **same** |
| Forward dtype           | fp16 (CUDA)      | bf16 (CUDA on L4) |
| Architecture wrapper    | TL `from_pretrained` with fold_ln/center | **HF direct** (TL OOMs on Gemma) |
| Save format             | JSON with `trials[]` list, each having `expA_{FVQ,CVQ}` | **same** |
| Output path             | `v3/results_vllm/attention_routing/<name>__normal.json` | **same** |
| Head-promoter aggregation | mean across trials of P(attend to last round) per (L, H), then sort by Δ | **same** |
| Bootstrap CI method     | Nonparametric, n=2000, seed=17, unpaired (trials are i.i.d.) | **same** |
| Top-k heads reported    | 15                | **same** |

**Caveats noted in Qwen run that also apply to Gemma run**:
- Trials are i.i.d. from the same distribution but not paired by seed
  (Python `hash()` randomization in v3 attention_routing.py); we now
  use deterministic seeding (`stable_seed` in data_gen.py) so the
  Gemma run is reproducible.
- bf16 vs fp16 dtype difference between Qwen and Gemma is below the
  signal scale (residuals differ by ~1e-3, deltas in attention are
  ~0.5).
- The Qwen Stage 3 followup (ablating the promoter heads) is *not*
  in this plan; that's a separate Section 7 follow-up that needs
  HF-direct attention head zeroing (see GEMMA_RUN_LOG §6 "Next steps").

---

## 9. Risks & open questions

- **Risk: HF attention output for Gemma3 may use different format.**
  Some HF VLM-style models (Gemma3 included) ship with grouped-query
  attention (GQA). `output_attentions=True` should still return a
  full [L, H_q, T, T] tensor where H_q = num_query_heads (8 for
  Gemma-3-4b). Verify in smoke test that the per-layer attention shape
  matches `(1, 8, T, T)` and not `(1, n_kv_heads, T, T)`.

- **Open: token-to-(round, key) mapping.** The simplest reliable approach
  is to tokenize each value separately and find its position in the
  concatenated prompt (search by token-id sequence). Stable for
  single-token values from ARBITRARY_SINGLE.

- **Open: tokenizer for merged model.** Use the tokenizer saved with
  `gemma_merged/` (which is Gemma-3's tokenizer with all 262k vocab
  entries; same as base).
