# Entropy-Lens findings — base vs LoRA vs from-scratch GPT-2

Entropy-Lens (arXiv:2502.16570) on the KV-interference task, run identically
(same FVQ/CVQ/IVQ query types, same K∈{2,4,6,8,10,12}×N∈{4,6,8,10,12} grid,
150 trials/cell/condition) across three models. Per-layer logit-lens entropy at
the answer position, normalised by log(vocab) and indexed by relative depth so
the 36-layer Qwen and 12-layer GPT-2 are comparable. Reproducible with
`PYTHONHASHSEED=0`; scratch weights from the canonical HF repo (bit-identical to
the local grokked checkpoint). Outputs: `entropy_profiles.png`,
`entropy_confusion.png`, `entropy_{base,lora,scratch}.json`, `entropy_summary.json`.

## Result 1 — two qualitatively different depth geometries

**Qwen (base & lora): expand → prune.** Entropy rises from ~0.45 (early) to a
peak ~**0.77 at relative depth 0.89**, then collapses sharply to ~0 at the
output (expansion amplitude +0.17–0.20, pruning drop ~0.73–0.77). This is the
canonical large-LM dynamic the paper describes.

**From-scratch GPT-2: flat → cliff.** Entropy sits ~**0.57 essentially flat
across layers 1–11 (expansion amplitude ~+0.01–0.05, i.e. none)**, then drops in
a single step at the final layer. The from-scratch, task-specialised model does
**not** use depth to progressively resolve uncertainty — it defers all
commitment to the last layer, and commits less far (output entropy 0.07–0.15 vs
Qwen's ~0.00–0.05).

## Result 2 — LoRA does NOT change Qwen's entropy geometry

base and lora profiles are near-identical in shape (same peak height, same
peak depth d≈0.89, same expand-prune). LoRA only prunes marginally harder at the
very end (output entropy → 0.00 vs base 0.02–0.06). So the adapter's effect is
**not** in the coarse token-space entropy dynamics. This dovetails with the
linear-probing result (`../linear_probing/FINDINGS_v3.md`): LoRA changes *what
is represented* (the correct answer value is far more linearly decodable) while
leaving the *entropy signature* of the computation unchanged.

## Result 3 — the "confusion" signature: does output uncertainty track correctness?

Output-layer normalised entropy, correct vs wrong trials (gap = wrong − correct):

| model | IVQ_d25 | IVQ_d50 | IVQ_d75 | CVQ | reading |
|---|---|---|---|---|---|
| base    | +0.009 | +0.009 | +0.005 | +0.014 | confidently wrong (≈0 gap) |
| lora    | +0.058\* | +0.039 | +0.042 | +0.058\* | slightly more calibrated |
| scratch | **+0.474** | **+0.418** | **+0.279** | **+0.373** | wrong ⇒ genuinely confused |

\* LoRA is almost always correct at easy conditions, so its wrong-subset n is
tiny there (CVQ n_wrong=1, IVQ_d25 n_wrong=14) — noisy; the reliable LoRA cells
are IVQ_d50/d75 (n_wrong=331/859), gap ~0.04.

**The from-scratch model's uncertainty tracks its correctness**: when it answers
wrong, output entropy stays high (~0.36–0.50 of max), i.e. it is visibly
"confused"; when right, ~0.02–0.08. **Pretrained Qwen is confidently wrong** —
output entropy is ~0 whether the answer is right or wrong (gap ≈0.01). LoRA is
marginally more calibrated than base (~0.04 gap on reliable cells) but still an
order of magnitude below the scratch model.

## Takeaways
1. Model *family / training regime*, not the LoRA adapter, determines the
   entropy-lens depth geometry (Qwen expand-prune vs scratch flat-cliff).
2. LoRA's mechanism is representational (answer content), not a change in the
   entropy dynamics — consistent across both experiments.
3. Only the from-scratch model's output confidence is calibrated to correctness
   on this task; the instruction-tuned LLM commits with near-zero entropy
   regardless of whether it is right — a concrete "confidently wrong" signature.

## Caveats
- Logit-lens entropy uses each model's own final norm + unembedding at every
  layer (standard). For Qwen the value pool is restricted to generation-safe
  single-token values so the free single-token correctness label is accurate.
- Qwen and scratch necessarily use different tokenizers/prompt encodings; only
  the task, cell, query-type and relative depth are matched (as in the paper's
  cross-family comparisons). Normalisation by log(V) and relative depth is what
  makes the curves comparable.
