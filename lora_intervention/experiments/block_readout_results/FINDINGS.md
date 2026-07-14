# Block vs LoRA — "two roads, one readout"

**Runs:** 2026-07-14, Colab L4, HF backend, Qwen2.5-3B, ARB single-token,
100 trials/condition. **Two cells, both confirm the pattern:**
`results_K2N30.json` (K=2, N=30) and `results_K10N50.json` (K=10, N=50 — the
harder cell where Block's format win is largest). Behavioral accs use the
prefix/contains matcher (multi-token-safe); logit-lens uses first-token P.

## Replication across cells (CVQ P(v_last) at readout L35; behavioral CVQ acc)

| cell | base_plain | base_block | lora_plain | mean\|block−lora\| vs \|block−base\| (L31–35) |
|---|---|---|---|---|
| K=2, N=30  | 0.37 (acc 0.41) | 0.96 (acc 0.96) | 0.99 (acc **1.00**) | 0.058 vs 0.450 |
| K=10, N=50 | 0.50 (acc 0.50) | 0.86 (acc 0.87) | 0.99 (acc **1.00**) | 0.075 vs 0.282 |

Both cells: base_plain's v_last stays **suppressed** at the readout; base_block
and lora_plain both **build and propagate** it, tracking each other ~4–8× closer
than either is to base_plain. **LoRA fully fixes CVQ (acc 1.00) at both cells** —
confirming this is a valid comparison cell (the earlier "0.58" was a single-token
exact-match artifact, since fixed).

Detailed K=2/N=30 layer trajectory below; K=10/N=50 shows the same shape.

## Question
App `block_locus` already showed Block and LoRA activate **different** attention
heads (Block barely touches the 15 L30–L33 promoter heads: pooled Pr(attend
v_last) 0.16 vs LoRA's 0.68). It never tested whether the two fixes **converge
at the readout**. This run adds the two methods block_locus skipped — logit-lens
+ CVQ-correctness probe — under the Block condition, for a 3-way comparison.

## Result: convergent readout, divergent routing

Per-layer **P(v_last)** for CVQ (logit lens), final layers:

| layer | base_plain | base_block | lora_plain |
|---|---|---|---|
| L32 | 0.03 | 0.49 | 0.33 |
| L33 | 0.18 | 0.75 | 0.70 |
| L34 | 0.19 | 0.74 | 0.79 |
| **L35** | **0.35** | **0.98** | **0.98** |

- **base_block and lora_plain track each other layer-for-layer** and both reach
  **0.98** at the readout. base_plain stays **suppressed at 0.35**.
- Quantified: mean |base_block − lora_plain| over L31–35 = **0.058**;
  mean |base_block − base_plain| = **0.451**. The two fixes are ~8× closer to
  each other than to the baseline.
- **CVQ-specific** (control): FVQ P(v_first) at L35 is high in *all* conditions
  (base_plain **0.87**). The base model already gets the first value — only the
  **last-value readout is suppressed**, and both interventions rescue exactly
  that.
- CVQ-correctness probe (secondary, all non-degenerate at N=30): max per-layer
  CV accuracy base_plain **0.76** → base_block **0.85**, lora_plain **0.86** —
  same direction, weaker signal than the logit lens.

## Interpretation
Combined with `block_locus`, the picture is **"two roads, one readout"**:
- **Divergent routing** — Block uses the explicit `[Update j]` markers as
  positional anchors; LoRA reweights the L30–L33 promoter heads. Different
  attention (block_locus).
- **Convergent readout** — both produce the *same* v_last propagation to the
  output (build at ~L32 → 0.98 at L35). Same computational fix (this run).

This directly instantiates the paper's title: the base model **tracks** v_last
(it surfaces, weakly, ~L33–35) but **suppresses** it (caps at 0.35); every
successful intervention — whether prompt-format or weights — **releases the same
suppressed readout**, via different upstream triggers. The shared signature of
"fixing it" is *released suppression*, not *better retrieval* — opposing the
naive view that the error is an attention-to-wrong-value retrieval failure.

## Paper impact
Strengthens/reframes the §5 "convergence" language that `block_locus` had forced
us to soften to "operationally distinct." Correct nuance: **operationally
distinct upstream (routing), functionally convergent downstream (readout).**
Candidate figure: the 3-condition per-layer P(v_last) overlay above.

## Caveats
- Single model (Qwen2.5-3B). Confirmed at two cells (K=2/N=30 and K=10/N=50);
  a cross-family replication (Gemma) is blocked by the HF-Xet download issue.
- **Metric note (resolved):** an earlier version of this run scored behavior by
  single-token argmax + exact match, which under-counted multi-token-under-Qwen
  values (reported base_block 0.60 / lora_plain 0.58). Fixed to multi-token
  greedy generation + prefix/contains match (matching evaluate.py); corrected
  accs are base_block 0.87–0.96, lora_plain **1.00**. The logit-lens uses the
  standard first-token P (matching the paper's §7 methodology), so the
  readout-convergence claim was never affected; the probe labels now use the
  corrected (proper-metric) correctness labels.
- Routing leg reused from `block_locus` (K=2, N=30) — same cell, so directly
  comparable.
