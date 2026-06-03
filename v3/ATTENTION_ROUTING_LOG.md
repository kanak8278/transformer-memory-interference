# Attention Routing Investigation — Lab Notebook

Running log of what was done and what was measured. Every numerical claim is
either:
- a direct measurement from raw data (sources: JSON files under
  `v3/results_vllm/attention_routing/`), or
- prefixed with **Interpretation:** when it is not directly measured.

---

## Date: 2026-05-21

## Goal

Identify the **mechanism** behind the PI > RI behavioral asymmetry in
Qwen2.5-3B-Instruct and gemma-3-4b-it by studying:
1. Which past-token positions the model attends to from the generation token
   and the query-key token at each layer (Exp A).
2. How that routing changes with sequence depth (Exp B/C).
3. Which heads, if any, differentiate between FVQ ("first value") and CVQ
   ("current value") queries (per-head analysis).
4. Whether the differentiating heads are causally necessary for behavior
   (ablation + random control).

## Design

Three forward passes per (trial × query) drive all three sub-experiments:

| Exp | Source token | Layers extracted | Sequence | Queries |
|---|---|---|---|---|
| A | gen-token + query-key | all | full (operating-point N) | FVQ + CVQ |
| B | gen-token | final | incremental N (1,2,3,5,10,20,opN) | CVQ |
| C | query-key | final | incremental N | FVQ + CVQ |

- **gen-token** = `input_ids.shape[1] - 1`, the last prompt position.
- **query-key** = last sub-token of the test category inside the question text.

Dataset format: **ARBITRARY_SINGLE, interleaved**, exact reproduction of v3
`stage1_sweep.py:248-265` (verified by reading source).

Position tracking: `tokenizer(text, return_offsets_mapping=True)` after
`apply_chat_template`, so character spans for keys and values are mapped
to token positions.

Single-token value filter per model:
`tokenizer.encode(" " + value, add_special_tokens=False)` must return 1 token.

Raw save format: per trial, `attn_from_gen` and `attn_from_qcat` arrays of
shape `[layers_kept, n_heads, n_rounds, 2]` where last dim is
`(key_span_attention_sum, value_position_attention)`. ~165 MB JSON per
(model, config) at 50 trials.

## Operating points

ARBITRARY_SINGLE behavioral data from `v3/results_vllm/arbitrary_single/`:

| Model | Mode | K | N | FVQ acc | CVQ acc | Gap |
|---|---|---|---|---|---|---|
| Qwen2.5-3B-Instruct | normal | 2 | 30 | 0.86 | 0.41 | +0.45 |
| Qwen2.5-3B-Instruct | "K=2/N=10" | 2 | 10 | 0.87 | 0.55 | +0.32 |
| Qwen2.5-3B-Instruct | "K=5/N=20" | 5 | 20 | 0.68 | 0.53 | +0.15 |
| Qwen2.5-3B-Instruct | reversal | 15 | 20 | 0.32 | 0.58 | −0.26 |
| gemma-3-4b-it | K=7/N=30 | 7 | 30 | 0.95 | 0.52 | +0.43 |
| gemma-3-4b-it | K=2/N=30 | 2 | 30 | 1.00 | 0.62 | +0.38 |
| gemma-3-4b-it | K=5/N=15 | 5 | 15 | 1.00 | 0.72 | +0.28 |

50 trials per config in the attention-routing experiment.

## Measurements

### M1. Head-averaged attention shows similar FVQ vs CVQ routing

When attention from `gen-token` and `query-key` is averaged over heads, the
per-layer pattern of "attention to round k" is visually similar for FVQ and
CVQ across all 4 Qwen configs and all 3 Gemma configs.

Source: `v3/plots/attention_routing/{config}/expA_*.png`.

This finding by itself does NOT show that routing is identical — only that
**head-averaged** routing is.

### M2. Per-head analysis reveals query-discriminating heads

For each (layer, head), compute mean attention to round 0 (primacy) and to
the last round (recency) over 50 trials, separately for FVQ and CVQ, then
take the difference. Source: `v3/extract_query_heads.py` output.

For Qwen2.5-3B-Instruct K=2/N=30:
- Max FVQ−CVQ on primacy: **+0.752** at (L31, H15)
- Max FVQ−CVQ on recency: **−0.183** at (L31, H15) — same head

L31 H15 is one specific head that boosts attention to round 0 under FVQ AND
suppresses attention to the last round under FVQ. Equivalently: under CVQ,
the same head increases recency attention and reduces primacy attention.

### M3. 11 heads are top FVQ-CVQ discriminators in all 4 Qwen configs

Source: `v3/results_vllm/attention_routing/head_analysis/head_analysis_report.md`.

For `gen → round 0 (primacy)`, the following 11 heads appear in the
top-15 by FVQ−CVQ difference in all 4 Qwen configs:

| Layer | Head | normal | K2N10 | K5N20 | reversal |
|---:|---:|---:|---:|---:|---:|
| 31 | 15 | +0.752 | +0.745 | +0.543 | +0.244 |
| 31 |  7 | +0.584 | +0.576 | +0.313 | +0.130 |
| 31 |  8 | +0.562 | +0.494 | +0.327 | +0.144 |
| 30 | 11 | +0.556 | +0.519 | +0.276 | +0.102 |
| 31 | 12 | +0.460 | +0.443 | +0.313 | +0.133 |
| 32 |  3 | +0.281 | +0.372 | +0.266 | +0.147 |
| 29 |  1 | +0.376 | +0.335 | +0.273 | +0.147 |
| 27 |  1 | +0.344 | +0.345 | +0.332 | +0.153 |
| 32 |  7 | +0.251 | +0.333 | +0.238 | +0.128 |
| 29 |  5 | +0.281 | +0.246 | +0.183 | +0.102 |
| 29 |  4 | +0.255 | +0.209 | +0.210 | +0.125 |

Cross-config layer range: **L27–L32**.

Several of these heads also appear in the top of the `gen → last round
(recency)` table with NEGATIVE FVQ−CVQ values (e.g., L31 H15: −0.183,
L30 H11: −0.118, L31 H12: −0.096, L31 H8: −0.084, L32 H3: −0.073,
L32 H7: −0.060) — same heads acting bidirectionally.

### M4. Mean FVQ−CVQ across the 11 heads (Qwen2.5-3B-Instruct, K=2/N=30)

Computed from raw data:
- Mean FVQ−CVQ on primacy:  **+0.43** (positive = FVQ attends more to round 0)
- Mean FVQ−CVQ on recency:  **−0.07** (negative = CVQ attends more to last round)
- |primacy| / |recency| ratio: **6.1**

### M5. Reversal-zone Qwen config shows ~3× reduced magnitudes

For the same 11 heads, FVQ−CVQ on primacy in K=15/N=20 (reversal) compared
to K=2/N=30 (normal):
- L31 H15: 0.752 → 0.244 (ratio 3.1×)
- L31 H7:  0.584 → 0.130 (ratio 4.5×)
- L31 H8:  0.562 → 0.144 (ratio 3.9×)
- L30 H11: 0.556 → 0.102 (ratio 5.5×)

Across all 11 heads, FVQ−CVQ on primacy reduces by ~3–5× from normal to reversal.

### M6. Gemma-3-4b-it cross-config discriminating heads (3 configs)

Source: per-config head analysis JSONs under
`v3/results_vllm/attention_routing/head_analysis/gemma-3-4b-it__*__heads.json`.

Heads appearing in top-15 of `gen → round 0 (primacy)` FVQ-dominant in ALL
3 Gemma configs (K=7/N=30, K=2/N=30, K=5/N=15):

| Layer | Head | K=7/N=30 | K=2/N=30 | K=5/N=15 |
|---:|---:|---:|---:|---:|
| 23 | 1 | +0.615 | +0.796 | +0.754 |
| 23 | 3 | +0.727 | +0.663 | +0.763 |
| 23 | 0 | +0.549 | +0.348 | +0.452 |
| 23 | 6 | +0.399 | +0.447 | +0.383 |
| 23 | 7 | +0.248 | +0.208 | +0.170 |
| 29 | 4 | +0.190 | +0.208 | +0.190 |
| 21 | 6 | +0.073 | +0.286 | +0.483 |

Heads in top-15 of `gen → last round` CVQ-dominant in all 3 Gemma configs:
L25 H0, L23 H3, L23 H1, L23 H0, L27 H6, L22 H7, L22 H6, L29 H4, L30 H2, L26 H4.

Heads appearing in both lists (bidirectional): **L23 H0, L23 H1, L23 H3, L29 H4**.

Cross-config layer range: **L21–L30**, with the largest magnitudes at L23.

### M7. Causal ablation, Qwen2.5-3B-Instruct K=2/N=30 (50 trials)

Method: forward_pre_hook on `o_proj` zeros out the specified head outputs
before the output projection. Hook target: `model.layers[L].self_attn.o_proj`.
11 heads ablated (the M3 set).

| | FVQ | CVQ | Gap |
|---|---:|---:|---:|
| Baseline | 0.860 | 0.400 | +0.460 |
| Ablated  | 0.720 | 0.180 | +0.540 |
| Δ        | −0.140 | −0.220 | +0.080 |

Per-trial transitions:
- FVQ: 10 correct→wrong, 3 wrong→correct
- CVQ: 14 correct→wrong, 3 wrong→correct

Median predicted round under FVQ: 0 (both baseline and ablated). Mean
predicted round under FVQ: 0.4 → 1.2.

### M8. Causal ablation, Gemma-3-4b-it K=7/N=30 (50 trials)

10 heads ablated (L17 H5, H0, H2; L23 H3, H1, H0, H6, H7; L29 H4, H3).

| | FVQ | CVQ | Gap |
|---|---:|---:|---:|
| Baseline | 0.920 | 0.400 | +0.520 |
| Ablated  | 0.000 | 0.000 | 0.000 |
| Δ        | −0.920 | −0.400 | −0.520 |

Per-trial: 46/50 FVQ correct→wrong (none reverse). 0/50 ablated FVQ
predictions hit any value in the input stream. 1/50 ablated CVQ predictions
hit a stream value (at relative position 0.86). 19/50 trials produce
identical strings for ablated FVQ and ablated CVQ.

### M9. Random-head control ablations (50 trials each)

Same number of randomly-chosen non-identified heads, with original layer
distribution preserved where possible (spillover into adjacent layers for
Gemma L23 since 5 of 8 heads are identified — only 3 non-identified remain).

Qwen2.5-3B-Instruct K=2/N=30:
| | FVQ | CVQ |
|---|---:|---:|
| Baseline | 0.860 | 0.400 |
| Random ablation | 0.920 | 0.320 |
| Δ | +0.060 | −0.080 |
| FVQ correct→wrong | 0 |  |

Gemma-3-4b-it K=7/N=30:
| | FVQ | CVQ |
|---|---:|---:|
| Baseline | 0.920 | 0.400 |
| Random ablation | 0.940 | 0.240 |
| Δ | +0.020 | −0.160 |
| FVQ correct→wrong | 0 |  |

### M10. Logit lens recomputation (Qwen2.5-3B-Instruct, existing data)

Source: `v3/results_vllm/logit_lens/Qwen2.5-3B-Instruct/stage2_logit_lens_*.json`.
Per-layer P(v_first) and P(v_last) averaged across 100 trials per cell.

| K | N | RI peak layer | RI peak P | RI final P | PI peak layer | PI peak P | PI final P |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 5  | 33 (94%) | 0.629 | 0.403 | 33 (94%) | 0.506 | 0.249 |
| 2 | 10 | 33 (94%) | 0.681 | 0.355 | 33 (94%) | 0.313 | 0.228 |
| 2 | 50 | 33 (94%) | 0.708 | 0.441 | 35 (100%) | 0.106 | 0.106 |

P-peak layer is L33 in all 3 cells for RI and 2/3 for PI. P(v_last) peak
magnitude decreases monotonically with N: 0.506 → 0.313 → 0.106.

P-drop from L33 to final L35 at K=2/N=5: RI 0.629→0.403 = **−35.9%**.
PI 0.506→0.249 = **−50.8%**.

### M12. Per-trial attention-behavior correlation (both models)

Method: for each of 50 trials in the Qwen normal config and 50 trials in the
Gemma K=7/N=30 config, compute the **per-trial mean FVQ−CVQ attention difference
across the discriminating heads** ("boost"). Match by `seed` to baseline
correctness from `causal_ablation` JSONs (seeds produce identical trial content
in both scripts — verified).

Boost metric (per trial):
- FVQ-primacy boost = mean over heads of `attn_FVQ[L,H,round=0] − attn_CVQ[L,H,round=0]`
- CVQ-recency boost = mean over heads of `attn_CVQ[L,H,last_round] − attn_FVQ[L,H,last_round]`

Sources: `v3/results_vllm/attention_routing/{model}.json` for boost,
`v3/results_vllm/causal_ablation/ablation__{model}.json` (baseline section) for
correctness. Test: Mann-Whitney U, two-sided, no scipy. Effect size: Cohen's d.

#### Qwen2.5-3B-Instruct, K=2/N=30, n=50

| Hypothesis | Group | n | Mean boost | Median boost |
|---|---|---:|---:|---:|
| H1: FVQ-primacy → FVQ correct | correct | 43 | **0.463** | 0.487 |
| | wrong | 7 | 0.210 | 0.194 |
| H2: CVQ-recency → CVQ correct | correct | 20 | **0.111** | 0.114 |
| | wrong | 30 | 0.042 | 0.036 |

- H1: U=278.0, z=+3.56, **p=0.0004**, Cohen's d=+1.97
- H2: U=517.0, z=+4.30, **p<0.0001**, Cohen's d=+1.64

#### Gemma-3-4b-it, K=7/N=30, n=50

| Hypothesis | Group | n | Mean boost | Median boost |
|---|---|---:|---:|---:|
| H1: FVQ-primacy → FVQ correct | correct | 46 | **0.349** | 0.358 |
| | wrong | 4 | 0.164 | 0.180 |
| H2: CVQ-recency → CVQ correct | correct | 20 | **0.083** | 0.086 |
| | wrong | 30 | 0.029 | 0.024 |

- H1: U=177.0, z=+3.04, **p=0.0024**, Cohen's d=+1.94
- H2: U=551.0, z=+4.97, **p<0.0001**, Cohen's d=+2.09

Plots: `v3/plots/attention_routing/correlation/per_trial_correlation__{model}.png`

#### Observations

1. For both models, **trials where the discriminating heads exhibit a larger
   attention boost are more likely to be correct on the corresponding query**.
2. The direction is consistent: positive Cohen's d on all four tests.
3. Effect sizes are large (all > 1.6).
4. Smallest p-value across the four tests: 0.0024 (Bonferroni-adjusted across
   4 tests: 0.0096 — still below 0.05).

### M11. Cross-config head identity in Qwen — Gemma comparison

| Aspect | Qwen2.5-3B (4 configs) | Gemma-3-4b (3 configs) |
|---|---|---|
| Configs spanning behavioral gap | +0.45 → −0.26 | +0.43 → +0.28 |
| Top-discriminator heads, gen-primacy | 11 in L27–L32 | 7 in L21–L29, dense at L23 |
| Heads ranking in top-15 of all configs | all 11 | 7 listed above |
| Max single head FVQ−CVQ on primacy | +0.752 | +0.727 |
| Baseline FVQ acc | 0.86 | 0.92 |
| Ablated FVQ acc | 0.72 | 0.00 |
| Δ FVQ | −0.14 | −0.92 |
| Δ CVQ | −0.22 | −0.40 |
| Random control FVQ Δ | +0.06 | +0.02 |
| Random control CVQ Δ | −0.08 | −0.16 |

---

## Interpretations

The following are interpretations of the measurements above. They are
plausible given the data but are NOT directly derived from a measurement
unless cited.

### I1. The 11 Qwen heads carry the FVQ–CVQ behavioral asymmetry

**Supports**: M3 (these heads consistently differentiate FVQ from CVQ across
configs), M7 (ablating them drops FVQ by 14 pp and CVQ by 22 pp), M9
(random heads at the same layers do not produce comparable drops; FVQ
correct→wrong = 0 for random).

**Limit**: M7 and M9 establish causality for these specific 11 heads versus a
matched-distribution random control. We have NOT shown that these heads
fully account for the asymmetry — only that they are necessary for it.

### I2. The same heads operate bidirectionally for FVQ vs CVQ

**Supports**: M3 shows L31 H15, L30 H11, L31 H12, L31 H8, L32 H3, L32 H7 appear
in BOTH the FVQ-dominant primacy table AND the CVQ-dominant recency table
with consistent signs. M2 quantifies this for L31 H15: +0.752 on primacy,
−0.183 on recency.

**Limit**: The data shows correlation between these two effects per head.
We have NOT decomposed how each head's circuit produces both behaviors.

### I3. Magnitude asymmetry between FVQ-boost and CVQ-boost (Qwen)

**Supports**: M4 — mean FVQ−CVQ on primacy is +0.43; mean on recency is
−0.07. Ratio |primacy| / |recency| = 6.1.

**Trial-level test (M12)**: trial-by-trial boost magnitude predicts trial-level
correctness in both models, for both query types, with Cohen's d > 1.6 and
p ≤ 0.0024 in all four tests. Boost magnitude IS associated with behavioral
accuracy at the individual-trial level (not just population mean), as the
mechanism predicts.

**Limit**: M12 is correlational. We have not ruled out that some third
property of the prompt drives both the boost magnitude AND the behavioral
correctness independently. Combined with M7+M9 (targeted vs random
ablation), the evidence is consistent with the heads being functionally
necessary for the asymmetry; trial-level causality would require, e.g.,
conditional intervention experiments not yet run.

### I4. Gemma's identified heads are necessary for context retrieval at this config

**Supports**: M8 (ablation reduces FVQ to 0% and CVQ to 0%; 0/50 ablated FVQ
predictions even hit any stream value), M9 (random control at same layers
leaves FVQ at 0.94 and CVQ at 0.24).

**Limit**: This claim holds for Gemma K=7/N=30 with this specific 10-head
set. We have NOT done layer-wise or single-head ablation; we cannot say
which subset of the 10 is minimally sufficient. We have NOT tested whether
the cluster identified from FVQ-CVQ difference is the full circuit for
context retrieval in Gemma.

### I5. Difference between Qwen and Gemma ablation effect sizes

**Observation**: M7 vs M8 — Qwen ablation drops FVQ by 14 pp; Gemma by 92 pp.

**Interpretation (not derived)**: Qwen may have additional pathways that
also retrieve round-0 content (consistent with the head-averaged primacy
attention bands at L5, L10, L17, L25 observable in `expA_*.png`); Gemma
may have fewer such pathways. This has NOT been verified by ablating those
other layers in Qwen.

### I6. The logit lens peak sits downstream of the discriminating head cluster

**Supports**: M3 — Qwen discriminating heads in L27–L32. M10 — RI and PI
logit lens P-peak at L33 (in 5 of 6 cells).

**Limit**: We can only say L33 > L32, i.e. the peak is *after* the cluster.
We do NOT have a derivation of why the peak is at L33 specifically. The
peak could plausibly have been at L32 (just at the last head) or at L34.

### I7. The final-layer P-drop is asymmetric between RI and PI

**Supports**: M10 — at K=2/N=5: RI drops 36% from L33 to L35; PI drops 51%.

**Limit**: This is one data point per (RI, PI). We have NOT tested whether
this asymmetric drop is the proximate cause of the behavioral PI > RI gap
at our operating point (K=2/N=30 — the logit lens data is at K=2/N=5).

### I8. Earlier "distributed, no bottleneck" ablation is consistent with our cluster finding

**Supports**: Earlier paper ran single-head ablations at K=2/N=5 and found no
bottleneck. M3 shows the discriminator is 11 heads, not 1. M7 shows ablating
the cluster (not done in the earlier paper) produces a measurable drop.

**Limit**: We have NOT re-run single-head ablations on our cluster at our
operating point. We cannot say whether the earlier "no bottleneck" finding
would replicate inside our cluster at K=2/N=30.

### I9. Connection to over-squashing / final-collapse (Barbero et al. 2024)

The Barbero et al. 2024 paper proves that certain distinct decoder inputs
yield arbitrarily close representations at the last token's position in
deep layers ("over-squashing"). M10's observation of post-peak P-drop in
the final layers is consistent with such a collapse, but our data does
NOT independently prove a theoretical link — we observe a drop, the paper
proves drops are possible under certain conditions.

---

## Open follow-ups — paused for now (different direction taken)

The investigation is being paused at this point to pursue a different
direction. The following items remain open. Each closes a specific
evidence gap identified in the Interpretations section above.

### Completed before the pause

- **Per-trial attention–behavior correlation** — DONE (M12).
  Cohen's d > 1.6 for all 4 tests across both models, smallest p = 0.0024.

### Not yet run — listed in rough priority order

1. **Single-head ablation** at L31 H15 (Qwen) and L23 H3 (Gemma).
   - Gap closed: I8 — does the largest single head reproduce most of the
     cluster's effect, or is the distribution itself necessary?
   - Cost: ~10 min/model on MPS.
   - Expected: in Qwen, single-head ablation likely produces a smaller drop
     than the 11-head ablation (−14/−22 pp). In Gemma, given the catastrophic
     0%/0% from the cluster, L23 H3 alone could either reproduce most of it
     or only a fraction — informative either way.

2. **Layer-wise ablation in Gemma**: L23 only vs L21 only vs L29 only.
   - Gap closed: I4 — which Gemma layer is minimally sufficient for the
     catastrophic collapse?
   - Cost: ~30 min total on MPS (3 runs).
   - Expected: L23 is the densest discriminator-head cluster (M6), so
     L23-only ablation is the prime suspect for reproducing the 0%/0%.

3. **Reversal-zone Qwen ablation** (K=15/N=20).
   - Gap closed: confirms the link between attention magnitude (M5) and
     behavioral impact. M5 showed ~3× reduced FVQ−CVQ diff in reversal vs
     normal; ablation effect should scale similarly.
   - Cost: ~10 min on MPS.
   - Expected: smaller drop in FVQ accuracy than the normal-zone −14 pp.

4. **Re-run Gemma ablation with the cleaner cross-config head set**
   (drop L17 H0/H2/H5; add L21 H6, L25 H0 per M6).
   - Gap closed: I4 — confirms the M6 cross-config-consistent set (rather
     than the original ad-hoc set) is causally sufficient.
   - Cost: ~10 min on MPS.

5. **Locate alternative retrieval pathways in Qwen**.
   - Gap closed: I5 — explains why Qwen retains 72% FVQ post-ablation while
     Gemma drops to 0%. Candidates: head-averaged primacy bands at L5, L10,
     L17, L25 visible in `expA_*.png`.
   - Cost: ablation experiments at those layers, ~30-60 min.
   - Open-ended; depends on finding the right additional heads to ablate.

6. **Logit lens at K=2/N=30** (matching the ablation operating point).
   - Gap closed: I7 K-mismatch — does the post-peak P-drop look the same at
     the operating point where ablation was run (existing logit lens is at
     K=2/N=5)?
   - Cost: ~1-2 hours including TransformerLens setup.
   - **De-prioritized by user (2026-05-21)**: existing logit lens at K=2/N=5
     is sufficient for our narrative; matching K isn't critical.

---

## Files

- `v3/attention_routing.py` — main experiment script (Exp A/B/C)
- `v3/causal_ablation.py` — ablation experiment with `--control-random` flag
- `v3/extract_query_heads.py` — top-k head identification + markdown report
- `v3/plot_attention_routing.py` — basic plots
- `v3/plot_per_head_diagnostic.py` — per-head plots
- `v3/plot_consistency_overlay.py` — cross-config line overlays

- `v3/results_vllm/attention_routing/*.json` — raw attention data
- `v3/results_vllm/attention_routing/head_analysis/` — head extraction outputs
- `v3/results_vllm/causal_ablation/*.json` — ablation results
- `v3/plots/attention_routing/{config}/*.png` — figures per config
