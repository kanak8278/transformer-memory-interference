# Qwen2.5-1.5B-Instruct Experiment Log

## Model Info
- **Model**: Qwen/Qwen2.5-1.5B-Instruct
- **Params**: 1.54B
- **Architecture**: 28 layers, 12 heads (GQA 12Q/2KV), d_model=1536
- **Data setup**: Single-token English values (2300 words) + 46 semantic categories
- **Purpose**: Cross-model validation of 0.5B findings

## Operating Points (Regime-Matched to 0.5B)

The 1.5B model reaches the same interference regimes at fewer updates than 0.5B.
PI collapse boundary shifts left — less interference needed to break PI on a larger model.
(Consistent with ACL paper: RI scales with size R²=0.49, PI does not R²=0.06)

| Regime | 0.5B config | 0.5B RI/PI | **1.5B config** | **1.5B RI/PI** | Description |
|--------|-------------|------------|-----------------|----------------|-------------|
| A (baseline) | 1k, 2u | 53%/47% | **1k, 2u** | 47%/47% | Both work, minimal gap |
| B (PI cracking) | 1k, 5u | 60%/17% | **1k, 3u** | 60%/17% | Clear asymmetry |
| C (PI collapsed) | 1k, 15u | 57%/10% | **1k, 5u** | 70%/13% | Deep PI failure |
| D (multi-key) | 2k, 5u | 43%/20% | **2k, 3u** | 47%/23% | Multi-key validation |

Source: recalibration sweep (`11c_recalibrate_sweep.py --model Qwen/Qwen2.5-1.5B-Instruct`)
Results: `results/Qwen2.5-1.5B-Instruct/0k_0u/recalibrate_sweep.json`

## Trial Strategy
- 100 trials per condition per experiment
- Same experiment suite as 0.5B (exp 12-23)

## Experiment Status

| Exp | Name | A (1k,2u) | B (1k,3u) | C (1k,5u) | D (2k,3u) |
|-----|------|-----------|-----------|-----------|-----------|
| 12 | Logit lens + DLA | DONE | DONE | DONE | DONE |
| 13 | Positional gradient | DONE | DONE | DONE | DONE |
| 14 | PI mass distribution | DONE | DONE | DONE | DONE |
| 15 | Activation patching | DONE | DONE | DONE | DONE |
| 16 | Head identification | DONE | DONE | DONE | DONE |
| 17 | Instruction sensitivity | DONE | DONE | DONE | DONE |
| 18 | Forced attention | DONE | DONE | DONE | DONE |
| 19 | Positional bias sweep | DONE | DONE | DONE | DONE |
| 19b | Bias attention proof | DONE | DONE | DONE | DONE |
| 20 | Minority override | DONE | DONE | DONE | DONE |
| 22 | Query patching granular | DONE | DONE | DONE | DONE |
| 23 | Ablation patching interaction | DONE | DONE | DONE | DONE |

## Key Results (Headline Numbers)

### Exp 12: Logit Lens
| Point | RI acc | PI acc | RI P(init) | PI P(final) |
|-------|--------|--------|------------|-------------|
| A (1k,2u) | 61% | 26% | 0.995 | 0.511 |
| B (1k,3u) | 49% | 20% | 0.989 | 0.345 |
| C (1k,5u) | 55% | 21% | 0.984 | 0.402 |
| D (2k,3u) | 61% | 6% | 0.978 | 0.091 |

### Exp 18: Forced Attention
- Points A, B, C: Forcing attention does NOT help PI (unlike 0.5B!)
- Point D: Marginal improvement
- Interpretation: On 1.5B, the OV circuit is ALSO part of the problem, not just QK

### Exp 22: Query Patching
- Point C (1k,5u): Late-layer patching works (+96%) — consistent with 0.5B
- Points A, B, D: Mixed/weak signal — needs investigation

### Notable Differences from 0.5B
1. **Forced attention doesn't help PI on 1.5B** — suggests OV circuit involvement at scale
2. **P(init) is much higher on 1.5B** (0.98-0.99 vs 0.56-0.87) — stronger RI representation
3. **Query patching shows negative recovery at L18** — different layer dynamics (28 vs 24 layers)
4. **1.5B reaches same regimes with fewer updates** — PI collapse boundary shifts left

## Exp 25a/25b/25c: Comprehensive Head Identification (2026-02-25)

Ran 3 independent methods at Point B (1k, 3u), 100 PI trials.

### 25a — Per-Head Causal Knockout (Gold Standard)

Baseline PI: 44%, mean_ld=2.48

| Rank | Head | Δld | KO accuracy |
|------|------|-----|-------------|
| 1 | **L8H3** | **+8.3** | **92%** |
| 2 | L0H7 | +3.5 | 75% |
| 3 | L0H3 | +3.1 | 66% |
| 4 | L0H11 | +2.9 | 64% |
| 5 | L0H1 | +2.8 | 60% |

Top retrieval: L16H7 (-3.9, KO→19%), L8H11 (-2.9, KO→21%), L15H7 (-2.2, KO→34%).

**Key finding: L8H3 is the dominant primacy head.** Knockout boosts PI from 44% to 92% (+48%). Massive single-head effect in a very early layer.

**4 of top 5 primacy heads are in L0.** The earliest layer has strong primacy-causing heads. This suggests the primacy bias is set up at the very beginning of the forward pass.

**Old L19H1 was wrong.** Not in top 15. The exp 16 attention-based classification was completely off.

### 25b — Attribution Patching

Top retrieval: L22H7 (+3.5), L26H1 (+1.3), L24H10 (+1.3) — all late layer.
Top primacy: L24H8 (-0.9), L25H4 (-0.9) — weak, late layer.

**25b completely misses L8H3.** Same failure mode as Gemma: first-order approximation can't capture indirect effects of early-layer heads propagating through 20+ downstream layers.

### 25c — Observational Metrics

Copy score ALL ZERO. DLA top: L26H7 (+2.3), L27H1 (+1.1). Retrieval by attention: L22H7 (0.55), L19H3 (0.45).

L8H3 does not appear in any observational top-10 list.

### Comparison to Old Exp 16 / Exp 18 Results

| | Old (exp 16/18) | New (exp 25a) |
|---|---|---|
| Primary primacy head | L19H1 (pi_primacy>0.7) | **L8H3** (Δld=+8.3, very early) |
| Location | Late layers (L19) | **L0-L8** (very early) |
| Forced attention result | "OV broken" | Likely forced wrong head (L19H1) |
| Single-head knockout | Not tested on L8H3 | **44%→92% PI** |

The forced attention confusion (exp 18: "OV broken on 1.5B") is now explained: we forced L19H1 which has negligible causal impact. The real primacy head (L8H3) was never tested.

### Cross-Model Pattern (25a gold standard)

| | Gemma 1B | Qwen 0.5B | **Qwen 1.5B** |
|---|---|---|---|
| Top primacy head | L14H2 (Δld=+20.2) | L12H0 (Δld=+1.8) | **L8H3** (Δld=+8.3) |
| KO effect | 17%→86% | 69%→70% | **44%→92%** |
| Distribution | Concentrated | Distributed | **Concentrated** |
| Location | L4-L14 | L3-L19 (spread) | **L0-L8 (very early)** |

Gemma and 1.5B share concentrated primacy bias (one dominant head). 0.5B is distributed. Larger models may concentrate primacy in fewer, earlier heads.

## Exp 17b/18b: Re-running with Causally-Identified Heads (2026-02-25)

Using L8H3 (from exp 25a) instead of L19H1 (from old exp 16).

### 17b — Instruction Sensitivity (L8H3 + top 4 primacy heads)

All 5 heads ignore the query word:

| Head | Attn to query word | Main target | Init attn | Final attn |
|------|-------------------|-------------|-----------|------------|
| **L8H3** | 1.84% | Instruction (59%) | 0.55% | 0.85% |
| L0H7 | 0.00% | Instruction (97%) | 0.00% | 0.00% |
| L0H3 | 0.00% | Instruction (97%) | 0.00% | 0.00% |
| L0H11 | 0.00% | Instruction (100%) | 0.00% | 0.00% |
| L0H1 | 0.00% | Instruction (96%) | 0.00% | 0.00% |

**L0 heads put literally 0.00% attention on values.** They attend exclusively to instruction/BOS tokens. This supports H1 (positional bias in weights) — they're injecting position-dependent signals from raw embeddings, not reading values at all.

**L8H3** puts 0.55% on init, 0.85% on final — slightly MORE attention to final (not primacy-biased by attention). Like Gemma's L14H2, its damage is through output corruption, not attention routing.

### 18b — Forced Attention (L8H3 only)

| Config | RI | PI | Gap | PI logit_diff |
|--------|----|----|-----|---------------|
| Baseline | 56% | 20% | +36% | +1.80 |
| **Knockout** | **24%** | **47%** | **-23%** | **+10.08** |
| Force correct | 56% | 21% | +35% | +2.04 |
| Force wrong | 55% | 21% | +34% | +2.04 |

**Mechanism: B (pure indirect corruption)**

- Forcing to correct position: PI stays at 21% — NO EFFECT.
- Forcing to wrong position: PI stays at 21% — NO EFFECT.
- Knockout: PI jumps 20% → 47% (+27%).
- **Attention direction is completely irrelevant for L8H3.** It doesn't matter WHERE this head looks — its damage is entirely through what it writes to the residual stream.
- Knockout HURTS RI: 56% → 24%. Head is essential for RI.

**This is the cleanest demonstration of mechanism B.** The head helps RI and hurts PI through the same output pathway, regardless of attention. It's an instruction-insensitive value processor that always promotes early-position information.

## Exp 25d: Knockout Validation — All 3 Tests Passed (2026-02-26)

Three-way validation of top 3 primacy heads (L8H3, L0H7, L0H3) at primary point (1k,3u) and secondary point (1k,5u).

### V1: RI + PI Knockout (200 trials, primary point)

| Head | RI base→KO | RI Δld | PI base→KO | PI Δld | Verdict |
|------|-----------|--------|-----------|--------|---------|
| **L8H3** | 100%→40% | -12.77 | 38%→91% | +8.38 | **PRIMACY** (helps PI, hurts RI) |
| **L0H7** | 100%→100% | -1.05 | 38%→72% | +3.57 | **PRIMACY** (helps PI, hurts RI) |
| L0H3 | 100%→98% | +0.05 | 38%→68% | +3.21 | PI-SPECIFIC (helps PI, RI neutral) |

**H3 (general capability loss) rejected for L8H3 and L0H7:** If knockout caused general degradation, RI would drop AND PI would drop. Instead RI drops and PI rises — effect is primacy-specific. L0H3 is different: it helps PI but barely touches RI — may be a PI-specific head rather than a dual-purpose primacy head.

### V2: Full Sweep at Secondary Point (50 trials, 1k,5u)

Baseline PI at secondary: 34%, mean_ld=4.65

| Rank | Head | Δld | KO acc | Note |
|------|------|-----|--------|------|
| 1 | **L8H3** | +7.494 | 60% | TARGET — rank 1/336 |
| 2 | **L0H7** | +4.498 | 58% | TARGET — rank 2/336 |
| 3 | **L0H3** | +3.730 | 54% | TARGET — rank 3/336 |
| 4 | L0H1 | +3.686 | 48% | |
| 5 | L0H11 | +3.249 | 44% | |

**All 3 target heads rank #1, #2, #3 at the secondary point.** Perfect stability across operating points.

### V3: 200-Trial CIs (Wilson score interval)

| Head | RI KO acc [CI] | PI KO acc [CI] |
|------|---------------|---------------|
| L8H3 | 40% [33%-46%] (baseline 100%) | **91% [86%-94%]** (baseline 38%) |
| L0H7 | 100% [97%-100%] (baseline 100%) | **72% [66%-78%]** (baseline 38%) |
| L0H3 | 98% [96%-99%] (baseline 100%) | **68% [61%-74%]** (baseline 38%) |

All PI CIs are well above baseline. No overlap with chance.

### Comparison to Gemma 25d Validation

| | Gemma 1B | **Qwen 1.5B** |
|---|---|---|
| Top head | L14H2 | **L8H3** |
| V1 (primacy-specific?) | Yes — all 3 heads | **Yes — L8H3 and L0H7** |
| V2 (stable across points?) | All rank top 6 | **All rank top 3/336** |
| V3 (tight CIs?) | PI=86% [81-91%] | **PI=91% [86-94%]** |
| New finding | — | **L0H3 is PI-specific (doesn't hurt RI)** |

**L0H3 is a new head type** not seen on Gemma: it helps PI when knocked out but doesn't hurt RI. This means it contributes to primacy bias without being essential for recency retrieval. The primacy circuit has at least two types of components: (1) dual-purpose heads that help RI and hurt PI (L8H3, L0H7), and (2) PI-specific heads that only hurt PI (L0H3).

### Updated Answers to Key Questions

1. ~~Does the same core circuit appear?~~ **ANSWERED: No. Primacy heads are model-specific and in early layers.**
2. Is the primacy cliff equally sharp? → Yes (all models show sharp RI cliff)
3. ~~Does forced attention fix PI?~~ → **ANSWERED: NO for 1.5B. Pure mechanism B — forcing attention has zero effect. Head damage is through output, not routing.**
4. Does late-layer query patching recover PI? → Yes at Point C, mixed elsewhere
5. ~~Are the DLA top heads condition-flipping?~~ → **DLA doesn't identify the real primacy heads. Causal knockout is needed.**
6. **NEW: Why are L0 heads in the primacy list?** → They attend 96-100% to instruction/BOS tokens. Likely positional bias in weight matrices (H1) — inject early-position signal from raw embeddings into residual stream at first layer, biasing all downstream computation.
