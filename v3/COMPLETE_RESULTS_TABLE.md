# Complete Results Table — All Models, All Stages

## Behavioral Results (Stage 1)

### Local Models at Matched Operating Points

| Model | Arch | Params | Config | RI% | PI% | Gap | Garbage |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | Qwen | 494M | 2k_5u | 82 | 34 | 48 | 9% |
| Qwen2.5-0.5B | Qwen | 494M | 2k_10u | 66 | 28 | 38 | 23% |
| Qwen2.5-1.5B | Qwen | 1.5B | 2k_5u | 100 | 56 | 44 | 0% |
| Qwen2.5-1.5B | Qwen | 1.5B | 2k_15u | 100 | 20 | 80 | 1% |
| Qwen2.5-1.5B | Qwen | 1.5B | 3k_40u | 92 | 6 | 86 | 3% |
| Qwen2.5-3B | Qwen | 3B | 2k_5u | 79 | 59 | 20 | 0% |
| Qwen2.5-3B | Qwen | 3B | 2k_15u | 76 | 23 | 53 | 3% |
| Qwen2.5-3B | Qwen | 3B | 5k_10u | 74 | 26 | 48 | 1% |
| Gemma-3-1B | Gemma | 1B | 2k_5u | 80 | 10 | 70 | 2% |
| Gemma-3-1B | Gemma | 1B | 2k_10u | 74 | 10 | 64 | 11% |
| Gemma-3-1B | Gemma | 1B | 3k_10u | 66 | 2 | 64 | 15% |
| Pythia-410M | GPT-NeoX | 410M | 2k_5u | 76 | 18 | 58 | 21% |
| Pythia-410M | GPT-NeoX | 410M | 2k_7u | 74 | 8 | 66 | 22% |

### API Models at High N

| Model | Config | RI% | PI% | Gap |
|---|---|---|---|---|
| Claude Haiku | 5k_100u | 100 | 65 | 35 |
| Claude Haiku | 10k_50u | 100 | 40 | 60 |
| Claude Haiku | 10k_100u | 100 | 10 | 90 |
| GPT-4.1-mini | 10k_100u | 100 | 65 | 35 |
| GPT-4.1-mini | 10k_200u | 100 | 75 | 25 |

## Logit Lens Results (Stage 2)

### Suppression Pattern: v_last Found Then Outcompeted

| Model | Layers | Op Point | Peak P(v_last) | Peak Layer | Final P(v_last) | Suppress | P(v_penult) | RI P(v_first) |
|---|---|---|---|---|---|---|---|---|
| Qwen 0.5B | 24 | 2k_5u | 0.236 | L21 (88%) | 0.038 | 0.199 | 0.293 | 0.917 |
| Qwen 0.5B | 24 | 2k_7u | 0.130 | L22 (92%) | 0.054 | 0.077 | 0.196 | 0.946 |
| Qwen 1.5B | 28 | 2k_5u | 0.140 | L26 (93%) | 0.014 | 0.125 | 0.302 | 1.000 |
| Qwen 1.5B | 28 | 2k_10u | 0.078 | L26 (93%) | 0.007 | 0.071 | 0.050 | 1.000 |
| Qwen 1.5B | 28 | 3k_10u | 0.054 | L24 (86%) | 0.000 | 0.054 | 0.130 | 0.997 |
| Qwen 3B | 36 | 5k_3u | 0.207 | L32 (89%) | 0.013 | 0.194 | 0.841 | 0.998 |
| Qwen 3B | 36 | 7k_5u | 0.079 | L32 (89%) | 0.011 | 0.067 | 0.410 | 0.999 |
| Gemma 1B | 26 | 2k_5u | 0.025 | L20 (77%) | 0.000 | 0.025 | 0.111 | 1.000 |

### Key Finding: Value Emergence in Last ~15% of Layers

All models show:
1. Zero signal for first 80-85% of layers
2. Values emerge at ~85-93% depth
3. P(v_first) rises monotonically to 0.92-1.00 (RI)
4. P(v_last) peaks weakly then gets suppressed by competitor (PI)

## Causal Results (Stage 3)

### Attribution Patching — Where Are the Important Heads?

| Model | Top 3 Attribution Heads | Relative Depth | N Heads/Layer |
|---|---|---|---|
| Qwen 1.5B | L21H6 (22.8), L18H5 (22.0), L18H0 (21.8) | 64-75% | 12 |
| Qwen 3B | L26H5 (30.4), L27H3 (26.3), L30H3 (26.1) | 72-83% | 16 |
| Gemma 1B | L11H3 (172), L11H0 (143), L11H2 (132) | 42% | 4 |

### Targeted Patching — RI→PI Head Restoration

| Model | Top Head | Delta P(v_last) | Positive Rate | N Heads/Layer |
|---|---|---|---|---|
| Qwen 1.5B | L19H6 | +0.038 | 92% | 12 |
| Qwen 3B | L26H3 | +0.032 | 64% | 16 |
| Gemma 1B | L15H2 | +0.071 | 82% | 4 |

### Ablation — Top-5 Head Removal

| Model | Normal P(v_last) | Ablated P(v_last) | Delta | Direction |
|---|---|---|---|---|
| Qwen 1.5B | 0.130 | 0.107 | -0.023 | Heads HELP |
| Qwen 3B | 0.308 | 0.264 | -0.044 | Heads HELP |
| Gemma 1B | 0.020 | 0.000 | -0.020 | Heads HELP |

### Key Finding: Distributed Mechanism

- No single head causes PI failure
- Fewer heads → larger per-head effects (Gemma 4 heads → 0.071 delta)
- Ablating important heads REDUCES P(v_last) — they help retrieval
- The mechanism is architectural, not a learned suppression circuit

## Error Position Analysis

### N-Dependent Error Distribution (Qwen 3B)

| N (updates) | Penultimate % | Near-last (>0.75) | First (v_0) | Pattern |
|---|---|---|---|---|
| 3 | 85% | 85% | 1% | Off-by-one |
| 5 | 61% | 61% | 0% | Off-by-one |
| 10 | 19% | 34% | 1% | Spreading |
| 15 | 17% | 35% | 0% | Diffuse |
| 20 | 10% | 19% | 1% | Very diffuse |

### Architecture-Dependent Failure Mode

| Model | Mean Error Pos | Top Error Zone | Primary Mode |
|---|---|---|---|
| Qwen 0.5B | 0.47 | Mixed | Moderate near-last |
| Qwen 1.5B | 0.61 | 0.8-1.0 (33%) | Near-last (recency imprecision) |
| Qwen 3B | 0.48 | Uniform | N-dependent |
| Gemma 1B | 0.24 | 0.0-0.2 (59%) | Primacy default |

Gemma defaults to primacy (first binding wins) while Qwen shows recency imprecision (tries for last but lands nearby). Different failure modes, same PI > RI outcome.
