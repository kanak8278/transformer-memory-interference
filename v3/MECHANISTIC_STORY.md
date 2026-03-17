# Mechanistic Story: Why Transformers Remember First, Forget Last

## The Puzzle

When LLMs see a key-value stream where a key gets updated N times:
- **RI (ask for first):** Near-perfect recall (~95-100% for capable models)
- **PI (ask for last):** Rapidly degrades with N (100% → 20% → 10%)

This asymmetry (PI > RI) appears across 39+ models, 3+ architectures, and persists from 0.5B to frontier scale.

## The Error Pattern (New V3 Finding)

The error pattern depends on BOTH N (update count) AND architecture:

**For Qwen models:** PI failures are primarily near-last (recency imprecision).
**For Gemma:** PI failures show significant primacy intrusion (59% at positions 0-0.2).
**For all models:** Primacy intrusion is secondary to the overall PI > RI gap.

The error distribution within Qwen is N-dependent:

| N (updates) | Where errors land | Interpretation |
|---|---|---|
| 3-5 | 61-85% at penultimate (v_{N-2}) | Off-by-one: positional confusion between adjacent late positions |
| 10 | Spread across positions, penultimate still mode (~19%) | Degrading positional discrimination |
| 15-20 | Nearly uniform distribution, weak penultimate peak | Complete positional addressing breakdown |

**This reveals two mechanisms operating at different scales:**

1. **Low N (off-by-one):** Model retrieves from the right region but can't distinguish v_{N-1} from v_{N-2}. RoPE encodes nearby positions nearly identically.

2. **High N (diffuse failure):** Model can't address ANY specific late position. Softmax attention becomes too dispersed over many key-value tokens.

## Mechanistic Evidence (Qwen2.5-3B Logit Lens)

### The Core Asymmetry (Fig 5 — money figure)

**RI retrieval trajectory:**
```
L0-30: P(v_first) ≈ 0 (no signal)
L31:   P(v_first) = 0.03
L32:   P(v_first) = 0.52
L33:   P(v_first) = 0.86
L34:   P(v_first) = 0.95
L35:   P(v_first) = 0.998 ← clean monotonic rise to certainty
```

**PI failure trajectory:**
```
L0-30: P(v_last) ≈ 0 (no signal)
L31:   P(v_last) = 0.03
L32:   P(v_last) = 0.21 ← PEAKS here
L33:   P(v_last) = 0.12 ← starts losing to competitor
L34:   P(v_last) = 0.08
L35:   P(v_last) = 0.01 ← crushed

Meanwhile P(v_penultimate):
L32:   0.52
L33:   0.73
L34:   0.83
L35:   0.84 ← wins decisively
```

**Key observation:** v_last IS found at L32 (P=0.21). The model does encode the correct value. But it immediately loses competition to v_penultimate in L33-35. For RI, v_first faces no such competition — it rises cleanly to 0.998.

### Why v_first Wins: Three Compounding Mechanisms

1. **Iterative attention path advantage** (Wu et al., ICML 2025)
   - Position 0 is attended by ALL subsequent positions at EVERY layer
   - Across L layers, early positions accumulate exponentially more paths
   - v_first gets L opportunities to be reinforced; v_last gets only 1

2. **Positional encoding smooth decay** (RoPE analysis)
   - RoPE encodes relative position as smooth rotation
   - d(pos_{N-1}, query) ≈ d(pos_{N-2}, query) → nearly identical keys
   - d(pos_0, query) is unique — no competitor has similar distance
   - As N grows, more competitors crowd the "near-last" region

3. **Attention sink amplification** (Barbero et al., 2025)
   - First token becomes attention sink — receives disproportionate attention
   - This is a learned mechanism to prevent over-mixing
   - Stronger in larger models (78% for 405B vs 46% for 8B)
   - Explains why RI resistance scales with model size (R²=0.49)

### Why the Mechanism is Distributed (Stage 3)

Attribution patching on 3B finds:
- Top head: L26H3 with ΔP(v_last) = +0.032
- Top 5 heads combined: ~0.085 total delta
- No single head causes or prevents PI failure

Ablating top 5 heads reduces P(v_last) from 0.308 → 0.264 (heads HELP retrieval, don't suppress it).

**Interpretation:** PI > RI is not a "bug" in a few heads — it's an architectural property. The entire positional addressing mechanism collectively fails at late positions. No targeted intervention can fix it because the issue is in the computation graph itself (causal masking + positional encoding + softmax dispersion).

## Predictions

1. RI accuracy is robust to N; PI accuracy degrades monotonically with N ✓
2. PI failures cluster at penultimate (off-by-one) for small N ✓
3. PI failures become diffuse for large N ✓
4. RI resistance scales with model size; PI does not ✓
5. Models WITHOUT causal attention (SSMs) should show different pattern → TODO
6. The suppression mechanism is distributed, not concentrated ✓

## What This Adds Over Prior Work

| Prior work | Their claim | Our contribution |
|---|---|---|
| Wu et al. (ICML 2025) | Iterative attention creates first-position bias | We show this manifests as PI > RI in competing key-value retrieval |
| Barbero et al. (2025) | First token is attention sink | We show this scales with model size AND correlates with RI resistance |
| Veličković et al. (ICML 2025) | Softmax dispersion causes representation collapse | We show this creates the N-dependent transition from off-by-one to diffuse failure |
| Liu et al. (2023) | Lost in the middle | We provide the mechanistic explanation AND show the error is position-dependent, not uniform |
| Our ACL paper | PI > RI across 39 models | We now explain WHY: layer trajectories, error positions, causal analysis |

## Paper Structure

1. **Behavioral:** PI > RI across models (ACL data + new 1.5B/3B/4B validation)
2. **Error characterization:** N-dependent failure pattern (off-by-one → diffuse)
3. **Logit lens:** Layer trajectories showing v_last found then suppressed
4. **Causal analysis:** Mechanism is distributed (no suppression heads)
5. **Theory:** Three compounding mechanisms predict all observations
6. **Architectural control:** SSM prediction (TODO)
