# Paper Results Section Draft

## Section 4: Behavioral Evidence — PI > RI Across Scales

### 4.1 Regime Map

We characterize each (keys, updates) configuration as one of four regimes:
- **A:** Both RI and PI succeed (>80%)
- **B:** RI succeeds (>60%), PI fails (<30%) — the interference zone
- **C:** PI > RI gap is clear but both degrade
- **D:** Both fail (<30%)

[Figure 1: Regime maps for Qwen-0.5B, Qwen-1.5B, Qwen-3B side by side]

Key observations:
1. As N (updates) increases, models transition from A → B → C
2. RI accuracy degrades slowly (stays >60% even at high N)
3. PI accuracy collapses rapidly (100% → 10% as N increases)
4. The critical N for PI collapse is model-size dependent: lower for 0.5B (~5-7), higher for 3B (~15-30), highest for Haiku (~50-100)

### 4.2 Error Characterization

[Figure 2: PI failure position distribution for 3 models × 3 N-levels]

When PI fails, where does the model's answer land?

**Finding 1: PI errors are NOT primacy intrusion.** Fewer than 5% of errors land on v_0 (the first value). This contradicts the simple "primacy bias" explanation.

**Finding 2: Error distribution depends on N:**
- At N=5: 61-85% of errors land on the penultimate value (v_{N-2}). Classic off-by-one.
- At N=10: Penultimate still modal (19-36%) but errors spread across positions.
- At N=20: Nearly uniform, with penultimate as weak mode (~10-21%).

**Finding 3: This reveals two compounding mechanisms:**
1. **Positional encoding precision** (dominates at low N): RoPE cannot distinguish adjacent late positions
2. **Softmax dispersion** (dominates at high N): Attention becomes too diffuse to address any specific late position

### 4.3 Cross-Model Consistency

[Table: RI, PI, Gap at matched difficulty points for 0.5B, 1.5B, 3B, Haiku, GPT-4.1-mini]

PI > RI appears at EVERY scale tested (0.5B–frontier). The threshold N for PI collapse increases with model size, but the qualitative pattern is identical.

## Section 5: Mechanistic Evidence — Inside the PI Failure

### 5.1 Logit Lens: Value Trajectories

[Figure 3: Layer-by-layer P(v_i) for RI correct vs PI failure, 1.5B and 3B side by side]

**The asymmetry is visible in probability space:**

For RI correct trials:
- P(v_first) emerges at ~85% depth and rises monotonically to ~1.0
- No competing value has significant probability
- Clean, steep sigmoid-like rise

For PI failure trials:
- P(v_last) emerges at ~85% depth, peaks weakly (0.05–0.21)
- P(v_penultimate) rises faster and outcompetes in the final ~5 layers
- P(v_last) is SUPPRESSED: it peaks then decreases (not just fails to grow)

### 5.2 The Suppression Phenomenon

[Figure 4: P(v_last) across layers for different N values — shows peak-then-crash]

At the critical depth (~90% of layers):
- 1.5B: P(v_last) peaks at 0.14 then drops to 0.01
- 3B: P(v_last) peaks at 0.21 then drops to 0.01

**The correct value IS found.** The model partially computes the right answer. But in the final layers, the competitor (usually v_{N-2}) wins. This is active suppression, not failure to encode.

### 5.3 Causal Analysis: No Bottleneck

[Figure 5: Head importance heatmap + patching results for 1.5B and 3B]

**Attribution patching** identifies heads at 60-80% depth (before value emergence) that most influence P(v_last).

**Targeted patching** shows small, distributed effects:
- 1.5B top head: L19H6, ΔP = +0.038 (92% of trials positive)
- 3B top head: L26H3, ΔP = +0.032 (64% of trials positive)

**Ablation** of top-5 heads REDUCES P(v_last):
- 1.5B: 0.130 → 0.107 (-0.023)
- 3B: 0.308 → 0.264 (-0.044)

**Interpretation:** These heads are retrieval helpers, not suppression agents. PI failure is not caused by a learned "interference circuit" that can be targeted. It is an architectural property of how causal attention + positional encoding computes value retrieval.

### 5.4 Cross-Model Scaling

[Table: Mechanistic metrics across 1.5B and 3B]

| Metric | 1.5B (28L) | 3B (36L) |
|---|---|---|
| Value emergence depth | 86-96% | 86-97% |
| P(v_last) peak | 0.14 | 0.21 |
| Final suppression | 0.13 | 0.19 |
| Attribution head depth | 57-79% | 67-83% |
| Max patching delta | 0.038 | 0.032 |

The mechanism is structurally identical across model sizes. The larger model achieves slightly higher peak P(v_last) (0.21 vs 0.14) and stronger penultimate competition (0.84 vs 0.30), but the qualitative story — found then suppressed — is the same.

## Section 6: Theoretical Framework

Three compounding mechanisms explain all observations:

### 6.1 Iterative Attention Path Advantage (Wu et al., ICML 2025)

Position 0 is attended by all subsequent positions at every layer. Across L layers, early positions accumulate exponentially more paths through the attention graph. This creates a structural advantage for v_first that grows with depth.

**Prediction:** RI should be robust to N → ✓ (RI stays >80% even at N=200 for Haiku)

### 6.2 Positional Encoding Precision (RoPE Smooth Decay)

RoPE encodes relative position as smooth rotation. Adjacent positions (N-1 vs N-2) have nearly identical encodings relative to the query position. As N grows, more values crowd the "late" region, making them harder to distinguish.

**Prediction:** PI errors should cluster at penultimate for low N → ✓ (85% at N=5)
**Prediction:** PI errors should diffuse for high N → ✓ (uniform-like at N=20)

### 6.3 Attention Sink Amplification (Barbero et al., 2025)

The first token becomes an attention sink, receiving disproportionate attention. This mechanism scales with model size and reinforces v_first retrieval.

**Prediction:** RI resistance should scale with model size → ✓ (R²=0.49 from ACL paper)
**Prediction:** PI should NOT scale with model size → ✓ (R²=0.06 from ACL paper)

## Section 7: Discussion

### What This Paper Adds

1. **Error characterization:** PI failure is recency imprecision (off-by-one → diffuse), not primacy intrusion
2. **Logit lens evidence:** The correct value IS found at ~90% depth but loses competition in final layers
3. **Causal analysis:** No bottleneck heads — mechanism is architectural, not a learned circuit
4. **Theoretical unification:** Three published mechanisms (path advantage, RoPE precision, attention sink) jointly explain the PI > RI asymmetry

### Limitations

1. All mechanistic models are from the Qwen family — cross-architecture validation needed
2. No SSM comparison yet (Mamba/RWKV prediction untested)
3. Logit lens assumes linear representation hypothesis
4. Stage 3 patching uses RI→PI direction, not matched-difficulty PI→PI
