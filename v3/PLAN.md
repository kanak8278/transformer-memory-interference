# V3 Experiment Plan: Positional Addressing Asymmetry in Transformers

Started: 2026-03-02

---

## Why V3 — What Changed

V2 experiments were built around the assumption that **PI failure = primacy intrusion**
(model outputs the first value when asked for the last). This shaped everything: the
logit_diff metric (logit(final) - logit(init)), the DLA framing ("which heads promote
init?"), the "primacy heads" naming, and the paper narrative.

**What the data actually shows (across 4 API models + 4 open-weight models):**

- PI failures cluster at position **0.74–0.89** (near-last), not position 0 (first)
- Primacy intrusion barely happens: 2–5% of failures on large models
- The model is **trying** to retrieve a recent value — it lands 1–2 positions early
- Dominant error type: intermediate_intrusion (near-last) + garbage
- High garbage rate (30–50%): model outputs tokens not in the value stream at all

**The real phenomenon: recency imprecision, not primacy promotion.**

The model can reliably retrieve position 0 (RI is easy) but cannot precisely address
position N-1 (PI is hard). It gets the direction right — attending toward the end — but
lands on v_{N-2} or v_{N-3} instead of v_{N-1}.

---

## Theoretical Framing

### Core Claim

**Causal attention creates a structural asymmetry in positional addressing: early
positions are uniquely identifiable, late positions are mutually confusable.**

### Mechanism (two compounding effects)

**1. Iterative attention compounds first-position advantage**
(Wu et al., ICML 2025 — "On the Emergence of Position Bias in Transformers")

In a multi-layer causal transformer:

- Position 0 is attended by all subsequent positions at every layer
- Across L layers, this compounds: early positions get exponentially more "paths"
  through the attention graph
- Late positions have fewer, shorter paths → less contextualized representations
- The causal mask ALONE (without positional encoding) creates first-position bias

**2. Positional encoding cannot discriminate near-last positions**
(Hopfield/associative memory retrieval bound: β·Δ > log(N-1))

- v_{N-1} and v_{N-2} have similar relative distances from the query position
- RoPE encodes position as smooth decay → nearby positions have near-identical encodings
- Δ (separation between target key and nearest competitor) is small for last vs penultimate
- As N (updates) increases, more near-last candidates compete → retrieval degrades

**3. First-token attention sink reinforces v_0**
(Barbero et al., 2025 — "Why do LLMs attend to the first token?")

- The first token becomes an attention sink — receives disproportionate attention
- This is a learned mechanism to prevent over-mixing, scales with model size
- Stronger attention sink in larger models (78% for 405B vs 46% for 8B)
- Consistent with RI resistance scaling with model size (R²=0.49 from ACL paper)

### Predictions

1. PI accuracy degrades as N increases (more near-last competitors) ✓ confirmed
2. RI accuracy is more robust to N (v_0 stays uniquely identifiable) ✓ confirmed
3. PI failure lands on v_{N-2} not v_0 (closest competitor, not furthest) ✓ confirmed
4. RI resistance scales with model size; PI does not ✓ confirmed (ACL paper R²)
5. Models WITHOUT causal attention (SSMs like Mamba) should show different pattern → TODO

### Key references

- Wu et al. (ICML 2025): "On the Emergence of Position Bias in Transformers"
- Barbero et al. (2025): "Why do LLMs attend to the first token?"
- Liu et al. (2023): "Lost in the Middle"
- Hsieh et al. (ACL 2024): "Found in the Middle"
- ICLR 2025: "When Attention Sink Emerges in Language Models"
- Ramsauer et al. (2021): "Hopfield Networks is All You Need" (attention = associative memory)

---

## Models

| Model | Size | Type | Role | Priority |
|-------|------|------|------|----------|
| Qwen2.5-1.5B-Instruct | 1.5B | Local (TL) | Primary mechanistic target | HIGH |
| Qwen2.5-3B-Instruct | 3B | Local (TL) | Scale validation within family | HIGH |
| Gemma-3-1b-it | 1B | Local (TL) | Cross-family validation | HIGH |
| Pythia-410m | 410M | Local (TL) | Base model control (no SFT) | MEDIUM |
| claude-4.5-haiku | — | API | Large model behavioral validation | DONE |
| gpt-4.1-mini | — | API | Large model behavioral validation | DONE |
| gpt-5-nano | — | API | Large model behavioral validation | DONE |
| gemini-2.5-flash | — | API | Large model behavioral validation | DONE |

Dropped: Qwen 0.5B (too weak, distributed effects, high garbage rate).

---

## Constraints

- **Minimum 5 updates per key** — avoids floor artifacts where the only wrong option is v_0
- **ARBITRARY_SINGLE** dataset for local models — single-token values for logit tracking
- **SEMANTIC_MULTI** for API models — richer values, matches ACL paper setup
- All metrics framed as "P(final) suppression" not "P(init) promotion"

---

## Staged Experiments

### Stage 1: Behavioral Sweep + Error Characterization

**Purpose:** Confirm PI > RI, find operating points, AND characterize WHERE failures land.

Unlike V2, we combine the behavioral sweep with error characterization in a single pass.
Every trial records:

| Field | What it captures |
|-------|-----------------|
| `predicted` | Raw model output (full text, not truncated) |
| `expected` | Correct answer |
| `error_type` | correct / intermediate_intrusion / primacy_intrusion / garbage |
| `predicted_idx` | Which value index was predicted (0 to N-1), or null if garbage |
| `predicted_relative_pos` | Normalized: 0.0 = first, 1.0 = last, null if garbage |
| `all_values` | Full value sequence for this key (for later analysis) |
| `output_length` | Number of tokens/words in the raw output |

**What we analyze from Stage 1:**

1. **Regime map:** Find regime B (RI works, PI fails) at ≥5 updates
2. **Failure position distribution:** histogram of `predicted_relative_pos` on PI failures
3. **Garbage rate:** what fraction of failures are not in the value stream?
4. **Output quality:** are outputs single words or multi-token garbage?
5. **Cross-model consistency:** does near-last error hold for all models?

**Operating point selection for Stage 2:**
Pick cells where PI accuracy is 10–40% (enough failures to analyze,
not total collapse) with ≥5 updates.

### Stage 2: Full Value Logit Lens

**Purpose:** Track P(v_i) for ALL values across ALL layers using TransformerLens.
This is the core mechanistic characterization — shows where in the network the model
commits to the wrong value and what the competition landscape looks like.

**Key difference from V2 exp 12:** V2 tracked only P(v_0) and P(v_{N-1}) — two values.
Stage 2 tracks ALL N values. This is what reveals the near-last pattern in probability
space and identifies the actual dominant competitor (which may not be v_{N-2}).

**What we run:**

- TransformerLens `model.run_with_cache(tokens)` — one forward pass per trial
- At every layer: project residual stream through W_U, get P(v_i) for all values
- At last layer: also record global vocab argmax and total_value_prob

**Operating points:** 3 per model, chosen from Stage 1 to vary update count:

| Point | Regime | Purpose |
|-------|--------|---------|
| Low N (e.g. 5 updates) | AB/B | Early failure — PI just starting to break |
| Medium N (e.g. 15 updates) | B | Core data — PI clearly failing |
| High N (e.g. 30 updates) | C | Saturation — does pattern hold at collapse? |

Fix keys=5 across all points, vary only updates. This isolates the effect of N.
Exact update levels adjusted per model based on Stage 1 regime map.

**Trials:** 100 per condition (RI + PI) per operating point.

**Budget:**

```
3 points × 2 conditions × 100 trials = 600 forward passes per model
4 models × 600 = 2400 total
At ~2-3 sec each (4B on GPU) ≈ 80-120 min total, parallelizable across GPUs
```

**What we save per trial:**

```python
{
    "seed", "condition", "query_word",
    "expected", "expected_idx", "expected_relative_pos",
    "predicted", "correct", "error_type",

    "all_values": ["ruby", "iron", ..., "coral"],       # N values
    "value_probs_by_layer": [[P(v_0,L0), ...], ...],    # shape [N_values, N_layers]

    "total_value_prob": 0.45,                  # sum P(v_i) at last layer
    "global_argmax_token": "pearl",            # full vocab winner at last layer
    "global_argmax_prob": 0.31,
    "global_argmax_is_value": True,
}
```

Storage: N_values × N_layers floats per trial = 10 × 36 = 360 floats ≈ 3KB.
For 600 trials per model ≈ 2MB per model. Negligible.

**Five analyses to run on the results:**

**Analysis 1 — Last-layer value probability distribution:**
Average P(v_i) at last layer across PI failures, plotted by position index.
Confirms near-last error in probability space (not just string match).

**Analysis 2 — Layer trajectory for top competing values:**
For PI failures: plot P(v_{N-1}), P(dominant competitor), P(v_0) across layers.
Reveals one of three patterns:

- (A) v_{N-1} never leads → "never found" — failure is in early representation
- (B) v_{N-1} leads mid-layers then gets overtaken → "late-layer corruption"
- (C) v_{N-1} and competitor are neck-and-neck → "never discriminated"

**Analysis 3 — Crossover layer histogram:**
For trials where v_{N-1} leads at some layer then loses: record the crossover layer.
If crossover clusters at layer ~25, that's where Stage 3 causal experiments should focus.

**Analysis 4 — Garbage threshold (total_value_prob vs predicted position):**
Scatter plot to find data-driven threshold separating real retrieval from noise.
Validates which trials are analyzable.

**Analysis 5 — RI vs PI comparison:**
Same trajectory plot for RI. P(v_0) should rise cleanly with no competition.
The contrast — clean RI retrieval vs messy PI competition — is the core paper figure.

**Paper figures from Stage 2:**

| Figure | Analysis | Story |
|--------|----------|-------|
| Main Fig 1 | #5 | RI vs PI trajectories side by side |
| Main Fig 2 | #1 | Value probability landscape, multi-model |
| Main Fig 3 | #2 | Layer trajectory showing where v_{N-1} loses |
| Appendix | #3 | Crossover layer histogram |
| Appendix | #4 | Garbage threshold scatter |

### Stage 2 → Stage 3 Transition: Questions to Answer Before Proceeding

After running Stage 2, answer these questions for each model. The answers become
Stage 3's config. Do NOT skip this — the answers vary by model.

**Q1: What is the failure pattern?**
Look at layer trajectories for PI failures. Which pattern?

- **(A) Never found:** P(v_last) is near-zero at ALL layers. The model never
  located the last value in the residual stream. → Stage 3 should focus on
  early/mid layers where the value should have been encoded.
- **(B) Overtaken:** P(v_last) rises at mid layers, then gets suppressed in
  late layers. Something actively destroys it. → Stage 3 should focus on the
  critical layers where suppression happens.
- **(C) Never discriminated:** P(v_last) and nearby values are neck-and-neck
  throughout. The model can't tell them apart. → Stage 3 should focus on
  what makes positional discrimination fail.

**Q2: Where are the critical layers?**
At which layers does P(v_last) change dramatically?

- If pattern B: identify the "crossover layers" where P(v_last) drops
- If pattern A: identify where P(v_0) rises (for RI comparison)
- Record: `critical_layer_start`, `critical_layer_end`

Example from Qwen 2.5-3B: layers 32-35 (last 4 of 36). P(v_last) goes from
0.21 at L32 to 0.01 at L35.

**Q3: Is there a single dominant competitor, or is probability diffuse?**
At the final layer, look at P(v_i) distribution across PI failures:

- **Concentrated:** One value (e.g. v_{N-2}) holds >50% of non-garbage mass.
  → logit_diff between v_last and v_competitor is a good metric.
- **Diffuse:** Probability spread across multiple values (e.g. v1=29%, v2=23%, v3=36%).
  → logit_diff between two specific tokens is misleading. Use P(v_last) directly
  as the metric — track what suppresses it, not what replaces it.

Example from Qwen 2.5-3B at 10k_5u: v1=29%, v2=23%, v3=36% → **diffuse**.
Example from Qwen3-4B at 3k_12u: v10=56% → **concentrated on v_{N-2}**.

**Q4: What is the garbage threshold?**
From the total_value_prob vs predicted_position scatter, at what threshold
does the position signal emerge? Trials below this are noise.

**Q5: What is P(v_last) suppression magnitude?**
Compare P(v_last) at its peak layer vs the final layer. If the drop is small
(<0.05), there may not be active suppression — just weak encoding. If the drop
is large (>0.10), something in the critical layers is actively suppressing it.

Example from Qwen 2.5-3B at 5k_3u: P(v_last) = 0.21 at L32 → 0.01 at L35.
Drop = 0.20 — strong active suppression.

**Summary table to fill per model:**

```
| Model | Pattern | Critical layers | Competitor | Diffuse? | Suppression |
|-------|---------|----------------|------------|----------|-------------|
| Qwen 2.5-3B | B (overtaken) | 32-35 | none (diffuse) | Yes | 0.21→0.01 |
| Qwen3-4B | TBD | TBD | TBD | TBD | TBD |
| ... | | | | | |
```

### Stage 3: Causal Mechanistic Experiments

**Purpose:** Identify WHICH components (heads, MLPs) are responsible for
P(v_last) suppression, and test whether intervening on them fixes PI.

**Input:** Stage 2 results file + the answers to Q1-Q5 above.

**The core metric depends on Q3:**

- If concentrated competitor: `logit_diff = logit(v_last) - logit(v_competitor)`
- If diffuse: `P(v_last)` directly — measured as `logit(v_last)` or its rank

#### Head identification: two sources, union of both

**Set A — Attribution patching (gradient-based, covers entire model):**

Attribution patching (Neel Nanda; AtP* from DeepMind, 2024) uses 1 forward +
1 backward pass to rank ALL heads at ALL layers by their gradient contribution
to the metric (P(v_last) or logit_diff). No destruction of model computation —
avoids the problem of early-layer knockout causing catastrophic cascading effects.

References:
- Nanda: "Attribution Patching: Activation Patching At Industrial Scale"
- Kramar et al. 2024 (AtP*): fixes false negatives in basic attribution patching
- Syed et al. 2024: "Attribution Patching Outperforms Automated Circuit Discovery"

Cost: 1 forward + 1 backward per trial = 50 trials ≈ 5 min on GPU.
Output: importance score for every (layer, head) pair → take top-K (e.g. K=20).

**Set B — Stage 2 critical layer heads:**

All heads in the layers where P(v_last) visibly drops (from Stage 2 Q2).
Example: all 16 heads at layers 32-35 for Qwen 2.5-3B = 64 heads.

**Why the union (A ∪ B):**

- A head in Set A but NOT Set B (e.g. at layer 12): something early sets up the
  suppression that becomes visible later. Attribution patching catches this because
  it traces gradients through the full computation graph.
- A head in Set B but NOT Set A (e.g. at layer 33): affects the trajectory but its
  gradient contribution to P(v_last) is indirect. Still worth testing causally.
- A head in both: convergent evidence — highest confidence target.

In practice, most Set A heads will be in the critical layers. The few that differ
are the most mechanistically interesting.

#### Experiment 3A: Attribution patching (fast head ranking)

For each PI trial:
- Forward pass: compute P(v_last) at the final layer
- Backward pass: compute gradient of P(v_last) w.r.t. each head's output
- The gradient magnitude = that head's importance for P(v_last)

Aggregate across trials → ranked list of heads.

Cost: ~50 trials × (1 fwd + 1 bwd) ≈ 5-10 min on GPU.

**Output:** `head_importance[layer, head]` — shape [n_layers, n_heads].
Top-K heads = Set A.

#### Experiment 3B: Targeted activation patching (causal validation)

For each head in **Set A ∪ Set B**:
- "Clean" run: RI trial where model correctly retrieves v_0
- "Corrupted" run: PI trial where model fails
- Patch: replace this head's output in the corrupted run with its clean-run value
- Measure: does P(v_last) increase? Does PI accuracy recover?

This causally validates attribution patching scores. A head with high attribution
that also shows high patching recovery = confirmed causally important.

Cost: |A ∪ B| × 50 trials ≈ 80 heads × 50 = 4,000 forward passes (~25 min).

#### Experiment 3C: Logit lens under ablation (trajectory change)

Using the top 5-10 confirmed heads from 3B:
- Run full logit lens trajectory with these heads zeroed out
- Compare P(v_last) trajectory: normal vs ablated
- Key question: does ablation raise P(v_last) at the final layer?
- Secondary: does it shift where P(v_last) peaks?

If ablating heads at layer 33 raises P(v_last) from 0.01 → 0.15 at the final
layer, those heads were directly responsible for the suppression seen in Stage 2.

Cost: ~50 trials with full cache ≈ 15 min.

#### Experiment 3D: Forced attention — only if Q3 = concentrated

For confirmed heads from 3B:
- Force them to attend to v_last's position in the sequence
- If PI improves → QK routing problem (heads looking at wrong position)
- If PI doesn't improve → OV problem (heads writing wrong thing regardless)

Skip for diffuse models — no single correct attention target.

Cost: ~50 trials × top-5 heads = 250 forward passes ≈ 5 min.

#### Stage 3 output per model

```python
{
    "model": "...",
    "operating_point": (keys, updates),
    "critical_layers_from_stage2": [32, 33, 34, 35],
    "pattern": "B",

    # Attribution patching: all heads ranked
    "attribution_scores": {
        "head_importance": [[...], ...],   # [n_layers, n_heads]
        "top_20_heads": ["L33H5", "L34H2", "L12H8", ...],
    },

    # Targeted patching: causal validation for Set A ∪ B
    "patching_results": {
        "L33H5": {"delta_p_last": +0.08, "in_set_A": True, "in_set_B": True},
        "L12H8": {"delta_p_last": +0.03, "in_set_A": True, "in_set_B": False},
        "L34H0": {"delta_p_last": +0.01, "in_set_A": False, "in_set_B": True},
        ...
    },

    # Ablation logit lens
    "ablation_trajectory": {
        "normal_p_last_final": 0.01,
        "ablated_p_last_final": 0.15,
        "heads_ablated": ["L33H5", "L34H2"],
    },

    # Convergence check: do Set A and Set B agree?
    "convergence": {
        "overlap_top20_A_with_B": 14,   # 14 of top-20 attribution heads are in critical layers
        "top_patching_heads_in_A": 8,   # 8 of top-10 patching heads were in Set A
    },
}

---

## Results Structure

```
v3/
├── PLAN.md                    # This file
├── observations/              # Running notes as we analyze
│   └── YYYY-MM-DD_topic.md
├── results/
│   ├── {model}/
│   │   ├── stage1_sweep_{ts}.json       # Behavioral sweep summary
│   │   ├── stage1_trials_{ts}.json      # All trial details (raw outputs)
│   │   ├── stage2_logit_lens_{ts}.json  # Per-layer P(v_i) tracking
│   │   └── ...
├── stage1_sweep.py            # Behavioral sweep (local models)
├── stage2_failure_analysis.py # Logit lens P(v_i) tracking
└── ...                        # Stage 3 scripts added later
```

---

## Open Questions

1. Is v_{N-2} consistently the dominant competitor, or does the "wrong" position
   shift with N? (Stage 1 will answer this)
2. Does the garbage rate correlate with model size? Smaller models → more garbage?
3. For Pythia (base model, no instruction tuning): does the same pattern hold?
   Or is the near-last error specific to instruction-tuned models?
4. Mamba/RWKV (no causal attention): do they show a different failure profile?
   The theory predicts they should — test this.
