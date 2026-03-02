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

### Stage 3: Causal Mechanistic Experiments (design AFTER Stage 2)

Only designed after Stage 2 confirms the pattern and identifies:
- The dominant competitor (which v_i beats v_{N-1})
- The crossover layer (where the failure happens)
- Which of the three failure patterns (A/B/C) dominates

Likely candidates from V2, with updated metrics:
- Head knockout (V2 exp 25a) — still the right causal method, now measuring
  effect on P(v_{N-1}) vs P(competitor) instead of P(init) vs P(final)
- Activation patching (V2 exp 15/22) — focused on crossover layer region
- Forced attention (V2 exp 18b) — tests QK vs OV at the identified heads

The key metric update: `logit_diff` becomes `logit(v_{N-1}) - logit(v_competitor)`
where `v_competitor` is identified empirically from Stage 2, not assumed to be v_0.

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
