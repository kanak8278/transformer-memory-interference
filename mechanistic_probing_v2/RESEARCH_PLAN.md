# Research Plan: Mechanistic Probing of PI > RI in Transformers (NeurIPS Expansion)

## Context

ACL paper showed PI > RI across 39 models (Cohen's d=1.73) using 46 categories and up to 300+ interference levels. NeurIPS expansion needs to explain WHY mechanistically. Preliminary work on SmolLM2-135M confirmed the effect at small scale and showed (via logit lens) that the initial value dominates the residual stream at all layers.

**Problem with current approach:** Single model (SmolLM2-135M), small sample sizes (25 trials), raw logit lens only, no causal evidence, GQA complicates head analysis, no cross-architecture validation.

**Goal:** Build a rigorous mechanistic story across multiple model sizes and families, establishing that attention heads performing value retrieval exhibit systematic positional primacy bias that causes PI > RI.

---

## Model Lineup

### Primary Models (Active)

| Model | Size | Heads (Q/KV) | Layers | Role | TransformerLens | Status |
|-------|------|--------------|--------|------|-----------------|--------|
| Qwen2.5-0.5B-Instruct | 0.5B | 14/2 | 24 | Primary probing target | Yes | DONE — full mechanistic suite |
| Qwen2.5-1.5B-Instruct | 1.5B | 12/2 | 28 | Size scaling within Qwen family (3x) | Yes | DONE — full mechanistic suite |
| Qwen2.5-3B-Instruct | 3B | 16/2 | 36 | Size scaling within Qwen family (6x) | Yes | Behavioral IN PROGRESS (Colab), mechanistic TODO |
| Gemma-3-1B-IT | 1B | TBD | TBD | Cross-family validation + SAE deep-dive | Yes (v3/TransformerBridge) | TODO |
| Pythia-160M (or 410M) | 160M/410M | MHA (clean) | 12/24 | Training dynamics + community anchor | Native (gold standard) | TODO |

### Model Selection Rationale

**Qwen 2.5 family (0.5B, 1.5B, 3B):** Primary data. 7 solid mechanistic findings on 0.5B+1.5B. Within-family scaling from 0.5B→3B. Instruction-tuned, 32K context. Limitation: no pre-trained SAEs, no training checkpoints, GQA complicates head analysis.

**Gemma 3 1B IT:** Best instruction-tuned model under 3B for interpretability. Has Gemma Scope 2 pre-trained SAEs + transcoders on every layer of the IT variant (not just base). Neuronpedia integration. Enables SAE feature analysis: "which features fire during PI failure?" Different architecture family from Qwen → cross-family validation. 32K context.

**Pythia 160M (or 410M):** Canonical mech interp model. Native TransformerLens support (gold standard). 154 intermediate training checkpoints (every 1000 steps) → enables Phase 3 training dynamics: "when does PI>RI emerge during training?" MHA (no GQA) → clean head-level analysis. Pre-trained SAEs available (EleutherAI/sparsify). Completion-only (not instruction-tuned) → requires few-shot pattern-completion reformulation (see Pythia Prompt Format below). Community trust: reviewers know Pythia, results are independently verifiable.

### Deprioritized Models

| Model | Size | Reason |
|-------|------|--------|
| SmolLM2-135M-Instruct | 135M | Preliminary work done, but no SAEs, limited community, GQA. Not investing further. Keep existing behavioral data as supplementary. |
| Gemma-2-2B-IT | 2.6B | Superseded by Gemma 3 1B IT — Gemma 3 has IT SAEs (Gemma 2 only has base SAEs for 2B), smaller (faster), newer tooling. |

### Pythia Prompt Format (Few-Shot Pattern Completion)

Pythia is completion-only — can't follow "What was the first value of X?" instructions. Reformulate as pattern completion:

```
art: impressionism
tool: hammer
gem: ruby
art: baroque
tool: screwdriver
gem: sapphire
art: cubism
tool: wrench
gem: emerald

The first value of art was: impressionism
The first value of tool was: hammer
The first value of gem was:
```

Model should complete with `ruby`. For PI (proactive interference):
```
The last value of gem was:
```

The few solved examples before the test query teach the model the task format in-context. This is a different prompt format but tests the same retrieval mechanism. Key considerations:
- Use 5-10 categories (not 46) — small completion models cap out earlier
- Short keys (3-5 chars): "art", "gem", "tool"
- Single-token values where possible
- N ∈ {1, 3, 5, 10, 20} updates — Pythia will saturate earlier than instruction-tuned models

### Data Setup

Two data configurations are used:

1. **Behavioral sweep** (Colab, synthetic values): Semantic keys + "Art375" style values. Multi-token. Used for full landscape mapping and paper behavioral figures.
2. **Mechanistic experiments** (local, single-token values): Semantic keys + single-token English words. Required for logit lens/DLA/attention analysis. Lower accuracy than synthetic values.

Key finding: single-token English words are significantly harder for the model than synthetic values. Operating points must be recalibrated per data setup.

---

## Phase 1: Comprehensive Behavioral Characterization (Per Model)

**Goal:** Map the full interference landscape for each model. This is NOT a quick screen — it's the first major analysis. The cracking pattern itself is data.

### Step 1.1: Install TransformerLens + Qwen2.5-0.5B

- Install transformer_lens in existing .venv
- Verify Qwen2.5-0.5B-Instruct loads correctly via TransformerLens HookedTransformer
- Verify MPS compatibility, memory footprint
- Run one sanity prompt to confirm instruction-following works

**Files:** `mechanistic_probing/experiments/10_setup_qwen.py` (setup validation script)

### Step 1.2: Full Behavioral Sweep

**Script:** `mechanistic_probing/experiments/11_behavioral_sweep.py`

**Design — parameterized for any model:**

Grid:
- Keys (categories): 2, 3, 5, 7, 10, 15, 20, 25, 30, 35, 40, 46
  (diff=5 after 15)
- Updates per key: 1, 3, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300
  (diff=5 from 10-20, diff=10 from 20-60, diff=20 from 60-300)
- Conditions: RI (recall first), PI (recall last)
- Trials per cell: minimum 30 (with different random seeds for category/value selection)
- Report: accuracy with 95% bootstrap CIs, error type classification

**Context length safety (pre-flight check):**
Before running any cell, generate ONE sample prompt, tokenize it, and check:
- `prompt_tokens < model_max_context * 0.90` (10% safety margin for generation)
- Get max context from `model.config.max_position_embeddings` or known values:
  - Qwen2.5-0.5B: 32,768
  - Qwen2.5-1.5B: 32,768
  - Qwen2.5-3B: 32,768
  - Gemma-3-1B: 32,768
  - Pythia-160M: 2,048 (much smaller — limits grid significantly)
  - Pythia-410M: 2,048
- If exceeds limit → skip cell, log it as "SKIPPED: context overflow (N tokens > limit)"
- Run the pre-flight for ALL cells first, print the feasible grid before starting trials

**Rough token estimate per cell:**
- Each key-value line ≈ 5-8 tokens ("category: value\n")
- Stream tokens ≈ N_keys × N_updates × 6 (average)
- Overhead (instructions + query) ≈ 40 tokens
- Example: 20 keys × 100 updates ≈ 12,000 + 80 ≈ 12,080 tokens (fits 32K)
- Example: 46 keys × 300 updates ≈ 82,800 tokens (won't fit anything)

The grid auto-truncates when:
- Pre-flight context check fails (see above)
- Both RI and PI hit 0% for 5+ consecutive update levels (model truly saturated)

**Value pool:** Use the SAME 46 categories from the original paper's `data/interleaved_dataset.json` where possible, with single-token filtering per tokenizer. This ensures comparability with the ACL results.

**Prompt format (NO few-shot — match original paper):**
```
Read the following key-value stream. Each key gets updated multiple times.

color: red
fruit: apple
color: blue
fruit: mango
color: green
fruit: grape

What was the first value of fruit?
```
No solved examples. Direct instruction + stream + single query per trial.
This matches the original 39-model study format for direct comparability.

**Controls built into the sweep:**
1. **Token frequency balance**: For each trial, log the mean log-frequency of initial and final value tokens. Report correlation between frequency and accuracy.
2. **Position distance control**: Track the token distance between initial value position and final value position. Interference should scale with distance.

### Step 1.2b: Few-Shot vs No-Few-Shot Comparison (subset)

Run AFTER Step 1.2, only at the cracking-edge operating points identified from the main sweep.

**Design:**
- Select 3-5 (keys, updates) cells at the PI cracking edge (where PI just starts failing but RI is fine)
- Run the SAME trials in two formats:
  - **No few-shot** (primary format, already have data from Step 1.2)
  - **With few-shot** (2 solved examples from other categories before the test query)
- 50 trials per cell per format

**What this tells us:**
- If few-shot helps PI more than RI → part of PI failure is comprehension ("last" is harder than "first")
- If few-shot doesn't change the asymmetry → PI failure is purely retrieval (model understands the question, attention can't access the answer)
- If few-shot helps BOTH equally → format disambiguation, not interference-specific

This cleanly separates comprehension failure from retrieval failure — important for the mechanistic narrative.

**Output per model:**
- Full keys × updates × condition accuracy grid with CIs
- Decay curves: RI accuracy vs N_updates, PI accuracy vs N_updates (for each key count)
- Cracking point identification: N where accuracy drops below 50%
- Error type breakdown: primacy intrusion vs other errors at each point
- Regime map: which (keys, updates) cells are in Regime A (both work), B (PI fails, RI works), C (both fail)

### Step 1.3: Compare Cracking Patterns Across Models

After running 1.2 on all models, compare:
- Do all models show PI > RI? At what (keys, updates) does PI crack for each model?
- Does RI cracking scale with model size (matching R²=0.49 from ACL paper)?
- Are decay curve shapes similar (exponential, polynomial, etc.)?
- Do error taxonomies match across models?

**This comparison is a paper figure.** A multi-panel plot: rows = models, columns = key counts, showing RI/PI accuracy vs updates with CI bands.

---

## Phase 2: Mechanistic Analysis Pipeline (Per Model, Multiple Operating Points)

**Critical:** Run these analyses at MULTIPLE points on the interference landscape, not just the "sweet spot." Specifically:

- **Point A** (no interference): keys=3, updates=1 — baseline where everything works
- **Point B** (PI cracking edge): the (keys, updates) where PI just starts failing but RI is fine
- **Point C** (deep asymmetry): where PI≈0% but RI>80%
- **Point D** (RI cracking edge): where RI starts degrading
- **Point E** (both failing): high interference where both are near 0%

The exact (keys, updates) values for B-E come from Phase 1 results and will differ per model.

**TransformerLens n_ctx:** Set to 8192 (default). This is just a buffer size — doesn't change model behavior. RoPE embeddings are computed on-the-fly from actual token positions. Well within Qwen2.5's 32K training context.

**Dependency chain:**
```
Token Tracker (core/token_tracker.py)     ← foundation, needed by everything below
    │
    ├── Step 2.1: Logit Lens              ← can run independently (only needs resid_post)
    │
    ├── Step 2.2: Attention Analysis       ← identifies retrieval heads
    │       │
    │       ├── Step 2.3: DLA             ← focused on identified retrieval heads
    │       │
    │       └── Step 2.4: Patching        ← focused on identified retrieval heads + positions
```

### Step 2.0: Token Position Tracker (prerequisite)

**Module:** `core/token_tracker.py`

The model sees a flat sequence of token positions. To analyze attention patterns meaningfully, we need to know which positions correspond to what semantic role. The token tracker takes a prompt + trial metadata and maps every token position to one of:
- `instruction` — "Read the following key-value stream..."
- `category_key` — "hat style", "tools", etc.
- `separator` — ":"
- `initial_value` — first value for a given category in the sequence
- `intermediate_value` — middle values (2nd through N-1th)
- `final_value` — last value for a given category in the sequence
- `query` — "What was the first value of hat style?"

**Handles multi-token values:** "Hat380" may tokenize as ["Hat", "380"]. The tracker marks both positions and downstream analysis sums attention across them.

**Implementation:** Tokenize the full prompt, then tokenize each value/key separately and search for those subsequences within the full token sequence, matching in order of appearance.

This module is shared infrastructure — every Phase 2 experiment imports it.

### Step 2.1: Logit Lens Analysis

**Script:** `mechanistic_probing/experiments/12_logit_lens_qwen.py`

**Design:**
- For each operating point (A through E), run 100 trials
- At each layer, project residual stream at answer position through unembedding
- Track: P(initial_value), P(final_value), rank of each, top-5 predictions
- Use TransformerLens cache for clean access: `cache["resid_post", layer]`

**Key questions this answers:**
- Does the "initial value dominates all layers" finding from SmolLM2 replicate on Qwen2.5?
- Does the pattern change across operating points? At Point D (RI cracking), does the initial value finally lose dominance?
- Is there a layer where the final value appears briefly before being suppressed? (Would change the narrative)

**Output:** Layer-by-layer probability curves for P(initial) and P(final), averaged across trials with CI bands. Separate plot per operating point. This is a paper figure.

### Step 2.2: Attention Pattern Analysis + Head Functional Classification

**Script:** `mechanistic_probing_v2/experiments/13_attention_analysis.py`

**Depends on:** Step 2.0 (token tracker). This is the most important mechanistic step — it identifies which heads are responsible for value retrieval and whether they have primacy bias.

**Core idea:**
At the last token position (where the model generates the answer), each of the 336 heads (24 layers × 14 heads) produces an attention distribution over all previous tokens. This distribution determines what information each head pulls from context. We analyze where each head looks to classify its functional role.

**Trial structure:**
For each operating point (A through E), run BOTH conditions with the SAME prompts:
- **RI trials** (100): "What was the **first** value of X?" — model should attend to initial value
- **PI trials** (100): "What was the **last** value of X?" — model should attend to final value

Same categories, same values, same interleaved stream — only the question word changes. This paired design isolates what changes (and what doesn't) when the instruction flips.

**Per-trial computation:**
Using TransformerLens, run forward pass with cache:

```
cache["pattern", layer] → shape [n_heads, seq_len, seq_len]
```

At the answer position (last token):
1. Use token tracker to classify each position: initial_value, intermediate_value, final_value, category_key, instruction, query
2. For each of the 336 heads, extract the attention vector at position -1 (last token)
3. Sum attention weights by semantic role (handling multi-token spans by summing across constituent positions)
4. Compute per-head scores:
   - `retrieval_score` = total attention to any value position (initial + intermediate + final) for the **test category**
   - `primacy_score` = attention_to_initial / (attention_to_initial + attention_to_final) for the test category
   - These are computed separately for RI and PI trials

**Aggregation across trials:**
For each head, average retrieval_score and primacy_score across all 100 trials per condition. Then compute:
- `mean_retrieval_score` = average across both conditions
- `primacy_score_RI` = average primacy score across RI trials
- `primacy_score_PI` = average primacy score across PI trials
- `condition_sensitivity` = |primacy_score_RI - primacy_score_PI|

A head with low condition_sensitivity behaves the same regardless of whether the instruction says "first" or "last" — it ignores the instruction.

**Head classification (based on aggregated scores):**
- **Retrieval heads**: mean_retrieval_score > threshold (heads that consistently attend to value tokens)
- **Primacy-biased retrieval heads**: retrieval heads with primacy_score_PI > 0.6 (attend more to initial value EVEN when asked for last — this is the failure mode)
- **Recency-biased retrieval heads**: retrieval heads with primacy_score_PI < 0.4 (correctly shift toward final in PI)
- **Condition-sensitive heads**: condition_sensitivity > threshold (shift behavior between RI/PI — responsive to instruction)
- **Non-retrieval heads**: low retrieval_score (attend to instruction, keys, or distribute broadly)

**GQA note:** Qwen2.5-0.5B has 14 query heads but only 2 KV heads (7:1 ratio). Attention patterns are per-query-head (14 distinct patterns per layer), but the value vectors are shared within each KV group. Track which KV group each query head belongs to — if all 7 query heads in a KV group show primacy bias, that's one shared mechanism, not 7 independent ones.

**Key questions:**
- What fraction of heads are retrieval heads? Does this change with model size?
- Do retrieval heads show primacy bias even in PI condition (where they SHOULD attend to final)? If yes → attention routing doesn't respond to "first" vs "last", explaining PI failure.
- Are there ANY recency-biased retrieval heads? If zero → the architecture has no mechanism for recency retrieval. Smoking gun.
- What is the distribution of condition_sensitivity? Near 0 for most heads → instruction has minimal effect on attention routing.
- Does the pattern change across operating points? At Point A (trivial) vs Point C (deep asymmetry) — do the same heads dominate?

**Expected outcomes and what they mean:**
1. **Most retrieval heads show primacy bias in both RI and PI** → Attention routing ignores instruction. PI fails because the right information is never retrieved. Clean story.
2. **Some heads shift between RI and PI, but not enough** → Partial instruction comprehension, primacy-biased heads dominate. More nuanced.
3. **Heads shift correctly but output is still wrong** → Problem is downstream (MLP or unembedding), not attention. Would redirect toward DLA.

**Visualization for paper:**
1. Scatter plot: per-head (attn_to_initial vs attn_to_final), colored by layer, **separate panels for RI vs PI** — shows whether the cloud shifts when instruction changes
2. Heatmap: layers × heads, colored by primacy_score, **side-by-side for RI and PI** — shows which heads adapt and which don't
3. Bar chart: distribution of head types (retrieval/non-retrieval, primacy/recency biased)
4. Histogram of condition_sensitivity across all retrieval heads — clustered near 0 = instruction-insensitive

### Step 2.3: Direct Logit Attribution (DLA) per Head

**Script:** `mechanistic_probing_v2/experiments/14_dla.py`

**Depends on:** Step 2.2 (retrieval head identification). Run DLA on ALL heads but interpret results through the lens of identified retrieval heads.

**Core idea:**
The model's output logit for any vocabulary token is literally a sum of contributions from every component (attention heads + MLPs) due to the residual stream architecture. Each component adds its output to the residual stream, and the final logit is `residual_stream @ unembedding_matrix`. So each component's individual contribution is `component_output @ unembedding_vector_for_that_token`.

**Terminology — initial_value vs final_value:**
In each trial, the test category (e.g., "hat style") appears multiple times in the stream:
```
hat style: Hat380    ← initial_value (first appearance in sequence)
hat style: Hat141    ← intermediate_value
hat style: Hat126    ← final_value (last appearance in sequence)
```
- RI condition: correct answer = initial_value (Hat380)
- PI condition: correct answer = final_value (Hat126)

**Per-trial computation:**
Using TransformerLens cache, decompose the logit into per-component contributions:

```python
# Per attention head contribution to logit of token t at the answer position
per_head_contribution = cache["attn_out", layer][:, -1, :] @ model.W_U[:, token_id]

# Per MLP layer contribution
per_mlp_contribution = cache["mlp_out", layer][:, -1, :] @ model.W_U[:, token_id]
```

For each trial, compute **logit_diff** per component — always defined as:
```
logit_diff = contribution_to_logit(initial_value) - contribution_to_logit(final_value)
```

- `logit_diff_per_head[layer, head]` = head's push toward initial vs final
- `logit_diff_per_mlp[layer]` = MLP layer's push toward initial vs final

**Sign convention (kept consistent across RI and PI):**
- **Positive logit_diff** = component pushes toward initial_value
- **Negative logit_diff** = component pushes toward final_value

In RI trials: positive = helping (promoting correct answer)
In PI trials: positive = hurting (promoting the WRONG answer — primacy intrusion)

This consistent convention means we can directly compare RI and PI heatmaps. Components that are strongly positive in BOTH conditions are the ones that can't adapt — they always promote the initial value regardless of instruction.

**Trial structure:**
Same as Step 2.2 — run both RI and PI trials (100 each) at each operating point, using the same prompts. Aggregate logit_diff across trials.

**Key questions:**
- **Convergence with Step 2.2:** Do the heads with highest logit_diff overlap with retrieval heads identified by attention analysis? If the same heads attend to the initial value AND promote it in the output → strong convergent evidence for a "primacy circuit."
- **Attention vs MLP:** Do MLPs contribute to the primacy bias, or is it purely attention?
  - If MLPs ≈ 0: bias is purely in attention routing. The model's stored knowledge isn't involved — it's a retrieval failure.
  - If MLPs contribute significantly: the model has learned "first = important" in its weights, not just its attention patterns. Deeper architectural bias, harder to fix.
- **Stability across operating points:** Do the same heads dominate at Point B (PI cracking edge) and Point C (deep asymmetry)? If yes → stable circuit. If different heads at different points → context-dependent mechanism.
- **Condition adaptation:** Does any component flip sign between RI and PI? (Positive in RI, negative in PI = component correctly adapts to instruction.) If no components flip → complete instruction-insensitivity in the output pathway.

**Visualization for paper:**
1. Heatmap of (layer × head) → mean logit_diff, **side-by-side for RI and PI**, with retrieval heads from Step 2.2 circled/marked
2. Bar chart: total logit_diff contribution from attention heads vs MLPs (aggregated) — shows which component class drives primacy
3. Scatter: per-head logit_diff_RI vs logit_diff_PI — points on the diagonal = no adaptation, points that flip quadrants = instruction-responsive

### Step 2.4: Activation Patching (Causal Evidence)

**Script:** `mechanistic_probing/experiments/15_activation_patching_qwen.py`

**Depends on:** Steps 2.2 + 2.3 (know which heads and positions matter). Attribution patching sweeps the full grid cheaply, but targeted patching focuses on retrieval heads and value positions identified earlier.

**Design:**

Two run types:
1. **Clean run**: Single value per key, no interference (e.g., "color: red" only). Model gets the answer trivially.
2. **Corrupted run**: Full interference (multiple updates). Model may fail (especially PI).

**Attribution patching (fast sweep):**
For each (layer, position), compute the gradient-based approximation of "how much would patching clean activation here change the output?" This is:
```
effect ≈ (clean_activation - corrupted_activation) · gradient_of_loss_wrt_activation
```
Sweep the full (layer × position) grid cheaply.

**Targeted causal patching (on top-20 from attribution):**
Actually patch clean activations into the corrupted run at the highest-effect positions. Measure real accuracy recovery.

**Patching variants:**
1. **Full residual stream** at (layer, position) — identifies critical positions
2. **Attention output only** at (layer, position) — isolates attention contribution
3. **MLP output only** at (layer, position) — isolates MLP contribution

**Key questions:**
- Can we RESTORE PI accuracy by patching? If patching the final value's position in late layers recovers accuracy → the information was there but inaccessible.
- Do RI and PI failures localize to DIFFERENT (layer, position) regions? → dual-process evidence
- Is the critical component attention or MLP?

**Visualization:** Two heatmaps (layer × token_position) colored by accuracy recovery. One for RI, one for PI. Token positions labeled by type (initial_value, update_1, update_2, ..., final_value, query).

---

## Phase 2B: Repeat Phase 2 on Additional Models

### Step 2B.1: Qwen2.5-1.5B-Instruct — DONE

Run Steps 2.1-2.4 on Qwen2.5-1.5B. COMPLETED — all 12 experiments at 4 operating points.

Key comparison:
- Does the primacy bias in retrieval heads weaken? (Would explain R²=0.49 for size→RI)
- Are there MORE recency-biased heads at larger scale?
- Does the DLA show the same circuit pattern?

### Step 2B.2: Qwen2.5-3B-Instruct — IN PROGRESS

Behavioral sweep running on Colab. Recalibration sweep DONE. Mechanistic experiments TODO.
See `QWEN_3B_EXPERIMENT_LOG.md` for details.

### Step 2B.3: Gemma-3-1B-IT — TODO (Cross-Family Validation + SAE)

Run Steps 2.1-2.4 on Gemma-3-1B-IT. Key comparison:
- Different architecture family (different training, different tokenizer, Google vs Alibaba)
- Same mechanistic story? If yes → architecture-general finding
- Does the primacy cliff, late-layer localization, and query-position corruption replicate?

**SAE feature analysis using Gemma Scope 2 (unique to this model):**
- Load pre-trained SAEs from Gemma Scope 2 (available for IT variant, every layer)
- Load transcoders (trace feature-to-feature across layers)
- For RI and PI trials, check which SAE features activate at the answer position
- Look for features corresponding to "initial value" vs "final value" concepts
- Check if "initial value" features are active even during PI failures → direct evidence for "information present but inaccessible"
- Use Neuronpedia to browse and label features interactively
- Compare SAE features across RI-correct, PI-correct, PI-failure trials
- This analysis is NOT possible on Qwen (no SAEs) — unique contribution of adding Gemma

**Dependencies:** `sae_lens` (SAELens library), `transformer_lens` v3 with TransformerBridge support

### Step 2B.4: Pythia-160M (or 410M) — TODO (Community Anchor + Training Dynamics)

**Behavioral sweep only (Phase 1).** Use few-shot pattern-completion format (see Pythia Prompt Format above).

Key questions:
- Does PI > RI appear in a completion model with no instruction tuning? If yes → the asymmetry is not an artifact of instruction tuning
- How does the cracking pattern compare to instruction-tuned models?
- At what interference level does Pythia-160M saturate? (2048 context limits the grid)

**Mechanistic probing (Phase 2) — if behavioral sweep confirms PI > RI:**
- Pythia is native TransformerLens → gold standard hook access, no GQA complications
- MHA architecture means each head has its own KV — cleaner head-level analysis than Qwen/Gemma
- Pre-trained SAEs available via EleutherAI/sparsify → can do SAE feature analysis similar to Gemma
- This would be the cleanest mechanistic analysis of any model in the study

**Training dynamics (Phase 3) — the unique Pythia contribution:**
- 154 intermediate checkpoints available (every 1000 steps)
- Run behavioral sweep at each checkpoint → track when PI > RI emerges
- See Phase 3, Step 3.2 for full design

---

## Phase 3: Architectural Controls & Theory Connection

**Only proceed here after Phase 2/2B shows convergent mechanistic findings across models.**

### Step 3.1: SSM Control (Mamba or RWKV)

Run behavioral sweep (Phase 1 only) on a state-space model:
- Mamba-130M or RWKV-169M (similar size to SmolLM2-135M)
- No mechanistic probing needed — just behavioral comparison
- Key prediction: SSMs should show different interference profile (possibly RI > PI or symmetric)
- If prediction holds → confirms attention-specificity of PI > RI

### Step 3.2: Training Dynamics (Pythia Checkpoints)

**Primary: Pythia-410M** (154 checkpoints, every 1000 steps from step 0 to step 143000):
- Run behavioral sweep (few-shot completion format) at ~20 evenly-spaced checkpoints
- Track: when does PI > RI emerge during training?
- If present from earliest checkpoint → architectural (causal masking imposes it from init)
- If develops over training → data/optimization effect (learned from training distribution)
- Pythia checkpoints available as HuggingFace branches: `revision="step-{N}"`

**Secondary: Pythia-160M** (same 154 checkpoints):
- Run same analysis at 160M for size comparison
- If PI > RI emerges at same training step across sizes → architecture-driven
- If larger model develops it later → capacity interaction

**Supplementary: SmolLM2-135M** intermediate checkpoints (~8 available, every ~250B tokens):
- Coarser granularity than Pythia but instruction-tuned checkpoints
- Can test whether instruction tuning changes when PI > RI appears

### Step 3.3: Theory Framing

Don't prove new theorem. Instead:
- Cite Wu et al. (2025, ICML) for formal proof: causal masking → primacy bias
- Cite Ramsauer et al. (2021) for attention = Hopfield retrieval
- OUR contribution: empirical demonstration that this theoretical bias manifests as PI > RI in associative recall, mechanistic identification of the responsible heads/layers, and causal evidence via patching

---

## Implementation Notes

### Shared Infrastructure

Create a reusable module: `mechanistic_probing_v2/core/`
- `model_loader.py` — Load any model via TransformerLens or fallback to HF hooks
- `dataset.py` — Generate interference prompts with controls (frequency balance, position tracking)
- `token_tracker.py` — Map token positions to value types (initial, intermediate, final, key, instruction)
- `analysis_utils.py` — Common functions: logit lens, attention extraction, DLA, patching

All experiment scripts import from core/ — no copy-pasting between experiments.

### File Structure

```
mechanistic_probing_v2/
├── core/
│   ├── __init__.py
│   ├── model_loader.py           # TransformerLens + HuggingFace loading
│   ├── dataset.py                # Trial generation, 46 categories, interleaving
│   ├── single_token_values.py    # Single-token value pools (per-tokenizer)
│   ├── token_tracker.py          # Position tracking by token ID
│   ├── output.py                 # Result saving utilities
│   └── analysis_utils.py         # Logit lens, attention, DLA, patching helpers
├── experiments/
│   ├── 10_setup_qwen.py          # Setup validation (Qwen)
│   ├── 10b_setup_gemma.py        # Setup validation (Gemma 3 1B IT)     [NEW]
│   ├── 10c_setup_pythia.py       # Setup validation (Pythia 410M/160M)  [NEW]
│   ├── 11_behavioral_sweep.py    # Phase 1: full grid sweep (instruction-tuned models)
│   ├── 11_behavioral_sweep_pythia.py  # Phase 1: few-shot completion format [NEW]
│   ├── 11b_behavioral_validation.py
│   ├── 11c_recalibrate_sweep.py
│   ├── 11d_neutral_keys_sweep.py
│   ├── 12_phase2_trial.py        # Phase 2: logit lens + attention + DLA
│   ├── 13_positional_gradient.py # Phase 2: P(v_i) across all value positions
│   ├── 14_pi_mass_distribution.py # Phase 2: where PI probability peaks
│   ├── 15_activation_patching.py  # Phase 2: causal patching
│   ├── 16_head_identification.py  # Phase 2: head classification + ablation
│   ├── 17_instruction_sensitivity.py # Phase 2: does model distinguish first/last?
│   ├── 18_forced_attention.py     # Phase 2: force heads to correct position
│   ├── 19_positional_bias_sweep.py # Phase 2: recency bias λ sweep
│   ├── 19b_bias_attention_proof.py
│   ├── 20_minority_override_analysis.py # Phase 2: why minority overrides majority
│   ├── 21a_logit_lens_under_ablation.py
│   ├── 21b_dla_split_by_outcome.py
│   ├── 21c_failure_output_classification.py
│   ├── 22_query_patching_granular.py
│   ├── 23_ablation_patching_interaction.py
│   ├── 24_ov_theory_tests.py
│   ├── 24b_single_head_force.py
│   ├── 25_gemma_sae_analysis.py   # Gemma Scope 2 SAE feature analysis  [NEW]
│   ├── 26_pythia_training_dynamics.py  # Phase 3: checkpoint sweep       [NEW]
│   └── 27_ssm_control.py         # Phase 3: Mamba/RWKV behavioral       [NEW]
├── results/
│   ├── Qwen2.5-0.5B-Instruct/    # Per-model, per-operating-point
│   ├── Qwen2.5-1.5B-Instruct/
│   ├── Qwen2.5-3B-Instruct/
│   ├── gemma-3-1b-it/             # [NEW]
│   ├── pythia-410m/               # [NEW]
│   ├── pythia-160m/               # [NEW]
│   └── *.json                     # Legacy flat results
├── figures/
└── notebooks/
    ├── behavioral_sweep_colab.ipynb
    └── cross_model_comparison.ipynb
```

### Dependencies to Install

```
transformer_lens    # Core mechanistic interpretability library (v3 for Gemma 3 support)
einops              # Required by TransformerLens
jaxtyping           # Required by TransformerLens
sae_lens            # For Gemma Scope 2 SAE analysis + Pythia SAEs
scipy               # For bootstrap CIs, curve fitting
```

### Execution Order

```
═══ DONE ═══════════════════════════════════════════════════════════════
Phase 1 (Qwen2.5-0.5B):  10 → 11 → operating points A-E identified
Phase 2 (Qwen2.5-0.5B):  12-24b at 4 operating points → mechanistic story
Phase 1 (Qwen2.5-1.5B):  11c recalibration → operating points
Phase 2B.1 (Qwen2.5-1.5B): 12-23 at 4 operating points → cross-model validation

═══ IN PROGRESS ════════════════════════════════════════════════════════
Phase 1 (Qwen2.5-3B):    11 behavioral sweep on Colab (partial)
                          11c recalibration DONE, operating points selected

═══ TODO ═══════════════════════════════════════════════════════════════
Phase 2B.2 (Qwen2.5-3B): 12-23 at 4 operating points (if time)

Phase 1 (Gemma-3-1B-IT):
  10b → 11 (behavioral sweep) → identify operating points
Phase 2B.3 (Gemma-3-1B-IT):
  12-20 (mechanistic suite) → cross-family validation
  25 (SAE feature analysis) → unique Gemma contribution

Phase 1 (Pythia-410M):
  10c → 11_pythia (few-shot behavioral sweep) → confirm PI > RI in completion model
Phase 2B.4 (Pythia-410M):
  12-20 (if behavioral confirms PI > RI) → cleanest mechanistic analysis (MHA, native TL)
Phase 3.2 (Pythia-410M + 160M):
  26 (training dynamics) → when does PI > RI emerge during training?

Phase 3 (controls):
  27 (SSM control) → Mamba/RWKV behavioral comparison
  Theory framing → cite Wu et al., Ramsauer et al.
```

### Verification

Each experiment script should:

1. Save raw results as JSON in `results/{model_name}/{keys}k_{updates}u/`
2. Print summary statistics to stdout
3. Generate at least one diagnostic plot
4. Log to `{MODEL}_EXPERIMENT_LOG.md` with date, config, results, observations

Cross-model comparison verification:

- Phase 1 complete when: all 5 models have behavioral grids (grid size varies by context limit)
- Phase 2 complete when: mechanistic findings consistent across ≥3 models (Qwen 0.5B + 1.5B + Gemma 3 1B)
- Phase 3 complete when: training dynamics mapped (Pythia) AND SSM control confirms attention-specificity

### Paper Figures (Target)

1. **Behavioral landscape**: Multi-panel grid showing RI/PI accuracy heatmaps across (keys × updates) for each model
2. **Decay curves**: RI and PI accuracy vs updates, with CI bands, across model sizes and families
3. **Logit lens trajectories**: P(initial) and P(final) across layers, for different operating points
4. **Attention head classification**: Scatter + heatmap showing retrieval heads with primacy bias
5. **DLA decomposition**: Which heads/MLPs promote initial vs final value
6. **Activation patching**: (layer × position) → accuracy recovery heatmaps for RI vs PI
7. **Positional gradient**: P(v_i) across all N value positions — sharp cliff for RI, gradual rise for PI
8. **PI mass distribution**: Where peak probability lands vs where it should land, across interference levels
9. **Cross-model consistency**: Same mechanistic finding across Qwen-0.5B, Qwen-1.5B, Gemma-3-1B (3 models, 2 families)
10. **SSM control**: Behavioral comparison showing different interference profile
11. **SAE feature analysis** (NEW): Which Gemma Scope 2 features activate during PI failure vs success
12. **Training dynamics** (NEW): PI > RI emergence across Pythia-410M training checkpoints
13. **Instruction-tuned vs completion** (NEW): PI > RI comparison between IT models (Qwen, Gemma) and completion model (Pythia)

---

## Phase 2 Findings So Far (Qwen2.5-0.5B-Instruct)

*Updated 2026-02-20 from trial runs.*

### Critical methodology: Single-token values

Multi-token synthetic values (e.g., "Art42", "Hat380") are **unusable for mechanistic probing** because:
- Values in the same category share the same first subtoken ("Hat410" and "Hat416" both start with "Hat")
- Logit lens can't distinguish initial from final value at the first-token prediction step
- DLA returns all zeros when initial_token_id == final_token_id

**Fix:** Use a pool of 96 common English words verified as single-token in Qwen2.5-0.5B's tokenizer (e.g., "red", "dog", "iron", "jade"). Each value maps to exactly one unique token ID.

**Tokenizer trap:** Values appear in prompts as `"category: value"`, so the tokenizer sees `" red"` (space-prefixed), not `"red"`. Space-prefixed `" red"` (tid=2518) and bare `"red"` (tid=1151) are **different tokens**. The logit lens must track both variants: space-prefixed dominates in early-mid layers, bare form takes over at the final layer.

**Files:** `core/single_token_values.py` — pool + verification. `core/token_tracker.py` — position tracking by token ID (no substring heuristics).

### Step 2.1 + 2.2 + 2.3: Logit lens, attention, DLA (trial results)

**Script:** `experiments/12_phase2_trial.py` — runs all three analyses on matched RI/PI trials.

**Key results (3 trials/condition, 3 operating points):**

| Point | RI correct | PI correct | RI retrieval primacy | PI retrieval primacy |
|-------|-----------|-----------|---------------------|---------------------|
| A (easy, 1 update) | 3/3 | 2/3 | 1.000 | 1.000 (same value) |
| B (5 updates) | 3/3 | 1/3 | **0.874** | **0.166** |
| C (10 updates, 3 keys) | 2/3 | 1/3 | **0.881** | **0.100** |

- RI retrieval heads attend to initial value (primacy ≈ 0.87-0.98) → correct for RI
- PI retrieval heads shift toward final value (primacy ≈ 0.1-0.3), but shift is **incomplete** in some heads → explains PI failures
- Key retrieval heads: **L16H0, L16H3, L16H7, L19H6** — consistently high retrieval scores across trials
- DLA shows real per-head contributions up to ±12 logits in late layers (L20-L23)
- MLPs contribute significantly (up to ±22 logit_diff) — not purely attention

### Step 2.5: Positional Gradient Analysis (NEW)

**Script:** `experiments/13_positional_gradient.py`

**Question:** Is primacy strictly "first value only" or a gradual positional decay?

**Result (5 trials, 10 updates, keys=2, Qwen2.5-0.5B):**

```
RI (recall first):
  P(v_i):  0.7844  0.0000  0.0001  0.0002  ...  0.0038
  v1/v0 prob ratio: 0.000  ← SHARP CLIFF

PI (recall last):
  P(v_i):  0.0004  0.0000  0.0000  ...  0.1913  0.3582  0.1266
  ← GRADUAL RISE toward end, spread across late positions
```

**Interpretation:** The model has **exact primacy retrieval** (P concentrated entirely on v0) but only **approximate recency retrieval** (P spread across the last ~25% of positions). This asymmetry is the mechanistic root of PI > RI.

### Step 2.6: PI Mass Distribution Across Interference Levels (NEW)

**Script:** `experiments/14_pi_mass_distribution.py`

**Question:** In PI, does the peak probability land on the literal last value, and how does this change with interference?

**Result (8 trials/level, keys=2, updates=3/5/10/15/20):**

| Updates | RI peak | PI peak | PI @last | PI accuracy |
|---------|---------|---------|----------|-------------|
| 3 | v0 (rel=0.00) | v1.9 (rel=0.94) | 7/8 | 88% |
| 5 | v0 (rel=0.00) | v3.5 (rel=0.88) | 4/8 | 50% |
| 10 | v0.9 (rel=0.10) | v6.8 (rel=0.75) | 3/8 | 38% |
| 15 | v3.6 (rel=0.26) | v12.0 (rel=0.86) | 4/8 | 50% |
| 20 | v1.4 (rel=0.07) | v15.5 (rel=0.82) | 2/8 | 25% |

**Key finding:** As interference increases, the PI peak **drifts away from the actual last position** — landing around the 75-85th percentile instead of position N-1. The model retrieves "a late value" not "the last value." Meanwhile, RI consistently peaks at v0 regardless of interference level.

### Step 2.7: Activation Patching (COMPLETED — key result)

**Script:** `experiments/15_activation_patching.py`

**Design:**
- **Clean run:** 1 value per category (no interference) → model gets answer trivially
- **Corrupted run:** N updates per category (full interference) → model may fail
- **Patched run:** corrupted input, but at (layer, answer_position), swap in clean residual stream

**Result (4 trials/condition, keys=2, updates=10):**

| Patched position type | RI recovery | PI recovery |
|-----------------------|-------------|-------------|
| Initial value position | ~0% | ~0% |
| Final value position | ~0% | ~0% |
| ALL test value positions | ~0% | ~0% |
| **Query position** | — | **100-139%** |

**KEY FINDING:** Patching at VALUE positions does nothing. Patching at the QUERY position ("What was the last value of...?") fully restores PI accuracy (100%+ recovery).

**Interpretation:** PI failure is a **query-routing problem, not a value-storage problem**. The model stores all values correctly in context, but can't decode "last" into the correct attention pattern when interference is present. "First" works because attention naturally defaults to the first occurrence — no explicit routing needed.

**Layer localization:**
- Recovery peaks at layers 21-23 (late layers only)
- Layers 0-15 show near-zero or negative recovery
- The query-to-retrieval mapping forms in late layers

### Step 2.8: Systematic Head Identification + Ablation (COMPLETED — key result)

**Script:** `experiments/16_head_identification.py`

**Design:**
- 30 trials × 2 conditions at operating point (keys=2, updates=5)
- For each of 336 heads (24 layers × 14 heads), measure average attention to initial vs final value positions across all trials
- Classify each head by its PI behavior:
  - Primacy-biased: PI_primacy > 0.6 (attends to initial even when asked for last)
  - Recency-responsive: PI_primacy < 0.4 (correctly shifts to final in PI)
  - Condition-sensitive: |RI_primacy - PI_primacy| > 0.2 (changes behavior with instruction)
- Ablation: zero out specific head groups, measure accuracy change

**Result (Qwen2.5-0.5B, 30 identification trials, 20 ablation trials):**

Head classification:
| Category | Count | % of retrieval heads |
|---|---|---|
| Total heads | 336 | — |
| Retrieval heads (>10% value attention) | 65 | 100% |
| Primacy-biased (PI primacy > 0.6) | **8** | 12% |
| Recency-responsive (PI primacy < 0.4) | **53** | 82% |
| Condition-sensitive | 54 | 83% |

**KEY FINDING:** Most retrieval heads (53/65 = 82%) correctly shift to attend to the final value in PI. Only 8 heads (12%) are primacy-biased — they always attend to the initial value regardless of instruction.

**Smoking gun head: L16H3** — PI_primacy=0.992, condition_sensitivity=0.008. This head completely ignores the "first"/"last" instruction.

Ablation results:
| Config | RI | PI | RI-PI gap |
|---|---|---|---|
| Baseline | 90% | 55% | 35% |
| Knockout L16H3 alone | 80% | 50% | 30% |
| **Knockout all 8 primacy-biased** | **60%** | **55%** | **5%** |
| Knockout all 53 recency-responsive | 0% | 0% | 0% |

**Removing the 8 primacy-biased heads nearly eliminates the PI > RI asymmetry** (gap shrinks from 35% to 5%). RI drops because those heads were also helping RI. Removing recency-responsive heads kills everything — they do the actual retrieval work.

**Interpretation:** PI > RI is caused by a small minority (12%) of instruction-insensitive retrieval heads that always promote the first value, overriding the correct recency signal from the majority (82%) of heads.

---

### Step 2.9: Instruction Sensitivity Probe (COMPLETED)

**Script:** `experiments/17_instruction_sensitivity.py`

**Question:** Does the model even distinguish "first" vs "last" in its internal representations?

**Result (20 trials, keys=2, updates=5):**

Part A — Representation divergence (||resid_RI - resid_PI|| normalized):
- Representations DO diverge, peaking at **layer 16** (cosine similarity = 0.886)
- But divergence at the **query token position itself is zero** — the "first"/"last" word creates no local difference
- The divergence only appears at the answer position, propagated through attention

Part B — L16H3 attention to query word:
```
Region          RI        PI       Diff
initial       0.659     0.520    -0.139
query_word    0.000     0.000     0.000  ← ZERO
```

**KEY FINDING:** All 8 primacy-biased heads put **literally zero attention on the query word "first"/"last".** They cannot be instruction-sensitive because they never read the instruction. They attend to initial value positions and instruction tokens, ignoring the query entirely.

### Step 2.10: Forced Attention Intervention (COMPLETED — key result)

**Script:** `experiments/18_forced_attention.py`

**Question:** Is the problem in the QK circuit (where heads attend) or OV circuit (what they output)?

**Result (20 trials, keys=2, updates=5):**

| Config | RI | PI | Gap |
|---|---|---|---|
| Baseline | 90% | 40% | +50% |
| Knockout primacy heads | 50% | 40% | +10% |
| **Force to correct position** | **100%** | **100%** | **0%** |
| Force to wrong position | 5% | 25% | -20% |

**KEY FINDING:** Forcing the 8 primacy-biased heads to attend to the correct value position gives **100% accuracy on both RI and PI**. The gap goes to zero.

**Interpretation:**
- The OV circuit is **perfectly functional** — these heads correctly copy any value they attend to
- The problem is **purely in the QK circuit** — the heads don't know WHERE to look
- These heads are powerful — when aimed correctly they boost PI from 40% to 100%
- When aimed at the wrong position, they destroy accuracy (control condition: 5% RI)

### Step 2.11: Head-Level QK Fine-Tuning (TODO — intervention experiment)

**Question:** Can we fix PI > RI by fine-tuning only the QK circuits of the 8 primacy-biased heads?

**Design:**
- Generate 100-200 PI training examples where the model currently fails
- Loss = -log P(correct_last_value)
- Backprop through full model but only update `W_Q` and `W_K` for the 8 identified heads
- Parameters to tune: 8 heads × 2 matrices × (d_model × d_head) = 8 × 2 × (896 × 64) = ~917K params (0.14% of 630M)
- Freeze all other parameters (embeddings, other heads' QKV, all OV circuits, all MLPs)
- Train for few epochs with small learning rate

**Evaluation:**
- Does PI accuracy improve without degrading RI?
- Do the fine-tuned heads now attend to the query word?
- Does the primacy score of these heads shift from ~1.0 to ~0.0 in PI condition?
- Does the fix generalize to different (keys, updates) operating points?

**What this would prove for the paper:**
- If 0.14% parameter surgery fixes PI → the asymmetry is caused by a minimal, identifiable circuit defect
- The architectural capacity for recency retrieval EXISTS (OV circuits work) — only the routing is broken
- Provides a concrete "fix" that validates the mechanistic explanation

---

## Phase 2 Summary: The Mechanistic Story

The complete mechanistic explanation of PI > RI in Qwen2.5-0.5B:

1. **Retrieval architecture:** 65 out of 336 attention heads function as value retrievers, attending to value token positions from the answer position.

2. **Majority works correctly:** 53/65 retrieval heads (82%) are recency-responsive — they shift attention from initial to final value when the query says "last" instead of "first."

3. **Minority is broken:** 8/65 retrieval heads (12%), led by L16H3, are primacy-biased — they always attend to the first value regardless of instruction. L16H3 has condition_sensitivity=0.008 (effectively zero).

4. **WHY they're broken:** These heads put **zero attention on the query word** "first"/"last" — they literally never read the instruction. Their QK circuits attend to early-position value tokens and instruction prefix tokens, completely ignoring the query. They CAN'T be instruction-sensitive.

5. **The broken minority overrides the majority:** Despite 82% of retrieval heads correctly shifting, the 8 primacy-biased heads produce a strong enough initial-value signal to dominate the output. Ablating them reduces the RI-PI gap from 35%→5%.

6. **The OV circuit is fine:** Forcing the 8 heads to attend to the correct position gives **100% accuracy on both RI and PI**. The problem is purely in QK routing, not value encoding. These heads are powerful retrievers — when aimed correctly, they boost PI from 40%→100%.

7. **The failure is at query routing, not value storage:** Activation patching at value positions does nothing. Patching at the query position fully recovers PI accuracy (100%+). The values are stored correctly; the model can't decode "last" into the right retrieval pattern under interference.

8. **Primacy is exact, recency is approximate:** The initial value gets P=0.78 with a sharp cliff (v1=0.00). The final value probability is spread across the last 25% of positions, peaking at ~80th percentile, not the actual last position.

---

### Updated file structure

```
mechanistic_probing_v2/
├── core/
│   ├── __init__.py
│   ├── model_loader.py           # TransformerLens + HuggingFace loading
│   ├── dataset.py                # Trial generation, 46 categories, interleaving
│   ├── single_token_values.py    # 96 single-token values for Phase 2
│   ├── token_tracker.py          # Position tracking by token ID
│   └── analysis_utils.py         # Logit lens, attention, DLA, patching helpers
├── experiments/
│   ├── 10_setup_qwen.py          # Setup validation
│   ├── 11_behavioral_sweep.py    # Phase 1: full grid sweep
│   ├── 12_phase2_trial.py        # Phase 2: logit lens + attention + DLA
│   ├── 13_positional_gradient.py # Phase 2: P(v_i) across all value positions
│   ├── 14_pi_mass_distribution.py # Phase 2: where PI probability peaks
│   ├── 15_activation_patching.py  # Phase 2: causal patching
│   ├── 16_head_identification.py  # Phase 2: head classification + ablation
│   ├── 17_instruction_sensitivity.py # Phase 2: does model distinguish first/last?
│   ├── 18_forced_attention.py     # Phase 2: force heads to correct position
│   ├── 19_positional_bias_sweep.py # Phase 2: recency bias λ sweep (negative result)
│   └── 20_minority_override_analysis.py # Phase 2: why 8 heads override 53
├── results/                       # JSON outputs
├── figures/
└── notebooks/
    ├── behavioral_sweep_colab.ipynb  # Phase 1 Colab runner
    └── test_colab.ipynb
```

### Remaining work

**Essential (must do before submission):**
1. Finish Phase 1 behavioral sweep (Colab, in progress)
2. Cross-model replication: run experiments 12-18 on Qwen-1.5B and Gemma-2B
3. SSM control: behavioral sweep on Mamba/RWKV (Phase 3, Step 3.1)
4. Head-level QK fine-tuning intervention (Step 2.11)

**Strengthening (if time permits):**
5. Scale up trial counts to 100+ for head identification
6. Prompt format experiments (Step 2.12)
7. Paper-ready visualizations (heatmaps, scatter plots)

### Step 2.13: Why Does the Minority Override the Majority? (TODO — critical mechanistic gap)

**Problem:** We claim 8/65 primacy-biased heads override 53/65 recency-responsive heads, but haven't explained WHY 12% can beat 82%. Without this, the story is descriptive, not mechanistic.

**Three testable hypotheses:**

**H1: Disproportionate DLA magnitude.**
Each primacy-biased head contributes more logit-difference per head than each recency-responsive head. Even outnumbered 8:53, if the average primacy head pushes 5× harder than the average recency head, the minority wins.
- **Test:** Run 30+ trials at operating point B (keys=2, updates=5). Compute per-head DLA (logit_diff = contribution to logit(initial) - logit(final)) for all 65 retrieval heads. Compare distributions: mean |logit_diff| for primacy-biased vs recency-responsive heads. Also compute total sum: Σ(primacy_heads DLA) vs Σ(recency_heads DLA) — which team pushes harder in aggregate?

**H2: OV circuit gain (static weight analysis).**
These heads may have larger output norms, meaning they amplify whatever they attend to more strongly. This is a property of the weight matrices, not the input.
- **Test:** For each head, compute `||W_OV||` = `||W_V @ W_O||` (Frobenius norm). Also compute the projection onto unembedding directions: for each head, measure `||W_OV @ W_U[:, initial_value_token]||` — how strongly the OV circuit projects onto the initial value's unembedding direction. Compare primacy-biased vs recency-responsive heads.

**H3: Layer position / last-word effect.**
L23H0 (the only primacy-biased head in the final layer) writes to the residual stream AFTER all recency-responsive heads. It gets the "last word." Even if layers 15-22 correctly point toward the final value, L23H0 overwrites it.
- **Test:** Single-head ablation of each of the 8 primacy-biased heads individually. If L23H0 has disproportionate causal impact (larger accuracy change when knocked out alone), the last-word effect is real. We already have L16H3 ablation data — need L0H4, L15H0, L16H9, L17H0, L17H7, L18H12, L23H0.

**These hypotheses aren't mutually exclusive.** The likely answer is a combination. The experiment should report all three measurements so we can quantify relative contributions.

**What this proves for the paper:**
- Completes the mechanistic explanation from "which heads" to "why those heads dominate"
- Transforms a descriptive finding into a quantitative mechanistic claim
- Provides specific numbers: "primacy-biased heads produce Xσ larger DLA per head, and L23H0 alone accounts for Y% of the primacy push"

**Script:** `experiments/20_minority_override_analysis.py`

---

### Step 2.15: Granular Query-Position Patching (TODO — disambiguate query routing claim)

**Problem with current Step 2.7 result:** We showed that patching the residual stream at the query position recovers PI 100%+. We framed this as "PI failure is a query-routing problem." But this framing may not survive scrutiny.

**Why:** Patching the full residual stream at the query position replaces EVERYTHING accumulated there — the query word representation AND any value information that earlier-layer attention heads routed to the query position. So the result is ambiguous between two claims:

- **(A) Query representation failure:** The model can't decode "last" into the correct retrieval pattern. The instruction processing itself is broken under interference.
- **(B) Value routing through query position:** Value information gets routed to the query position in early/mid layers (via cross-position attention), and THAT accumulated signal is what's corrupted. The query word processing is fine, but the values it gathered are wrong.

These have different implications. (A) means the problem is in instruction comprehension under load. (B) means the problem is in value transport that happens to pass through the query position. A NeurIPS reviewer could point out this ambiguity.

**Experiments to disambiguate:**

**(1) Component-specific patching at query position:**
- Patch ONLY attention output (`hook_attn_out`) at query position → tests whether the attention mechanism's value routing through this position is the issue
- Patch ONLY MLP output (`hook_mlp_out`) at query position → tests whether the MLP's local processing of the query word is the issue
- If attention patching recovers PI but MLP doesn't → problem is cross-position value routing (supports B)
- If MLP patching recovers PI → problem is in local query processing (supports A)

**(2) Layer-specific patching at query position:**
- Patch at early layers (0-7) only → these process the query word locally, before much cross-position information has accumulated
- Patch at mid layers (8-15) → cross-position routing is starting
- Patch at late layers (16-23) → full accumulated representation
- If early-layer patching recovers PI → the query word representation itself is already corrupted from the start (strong support for A)
- If only late-layer patching works → the problem is in accumulated cross-position information (supports B)

**(3) Combined: layer × component sweep at query position:**
For each layer in [0, 4, 8, 12, 16, 20, 23], patch attn_out and mlp_out separately. This gives a 2D map of where the critical information sits.

**Script:** `experiments/22_query_patching_granular.py`

**Result (10 trials, keys=2, updates=10, PI condition):**

| Layer range | resid_post | attn_out | mlp_out |
|---|---|---|---|
| Early (L0-8) | -1% | -0% | +0% |
| Late (L16-23) | +66% | +6% | +1% |
| L22 | +99% | +1% | +5% |
| L23 | +100% | -3% | -4% |

**Conclusion: Hypothesis (B) confirmed, (A) rejected.**
- Early-layer query patching gives 0% recovery → the query word "last" is processed correctly in early layers
- Only late-layer full-residual patching works (L16+ → +66%, L22 → +99%)
- Neither attn_out nor mlp_out alone recovers PI — the corruption is in the CUMULATIVE residual stream
- The corruption starts at L16, exactly where the primacy-biased heads (L15-18) operate

**Revised framing:** "PI failure is NOT a query-routing problem in the traditional sense. The model correctly processes 'last' in early layers. But by L16-23, primacy-biased heads write initial-value information into the residual stream, which accumulates at the query position through cross-position attention. By the time the model generates an answer at the final layer, the query position's representation has been overwritten with primacy-biased value information. The correct 'last' signal is drowned out by accumulated wrong-value signals, not by a failure to understand the instruction."

This is a more nuanced and defensible claim than "query routing failure" — it locates the corruption in specific layers (L16+) and connects it to the identified primacy-biased heads.

### Step 2.15b: Ablation + Patching Interaction (COMPLETED — partial causal chain confirmed)

**The gap:** We showed (a) ablating 8 primacy heads closes the PI-RI gap, and (b) patching query position at L16+ recovers PI. But we haven't shown these are the SAME mechanism.

**Script:** `experiments/23_ablation_patching_interaction.py`

**Result (10 PI trials, keys=2, updates=10):**

Both normal and ablated baselines show 0% PI accuracy (harder operating point than previous experiments).

| Layer | Normal Recovery | Ablated Recovery | Difference |
|-------|----------------|-----------------|------------|
| L0 | +0% | +1% | +1% |
| L4 | +1% | +2% | +2% |
| L8 | +2% | +10% | +9% |
| L12 | +2% | +5% | +3% |
| **L16** | **+36%** | **+5%** | **-31%** |
| **L18** | **+35%** | **+9%** | **-26%** |
| L20 | +89% | +81% | -8% |
| L22 | +90% | +96% | +6% |
| L23 | +100% | +100% | +0% |

**Interpretation — Two-layer causal story:**

1. **L16-L18: Primacy heads ARE the cause.** Ablating the 8 heads reduces query-position patching recovery from 35% → 5-9% at these layers. This is a ~80% reduction. 6 of the 8 primacy heads live in L15-L18 (L15H0, L16H3, L16H9, L17H0, L17H7, L18H12). The causal chain at L16-L18 is confirmed: **primacy heads corrupt the query-position representation, which downstream layers read.**

2. **L20-L23: Independent corruption source.** Both normal and ablated runs show 81-100% recovery when patching at L20+. Ablation doesn't reduce this. Something ELSE is corrupting the query position at L20+, independent of the 8 primacy heads. This could be:
   - MLP layers in L20-23 that encode positional bias
   - Other non-retrieval heads that contribute to the corruption
   - The residual stream's accumulated drift from many small biases

**Revised mechanistic story:** PI failure involves at least two layers of corruption at the query position:
- Layer 1 (L16-L18): Primacy heads inject initial-value bias → confirmed causal
- Layer 2 (L20-L23): Additional corruption from other components → source unknown

**Open question:** What causes the L20+ corruption? Candidates: MLP layers, non-retrieval heads, or accumulated small biases from many components. This could be investigated by patching individual components (attn_out vs mlp_out) at L20+ with heads ablated.

---

### Step 2.13b: Open Question — How Does the Minority Actually Override? (TODO — needs larger runs)

**The puzzle:** DLA shows recency heads win the logit-space tug of war overwhelmingly (recency sum=-11.00 vs primacy sum=+0.83, net=-12.86 toward final value). Yet PI accuracy is only 55%. The ablation result (gap 35%→5% when primacy heads removed) is real. But the mechanism is NOT raw logit magnitude.

**Hypotheses rejected (30 trials, Step 2.13):**
- H1 (DLA magnitude): Primacy heads push 0.3x per head, not stronger. REJECTED.
- H2 (OV gain): Primacy heads have 0.46x W_OV norm. REJECTED.
- H3 (Last-word effect): L23H0 knockout has 0% PI change. REJECTED.

**Hypotheses to test with larger runs (100+ trials):**

1. **High variance / bimodal behavior.** DLA is averaged over 30 trials. On the ~45% of PI trials that FAIL, primacy heads might dominate. On the ~55% that succeed, recency wins. The average hides bimodal behavior.
   - Test: Split DLA by trial outcome (correct vs incorrect). If primacy DLA is much larger on incorrect trials → the override is trial-dependent, not universal.

2. **Nonlinear / indirect effects.** DLA assumes linear additivity. But heads in L15-18 write to the residual stream, and heads in L19-23 READ from it. Primacy heads in L16 could corrupt the intermediate representation, causing downstream recency heads to retrieve wrong information. This chain effect wouldn't show up in DLA.
   - Test: Run logit lens WITH vs WITHOUT the 8 primacy heads (ablation). Track P(initial) and P(final) layer by layer. If ablating the 8 heads changes the trajectory starting at layer 16 → confirms indirect/information-flow mechanism.

3. **The model outputs a THIRD token.** PI failures might not be "initial value wins over final value." The model might output something entirely different — neither initial nor final. The DLA measures initial-vs-final, but the actual failure mode could be something else entirely.
   - Test: On PI failure trials, classify the actual output: is it the initial value (primacy intrusion), an intermediate value, or something unrelated? We have this from Step 2.6 (PI peak at ~80th percentile), but should connect it to the DLA analysis.

4. **Ablation result is noisy.** The gap-closing result (35%→5%) used only 20 trials. Today's baseline was RI=80%, PI=55% (gap=25%, not 35%). With 20 trials, 95% CI on accuracy is ±~20%. Need 100+ trials for reliable ablation.
   - Test: Rerun ablation with 100 trials.

**Priority:** Run hypotheses 1 and 2 first (split DLA by outcome + logit lens ablation). These directly address the mechanism. Hypothesis 4 (more trials) should be done for ALL experiments when scaling up for the real runs.

**Script:** `experiments/21_override_mechanism.py` (TODO)

---

### Step 2.16: Copy-Suppression Hypothesis Test (TODO — CRITICAL, highest-payoff single experiment)

**Script:** `experiments/28_copy_suppression_test.py` (NEW)

**The puzzle this solves:** DLA shows recency heads win overwhelmingly (sum=-11.00 vs primacy sum=+0.83). Yet PI accuracy is only 55%, and ablating the 8 primacy heads closes the gap from 35%→5%. The primacy heads are causal (ablation proves it) but aren't winning through raw logit magnitude (DLA proves it). How?

**Hypothesis (from McDougall et al., BlackboxNLP 2024, [arxiv:2310.04625](https://arxiv.org/abs/2310.04625)):** The primacy-biased heads are not promoting V1 (initial value). They are **suppressing V2** (final value). Copy-suppression heads have negative eigenvalues in their OV weight matrix — they flip the sign of what they read, effectively negating downstream predictions.

If true, the mechanism is:
1. Recency heads (majority) build up a V2 signal in the residual stream (DLA = -11.00 toward V2)
2. Primacy heads (minority) attend to positions carrying V2-related information and write a **negation** of that signal back into the residual stream
3. The V2 signal is cancelled. V1 wins by default (because causal masking has embedded V1 deeply in the residual stream from early layers, and nobody is suppressing V1)
4. DLA sees weak positive from primacy heads because suppressing V2 ≠ promoting V1 in logit space — it shows up as a small indirect push toward V1

This resolves every contradiction in the data:
- DLA says recency wins → correct, before suppression
- PI accuracy is 55% → V2 signal is partially cancelled by suppression
- Ablating primacy heads improves PI → removing the suppression lets V2 through
- Primacy heads have weak DLA → because suppression creates an indirect effect, not a direct V1 logit push

**Three tests (in order of conclusiveness):**

**Test 1: OV Eigenspectrum (static weight analysis, zero forward passes)**

For each of the 65 retrieval heads (all operating points, both 0.5B and 1.5B):
```
W_OV = W_V[layer, head] @ W_O[layer, head]   # [d_head, d_head]
eigenvalues = eigvals(W_OV).real
dominant_sign = sign(eigenvalue with largest absolute value)
negative_fraction = count(eigenvalues < 0) / total
```

Expected result if hypothesis is correct:
- Primacy-biased heads (L16H3, L19H1): dominant eigenvalues are NEGATIVE
- Recency-responsive heads: dominant eigenvalues are POSITIVE
- Clear separation between the two groups

**Test 2: Directional Suppression Score (static weight analysis)**

For each head, project the OV circuit onto V2's unembedding direction:
```
v2_unembed = model.W_U[:, v2_token_id]   # [d_model]

# Full OV projection: what does this head do to V2-like information?
W_OV_full = model.W_V[layer, head] @ model.W_O[layer, head]

# Suppression score: negative = head suppresses V2
suppression_score = v2_unembed @ W_OV_full @ v2_unembed
```

Run across 100 trials (different V2 tokens each time) and average. Compare distributions for primacy-biased vs recency-responsive heads.

Expected result: primacy heads have negative suppression scores, recency heads have positive.

**Test 3: DLA Decomposed into V1-promotion vs V2-suppression (requires forward passes)**

During 100 PI trials at each operating point, for each head compute:
```
head_output = cache["result", layer][:, -1, head, :]   # [d_model]

dla_v1 = head_output @ model.W_U[:, v1_token_id]   # contribution to V1 logit
dla_v2 = head_output @ model.W_U[:, v2_token_id]   # contribution to V2 logit
```

Standard DLA (logit_diff = dla_v1 - dla_v2) hides the mechanism. Decomposing reveals it:
- Normal retrieval head helping PI: dla_v1 ≈ 0, dla_v2 >> 0 (promotes V2)
- Copy-suppression head hurting PI: dla_v1 ≈ 0, dla_v2 << 0 (suppresses V2)
- Direct V1 promoter: dla_v1 >> 0, dla_v2 ≈ 0 (promotes V1)

If primacy heads show (dla_v1 ≈ 0, dla_v2 << 0) → copy-suppression confirmed. They're not promoting V1 at all — they're tearing down V2.

**Visualization for paper:**
1. Scatter: DLA_v1 vs DLA_v2 per head, colored by primacy classification. Copy-suppression heads appear in the bottom-left quadrant (low V1, negative V2).
2. Eigenvalue distributions: violin plot comparing primacy vs recency heads.
3. Suppression score histogram: primacy heads in negative territory, recency heads in positive.

**Outcome matrix:**

| OV eigenvalues | DLA_v2 of primacy heads | Interpretation |
|---|---|---|
| Negative dominant | Strongly negative | **Copy-suppression confirmed.** Full mechanism identified. Paper headline: "Primacy arises from active suppression of recent values, not retrieval of old ones." |
| Negative dominant | Near zero | Partial — OV structure supports suppression but effect is weak in practice. Report as supporting evidence, not main claim. |
| Positive dominant | Near zero or positive | **Copy-suppression rejected.** Primacy heads are weak V1 promoters, not V2 suppressors. Fall back to information-flow corruption (exp 23) or accept the mechanism is indirect/nonlinear. |
| Mixed | Mixed | Inconclusive. Some heads may suppress, others promote. Report the heterogeneity. |

**Models:** Run on both Qwen2.5-0.5B and 1.5B. Focus on the strictly-classified heads (L16H3, L19H1) plus the broader set of retrieval heads for context.

**Dependencies:** None — Test 1 and 2 are pure weight analysis. Test 3 reuses the same forward passes as exp 12.

**Effort:** 1 day (mostly analysis code + interpretation)

---

### Step 2.14: Positional Recency Bias Sweep (COMPLETED — mixed result)

**Script:** `experiments/19_positional_bias_sweep.py`

**Design:** Add linear ramp `λ × (j / seq_len)` to pre-softmax attention scores of 8 primacy-biased heads at the answer position. Sweep λ ∈ [0, 0.25, 0.5, ..., 10.0]. Two modes: blind (all positions) and oracle (value positions only).

**Result (blind mode, 20 trials, keys=2, updates=5):**

| λ | RI | PI | Gap |
|---|---|---|---|
| 0 (baseline) | 95% | 65% | +30% |
| 2.0–5.0 | 95% | 70% | +25% |
| 10.0 | 85% | 65% | +20% |

**Result — blind mode (20 trials, keys=2, updates=5):**

| λ | RI | PI | Gap |
|---|---|---|---|
| 0 (baseline) | 95% | 65% | +30% |
| 2.0–5.0 | 95% | 70% | +25% |
| 10.0 | 85% | 65% | +20% |

Blind mode: only +5% PI at best. At high λ, heads attend to query tokens instead of final values. No sweet spot.

**Result — oracle mode (20 trials, keys=2, updates=5):**

| λ | RI | PI | Gap |
|---|---|---|---|
| 0 (baseline) | 90% | 70% | +20% |
| 5.0 | 90% | 80% | +10% |
| **10.0** | **90%** | **90%** | **+0%** |

Oracle mode: **PI goes from 70% to 90% with ZERO RI degradation.** Gap closes completely at λ=10. RI stays rock-solid at 90% across all λ values.

**Key insight:** Blind bias fails because it's diluted across non-value tokens (query, instruction) — at high λ the heads attend to query tokens at the end. Oracle bias concentrates the correction on value positions only, so the recency ramp pushes attention specifically toward late VALUE tokens.

**Proof (Step 2.14b, `experiments/19b_bias_attention_proof.py`):**
Extracted attention patterns of the 8 primacy heads under all three modes (20 PI trials):

| Mode | → Initial | → Final | → Other Values | → Non-value |
|---|---|---|---|---|
| Baseline | 12.6% | 2.9% | 13.7% | 70.9% |
| Blind λ=10 | 5.0% | 6.9% | 9.4% | **78.7%** |
| Oracle λ=10 | 10.6% | **40.7%** | **44.6%** | **4.1%** |

Blind bias shifts attention from initial value to NON-VALUE tokens (78.7%), not to the final value. Oracle bias shifts attention to FINAL VALUE (40.7%) + other values (44.6%). This directly proves: (a) the primacy heads' default attention goes to non-value tokens + initial value, (b) blind bias pushes them to the wrong place (later non-value tokens), (c) oracle bias pushes them to the right place (later value tokens). The blind/oracle contrast is not about bias strength — it's about semantic targeting.

**What this proves:**
- The primacy bias IS overcomable with positional correction, but only when the bias is semantically targeted (value positions only)
- Pairs with forced attention result: both show OV circuits work, problem is purely QK routing
- Oracle bias is a softer version of forced attention — doesn't force to a single position, just nudges the distribution toward later values
- RI doesn't degrade because the 53 condition-sensitive heads overwhelm any recency bias in RI

---

### Step 2.12: Prompt Format Experiments (TODO — low priority)

**Question:** Can changing the prompt format fix PI without any model intervention?

Three variants to test:

**(a) Query BEFORE stream (WORTH DOING):**
```
You will need to recall the LAST value of color.

Read the following key-value stream:
color: red
animal: dog
color: blue
...

What was the last value of color?
```
The instruction appears early in the sequence where primacy-biased heads actually attend (they're biased toward early positions). If this fixes PI → the heads CAN process the instruction, they just can't see it when it's at the end. This has mechanistic implications — it means the QK routing failure is positional, not semantic.

**(b) Repeated emphasis (PROBABLY WON'T WORK):**
```
...stream...

IMPORTANT: Recall the LAST (most recent) value.
What was the last value of color?
```
Logic: give the instruction a second chance to be seen. However, our instruction sensitivity results show the 8 primacy-biased heads put zero attention on the query word regardless — they don't attend to instruction-type tokens at the end of the sequence. Adding more emphasis at the same position is unlikely to change this. Only worth testing if (a) shows position matters.

**(c) Completion format (DIFFERENT TASK):**
```
...stream...

The last value of color was:
```
No question, just sentence completion. Bypasses query routing entirely — the model fills in the blank via pattern completion rather than answering a question. If this fixes PI it proves the failure is in question-answering format specifically, but it's not really "fixing PI" — it's avoiding the PI task. Less useful for the mechanistic narrative. Include as a control if time permits.

**Theory (writing, not code):**
8. Hopfield framing: cite Ramsauer et al. + Wu et al., present empirical findings as mechanistic validation of theoretical primacy bias prediction

---

## Current Status (2026-02-22): Qwen 0.5B + 1.5B Complete, Expanding to Gemma 3 + Pythia

### What's Done

Ran 12 experiments × 4 operating points × 2 models = 96 experiment runs (all at 100 trials).

**Models completed:** Qwen2.5-0.5B-Instruct (24L, 14H) and Qwen2.5-1.5B-Instruct (28L, 12H)
**Model in progress:** Qwen2.5-3B-Instruct (behavioral sweep on Colab, recalibration done)
**Models planned:** Gemma-3-1B-IT (cross-family + SAE), Pythia-410M + 160M (training dynamics + community anchor)

**Data setup:** Single-token English words (2300 pool) + 46 semantic categories. Operating points regime-matched between models (1.5B reaches same regimes with fewer updates).

### Solid Findings (not dependent on head classification)

| # | Finding | Evidence | Status |
|---|---------|----------|--------|
| 1 | PI internal representation degrades with interference, RI stays strong | Logit lens: RI P(init) 0.56-0.99, PI P(final) 0.86→0.25 | Solid, both models |
| 2 | Primacy is a sharp cliff (first position only) | v1/v0 ratio 0.000-0.104 across all points | Solid, both models |
| 3 | Late-layer patching at query position recovers PI ~100% | Activation patching at L21+/L27+ | Solid, both models |
| 4 | Only late-layer query patching works, early layers do nothing | Exp 22: early +0%, late +48-82% (0.5B); late +96% at Point C (1.5B) | Solid on 0.5B, partial on 1.5B |
| 5 | Heads ignore the literal query word "first"/"last" | Attention to query word ≈ 0.003 in both conditions | Solid, both models |
| 6 | Linear positional bias correction doesn't help | Exp 19: no sweet spot found | Solid, both models |
| 7 | Circuit lives in last ⅓ of network | DLA: 88-91% of total DLA in final third of layers | Solid, both models |

### The Head Classification Problem

**Issue:** We used three different criteria across experiments and got different answers each time.

- Exp 16 (attention-based, retr>0.1, pi_prim>0.6): 6 heads on 0.5B, 5 on 1.5B
- Stricter filter (retr>0.3, pi_prim>0.7, DLA>0): 1 head on each (L16H3, L19H1)
- DLA-based top-5: different heads again

**This caused a false alarm:** Exp 18 forced all 5 attention-classified heads → PI improved on 0.5B (+18%) but WORSENED on 1.5B (-19%). We spent significant effort investigating "why is the mechanism different across models?" (theories about QK vs OV circuits, residual contamination, etc.)

**Resolution:** Exp 24b forced ONLY the single strictly-classified head → both models improve +20%. The divergence was caused by forcing misclassified non-primacy heads that were actually helping PI. The mechanism IS the same on both models.

**But the resolution is not satisfying.** We found the right answer by stacking filters until we got consistency. That's post-hoc. A reviewer asking "how did you identify primacy heads?" would see we tried multiple criteria.

### Options Going Forward

**Option A: Drop head-level claims from the paper.**
- Frame the story around findings 1-7 (all solid, no head classification needed)
- Narrative: "PI fails because the final value's representation degrades in late layers. Causal patching confirms the information exists but is inaccessible. The failure is architectural — caused by how causal attention accumulates positional bias."
- Pros: Clean, well-supported, no attack surface on methodology
- Cons: Less mechanistic depth, doesn't explain WHY specific heads behave this way

**Option B: Do head classification properly using established methods.**
- Use Wu et al. (2024) retrieval score (copy-paste frequency, threshold >0.1) adapted for our task
- Or use Voita et al. (2019) criterion (>90% of max attention to fixed relative position)
- Define a novel "primacy bias score" = E[attn to first value pos] - E[attn to last value pos] in PI, averaged over 200+ trials
- Validate with shuffled position controls (separate position bias from content bias)
- Pros: Stronger mechanistic claims, deeper story
- Cons: Needs justification for any novel metric, more experiments, reviewer attack surface on thresholds

**Option C: Use DLA as the primary metric (following Wang et al. 2022 IOI circuit).**
- Classify heads by their causal contribution to logit(init) - logit(final)
- No threshold needed — report continuous DLA values for all heads
- Show distribution, identify natural clusters if they exist
- The "primacy head" question becomes "which heads have positive PI DLA?" (promote init in PI condition)
- Pros: Causal, continuous, no arbitrary threshold, follows established IOI methodology
- Cons: DLA is per-trial noisy, need many trials for stable estimates

**Current recommendation:** Option A or C. Option A is the safe path — we already have 7 solid findings. Option C adds depth using an established methodology without inventing new metrics. Option B is the strongest but highest risk/effort.

### Known Issues (Must Resolve Before Final Results)

#### Issue 1: Single-Token Value Pool Contamination

**Problem:** Our "single-token" value pool (2,300 words) was verified with space-prefix tokenization (`" storm"` = 1 token). But during generation, the model outputs tokens WITHOUT space prefix (`"storm"` may be 1 or 2 tokens depending on the word). ~34% of words in the pool are multi-token when decoded without space prefix (e.g., `"grain"` → `["gr", "ain"]`).

**Impact on accuracy:** ~25% of trials scored as "wrong" are actually correct — the model predicted the right first sub-token but we compared it to the full word. Reported accuracy ~46% is actually ~72%.

**Impact on logit lens/DLA:** Logit lens checks P(space-prefixed token ID) at position -1. This is the CORRECT thing to check — the model's prediction logits at position -1 correspond to the first generated token, and for single-token words (with space prefix), that IS the answer token. But for multi-token words, P(space-prefixed ID) might be low even when the model is "trying" to produce that word via sub-tokens.

**Impact on failure classification (exp 21c):** Many "garbage" and "initial_prefix" failures are actually correct answers with multi-token words. The primacy intrusion rate (3-5%) may be even lower.

**Fix options:**

Option A — Strict pool filter (RECOMMENDED):
- Filter to 1,292 words that are single-token BOTH with and without space prefix
- All evaluation and probing is correct by construction
- Pool is large enough for all operating points
- Requires re-running all experiments

Option B — Fix evaluation only:
- Generate full answer autoregressively (up to EOS)
- Compare decoded full text to expected answer (string comparison)
- Probing still at position -1 (correct for logit lens — this is where prediction happens)
- Faster fix, no pool change, but multi-token words still have noisy logit lens signal

Option C — Generate + locate answer token:
- Generate full answer, tokenize it, find the answer token(s) in generated sequence
- For probing: run model on (input + generated prefix) to get cache at the answer prediction position
- Most general, handles any answer format
- But significantly more complex pipeline, and for single-token answers it reduces to position -1 anyway

**Current status:** Not resolved. All existing results (0.5B × 4 points, 3B × 1 point) have this issue. Directional findings (PI > RI, primacy cliff, late-layer patching) are valid but absolute accuracy numbers are wrong.

**Recommendation:** Option A. It's the cleanest fix. 1,292 words is sufficient. Same tokenizer for 0.5B and 3B (verified). All experiments need re-running anyway if we change the pool.

#### Issue 2: Answer Position Assumption

**Problem:** All mechanistic experiments use `answer_position = -1` (last input token). This works because:
1. The chat template adds `<|im_start|>assistant\n` at the end
2. Position -1 is the `\n` after `assistant`
3. `logits[0, -1, :]` predicts what comes AFTER this `\n` — which is the first generated token
4. The system prompt ("Answer with ONLY the exact value") ensures the first generated token IS the answer

**Verification:** Tested with autoregressive generation on 20 trials — model outputs just the value, no preamble. 18/20 outputs were exactly the expected word (or multi-token spelling of it). 0/20 had explanatory text before the answer.

**Risk:** This assumption holds for Qwen2.5 instruction-tuned models with our system prompt. It may NOT hold for:
- Non-instruction-tuned models (Pythia) — these need few-shot format, different answer position
- Models that don't follow "answer with ONLY the value" instruction
- Different prompt formats

**Status:** Valid for current Qwen experiments. Must be re-verified for each new model family. Pythia will need a completely different approach (pattern completion, not question answering).

#### Issue 3: Hardcoded Layer/Threshold Decisions

**Fixed:**
- Layer sampling in exp 22/23: now dynamic based on `model.cfg.n_layers`
- Early/late layer boundary in exp 22/23: now derived from exp 15 retrieval onset layer (data-driven, per model)
- Hardcoded layer reference in exp 20: now uses `max(layer)` from data
- Fallback primacy heads in exp 17/18/19/21a: now raises error instead of using wrong heads

**Still arbitrary (acceptable for now, document in paper):**
- Head classification thresholds (exp 16): `RETRIEVAL_THRESHOLD=0.10`, `PRIMACY_HIGH=0.6`, `PRIMACY_LOW=0.4` — will be revisited when head classification method is decided (Option A/B/C)
- Interpretation thresholds in exp 22 (recovery > 0.3), exp 18 (improvement > 0.1) — only affect printed interpretation, not saved data. A reviewer can re-interpret the numbers.
- Lambda sweep values in exp 19 — log-spaced 0 to 10, capped because exp 19b showed λ=10 saturates attention to non-value tokens
- Retrieval onset threshold (10% recovery) — could sweep 5-20% to show onset layer is stable

### Remaining Work

**Must do:**
- Resolve Issue 1 (value pool) — decide Option A/B/C, re-run affected experiments
- Decide on Option A/B/C for head classification
- Gemma-3-1B-IT: setup, behavioral sweep, mechanistic suite (Phase 2B.3)
- Gemma-3-1B-IT: SAE feature analysis with Gemma Scope 2 (exp 25)
- Pythia-410M: setup, behavioral sweep with few-shot format (Phase 2B.4)
- Cross-model comparison analysis (Qwen 0.5B + 1.5B + Gemma 3 1B, minimum 3 models)
- Paper-ready visualizations

**Should do:**
- Pythia training dynamics: sweep 154 checkpoints (Phase 3, Step 3.2)
- Pythia-160M: same as 410M for size scaling within Pythia family
- Qwen2.5-3B: finish behavioral sweep, mechanistic experiments if time
- SSM control: Mamba/RWKV behavioral sweep (Phase 3, Step 3.1)
- Pythia mechanistic probing (if behavioral confirms PI > RI — cleanest analysis due to MHA)

**Nice to have:**
- QK fine-tuning intervention (Step 2.11)
- Prompt format experiments (Step 2.12)
- Pythia SAE analysis (EleutherAI/sparsify pre-trained SAEs available)
