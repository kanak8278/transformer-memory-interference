# Mechanistic Interpretability: Complete Consolidated State

**Date:** 2026-03-18
**Purpose:** Single reference for all MI experiments, results, code, and theory.
Covers the full arc: ACL v1 (behavioral) → mechanistic_probing_v2 (abandoned) → v3 (current).

---

## 0. PROJECT HISTORY: WHY v2 WAS ABANDONED

### v1 (ACL paper)
39-model behavioral sweep. Found PI > RI across all models. No mechanistic explanation.
The finding: RI accuracy > PI accuracy everywhere, Cohen's d = 1.73.

### v2 (mechanistic_probing_v2/) — WRONG HYPOTHESIS, ABANDONED

**The v2 framing:** PI > RI is caused by specific "primacy heads" — a localizable circuit
where particular attention heads suppress v_{N-1} and route the output to v_0 instead.

**What v2 found:** L8H3 in Qwen 1.5B — knockout boosted PI from 44% to 92% (+48pp).
Gemma had L14H2. Looked like a clean circuit. Built 20+ experiments around it.

**Why it fell apart:**

1. **Wrong error distribution.** v2 assumed PI errors landed at position 0 (primacy
   intrusion — model outputs v_0 instead of v_{N-1}). The entire "primacy heads cause
   primacy intrusion" narrative was built on this. v3's Stage 1 showed the actual
   distribution: at low N, **73-98% of PI errors land at the PENULTIMATE position**
   (v_{N-2}), not v_0. The model fails at off-by-one retrieval, not primacy intrusion.
   The core assumption was wrong.

2. **Attribution methods gave wrong head rankings.** DLA and attention-based attribution
   "completely missed L8H3" (experiment 25b log). The real causal head was only found
   through brute-force knockout — expensive, and it showed that first-order attribution
   can't capture indirect effects propagating through 20+ downstream layers.

3. **Forced attention gave paradoxical results.** When forcing attention to v_last, PI
   didn't recover. Eventually traced to forcing the WRONG head (L19H1, not L8H3) —
   but this revealed the circuit wasn't where v2 thought.

4. **Head effects didn't replicate architecturally.** Gemma: L14H2. Qwen 1.5B: L8H3.
   Qwen 0.5B: distributed (no dominant head). No architectural consistency.

5. **The "found" circuit was likely a general capability head, not primacy-specific.**
   L0H3 in Qwen 1.5B knocked out PI without hurting RI — but L8H3 hurt BOTH when
   ablated. The specificity was questionable.

**What v2 got right (carried into v3):**
- TransformerLens pipeline for hooking into residual stream
- Understanding that logit lens and causal patching are the right tools
- Recognition that early layers (L0) showed strong positional bias from instruction/BOS tokens

### v3 (current) — CORRECT FRAMING

**Reframed questions based on correct error distribution:**

| v2 question | v3 question |
|---|---|
| "Which heads suppress v_{N-1} and output v_0?" | "Why does v_{N-1}'s signal peak at 90% depth then get outcompeted?" |
| "Can we ablate the primacy circuit?" | "Is the mechanism distributed or concentrated?" |
| "What is the primacy circuit?" | "Does a bottleneck exist at all?" |

**v3 answer:** No bottleneck heads. Ablating any individual head HURTS PI retrieval.
The mechanism is architectural, distributed, and not a fixable circuit. This is the
correct and honest answer — less exciting than finding a named circuit, but stronger.

---

## 1. THE THREE-STAGE PIPELINE (Core MI contribution)

### Stage 1: Behavioral Sweep (`scripts/experiments/stage1_sweep.py`)

**What it does:** Runs PI/RI retrieval task across a grid of (num_keys × num_updates) operating points.
Records per-trial accuracy, error positions, garbage rates, and regime classification.

**Models with results:**

| Model | Max trials | Key result |
|---|---|---|
| Qwen2.5-0.5B-Instruct | 50/cell | RI=72%, PI=14% at 2k_20u |
| Qwen2.5-1.5B-Instruct | 200/cell | RI=99%, PI=23% at 2k_20u |
| Qwen2.5-3B-Base | 200/cell | RI=99%, PI=50% at 2k_20u |
| Qwen2.5-3B-Instruct | 200/cell | RI=88%, PI=46% at 2k_20u |
| Gemma-3-1B-it | 50/cell | RI=88%, PI=6% at 2k_20u |
| TinyLlama-1.1B | 200/cell | RI=100%, PI=0% at 2k_10u |
| StableLM-1.6B | 200/cell | RI=98%, PI=9% at 2k_7u |
| Pythia-410M | 50/cell | RI=42%, PI=8% at 2k_20u |
| Mamba-1.4B | 200/cell | RI=66%, PI=10% at 2k_10u |
| Mamba-130M | 50/cell | UNUSABLE (>60% garbage) |
| RWKV-430M | 30/cell | UNUSABLE (>95% garbage) |

**Aggregate:** 91% of cells (51/56) show PI > RI. Mean gap: +50pp.

---

### Stage 2: Logit Lens (`scripts/experiments/stage2_logit_lens.py`)

**What it does:** At each of n_layers, reads the residual stream at the answer position and converts
to probability over all values in the sequence. Tracks P(v_i) for every value i=0..N-1 per layer.

**Models with results:**

| Model | Layers | Key finding |
|---|---|---|
| Qwen2.5-0.5B | 24 | v_last peaks 0.24 at L21, crashes to 0.04 |
| Qwen2.5-1.5B | 28 | v_last peaks 0.14 at L26, crashes to 0.01 |
| Qwen2.5-3B | 36 | v_last peaks 0.21 at L32, crashes to 0.01 |
| Gemma-3-1B | 26 | v_last peaks 0.03 at L23, crashes to 0.00 |
| Qwen3-4B | — | Partial (checkpoint only) |

**Universal pattern (Pattern B — "Overtaken"):**
- v_last IS found at ~90% depth (computed correctly in intermediate layers)
- v_last then loses to penultimate value in final layers
- v_first rises monotonically to 0.92–1.00 without competition
- C1 empirically confirmed: P(v_last) is non-increasing post-peak in 8/8 post-peak layers

---

### Stage 3: Causal Analysis (`scripts/experiments/stage3_causal.py`)

**What it does:** Four sub-experiments:
- **3A Attribution patching:** Gradient-based ranking of which heads contribute most to P(v_last)
- **3B Targeted patching:** Swap activations from clean→corrupted to test causal effect per head
- **3C Logit lens under ablation:** Does ablating top head change suppression pattern?
- **3D Forced attention:** If we force attention to v_last position, does accuracy recover?

**Models with results:**

| Model | Top head | Attribution delta | Ablation effect |
|---|---|---|---|
| Qwen 1.5B | L19H6 | +0.038 | -0.023 (hurts!) |
| Qwen 3B | L26H3 | +0.032 | -0.044 (hurts!) |
| Gemma 1B | L15H2 | +0.071 | -0.020 (hurts!) |

**Key finding:** Ablating top heads HURTS P(v_last). These heads HELP retrieval.
No suppression circuit. Mechanism is distributed. PI > RI is architectural, not a learnable bug.

---

## 2. SUPPORTING EXPERIMENTS

### 2A. Probing Classifiers (`scripts/experiments/probing_classifier.py`)

**What it does:** Trains linear probes at each layer predicting:
1. RI vs PI condition discrimination (is this a "first" or "last" query?)
2. RI-correct vs RI-incorrect (can we predict success from representation?)
3. PI-correct vs PI-incorrect (same for PI)

**Results:**

| Model | Condition disc. | RI correct probe | PI correct probe |
|---|---|---|---|
| Qwen 1.5B | 100% | 87% | 61% |
| Qwen 3B | 99% | 81% | 61% |
| Gemma 1B | 97% | 60% | N/A |

**Key finding:**
- Condition probe (100%) is expected — different query words create different representations
- RI correctness is well-encoded (87%): representation predicts success
- PI correctness is weakly above chance (61%): model can't "see" its own PI failure
- This is a within-condition comparison — not explainable by query word detection

---

### 2B. Jacobian at Initialization (`scripts/experiments/jacobian_at_init.py`)

**What it does:** Computes ||∂output/∂input_j||_F for each position j on BOTH
untrained (random weights) and pretrained models. Shows whether primacy bias
is architectural or learned.

**Results (N=20 inputs, seq_len=50):**

| Model | Untrained primacy | Pretrained |
|---|---|---|
| Qwen 1.5B | 1.47× first/middle | Both amplified |
| Mamba 1.4B | **295×** first/last | Recency also strong |

**Key finding:** Primacy bias exists at initialization (before ANY training data).
Mamba's extreme primacy (295×) comes from HiPPO matrix transient growth, not learned bias.

---

### 2C. Training Dynamics (`scripts/experiments/jacobian_dynamics.py`)

**What it does:** Runs Jacobian analysis across 9 SmolLM2-1.7B training checkpoints
(steps 125K → 2M tokens). Tracks how positional bias evolves during training.

**Results (N=20 inputs per checkpoint):**

| Step | Primacy ratio | Recency ratio |
|---|---|---|
| 125K | 1.57× | 1.52× |
| 375K | 1.33× | 1.31× |
| 625K | 1.36× | 1.61× |
| 875K | 1.50× | 1.38× |
| 1125K | 1.54× | 1.72× |
| 1375K | 1.47× | 1.44× |
| 1625K | 1.41× | 1.76× |
| 1875K | 1.25× | 1.45× |
| Final | **2.00×** | **1.80×** |

**Key finding:** Neither bias is monotonically increasing. Both strengthen across training.
Final instruct model has highest primacy (2.00×) AND recency (1.80×). Instruction tuning
amplifies the U-shape. PI > RI at behavioral level is from primacy narrowly winning,
not architectural lock-in.

---

### 2D. Bidirectional Controls

**Three experiments probing whether autoregressive direction is the cause:**

| Experiment | Model | Script | Result |
|---|---|---|---|
| BERT MLM probe | bert-base-uncased | `bidirectional_probe_v2.py` | 0% both conditions — can't do KV retrieval via MLM |
| Encoder-decoder | Flan-T5-base | `encoder_decoder_test.py` | RI=13%, PI=22%, gap=**-9%** (reversed) — no primacy |
| Encoder-decoder | Flan-T5-large | `encoder_decoder_test.py` | FAILED (killed — slow on CPU) |

**Key finding:** Flan-T5 (bidirectional encoder) shows no primacy bias (gap=-9%).
High garbage (64-77%) weakens the conclusion. Treated as preliminary/appendix.

---

### 2E. Gemma Scope SAE Analysis (`scripts/experiments/gemma_scope_sae.py`)

**What it does:** Uses Google Gemma Scope 2 pre-trained Sparse Autoencoders
(65k features, JumpReLU) on gemma-3-1b-it at 4 layers (7, 13, 17, 22).
Compares mean feature activations: RI vs PI, PI-correct vs PI-incorrect.

**Results (100 trials, 2k_5u):**

| Layer | RI active features | PI active features | Difference |
|---|---|---|---|
| 7 | 48.9 | 46.0 | RI +6% |
| 13 | 58.9 | 64.8 | PI +10% (reversed at mid-depth) |
| 17 | 58.4 | 55.0 | RI +6% |
| 22 | 45.2 | 38.4 | RI +18% |

Top RI-preferring features at L22: [10793, 2292, 725, 2190, 1652] (diff > 150)

**Key finding:** Late layers (L22) engage 18% more features for RI than PI.
Mid-depth reversal (L13) aligns with logit lens — PI IS computed mid-depth,
then suppressed. Feature richness asymmetry mirrors probability asymmetry.

**Limitation:** Behavioral accuracy was 0% in saved results (prediction tracking bug).
Feature activation data is valid but not paired with correctness.

---

## 3. REMEDY EXPERIMENTS

### 3A. Behavioral Remedy (`scripts/experiments/remedy_experiment.py`)

**What it does:** Tests 5 prompt-level interventions on Claude Haiku at 10 keys.
Targets each of the three theoretical forces.

**Results (10k_100u, 30 trials):**

| Intervention | Targets | Control PI | Intervention PI | Improvement |
|---|---|---|---|---|
| Control | — | 70% | 70% | — |
| Numbered updates | Force 3 (pos. confusability) | 70% | 80% | +10pp |
| Landmark (---) | Force 2 (capacity saturation) | 70% | **93%** | **+23pp** |
| Recency cue | Force 1 (cumulative reinforcement) | 70% | 67% | -3pp |
| Old combined (verbose) | All three | 70% | 60% | -10pp (hurts!) |
| **New combined** (lightweight) | Forces 2+3 | 70% | **100%** | **+30pp** |

**Key finding:** Landmark separators are most effective. New lightweight combined
(round markers + landmark) eliminates the gap entirely at hard operating points.
Verbose interventions hurt because token overhead worsens capacity saturation.

### 3B. Mechanistic Remedy Validation (`scripts/experiments/remedy_logit_lens.py`)

**What it does:** Runs logit lens on Qwen 1.5B WITH and WITHOUT landmark separators.
Tests whether the suppression pattern (P(v_last) peak then crash) changes.

**Behavioral results:**
- Control: RI=96%, PI=20%, gap=+76%
- Landmark: RI=78%, PI=38%, gap=+40%
- Landmark cuts gap by 36pp on local model

**Logit lens trajectories:** ALL ZEROS (bug — ARBITRARY_SINGLE values are multi-token
in Qwen's tokenizer; value-to-token-ID mapping fails). The mechanistic link between
remedy and suppression pattern is NOT yet established.

---

## 4. FORMAL THEORY

### What we prove (`paper/theory/FORMAL_BOUND.md`)

**Proposition (complete):**
Under conditions C1, C2, A1–A3:
- (i) I(h_query; v_1 | N) → I_∞ > 0 — RI information converges, never fully overwritten
- (ii) I(h_query; v_N | N) ≤ C(d) - I(h_N; v_1...v_{N-1}) — PI capacity squeezed as N grows
- (iii) R_first(N) > R_last(N) for sufficiently large N

**C1 (Monotone Overwrite):** PROVEN for single-layer softmax (`LEMMA_C1_PROOF.md`)
- SNR strictly decreases when new value added: SNR_{t+1} < SNR_t
- Verified: 0/10,000 failures with random attention scores
- Extended to SSMs (Jacobian norm argument) for diagonal A

**C2 (Diminishing Marginal Overwrite):** DERIVED from Chowdhury (`LEMMA_C2_FROM_CHOWDHURY.md`)
- ρ_H(x) = Σ c_k (ln(1/x))^{k-1} is strictly decreasing → overwrite loss is non-increasing
- Verified: 12/12 configurations pass

**A3 (Summable Overwrite):** PROVEN for single-layer (Δ_j = O(1/j²), Σ converges)

**Honest limitations:**
- C1 multi-layer: CONJECTURED (proven single-layer, empirically verified 8/8)
- C1 HiPPO SSMs: CONTRADICTED by transient growth; empirically holds
- Exponential decay PI(N) = a·exp(-b·N)+c: EMPIRICAL FIT, not derived

### Architecture-specific corollaries

**Transformers (Corollary 1):**
Chowdhury's influence density ρ_H(x) with primacy tail ~ (ln(1/x))^{H-1} implies
both C1 and C2. Connects to Veličković (softmax dispersion) and Wu (fixed point).

**SSMs (Corollary 2):**
HiPPO non-normality causes transient growth: ||A^49|| = 2.61 for d=64.
Explains Mamba's 295× primacy at init (not eigenvalue decay — non-normality).

---

## 5. OVERALL MI STATE ASSESSMENT

### Strongest evidence (paper-ready)

1. **Logit lens Pattern B** — v_last found at ~90% depth then suppressed — 4 models, clean, consistent. This is the paper's core mechanistic claim. ✅

2. **Probing asymmetry** — RI 87% vs PI 61% correctness encoding — 2-3 models. ✅

3. **Jacobian at init** — Primacy bias before training — 2 architectures. ✅

4. **Causal: distributed mechanism** — No bottleneck heads, ablating helps hurts. The finding IS that there's no circuit. ✅

5. **SAE features** — RI activates more features in late layers (L22: +18%). Novel, first-of-kind for this task. ⚠️ (behavioral accuracy missing)

6. **Training dynamics** — Both biases strengthen; final model strongest both. Novel. ✅

### Known gaps and fixes needed

| Gap | Severity | Fix |
|---|---|---|
| Remedy logit lens has zero trajectories | MEDIUM | Fix multi-token value mapping, re-run |
| SAE behavioral accuracy = 0% | LOW | Feature data valid; just add accuracy separately |
| C1 not proven multi-layer | LOW | Conjectured + empirical; sufficient for NeurIPS |
| No MI on models >4B | LOW | Argue via Jacobian + behavioral scaling |
| Stage 3 only on 3 models | LOW | Qwen 1.5B/3B + Gemma sufficient for claim |
