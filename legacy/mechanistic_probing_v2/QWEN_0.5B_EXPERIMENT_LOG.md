# Qwen2.5-0.5B-Instruct — Experiment Log

**Model:** Qwen/Qwen2.5-0.5B-Instruct (494M params, 24 layers, 14 heads, GQA 14Q/2KV)
**Started:** 2026-02-20

---

## Operating Points

Selected from Phase 1 behavioral sweep. All Phase 2 experiments run at each point.

| Point | Keys | Updates | RI Acc | PI Acc | Gap | Regime | Tests |
|-------|------|---------|--------|--------|-----|--------|-------|
| **A** | 3 | 1 | 100% | 90% | +10% | A | Baseline — no real interference |
| **B** | 2 | 5 | 90% | 40% | +50% | B | PI cracking edge (depth) |
| **C** | 2 | 30 | 83% | 3% | +80% | B | Maximum gap, PI dead |
| **D** | 10 | 80 | 47% | 3% | +43% | C | RI breaking down |
| **E** | 25 | 3 | 80% | 3% | +77% | B | PI dead from key count, not updates |

**Trial counts:** 100 per condition for all experiments. If a specific result is marginal (e.g., ablation effect p=0.08), bump that experiment to 200. Don't pre-optimize trial counts — the actual bottleneck is getting consistent findings across models, not per-experiment power.

**Result path:** `results/Qwen2.5-0.5B-Instruct/{keys}k_{updates}u/{experiment}.json`

**Script interface:** `--keys K --updates U --trials N --model Qwen/Qwen2.5-0.5B-Instruct`

---

## Phase 1: Behavioral Sweep — COMPLETE

| Field | Value |
|-------|-------|
| **Script** | `experiments/11_behavioral_sweep.py` |
| **Result** | `results/behavioral_sweep_Qwen2.5-0.5B-Instruct_summary.json` |
| **Config** | keys=[2-46], updates=[1-200], 30 trials/cell, 135 cells, 8100 trials |

### Findings

**Regime breakdown:** 17 A (both work) | 87 B (PI fails, RI works) | 31 C (both fail)

- PI collapses at updates=3-5 across ALL key counts. At keys=25+, going from 1→3 updates drops PI from 80-90% to 3-10%.
- RI stays 60-80% out to updates=30-50. Only collapses at extreme load.
- At updates=1 (no interference), PI ≥ RI for keys=15+. Asymmetry ONLY appears with interference.
- RI degrades slowly; PI dies abruptly. Different failure modes.

**Cracking points:**

| Keys | PI cracks at | RI cracks at | Ratio |
|------|-------------|-------------|-------|
| 2 | updates=5 | updates=100 | 20× |
| 5 | updates=10 | updates=60 | 6× |
| 10 | updates=3 | updates=80 | 27× |
| 25 | updates=3 | updates=50 | 17× |

---

## Phase 1.5: Behavioral Validation (Single-Token Values + 46 Categories) — TODO

**Purpose:** Confirm PI > RI still holds when using the mechanistic data setup (single-token values from `core/single_token_values.py` + 46 categories from `core/dataset.py`) instead of the behavioral sweep's multi-token synthetic values. If this fails, the mechanistic analysis results won't connect to the behavioral findings.

**Config:** 30 trials per cell, 2 conditions (RI/PI), 5 operating points.

| Point | Keys | Updates | Sweep RI | Sweep PI | Validation RI | Validation PI | Status |
|-------|------|---------|----------|----------|---------------|---------------|--------|
| A | 3 | 1 | 100% | 90% | | | TODO |
| B | 2 | 5 | 90% | 40% | | | TODO |
| C | 2 | 30 | 83% | 3% | | | TODO |
| D | 10 | 80 | 47% | 3% | | | TODO |
| E | 25 | 3 | 80% | 3% | | | TODO |

**Pass criteria:** PI > RI asymmetry present at all 5 points. Exact numbers may differ but regime classification (A/B/C) must match.

---

## Phase 2: Mechanistic Analysis — Run Matrix

### Exp 12 — Logit Lens (Layer-by-Layer Value Tracking)

**Question:** At which layer does P(initial) vs P(final) diverge? Does the divergence point shift across operating points?

| Point | Status | Trials | Result file | Findings |
|-------|--------|--------|-------------|----------|
| A (3,1) | TODO | 100 | | |
| B (2,5) | TODO | 100 | | |
| C (2,30) | TODO | 100 | | |
| D (10,80) | TODO | 100 | | |
| E (25,3) | TODO | 100 | | |

**Cross-point analysis:** _Not yet done._

---

### Exp 13 — Positional Gradient Analysis

**Question:** Sharp primacy cliff or gradual decay? How does the gradient shape change with interference level?

| Point | Status | Trials | Result file | Findings |
|-------|--------|--------|-------------|----------|
| A (3,1) | TODO | 100 | | |
| B (2,5) | TODO | 100 | | |
| C (2,30) | TODO | 100 | | |
| D (10,80) | TODO | 100 | | |
| E (25,3) | TODO | 100 | | |

**Cross-point analysis:** _Not yet done._

---

### Exp 14 — PI Mass Distribution

**Question:** Where does probability mass land on PI failures? First value, intermediate, or garbage?

| Point | Status | Trials | Result file | Findings |
|-------|--------|--------|-------------|----------|
| A (3,1) | TODO | 100 | | |
| B (2,5) | TODO | 100 | | |
| C (2,30) | TODO | 100 | | |
| D (10,80) | TODO | 100 | | |
| E (25,3) | TODO | 100 | | |

**Cross-point analysis:** _Not yet done._

---

### Exp 15 — Activation Patching (Clean → Corrupted)

**Question:** Which (layer, position, component) restores PI when patched from clean run? Does the critical region shift across points?

| Point | Status | Trials | Result file | Findings |
|-------|--------|--------|-------------|----------|
| A (3,1) | TODO | 100 | | |
| B (2,5) | TODO | 100 | | |
| C (2,30) | TODO | 100 | | |
| D (10,80) | TODO | 100 | | |
| E (25,3) | TODO | 100 | | |

**Cross-point analysis:** _Not yet done._

---

### Exp 16 — Head Identification & Ablation (HEADLINE)

**Question:** Which heads are retrieval heads, primacy-biased, recency-responsive? Do the SAME heads appear at all 5 points?

| Point | Status | Trials | Result file | Findings |
|-------|--------|--------|-------------|----------|
| A (3,1) | TODO | 100 | | |
| B (2,5) | TODO | 100 | | |
| C (2,30) | TODO | 100 | | |
| D (10,80) | TODO | 100 | | |
| E (25,3) | TODO | 100 | | |

**Cross-point analysis:** _Not yet done._

**Key question:** Are the primacy-biased heads consistent across points? If the same heads appear everywhere, the mechanism is general. If different heads at different points, the story is more nuanced.

---

### Exp 17 — Instruction Sensitivity Probe

**Question:** Where do RI and PI representations diverge? Does the divergence layer shift across points?

| Point | Status | Trials | Result file | Findings |
|-------|--------|--------|-------------|----------|
| A (3,1) | TODO | 100 | | |
| B (2,5) | TODO | 100 | | |
| C (2,30) | TODO | 100 | | |
| D (10,80) | TODO | 100 | | |
| E (25,3) | TODO | 100 | | |

**Cross-point analysis:** _Not yet done._

---

### Exp 18 — Forced Attention Intervention (HEADLINE)

**Question:** Does forcing primacy heads to attend to the correct value restore PI at ALL operating points?

| Point | Status | Trials | Result file | Findings |
|-------|--------|--------|-------------|----------|
| A (3,1) | TODO | 100 | | |
| B (2,5) | TODO | 100 | | |
| C (2,30) | TODO | 100 | | |
| D (10,80) | TODO | 100 | | |
| E (25,3) | TODO | 100 | | |

**Cross-point analysis:** _Not yet done._

---

### Exp 19 — Positional Bias Sweep (Blind vs Oracle) (HEADLINE)

**Question:** Does oracle bias (value-position-specific) fix PI while blind bias fails? Consistent across points?

| Point | Status | Trials | Result file | Findings |
|-------|--------|--------|-------------|----------|
| A (3,1) | TODO | 100 | | |
| B (2,5) | TODO | 100 | | |
| C (2,30) | TODO | 100 | | |
| D (10,80) | TODO | 100 | | |
| E (25,3) | TODO | 100 | | |

**Cross-point analysis:** _Not yet done._

---

### Exp 19b — Bias Attention Proof

**Question:** Where does attention actually go under blind vs oracle bias?

| Point | Status | Trials | Result file | Findings |
|-------|--------|--------|-------------|----------|
| A (3,1) | TODO | 100 | | |
| B (2,5) | TODO | 100 | | |
| C (2,30) | TODO | 100 | | |
| D (10,80) | TODO | 100 | | |
| E (25,3) | TODO | 100 | | |

**Cross-point analysis:** _Not yet done._

---

### Exp 20 — Minority Override Analysis (DLA)

**Question:** How do primacy heads override recency heads despite weaker DLA? Does the override pattern change across points?

| Point | Status | Trials | Result file | Findings |
|-------|--------|--------|-------------|----------|
| A (3,1) | TODO | 100 | | |
| B (2,5) | TODO | 100 | | |
| C (2,30) | TODO | 100 | | |
| D (10,80) | TODO | 100 | | |
| E (25,3) | TODO | 100 | | |

**Cross-point analysis:** _Not yet done._

---

### Exp 22 — Query Patching Granular

**Question:** Layer-by-layer, component-by-component recovery at query position. Where is the critical corruption layer?

| Point | Status | Trials | Result file | Findings |
|-------|--------|--------|-------------|----------|
| A (3,1) | TODO | 100 | | |
| B (2,5) | TODO | 100 | | |
| C (2,30) | TODO | 100 | | |
| D (10,80) | TODO | 100 | | |
| E (25,3) | TODO | 100 | | |

**Cross-point analysis:** _Not yet done._

---

### Exp 23 — Ablation + Patching Interaction

**Question:** Does ablating primacy heads reduce the need for query patching? One mechanism or two?

| Point | Status | Trials | Result file | Findings |
|-------|--------|--------|-------------|----------|
| A (3,1) | TODO | 100 | | |
| B (2,5) | TODO | 100 | | |
| C (2,30) | TODO | 100 | | |
| D (10,80) | TODO | 100 | | |
| E (25,3) | TODO | 100 | | |

**Cross-point analysis:** _Not yet done._

---

## Phase 2: Experiments NOT YET WRITTEN

| Exp | Name | Question | Priority |
|-----|------|----------|----------|
| **21a** | Logit lens under ablation | Does ablating primacy heads change P(initial)/P(final) trajectory? | CRITICAL |
| **21b** | DLA split by outcome | On PI failures vs successes, is primacy DLA larger? | CRITICAL |
| **21c** | Failure output classification | What does model output on PI failures? Primacy intrusion? Garbage? | CRITICAL |
| **24** | QK fine-tuning | Fine-tune W_Q/W_K of primacy heads — does 0.14% param fix PI? | CRITICAL |
| **25** | OV/QK circuit decomposition | WHY are primacy heads position-biased? Decompose W_OV, W_QK | IMPORTANT |
| **26** | Path patching | Prove circuit connectivity (e.g., L0H4 → L16H3) | IMPORTANT |
| **27** | Mean ablation comparison | Replace head output with dataset mean (not zero) — more standard | IMPORTANT |

---

## Cross-Point Synthesis

_To be filled after all 5 points are run for each experiment._

### Key questions to answer:
1. **Head consistency:** Do the same primacy-biased heads appear at all 5 points?
2. **Divergence layer:** Does L16 remain the critical layer, or does it shift?
3. **Forced attention:** Does 100% PI hold at hard points (C, D, E)?
4. **Depth vs breadth:** Same mechanism at Point C (many updates) vs Point E (many keys)?
5. **RI failure:** At Point D where RI breaks, what changes mechanistically?

---

## Exp 25a/25b/25c: Comprehensive Head Identification (2026-02-25)

Ran 3 independent methods at Point B (1k, 5u), 100 PI trials.

### 25a — Per-Head Causal Knockout (Gold Standard)

Baseline PI: 69%, mean_ld=6.77

| Rank | Head | Δld | KO accuracy |
|------|------|-----|-------------|
| 1 | L12H0 | +1.8 | 70% |
| 2 | L11H3 | +1.6 | 74% |
| 3 | L12H13 | +1.6 | 79% |
| 4 | L19H8 | +1.5 | 72% |
| 5 | L19H6 | +1.4 | 62% |

Top retrieval: L8H3 (-10.1, KO→0%), L6H9 (-9.3, KO→1%), L4H12 (-8.1, KO→0%).

**Key finding: NO single dominant primacy head.** Best head (L12H0) only shifts PI by +1.8 logits and +1% accuracy. The primacy bias is genuinely distributed across many weak heads on 0.5B.

**Old L16H3 was wrong.** Ranks 13th (Δld=+0.6). Not a primacy head by causal evidence.

### 25b — Attribution Patching

Top retrieval: L20H7 (+2.1), L20H12 (+1.8), L16H7 (+1.3) — all late layer.
Top primacy: L23H12 (-1.3), L20H9 (-0.9) — also late layer.

**25b finds different heads than 25a.** Same pattern as Gemma: attribution patching misses early-layer indirect effects.

### 25c — Observational Metrics

Copy score is ALL ZERO — no head does copy-paste at this operating point (PI=69% means model mostly gets it right without retrieval heads doing copy-paste).

Spearman correlations weak (most < 0.3). Methods disagree on what "matters."

### Comparison to Old Exp 16 Classification

| | Old (exp 16) | New (exp 25a) |
|---|---|---|
| "Smoking gun" | L16H3 (pi_primacy=0.992) | None — distributed |
| Top head effect | Ablating 8 heads: gap 35%→5% | Best single head: +1.8 logits |
| Head location | L15-L23 (late) | L3-L19 (spread) |
| Mechanism | "Concentrated minority overrides majority" | **Distributed — no single head dominates** |

The old exp 16 story about 8 primacy-locked heads was based on attention thresholds. The causal evidence (25a) shows the attention pattern didn't translate to causal impact. L16H3 has pi_primacy=0.992 (always attends to initial) but Δld=+0.6 (negligible causal effect on PI).

---

## Paper Figures (Blocked on production runs)

| # | Figure | Data needed | Status |
|---|--------|-------------|--------|
| 1 | Behavioral heatmap (keys × updates, RI vs PI) | Phase 1 sweep | DATA READY |
| 2 | Logit lens trajectories across operating points | Exp 12 × 5 points | BLOCKED |
| 3 | Head classification scatter (primacy score RI vs PI) | Exp 16 × 5 points | BLOCKED |
| 4 | Ablation bar chart per operating point | Exp 16 × 5 points | BLOCKED |
| 5 | Forced attention results per operating point | Exp 18 × 5 points | BLOCKED |
| 6 | Positional gradient curves | Exp 13 × 5 points | BLOCKED |
| 7 | Oracle vs blind bias results | Exp 19 × 5 points | BLOCKED |
| 8 | Cross-model comparison | Pending second model | BLOCKED |
| 9 | SSM control comparison | Pending SSM experiment | BLOCKED |
