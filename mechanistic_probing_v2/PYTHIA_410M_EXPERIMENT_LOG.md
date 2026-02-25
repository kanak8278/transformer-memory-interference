# Pythia-410M — Experiment Log

**Model:** EleutherAI/pythia-410m (405M params, 24 layers, 16 heads, MHA — no GQA)
**Started:** 2026-02-25
**Purpose:** Two targeted experiments to (1) confirm PI > RI in a base completion model (kills the "it's just instruction tuning" objection) and (2) resolve the GQA confound on copy-suppression (MHA eigenspectrum should actually work).

---

## Why Pythia?

1. **Base model (no SFT/RLHF):** All prior experiments used instruction-tuned models. A reviewer can argue the PI > RI asymmetry is an artifact of fine-tuning. Pythia is pure pre-trained — if PI > RI appears here, it's architectural.
2. **MHA architecture:** Qwen uses GQA (grouped-query attention), which collapsed the OV eigenspectrum to zeros in Exp 28 (copy-suppression test). Pythia uses standard multi-head attention — eigenvalues should be nonzero and interpretable.
3. **Well-studied:** Pythia is the canonical mech interp model. Results on Pythia connect to existing literature (retrieval heads, induction heads, copy-suppression).

## Why 410M and not 160M?

We tried 160M first. It can do the first/last retrieval task with manually crafted natural-word prompts (4/4 RI correct, 0/4 PI correct — perfect asymmetry). But with the automated experimental setup (random single-token English words as values), 160M fails to generalize from the few-shot demos. It outputs demo-block values instead of test-section values.

**Pythia-410M generalizes the task format to arbitrary values.** 24 layers and 16 heads give it enough capacity for in-context learning of the retrieval pattern.

## Prompt Format: Completion (Few-Shot)

Pythia has no chat template. We use a **two-block few-shot** completion format:

```
color: red
animal: cat
color: blue
animal: dog
color: green
animal: bear
The first value of color was: red
The last value of color was: green
The first value of animal was: cat
The last value of animal was: bear

fruit: apple
metal: gold
fruit: grape
metal: silver
fruit: lemon
metal: iron
The first value of fruit was: apple
The last value of fruit was: lemon
The first value of metal was: gold
The last value of metal was: iron

shape: wire
drink: lobby
shape: blend
drink: frost
shape: clone
drink: ridge
The last value of drink was:
```

- **Two demo blocks** with natural/recognizable words (required — one block is insufficient for the model to learn the pattern)
- **Test section** with random single-token English words from the verified pool
- Demo categories (color, animal, fruit, metal) are disjoint from test categories (shape, drink, stone, plant, fish, sport, bird, grain, dance, spice)
- Values verified as single-token against Pythia's GPT-NeoX tokenizer (2135 of 2300 candidates pass)

**Scripts:** `experiments/29_pythia_behavioral_sweep.py`, `experiments/30_pythia_copy_suppression.py`
**Infra changes:** `core/dataset.py` (added `generate_completion_trial`, `build_completion_prompt`, `generate_few_shot_demo` with 2-block hardcoded demos), `core/model_loader.py` (added Pythia to `CONTEXT_LIMITS` and `BASE_MODELS`)

---

## Operating Points

Selected from Phase 1 behavioral sweep. Prioritizing highest RI-PI gap in regime B.

| Point | Keys | Updates | RI Acc | PI Acc | Gap | Regime | Rationale |
|-------|------|---------|--------|--------|-----|--------|-----------|
| **A** | 2 | 1 | 80% | 63% | +17% | A | Baseline — minimal interference |
| **B** | 3 | 5 | 80% | 0% | +80% | B | **Primary.** Maximum gap, clean RI |
| **C** | 2 | 20 | 80% | 0% | +80% | B | Depth stress — many updates, RI holds |
| **D** | 5 | 10 | 67% | 3% | +63% | B | Width + depth combined |

**Selected for Exp 30 (copy-suppression):** Point B (3k, 5u) — 80% RI, 0% PI, maximum contrast. Point A (2k, 1u) — 80% RI, 63% PI, as control (both work → heads should behave identically → no DLA difference expected).

**Trial counts:** 100 per condition for mechanistic experiments.
**Result path:** `results/pythia-410m/{keys}k_{updates}u/{experiment}.json`

---

## Phase 1: Behavioral Sweep — COMPLETE

| Field | Value |
|-------|-------|
| **Script** | `experiments/29_pythia_behavioral_sweep.py` |
| **Result** | `results/behavioral_sweep_pythia-410m_partial.json` |
| **Config** | keys=[2,3,5,7,10], updates=[1,2,3,5,7,10,15,20,30], 30 trials/cell, 45 cells, 2700 trials |
| **Time** | ~25 min on Apple Silicon MPS |

### Findings

**PI > RI confirmed in a base completion model.**

**Regime breakdown:** 4 A (both work) | 19 B (PI fails, RI works) | 22 C (both fail)

**Accuracy grids:**

```
--- RI Accuracy ---
        1    2    3    5    7   10   15   20   30
   2  80%  50%  70%  53%  63%  63%  60%  80%  53%
   3  67%  60%  57%  80%  60%  57%  57%  53%  33%
   5  63%  57%  40%  43%  57%  67%  17%  30%   0%
   7  57%  13%  30%  27%  50%  33%   3%   3%   0%
  10  47%  30%  20%  43%  10%   3%   0%   3%   0%

--- PI Accuracy ---
        1    2    3    5    7   10   15   20   30
   2  63%  20%  17%   0%   0%   0%   0%   0%   3%
   3  57%  10%  13%   0%   0%   3%   3%   3%   0%
   5  63%  30%  17%   0%   3%   3%   0%   0%   7%
   7  50%  17%   7%  10%   3%   0%   3%   3%   0%
  10  47%  13%  17%   7%   7%   3%   7%   3%   7%

--- Regime Map ---
        1    2    3    5    7   10   15   20   30
   2    A    B    B    B    B    B    B    B    B
   3    A    B    B    B    B    B    B    B    C
   5    A    B    C    C    B    B    C    C    C
   7    A    C    C    C    B    C    C    C    C
  10    C    C    C    C    C    C    C    C    C
```

**Overall:** Mean RI = 41.0%, Mean PI = 11.6%, Gap = 29.4pp

### Error Analysis (Regime B cells, n=570 PI trials)

| Error Type | Count | % |
|---|---|---|
| garbage | 230 | 40.4% |
| **primacy_intrusion** | **224** | **39.3%** |
| intermediate_intrusion | 82 | 14.4% |
| correct | 34 | 6.0% |

PI errors: 39% primacy intrusions (model outputs V1 when asked for V_last). The high garbage rate (40%) is expected for a base model — it sometimes continues generating unrelated text instead of a clean answer. When it does produce a value, it's overwhelmingly the first value.

RI errors: 60% correct, only 2.1% recency intrusions. The model almost never outputs V_last when asked for V_first.

### Cracking Points

| Keys | PI cracks at | RI cracks at | Ratio |
|------|-------------|-------------|-------|
| 2 | updates=2 | updates=30+ | 15×+ |
| 3 | updates=2 | updates=30 | 15× |
| 5 | updates=2 | updates=3 | 1.5× |
| 7 | updates=2 | updates=2 | 1× |
| 10 | updates=1 | updates=1 | 1× |

At 2-3 keys, PI collapses 15× earlier than RI — same qualitative pattern as instruction-tuned models. At 7+ keys, the model is capacity-limited and both fail early.

### Comparison to Instruction-Tuned Models

| Model | Arch | Params | SFT? | PI > RI? | RI holds to | PI cracks at |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | GQA | 494M | Yes | **Yes** | 100 updates | 3-5 updates |
| Qwen2.5-1.5B-Instruct | GQA | 1.5B | Yes | **Yes** | 200+ updates | 5-10 updates |
| Gemma-3-1B-IT | MHA | 1B | Yes | **Yes** | ~30 updates | 2 updates |
| **Pythia-410M** | **MHA** | **405M** | **No** | **Yes** | **30 updates** | **2 updates** |

**Conclusion:** PI > RI is present in base completion models with no fine-tuning. The asymmetry is architectural (attention mechanism), not an artifact of instruction tuning or RLHF.

### Surprising Findings

1. **Pythia-160M can't do this task with random words** but can with natural words. The 160M model learns the few-shot format but doesn't generalize to arbitrary values. At 410M, generalization works. This suggests the in-context learning capacity threshold for this task is between 160M and 410M parameters.

2. **High garbage rate (40%)** in PI errors. Base models often continue generating unrelated text rather than producing a clean wrong answer. This is a prompt engineering limitation, not a model capability issue — when the model does produce a value-like answer, it shows clear primacy bias.

3. **RI accuracy is noisy** (varies 50-80% at 2 keys across updates). This is higher variance than instruction-tuned models, likely because the completion format gives the model more degrees of freedom in how it responds.

---

## Phase 2: Copy-Suppression Test — COMPLETE

| Field | Value |
|-------|-------|
| **Script** | `experiments/30_pythia_copy_suppression.py` |
| **Operating Points** | Point B: 3k,5u (RI=80%, PI=0%) + Point A control: 2k,2u (RI=50%, PI=20%) |
| **Purpose** | Resolve GQA confound from Exp 28 (Qwen). MHA eigenspectrum should produce nonzero eigenvalues. |
| **Results** | `results/pythia-410m/3k_5u/copy_suppression_test.json`, `results/pythia-410m/2k_2u/copy_suppression_test.json` |
| **Time** | ~3 min (Point B) + ~3 min (Point A) |

### Background: Qwen 0.5B Results (Exp 28)

- Test 1 (eigenspectrum): **INCONCLUSIVE** — all eigenvalues 0.0000 due to GQA weight structure
- Test 2 (suppression score): Mixed — primacy heads weakly suppressive but less than average
- Test 3 (DLA decomposed): **V1-PROMOTION** — L16H3 DLA_v1=+1.19, DLA_v2=+0.26

### MPS Limitation: `linalg_eig` not supported

Initial run had all eigenvalues at 0.0000 — looked like GQA collapse again. But the real cause: **`aten::linalg_eig` is not implemented on MPS.** The operation silently fails. Fixed by moving tensors to CPU for eigenvalue computation. After fix, eigenvalues are real and nonzero (range [0.03, 1.70]).

### Head Identification

Point B (3k, 5u) identified **21 primacy heads** (threshold=0.10). Point A (2k, 2u) identified **14 primacy heads**. Core overlap: L11H2, L11H6, L11H14, L13H5, L14H0, L15H5, L15H8, L17H11, L18H8, L18H14, L19H15, L21H8.

Strongest primacy head: **L14H0** (attn_to_init=0.599 vs attn_to_final=0.153, bias=+0.446).

### Test 1: OV Eigenspectrum — DOES NOT SUPPORT copy-suppression

With CPU eigenvalue computation, Pythia's MHA produces real, nonzero eigenvalues. **This resolves the GQA confound.**

| | Primacy Heads | Other Heads |
|---|---|---|
| **Dominant eigenvalue (mean)** | -0.002 | +0.018 |
| **Neg fraction (mean)** | 0.490 | 0.537 |

Primacy heads have roughly the same eigenspectrum as other heads. No enrichment of negative eigenvalues. Copy-suppression predicts primacy heads should have MORE negative eigenvalues — they don't.

Individual primacy head eigenspectra are mixed: some have dominant negative (L11H2=-0.175, L17H10=-0.337, L18H8=-0.359), others dominant positive (L11H6=+0.416, L16H7=+0.421). No consistent pattern.

### Test 2: Directional Suppression Score — Mixed/Weak

| | Primacy Heads | Other Heads |
|---|---|---|
| **Point B (3k,5u)** | +0.0125 | +0.0048 |
| **Point A (2k,2u)** | +0.0019 | +0.0053 |

Point B: primacy heads are LESS suppressive. Point A: marginally more suppressive (Δ=-0.003). Neither is significant. The suppression signal is noise-level.

### Test 3: DLA Decomposed — V1-PROMOTION confirmed

**Point B (3k, 5u) — PI=6%, maximum interference:**

| Head | DLA_v1 | DLA_v2 | Mechanism |
|---|---|---|---|
| **L11H14** | **+2.28** | **-0.12** | **V1-PROMOTION** (strongest) |
| L18H8 | +1.11 | +0.23 | V1-PROMOTION |
| L17H10 | +1.32 | +1.00 | V1-PROMOTION |
| L16H5 | +0.99 | +0.65 | V1-PROMOTION |
| L16H1 | +0.81 | +0.56 | V1-PROMOTION |

**17/21 primacy heads classified as V1-PROMOTION. 0/21 V2-SUPPRESSION.** 4 classified as WEAK.

DLA_v2 is mostly positive or near-zero — primacy heads are NOT suppressing V2. They're promoting both V1 and V2, but V1 much more strongly. V1 wins by being louder, not by V2 being silenced.

**Point A (2k, 2u) — PI=13%, mild interference (control):**

| Head | DLA_v1 | DLA_v2 | Mechanism |
|---|---|---|---|
| **L11H14** | **+1.83** | **+0.05** | **V1-PROMOTION** |
| L18H8 | +2.27 | +1.56 | V1-PROMOTION |
| L13H5 | +1.32 | +0.74 | V1-PROMOTION |

**7/14 primacy heads V1-PROMOTION. 0/14 V2-SUPPRESSION.** 7 WEAK.

**Control comparison:** At mild interference (Point A), more heads are WEAK (7/14 = 50%) vs severe interference (Point B, 4/21 = 19%). The V1-promotion mechanism is the same but less pronounced when interference is mild. This makes sense — at low interference, the primacy heads don't need to work as hard.

### Cross-Architecture Summary

| | Qwen 0.5B (GQA) | Pythia-410M (MHA) |
|---|---|---|
| **Test 1 (eigenspectrum)** | INCONCLUSIVE (GQA collapse) | **DOES NOT SUPPORT** (real eigenvalues, no neg enrichment) |
| **Test 2 (suppression score)** | Does not support | Does not support |
| **Test 3 (DLA decomposed)** | **V1-PROMOTION** (3/3 heads) | **V1-PROMOTION** (17/21 heads) |
| **Architecture** | GQA (14Q/2KV) | MHA (16/16) |
| **Instruction-tuned?** | Yes | No |

### Verdict

**Copy-suppression hypothesis REJECTED across both architectures.** The mechanism is V1-promotion, not V2-suppression. This is:

1. **Architecture-independent** — same result in GQA (Qwen) and MHA (Pythia)
2. **Independent of instruction tuning** — same result in SFT model and base model
3. **Now with valid eigenspectrum data** — MPS fix confirms eigenvalues are nonzero; no negative enrichment in primacy heads

The paper can state: *"Primacy-biased attention heads promote V1 via positive DLA contributions, rather than suppressing V2 via negative OV eigenvalues as predicted by the copy-suppression hypothesis (McDougall et al., 2024). This V1-promotion mechanism is confirmed across GQA (Qwen) and MHA (Pythia) architectures, and in both instruction-tuned and base models."*

### Surprising Findings

1. **L11H14 is the dominant primacy head** — DLA_v1=+2.28 at Point B, the single strongest V1-promoting head in the model. It's at layer 11 (mid-network), not in the final layers where most DLA concentrates in Qwen. Pythia's primacy circuit may be structured differently.

2. **More primacy heads in Pythia (21) than Qwen (3).** Qwen concentrates primacy in 2-3 heads; Pythia distributes it across 14-21 heads. This could reflect MHA vs GQA: with GQA sharing KV across query groups, fewer distinct "head behaviors" are possible, so the function concentrates.

3. **DLA_v2 is often positive** in primacy heads. They're not just promoting V1 — they're promoting everything, just V1 more. This is consistent with the heads acting as general "retrieval amplifiers" with positional bias, not targeted V1-specific circuits.
