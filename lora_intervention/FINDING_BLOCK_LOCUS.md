# Finding — Block format and LoRA do NOT share an internal locus (#6)

**Status**: complete. Tests the paper's implicit "common locus" claim
(`paper/main.tex` lines 536-543: "the format and LoRA interventions
converge... the capability is therefore present in the pretrained
architecture but not dispatched by default").

**Outputs**:
- `lora_intervention/results/block_locus_comparison.txt` — full per-head table
- `lora_intervention/results/block_locus_comparison.json` — structured data
- `v3/results_vllm/attention_routing/Qwen2.5-3B-Instruct-baseline-block__block.json` — new data (50 trials)

---

## The hypothesis

`H1` (paper's implicit claim): Block-format input on the base
(no-LoRA) Qwen model activates the same L30-L33 promoter heads that
post-LoRA activates on plain-format input.

`H0` (devil's-advocate): Block and LoRA produce the same behavior
via different internal mechanisms; their attention patterns differ.

---

## Experimental design

Three conditions on Qwen2.5-3B-Instruct, K=2/N=30, 50 trials/condition,
expA_CVQ:

| | Model | Prompt format | Source |
|---|---|---|---|
| A | base | Plain | existing `Qwen2.5-3B-Instruct__normal.json` |
| B | base | **Block** | NEW (`Qwen2.5-3B-Instruct-baseline-block__block.json`) |
| C | base + main LoRA | Plain | existing `Qwen2.5-3B-Instruct-LoRA__normal.json` |

For each of the **15 LoRA-promoter heads** in L30-L33 (identified by
the existing §5.1 attention-routing analysis, see `compute_cis.py`):
compute P(attend to v_last round) per condition, then Δ(B-A) and
Δ(B-C) with bootstrap 95% CIs.

---

## Behavioral pre-check

Before the mechanism probe, confirmed Block format does close the gap
on Qwen at K=2/N=30 (`lora_intervention/eval_block_behavior.py`, 100
trials/cell):

| Format | FVQ | CVQ | gap |
|---|---|---|---|
| Plain | 89% | **49%** | +40% |
| Labeled | 93% | 53% | +40% |
| Landmark | 92% | 69% | +23% |
| **Block** | 98% | **74%** | +24% |

Block raises CVQ from 49% → 74% on this cell (Δ = +25 pts).
Substantial behavioral effect, even if not full closure to ceiling.
The mechanism question is well-posed: at minimum, *some* internal
change is responsible for those +25 pts.

---

## Results

Pooled across the 15 LoRA-promoter heads (Qwen2.5-3B-Instruct,
K=2/N=30, expA_CVQ, bootstrap n=2000, seed=17):

|     | mean P(attend last round) |
|-----|---|
| **A** (Base + Plain) | 0.103 |
| **B** (Base + Block) | **0.160** (Δ vs A: **+0.056**) |
| **C** (LoRA + Plain) | **0.676** (Δ vs A: +0.573) |

| Comparison | mean Δ | Interpretation |
|---|---|---|
| Δ(B − A) | **+0.056** | Block barely touches LoRA-promoter heads |
| Δ(C − A) | +0.573 | LoRA strongly activates them |
| Δ(B − C) | **−0.516** | Block falls **51 points short** of LoRA's activation |

### Per-head verdict (pre-registered criteria)

| Criterion | Met for | Verdict |
|---|---|---|
| "B matches C" (95% CI of Δ_BC contains 0 AND p_B > 0.4) | **0 of 15 heads** | — |
| "B matches A" (95% CI of Δ_BA contains 0) | 4 of 15 heads | — |
| Block activation lifts slightly but not to LoRA level | 11 of 15 heads | **PARTIAL OVERLAP** |

The 4 heads where Block leaves baseline essentially unchanged (B matches A):
L31H15, L32H10, L32H14, L33H14.

The 11 heads where Block activates them somewhat but stays far below
LoRA: rest of the cluster.

**No head reaches LoRA's activation level under Block format.**

---

## Verdict

→ **DISTINCT MECHANISMS (with partial overlap)**.

Block format and the main LoRA produce overlapping behavioral
recovery via **substantially different internal routes**. Block
nudges the L30-L33 cluster by +0.056 (compared to LoRA's +0.573).
Block must be doing most of its work through some other pathway —
likely the explicit `[Round j]` headers giving the model a
discrete-step lookup structure that doesn't require building the
L30-L33 "attend-to-final-round" pattern from scratch at inference
time.

---

## Implications for the paper

### The current text (lines 536-543 of `main.tex`)

> "The format and LoRA interventions converge: both close the
> FVQ-CVQ gap and restore intermediate-position retrieval. The
> first does so at inference time without gradients; the second
> with a fraction of a percent of the model's parameters. **The
> capability is therefore present in the pretrained architecture
> but not dispatched by default.**"

The bold sentence is consistent with this finding (the capability
IS in the architecture; both interventions surface it). The
preceding "converge" sentence is fine if read as a behavioral claim
("both close the gap") but ambiguous if read as mechanistic ("both
operate through the same locus") — which a careful reader will
infer.

### Proposed revision

Add the following to §5 as a short paragraph:

> "We tested whether the format and LoRA interventions share an
> internal locus by computing attention-routing under Block format
> on the base model. Block format does not activate the L30-L33
> promoter cluster that LoRA recruits: pooled mean P(attend to last
> round) across the 15 LoRA-target heads is 0.16 under Block (95% CI
> [0.13, 0.19]) versus 0.68 under LoRA (95% CI [0.66, 0.71]) — a 51
> percentage point gap. **The two interventions are
> operationally distinct: each closes the gap, but they recruit
> different internal circuits.** Block presumably leans on the
> explicit `[Round j]` headers to provide a discrete-step lookup
> structure, bypassing the attention reweighting LoRA installs."

This is honest and **strengthens the paper**: it (a) reports a
nontrivial result a reviewer would otherwise extract, (b)
disambiguates the "convergence" framing the abstract uses, (c)
positions Block as a complementary intervention rather than a
free-of-gradient version of LoRA.

---

## Caveats

1. **Behavioral effect of Block at K=2/N=30 is moderate** (CVQ
   49% → 74%, not full closure). At K=10/N=50, Block closes the
   gap fully on Qwen (0.23 → 0.92 per `tab_format.tex`). We
   probed mechanism at K=2/N=30 because that matches the existing
   Qwen attention-routing baseline. A natural follow-up: probe at
   K=10/N=50 where Block's behavioral effect is dramatic. We
   expect the conclusion to hold (Block doesn't activate L30-L33
   even more strongly there because the heads were chosen to be
   LoRA-specific), but it would be a cleaner demonstration.

2. **Trials not paired by seed.** Seeds 1000-1049 used for all
   three conditions, but the prompt format change means tokenized
   prompts differ between A and B → trial-level pairing has
   limited interpretability. Pooled means + unpaired bootstrap CIs
   are what the table reports.

3. **The 15 LoRA-promoter heads are the right set to test** because
   those are the heads §7 identifies as carrying the gap-closure
   signal post-LoRA. If Block activates *different* heads we
   haven't identified, our probe won't see them — but that's the
   nature of the test (asking "does Block use *these specific*
   heads", not "what does Block use"). Identifying Block's actual
   mechanism is a follow-up.

---

## Reproduce

```
# A and C already exist in v3/results_vllm/attention_routing/

# B (new):
python lora_intervention/run_attention_routing_hf.py \
    --merged_path Qwen/Qwen2.5-3B-Instruct \
    --base Qwen/Qwen2.5-3B-Instruct \
    --out_name Qwen2.5-3B-Instruct-baseline-block \
    --K 2 --N 30 --trials 50 \
    --prompt-format block

# Comparison:
python lora_intervention/compute_block_locus_probe.py
```
