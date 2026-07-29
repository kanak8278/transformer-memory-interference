# Do LLMs Remember First and Forget Last — Even in Real Text?

**Kanak Raj** | NeurIPS Expansion Work | March 2026

---

## The Core Finding (from prior work)

When language models process a sequence of key-value updates — *"Alice's score is 4... Alice's score is 7... Alice's score is 2"* — they exhibit a striking asymmetry:

- **RI (Retroactive Interference):** Ask for the *first* value → models answer correctly ~70% of the time
- **PI (Proactive Interference):** Ask for the *last* value → models answer correctly ~30% of the time

Models are dramatically better at recalling the *first* thing they saw than the *most recent* update. This holds across 39 models from 7B to frontier scale. The gap (RI accuracy − PI accuracy) averages +18 percentage points. We call this **PI > RI**: primacy beats recency.

---

## The Question This Work Answers

The original experiments used synthetic, stripped-down sequences. **Does the same primacy bias appear when models read naturalistic, realistic text?**

We built three narrative generators and ran the same RI/PI experiment on each.

---

## Three New Domains

| Domain | Format | Example entity | Tracked attribute |
|--------|--------|---------------|-------------------|
| **Wildlife tracking** | GPS-collar field reports | WF-2847 (gray wolf) | Weight (kg) |
| **Hospital ICU** | Clinical shift handoffs | Mrs. Liu, Bed 4 | Heart rate, lactate |
| **Air traffic control** | Sector transcripts | DAL1472 | Assigned altitude |

Each domain generates narratives where multiple entities receive sequential attribute updates — structurally identical to synthetic KV pairs, but embedded in realistic, domain-appropriate prose.

---

## Results

Tested on **Qwen2.5-1.5B** (a model showing strong primacy bias on synthetic data) and **Claude Haiku** (a frontier model that resists the effect on synthetic data).

### Wildlife Tracking — Effect transfers, amplified

| | Qwen 1.5B | Claude Haiku |
|--|-----------|--------------|
| PI>RI cells | **10/11** | 3/8 |
| Mean gap | **+27%** | −2% |
| Significant | **5 cells** | 0 |

Qwen 1.5B shows stronger primacy bias on wolf-tracking narratives than on synthetic data (+27% vs +19%). At 2 wolves × 3 observations: RI = 92%, PI = 34%, gap = **+58%**. The effect is not a synthetic artifact.

### Hospital ICU — Null result

| | Qwen 1.5B | Claude Haiku |
|--|-----------|--------------|
| PI>RI cells | 5/15 | 0/8 |
| Mean gap | −4% | −4% |

No effect for either model. Why? ICU notes label every value explicitly: *"BUN = 53 mg/dL"*, *"HR 125 bpm"*. When values are labeled, both first and last are equally easy to retrieve regardless of position. **Explicit labeling eliminates interference.**

### Air Traffic Control — Model-dependent reversal

| | Qwen 1.5B | Claude Haiku |
|--|-----------|--------------|
| PI>RI cells | 10/15 | **0/8** |
| Mean gap | +6% | **−34%** |

Haiku shows a *strong reversal*: PI is dramatically *easier* than RI (gap = −34%, consistent across all cells). ATC transcripts end with the most recent instruction. Haiku exploits this document structure — scanning backward to find the last assignment — while Qwen 1.5B, processing more sequentially, still shows directional primacy.

---

## What This Means

**Three findings, each significant on its own:**

1. **Primacy bias is real, not a format artifact.** Wildlife narratives reproduce the same effect with larger gaps than synthetic data.

2. **Format determines whether interference appears.** Explicit value labels (ICU) eliminate the effect by equalizing retrieval difficulty. The underlying architecture hasn't changed — only how information is packaged.

3. **Strong models fail differently.** Frontier models don't show primacy bias, but they *do* show document-level recency bias in transcript-formatted text. This is a different mechanism: strategic document navigation rather than attention-driven primacy.

---

## The Bigger Picture

| Model | Synthetic KV | Wildlife narrative | ICU notes | ATC transcript |
|-------|-------------|-------------------|-----------|----------------|
| Qwen 1.5B | PI>RI +19% | **PI>RI +27%** | No effect | +6% (noisy) |
| Claude Haiku | No effect | No effect | No effect | **Reversed −34%** |

The primacy bias is an architectural property that persists into naturalistic text — but only in formats that require sequential scanning without structural shortcuts. Stronger models escape primacy but become susceptible to recency when document structure enables it.

---

## What We Built

- **Three narrative generators** (wildlife, ICU, ATC) producing realistic multi-entity sequential-update narratives, validated on 3,000 trials
- **Experiment infrastructure** for local and API model evaluation with Wilson CIs
- **Draft paper section** with full statistical analysis

*Code: `narrative_generator/` | Results: `v3/results/narrative/` | Draft: `v3/paper/drafts/NARRATIVE_TRANSFER_RESULTS.md`*
