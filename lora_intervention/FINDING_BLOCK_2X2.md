# Finding — Block format 2×2 ablation (#9)

**Status**: complete (re-analysis of existing proprietary data, no new
runs). Open-weight side deferred pending `ucurve_vllm/` data sync.

**Data**: `experiments_cloud/results/ucurve_proprietary_results.csv`,
36 (model, K, N) triples × 4 formats × 200 trials each. Six
proprietary models (Claude Haiku/Sonnet, Gemini Flash/Pro, GPT-4.1/4.1-mini).

**Analysis script**: `lora_intervention/compute_2x2_ablation.py`
**Outputs**:
- `lora_intervention/results/block_2x2_table.txt`
- `lora_intervention/results/block_2x2_data.json`

---

## Headline (pooled across 36 cells × 6 models, bootstrap n=2000, seed=17)

### FVQ axis (first-value retrieval)

|       Format | mean acc [95% CI] |
|--------------|-------------------|
| Plain        | 0.998 [0.994, 1.000] |
| Labeled      | 1.000 [0.999, 1.000] |
| **Landmark** | **0.856 [0.785, 0.913]** ← drops |
| Block        | 0.998 [0.995, 1.000] |

| Contrast (FVQ) | Δ [95% CI] | sig |
|---|---|---|
| Numbering, no grouping (Labeled − Plain) | +0.002 [-0.000, +0.005] | no |
| Numbering, with grouping (Block − Landmark) | **+0.143 [+0.085, +0.213]** | yes |
| Grouping, no numbering (Landmark − Plain) | **-0.142 [-0.210, -0.086]** | yes (negative) |
| Grouping, with numbering (Block − Labeled) | -0.001 [-0.005, +0.001] | no |
| **Super-additivity** (Block − Labeled − Landmark + Plain) | **+0.141 [+0.085, +0.210]** | yes |

### CVQ axis (last-value retrieval — where the gap is)

|       Format | mean acc [95% CI] |
|--------------|-------------------|
| Plain        | 0.855 [0.810, 0.896] |
| Labeled      | 0.962 [0.944, 0.976] |
| Landmark     | 0.991 [0.984, 0.997] |
| Block        | 0.998 [0.994, 1.000] |

| Contrast (CVQ) | Δ [95% CI] | sig |
|---|---|---|
| Numbering, no grouping (Labeled − Plain) | **+0.106 [+0.065, +0.153]** | yes |
| Numbering, with grouping (Block − Landmark) | +0.006 [-0.001, +0.014] | no |
| Grouping, no numbering (Landmark − Plain) | **+0.136 [+0.099, +0.177]** | yes |
| Grouping, with numbering (Block − Labeled) | **+0.036 [+0.021, +0.054]** | yes |
| **Super-additivity** (Block − Labeled − Landmark + Plain) | **-0.100 [-0.143, -0.062]** | yes (sub-additive) |

---

## Interpretation (what changes for the paper)

### The paper's current prose (lines 1102-1107 of `main.tex`):

> "Read together, Labeled and Landmark localise Block's two effective
> ingredients: round structure provides salient boundaries for the
> lookup region (without which CVQ stays floored), while explicit
> round numbers anchor that structure to the query (without which
> open-weight models fall back to Plain-format behaviour). Block
> combines both."

### What the data actually shows:

The "Block combines both" framing is **partly right but the
mechanism is different in two important ways**:

1. **Landmark hurts FVQ.** Adding round separators *without*
   numbers damages first-value retrieval (FVQ drops 0.998 → 0.856,
   Δ = −0.142, CI excludes zero). This is a real cost, not just a
   neutral baseline. Round boundaries disrupt the model's implicit
   "first value is at the start of the stream" cue, and without
   round numbers there's no alternative position signal. The paper
   currently doesn't note this.

2. **CVQ recovery is sub-additive, not super-additive.** Numbering
   alone (+0.106) and grouping alone (+0.136) each independently
   close most of the CVQ gap. Combining them adds only +0.007 on top
   of Landmark (Block − Landmark on CVQ, CI includes zero). This is
   a ceiling effect — each intervention pushes CVQ near 1.0 on
   proprietary models, so combining can't go further.

### Refined story (proposed §5.1 prose):

> "Decomposing Block into its two ingredients via Landmark
> (grouping, no numbers) and Labeled (numbers, no grouping) reveals
> that *either ingredient alone* approximately closes the CVQ gap
> on proprietary models (Labeled: +0.106 CVQ; Landmark: +0.136
> CVQ). Block's marginal benefit over the better of the two on CVQ
> is +0.006 (CI includes zero) — a ceiling effect. **Block's value
> is asymmetric across axes**: on CVQ it merely matches the better
> of {Labeled, Landmark}, but on FVQ it is the only format that
> avoids damage. Landmark alone *drops* FVQ from 0.998 to 0.856
> (−0.142, CI excludes zero) — round separators without
> orderability cues destroy first-value retrieval. Block's
> numbering reverses this loss. So Block isn't superior because it
> combines complementary fixes; it is superior because it provides
> the *only* format that fixes CVQ without sacrificing FVQ."

This is a stronger and more accurate claim than the current prose.

### What this does NOT show (limitations)

1. **Open-weight models not yet covered.** The
   `ucurve_proprietary_results.csv` file has the proprietary 2×2
   complete; the equivalent `ucurve_vllm` data for open-weight
   models is referenced by `paper/scripts/generate_main_results.py`
   but not present in this repo checkout. The open-weight side may
   tell a different story — main.tex line 1097 hints that Landmark
   leaves Gemma-3-4b-it at "CVQ ≈ 0.00", so the proprietary
   ceiling-effect interpretation may not generalise. Open-weight
   2×2 is the natural next step.

2. **Super-additivity sign flips across axes.** FVQ shows
   super-additivity (+0.141), CVQ shows sub-additivity (−0.100).
   The two interventions don't act on the same internal mechanism
   uniformly — they have asymmetric effects on the two query types.
   This is reportable but doesn't simplify into "Block combines two
   independent ingredients" or "Block exploits interaction" — it's
   axis-specific.

3. **The proprietary set is small.** Six models × six cells = 36
   cells. Per-model breakdowns show some variation (e.g.,
   Gemini-2.5-pro is at ceiling everywhere; Claude-Sonnet has
   different patterns than the others). The pooled CI is reasonably
   tight but per-model effects vary.

---

## What the critique got right and wrong

The original critique said:
> "Block beats Labeled and Landmark, but you never run the 2×2
> (grouping vs numbering vs both) to identify which property
> matters. Fix: a one-day experiment on 2 models, structure-only vs
> number-only."

- **Wrong**: the 2×2 IS in the existing data; no new experiment was
  needed for the proprietary side
- **Right**: the paper never explicitly formalized the 2×2 with
  contrasts and CIs; this analysis fills that gap

What the paper got right (lines 1102-1107):
- The intuition that Labeled and Landmark "localise Block's two
  effective ingredients" matches the marginal effects on CVQ

What the paper got wrong (lines 1102-1107):
- "Block combines both" suggests super-additive combination on
  CVQ. Data shows sub-additive (ceiling effect) on CVQ and
  super-additive on FVQ, with the asymmetry driven by Landmark's
  damage to FVQ — a finding the current text doesn't capture.

---

## Next step (deferred)

Once `experiments_cloud/results/ucurve_vllm/` data is synced from
the user's Mac, re-run `compute_2x2_ablation.py` with the
open-weight CSV path added. Expect the open-weight CVQ contrasts
to look different (Landmark not a CVQ fix there per main.tex line
1097) — that will further refine the story.
