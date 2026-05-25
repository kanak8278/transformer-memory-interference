# Plan — Block format 2×2 ablation (grouping × numbering) (§5.1)

**Status**: planned, can be executed as pure re-analysis (no GPU).
The data already exists; the paper already states the 2×2
intuition in prose (lines 1102-1107) but doesn't formalize it
statistically.

---

## 1. The critique under test

> "Block beats Labeled and Landmark, but you never run the 2×2
> (grouping vs numbering vs both) to identify which property matters.
> Devil's Advocate frames Block as 'providing an explicit
> answer-location map.' Fix: a one-day experiment on 2 models,
> structure-only vs number-only. Even negative results tighten the
> claim."

The critique is *partially* incorrect: the 2×2 design IS already in
the paper, just not framed that way explicitly.

| | NO grouping | WITH grouping |
|---|---|---|
| NO numbering | **Plain** (`flat_nolabel`) | **Landmark** (`landmark`) — `---` between rounds |
| WITH numbering | **Labeled** (`flat_verbose`) — per-entry `(update j)` | **Block** (`block`) — `[Round j]` header + grouped entries |

So we have the 4 cells of the 2×2 already. What's missing:

1. Formal **pairwise contrasts** with CIs:
   - Plain → Labeled (numbering effect, no grouping)
   - Plain → Landmark (grouping effect, no numbering)
   - Landmark → Block (numbering effect, with grouping)
   - Labeled → Block (grouping effect, with numbering)
2. **Marginal effect** decomposition: Δ(numbering) and Δ(grouping)
   averaged across the levels of the other factor
3. **Interaction test**: is Block > Labeled + Landmark − Plain? I.e.,
   do numbering and grouping interact super-additively (Block uses
   BOTH), or are they independent (Block = sum of independent
   effects)?

## 2. What the paper already says (lines 1102-1107)

> "Read together, Labeled and Landmark localise Block's two effective
> ingredients: round structure provides salient boundaries for the
> lookup region (without which CVQ stays floored), while explicit
> round numbers anchor that structure to the query (without which
> open-weight models fall back to Plain-format behaviour). Block
> combines both."

This is the right intuition. The proposed addition is the table that
backs it up.

## 3. Data sources confirmed

### Proprietary models — `experiments_cloud/results/ucurve_proprietary_results.csv`

- 2,160 rows; columns include `format ∈ {flat_nolabel, flat_verbose,
  block, landmark}` and `position ∈ {1..N, "last"}`
- 36 unique (model, K, N) triples with all 4 formats represented
- Models: claude-4.5-haiku, claude-sonnet, gemini-2.5-flash,
  gemini-2.5-pro, gpt-4.1, gpt-4.1-mini

### Open-weight models — `experiments_cloud/results/ucurve_vllm/<model>/ucurve_*.json`

**NOT currently on this machine.** The path is referenced by
`paper/scripts/generate_main_results.py:load_format_open`, but the
directory `experiments_cloud/results/ucurve_vllm/` does not exist in
this checkout. The paper's open-weight format table renders from
that data on the user's Mac.

For #9 v1 we proceed with proprietary-only; v2 (when ucurve_vllm
data is pushed) adds the open-weight side.

### Mapping to FVQ / CVQ

The ucurve CSV stores per-position accuracy. From the data:
- `position = 1` → FVQ (first value query)
- `position = "last"` → CVQ (last value query) [literal label]
- `position = k ∈ {2..N}` → IVQ at position k

For the 2×2 ablation we only need FVQ and CVQ rows.

## 4. Implementation: `compute_2x2_ablation.py`

```python
"""
2×2 ablation analysis: grouping × numbering effects on FVQ/CVQ.

Outputs: lora_intervention/results/block_2x2_table.txt
         lora_intervention/results/block_2x2_data.json
"""
import csv
from collections import defaultdict
import numpy as np

# --- Load data ---
rows = list(csv.DictReader(open("experiments_cloud/results/ucurve_proprietary_results.csv")))

# Bucket by (model, K, N, format), keep only FVQ (pos=1) and CVQ (pos="last")
def load_fvq_cvq():
    out = defaultdict(dict)  # (model, K, N) → {format: {"fvq": .., "cvq": .., "n": ..}}
    for r in rows:
        k = (r["model"], int(r["num_keys"]), int(r["num_updates"]))
        fmt = r["format"]
        if fmt not in {"flat_nolabel", "flat_verbose", "block", "landmark"}:
            continue
        slot = out[k].setdefault(fmt, {"fvq": None, "cvq": None, "n": int(r["n_trials"])})
        if r["position"] == "1":
            slot["fvq"] = float(r["accuracy"])
        elif r["position"] == "last":
            slot["cvq"] = float(r["accuracy"])
    # Keep only cells with all 4 formats and both FVQ/CVQ populated
    full = {}
    for k, v in out.items():
        if len(v) == 4 and all(s["fvq"] is not None and s["cvq"] is not None for s in v.values()):
            full[k] = v
    return full

# --- Contrasts per cell ---
def contrasts(cell):
    plain    = cell["flat_nolabel"]
    labeled  = cell["flat_verbose"]
    landmark = cell["landmark"]
    block    = cell["block"]

    # Effect of numbering, holding grouping fixed
    num_no_group  = {"fvq": labeled["fvq"] - plain["fvq"],    "cvq": labeled["cvq"] - plain["cvq"]}
    num_w_group   = {"fvq": block["fvq"]   - landmark["fvq"], "cvq": block["cvq"]   - landmark["cvq"]}
    # Effect of grouping, holding numbering fixed
    group_no_num  = {"fvq": landmark["fvq"] - plain["fvq"],   "cvq": landmark["cvq"] - plain["cvq"]}
    group_w_num   = {"fvq": block["fvq"]   - labeled["fvq"],  "cvq": block["cvq"]   - labeled["cvq"]}
    # Interaction: Block − (Labeled + Landmark − Plain); positive = super-additive
    interaction   = {
        "fvq": block["fvq"] - (labeled["fvq"] + landmark["fvq"] - plain["fvq"]),
        "cvq": block["cvq"] - (labeled["cvq"] + landmark["cvq"] - plain["cvq"]),
    }
    return {
        "num_no_group":  num_no_group,
        "num_w_group":   num_w_group,
        "group_no_num":  group_no_num,
        "group_w_num":   group_w_num,
        "interaction":   interaction,
    }
```

Aggregate across cells: mean and bootstrap 95% CI for each contrast
on each (FVQ, CVQ) axis. Report per (K, N) cell and pooled across
all 36 cells × 6 models.

## 5. What we expect to find

From the paper's existing numbers at K=10/N=50 on
Qwen2.5-3B-Instruct (FVQ / CVQ format=block from `tab_format.tex`):

|         | Plain      | Labeled    | Block      | Landmark   |
|---------|------------|------------|------------|------------|
| FVQ/CVQ | 0.49/0.23  | 0.34/0.39  | 0.84/0.92  | 0.71/0.27  |

Computing contrasts for **CVQ** (the gap is in CVQ, so the marginal
effects on CVQ tell the story):
- Numbering effect, no grouping (Labeled − Plain CVQ) = 0.39 − 0.23 = **+0.16**
- Numbering effect, with grouping (Block − Landmark CVQ) = 0.92 − 0.27 = **+0.65**
- Grouping effect, no numbering (Landmark − Plain CVQ) = 0.27 − 0.23 = **+0.04**
- Grouping effect, with numbering (Block − Labeled CVQ) = 0.92 − 0.39 = **+0.53**
- Interaction = 0.92 − (0.39 + 0.27 − 0.23) = 0.92 − 0.43 = **+0.49**

**Strong super-additivity** on this one cell. Numbering alone helps
a little (+0.16). Grouping alone helps even less (+0.04). But
together they help massively (+0.69 over Plain). The interaction
term is +0.49 — Block is much more than the sum of its parts.

If this pattern holds across all 36 cells × 6 models, the
**paper's "Block combines both" framing is statistically supported**.
If the interaction is near zero pooled across models, the story
needs to be reframed as "two independent additive effects."

Either outcome is publishable.

## 6. Cleanup needed in §5 (after analysis)

Replace prose at lines 1102-1107 with:
- A reference to the new App. table showing the 2×2 contrasts
- A line stating the interaction direction with its CI
- Concrete numbers: "Labeled adds +X.XX [CI] to CVQ; Landmark
  +Y.YY; Block +Z.ZZ [super-additive Δ = +W.WW [CI]]"

## 7. Effort estimate

| Step | Time |
|---|---|
| Load CSV, build per-cell 2×2, compute contrasts | 30 min |
| Bootstrap 95% CIs per contrast per cell (n=200 trials in source data) | 30 min |
| Aggregate across cells × models | 30 min |
| Generate table + figure | 30 min |
| Write up + commit | 30 min |
| **Total** | **~2.5 h** (no GPU) |

## 8. Open-weight gap

Open-weight format data (`experiments_cloud/results/ucurve_vllm/...`)
isn't in this checkout. Two paths:

- **A:** ship v1 with proprietary-only analysis. Note in the table
  caption that "open-weight 2×2 deferred pending data sync."
- **B:** request the ucurve_vllm data, re-run the analysis on both
  open-weight and proprietary sets.

Recommendation: do (A) first since the proprietary numbers will
either confirm or refute the super-additivity pattern; add (B) once
the data lands.

## 9. Reproducibility

All inputs are CSV. Output script will be committed to the repo
along with the table. Anyone can re-run with:

```
python lora_intervention/compute_2x2_ablation.py
```

and regenerate the table from raw data.
