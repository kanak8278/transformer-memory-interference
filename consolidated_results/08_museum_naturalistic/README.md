# 08 — Does the effect survive naturalistic prose?

**Question.** Every other theme uses a synthetic key-value stream —
`visual art: turquoise`, one labelled update per line. Is the whole phenomenon
an artefact of that format? This theme re-asks three of the project's questions
against the **museum M0 narrative**: continuous prose about visitors moving
through galleries, where an update is a sentence, not a labelled line.

| | |
|---|---|
| files | `endpoint.csv` (116) · `ivq.csv` (96) · `cot_ivq.csv` (288). All `raw`. |
| per-model slices | `by_model_endpoint/`, `by_model_ivq/`, `by_model_cot/` (generated) |
| models | claude-4.5-haiku (all three sweeps) · claude-4.5-opus (**`nocot` arm of `cot_ivq.csv` only**) |
| dataset | `museum_m0` |
| prompt_format | `narrative_m0` |
| variant | `base` (endpoint, ivq) · `nocot` / `cot_thinking` (cot_ivq) |

| file | mirrors | grid |
|---|---|---|
| `endpoint.csv` | theme 01 (FVQ vs CVQ) | **58 cells** — K ∈ {1,2,5,10,15,20,25,30,40,45} × N ∈ {5,10,15,20,30,50} |
| `ivq.csv` | theme 02 (per-position) | 6 cells — K ∈ {5,10} × N ∈ {10,20,50}, 12–18 probed positions |
| `cot_ivq.csv` | theme 07 (CoT vs no-CoT) | 6 cells, same grid — haiku both arms, opus `nocot` only |

`endpoint.csv` is the widest grid in the whole corpus — wider than the synthetic
sweeps it mirrors.

## Coverage is ragged — check it before any groupby

| model | `nocot` | `cot_thinking` | `endpoint` | `ivq` |
|---|---|---|---|---|
| haiku | ✓ 13,640 trials | ✓ 17,160 trials | ✓ | ✓ |
| opus | ✓ 13,744 trials, 3.9% malformed — **consolidated** | ✗ never run (smoke only) | ✗ | ✗ |
| sonnet | ⚠ **54.1% of attempts malformed** (9,757/18,024) — excluded | ✗ died 64 trials into cell 1 of 6, **no checkpoint** | ✗ | ✗ |

So `cot_ivq.csv` has a **two-model `nocot` arm and a one-model `cot_thinking`
arm**. `df.groupby("variant").accuracy.mean()` silently compares different model
sets. Haiku is the only model where the CoT contrast is paired:

```python
df[df.model == "claude-4.5-haiku"].groupby("variant")   # the CoT effect
df[df.variant == "nocot"].groupby("model")              # the model effect
```

`endpoint.csv` and `ivq.csv` remain haiku-only.

**Sonnet is excluded on validity grounds, not convenience.** Its 54.1% malformed
rate against haiku's 0.0% and opus's 3.9% means its reported 0.350 accuracy
rests on a self-selected 46% of trials. The ≥20% off-stream rule that removed
seven open-weight models from theme 01 removes this arm too. Its `cot_thinking`
crash is worse than a gap because the runner logged `exit 0` — the log claims
success. `PROVENANCE.md` records both.

**Opus is a gap, not a failure.** `claude-opus-4-5-20251101__cot_thinking__smoke`
(2 trials × 6 positions on the hardest cell, K=10/N=50) runs clean: 12/12 calls
returned, **0 thinking-budget truncations**, output 310–2,715 tokens against a
3,000 budget. So the full run is safe to launch. Projected cost, from haiku's
actual 17,160 calls repriced at opus rates, is **~$420 as a floor**; opus emitted
1.44× haiku's output tokens per call in the smoke, so budget $500–600 and 3–4 h.

## Headline 1 — the endpoint gap replicates almost exactly

| | theme 01 (synthetic) | theme 08 (narrative) |
|---|---|---|
| mean FVQ−CVQ gap | **+0.19** | **+0.189** (on-stream) |
| positive in | 19/20 series | **54/58 cells** |

That is about as clean a cross-format replication as this project has. The
first-value advantage is not a property of labelled key-value lines.

**But read the raw number with the correction applied.** Off-stream failure in
this sweep is wildly asymmetric between the two conditions:

| condition | off-stream |
|---|---|
| FVQ | **1.0%** (104/10,076) |
| CVQ | **14.8%** (1,492/10,076) |

And it grows with depth — all **17** cells above the 20% threshold are CVQ, up to
**58%** at (20,50). So part of the apparent CVQ collapse is the model failing to
name any value in the narrative, not retrieving the wrong one. At K=10:

```
N            5     10     15     20     30     50
CVQ raw    0.771  0.661  0.557  0.516  0.417  0.372   <- looks like a cliff
CVQ onstrm 0.771  0.672  0.626  0.627  0.606  0.736   <- roughly flat
CVQ off%     0%     2%    11%    18%    31%    49%
```

Correcting for it shrinks the mean gap from **+0.260 to +0.189** and flips the
sign in 2 of 58 cells — (10,50) goes +0.267 → −0.087 and (5,50) +0.214 → −0.026.
**Use `accuracy_onstream` for any depth claim**, and quote the raw column only
alongside `n_offstream`.

## Headline 2 — the U-curve is there, and so is the ordinal/semantic split

`ivq.csv`, K=10, N=50:

```
pos     1     2     3     4     5     6     7     8    14    20    26    31    37    43    49    50 | first  last
acc  0.82  0.51  0.23  0.20  0.14  0.06  0.03  0.03  0.01  0.00  0.01  0.01  0.01  0.00  0.00  0.01 | 0.76  0.26
```

Interior on the floor by position 6 and it never recovers — including at
position 50, the final update, when asked **by number** (0.01). Ask for the same
item as "the last thing" and it jumps to 0.26. Averaged over all six cells:

| condition | ordinal | semantic |
|---|---|---|
| FVQ | 0.806 | 0.789 |
| IVQ | 0.123 | — |
| CVQ | **0.038** | **0.478** |

A 12× difference on the identical target item, from wording alone. Same
decomposition as theme 02 — a *counting* failure separate from a *retrieval*
failure — reproduced on prose.

## Headline 2b — the counting failure is haiku's; the interior failure is not

With opus on the `nocot` arm, the museum result splits in two. Mean accuracy,
ordinal queries:

| model | arm | FVQ | IVQ | CVQ |
|---|---|---|---|---|
| haiku | `nocot` | 0.827 | **0.142** | **0.051** |
| opus | `nocot` | 0.631 | **0.107** | **0.642** |

Haiku shows a huge endpoint asymmetry on numbered queries — 0.83 at the first
update, 0.05 at the last. Opus is essentially flat, 0.63 vs 0.64. So *"cannot
count to N"* is a **haiku** failure, not a property of the narrative. But the
interior is on the floor for both (0.142 and 0.107), so *"cannot reach the
interior"* holds across models.

This is the same split theme 07 found on the synthetic stimulus, where opus
`nocot` scored 0.554 at CVQ-ordinal against haiku's 0.010 — so it is a model
property that survives the format change, in both directions.

Worded endpoints agree across models (`first` 0.839 haiku / 0.832 opus; `last`
0.530 / 0.559), which is what isolates the effect to *numbered* addressing.

## Headline 3 — CoT rescues the interior here too, but only halfway

`cot_ivq.csv`, ordinal queries, mean accuracy — **haiku only**, the one model
with both arms:

| arm | FVQ | IVQ | CVQ |
|---|---|---|---|
| `nocot` | 0.827 | **0.142** | 0.051 |
| `cot_thinking` | 0.889 | **0.579** | 0.658 |

The direction matches theme 07 — thinking turns an interior floor into real
retrieval — but the ceiling is much lower. Theme 07 reached **0.97–1.00** on
Semantic-Multi; the narrative tops out at **0.579**. So CoT's interior rescue is
partly format-dependent: counting sentences in prose is harder than counting
labelled lines, and the same 3,000-token budget buys less.

Worded endpoints also gain: `last` goes 0.530 → 0.904 with thinking.

> **7 of 96 `cot_thinking` rows are lower bounds.** Those positions had ≥1 call
> hit the 3,000-token thinking budget, which the API does not report — the sweep
> derives it per call and the row says so in `notes`. Truncation lands on deep
> positions and never on shallow ones, so it depresses exactly the numbers under
> test. A 28-call probe at budget 16,000 found 50% of calls exceeding 3,000 on
> deep positions of K10/N50.

## Two source properties that do not hold elsewhere

- **Seeds are not reproducible for `endpoint` and `ivq`.** Both record
  `seed_formula: abs(hash((nk,nu,t)))%(2**31)`, and Python salts `hash` per
  process for the string in that tuple — the exact defect `cot_ivq_prompts.py`
  switched to blake2b to avoid. The accuracies are sound; the individual stimuli
  cannot be regenerated. Every row from those two files carries the warning in
  `notes`. `cot_ivq` is clean (`seed_version=museum_cot_v1`).
- **`ivq.csv` has no early stopping.** n=200 flat on every position, unlike every
  other sweep here. Convenient — cross-position comparisons need no weighting.

## Caveats

- **Two models at most, one on most sweeps.** Nothing here supports a scaling
  or cross-family claim: the synthetic themes span 14 models, this spans two on
  one arm of one file and one everywhere else.
- **Grids do not line up cell-for-cell** with the themes they mirror, so compare
  shapes and trends, not paired cells. `ivq.csv` names its intended comparator
  in `notes`: `plain flat_nolabel in ucurve_proprietary_results.csv` (theme 02).
- **`legacy/` in the `ivq` source path is historical.** That run is dated
  2026-07-22, contemporaneous with the other two; the directory is not a
  statement about the data.

## Sources

`experiments_cloud/results/museum_endpoint/claude-haiku/`,
`legacy/results/museum_ivq/claude-haiku/`,
`experiments_cloud/results/museum_cot/claude-haiku-4-5-20251001__{nocot,cot_thinking}/`,
`experiments_cloud/results/museum_cot/claude-opus-4-5-20251101__nocot/`.
Producers: `museum_endpoint_sweep.py`, `museum_ivq_haiku.py`,
`museum_cot_sweep.py` + `museum_cot_prompts.py` + `museum_cot_score.py`,
`run_museum_cot.sh`. See `../PROVENANCE.md` or `../build/build_08_museum.py`.
