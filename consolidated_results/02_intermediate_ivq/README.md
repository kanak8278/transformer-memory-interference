# 02 — Intermediate-position retrieval (IVQ), standalone base models

**Question.** The first and last values are the easy cases. What happens when
you ask for update number 7 of 50, on an unmodified model?

| | |
|---|---|
| file | `ivq.csv` — 2,160 rows, all `raw` |
| per-model slices | `by_model/` — 6 files (generated; do not edit) |
| models | **6 proprietary only** — claude-4.5-{haiku,sonnet}, gpt-4.1, gpt-4.1-mini, gemini-2.5-{flash,pro} |
| dataset | `semantic_multi` |
| variant | `base` (no fine-tuning — these are API models) |

## Scope note — why proprietary-only

This theme holds only **standalone** base-model interior sweeps. Every
open-weight IVQ experiment we have (behavioral block-vs-LoRA, dense-IVQ, E1
extrapolation) was run as a **base-vs-LoRA comparison**, so it lives with the
LoRA work in **`../04_lora/lora_ivq.csv`**, not here. The proprietary U-curve
is the one interior sweep with no LoRA arm — you cannot fine-tune an API model —
so it is the only genuinely standalone IVQ data, and it stays.

If you want the open-weight U-shape / block-flattens-it / LoRA-doesn't story,
it is now in `../04_lora/README.md`.

## The run

One fully-crossed sweep, **identical grid across all 6 models**:

- **4 formats** — `flat_nolabel`, `flat_verbose`, `block`, `landmark`
- **6 cells** — K ∈ {5, 10} × N ∈ {10, 20, 50}
- **~15 probed positions** per cell (1–8, then a spread, then `last`)
- **40–200 trials/cell** — the only thing that varies across models, via Wilson
  early stopping (a saturated cell stops at 40; an uncertain one runs to 200)

## Headline: the interior is the floor, and format decides how deep

Mean accuracy by which value is targeted, `flat_nolabel` (indices hidden):

| | FVQ (first) | **IVQ (interior)** | CVQ (last, semantic) |
|---|---|---|---|
| proprietary, `flat_nolabel` | 1.00 | **0.47** | 0.62 |

Per-position shape, GPT-4.1, `flat_nolabel`, K=10, N=50:

```
pos    1     2     3     4     5     6     7    14    20    26    31    37    43    49    50   last
acc  1.00  0.85  0.34  0.14  0.04  0.04  0.06  0.01  0.01  0.03  0.01  0.01  0.01  0.01  0.01  0.69
```

Accuracy collapses by position 5 and never recovers — except at `last`. The
`block` and `flat_verbose` formats lift the interior sharply (see theme 03);
`flat_nolabel` and `landmark` leave it on the floor.

## The ordinal / semantic split — read before using `condition`

`condition` = *which* value (FVQ/CVQ/IVQ); `query_type` = *how* it was asked
(`semantic` "the last value of X" vs `ordinal` "the 7th value of X"). The final
update is queried both ways, and the answers diverge enormously:

| model | format | ordinal | semantic |
|---|---|---|---|
| gpt-4.1 | flat_nolabel | **0.012** | **0.832** |
| claude-4.5-haiku | flat_nolabel | 0.035 | 0.656 |
| *all six* | block | 0.99–1.00 | 0.99–1.00 |
| *all six* | flat_verbose | 0.99–1.00 | 0.88–1.00 |

The gap appears only under `flat_nolabel` / `landmark` (the model must count
occurrences) and vanishes under `block` / `flat_verbose` (indices given). So the
interior failure decomposes into a **counting** failure and a **retrieval**
failure. **Never pool the two query types into one mean:**

```python
df[(df.condition == "CVQ") & (df.query_type == "semantic")]
```

## Caveats

- Interior means are over unevenly sampled positions (16 probed of N).
- Trial counts vary per cell (40–200); read `n_trials` before comparing.
- These rows also appear in `../03_prompt_format/format.csv`, viewed along the
  format axis. Don't concatenate the two without de-duplicating.

## Sources

`experiments_cloud/results/ucurve_proprietary_results.csv`. See
`../PROVENANCE.md` or `../build/build_02_ivq.py`.
