# Theme 1 — exact configurations per model

What each model was actually run on. Regenerate with
`python3 ../build/report_configurations.py`.

**Both model families were run on both datasets.** What differs is which raw
data survived, and that is what shapes this file:

| | `ARBITRARY_SINGLE` | `SEMANTIC_MULTI` |
|---|---|---|
| open-weight | **raw, 8 models × up to 79 cells** | ran, but raw **not local** → survives only as 1 accuracy cell + 4 gap-only cells per model, back-extracted from the paper |
| proprietary | ran as `ARBITRARY_**MULTI**` (a different dataset), 2 models only, March 2026 — **not ingested**, see below | **raw aggregate, 6 models × 69 cells** |

So theme 1 is open-weight-on-Arbitrary vs proprietary-on-Semantic **not by
design, but by what is on disk.** The two blocks below are therefore not
comparable cell-for-cell — they differ in dataset, grid, prompt and trial policy:

| | open-weight | proprietary |
|---|---|---|
| dataset | `ARBITRARY_SINGLE` | `SEMANTIC_MULTI` |
| backend | vLLM, local GPU | vendor APIs |
| grid | K∈{2,3,5,7,10,15,20,25,30} × N∈{5,7,10,15,20,30,50,75,100} = 81 | K∈{1,2,5,10,15,20,25,30,40,45} × N∈{1,5,10,15,20,30,50} = 70 |
| trials/cell | **100, fixed** | **40–200, early-stopped** |
| decoding | greedy (temp 0, top_p 1.0, top_k −1, no penalties) | vendor default, temp 0 |
| max_new_tokens | 30 | (not recorded in the surviving CSV) |
| min updates | 5 | 1 |

**Not comparable cell-for-cell.** Overlapping (K, N) coordinates are
coincidental; the tasks differ.

### Why proprietary Arbitrary data is not here

`experiments_cloud/sweep_arbitrary.py` sets `DATASET_TYPE = "ARBITRARY_MULTI"` —
a *different* dataset from the open-weight `ARBITRARY_SINGLE`, so it would not
make the comparison possible anyway. Only two models were run
(claude-haiku, gpt-4.1-mini), at 10–20 trials per cell, in March 2026, and one
of the four files is 100% API auth failures. It is excluded as not comparable
and not usable, not as an oversight.

**Where proprietary Arbitrary-style data does exist:** nowhere in this
consolidation. Themes 2 and 3 use the proprietary U-curve, which also ran on
`SEMANTIC_MULTI` (`run_ucurve_proprietary.sh` passes `--dataset SEMANTIC_MULTI`).

---

## ⚠ The prompt format is NOT constant across open-weight models

Four models ran with a chat template, four with few-shot completion, chosen by
whether the checkpoint is instruction-tuned:

| `prompt_format` | models |
|---|---|
| `chat_template` | Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct, gemma-3-1b-it, gemma-3-4b-it |
| `completion_few_shot` | Qwen2.5-3B *(base)*, Qwen3.5-2B, Qwen3.5-4B, Qwen3.5-9B |

The two prompts are different text (`v3/scripts/experiments/stage1_sweep.py:264`
vs `:279`), and the completion variant additionally prepends
`FIXED_COMPLETION_DEMOS` — few-shot examples the chat models never see.

**This is a live confound for any cross-model claim in theme 1**, and theme 3
is the reason to take it seriously: formatting moves the FVQ−CVQ gap by up to
0.94 on these same models. So a Qwen3.5-9B vs Qwen2.5-3B-Instruct comparison
varies model family, instruction-tuning, *and* prompt format at once.

It also aligns almost perfectly with the Qwen3.5 line, which means **"Qwen3.5
behaves differently" and "completion format behaves differently" are not
separable in this data.**

Safe comparisons: within a `prompt_format` group. Unsafe: across them, without
saying so.

---

## Open-weight — `ARBITRARY_SINGLE`, 100 trials/cell, greedy, 30 new tokens

Cell coverage. `X` = ran, `.` = absent.

**Two of the 81 cells are impossible by construction**, not skipped:
`ARBITRARY_SINGLE` draws K×N values without replacement from a **shared
2,300-word pool**, so K×N ≤ 2300 — ruling out K=25/N=100 (2,500) and
K=30/N=100 (3,000). **Maximum achievable is 79.**

So Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct, Qwen3.5-2B and Qwen3.5-9B ran the
**complete grid**; their "79/81" is the ceiling, not a truncation. Only the other
four lost cells beyond that, to context limits or an interrupted run. Full
derivation in [`../DATASETS.md`](../DATASETS.md).

```
                                    N=   5    7   10   15   20   30   50   75  100
Qwen2.5-1.5B-Instruct  chat        79/81 cells
  K=2,3,5,7,10,15,20                    X    X    X    X    X    X    X    X    X
  K=25,30                               X    X    X    X    X    X    X    X    .

Qwen2.5-3B-Instruct    chat        79/81 cells
  K=2,3,5,7,10,15,20                    X    X    X    X    X    X    X    X    X
  K=25,30                               X    X    X    X    X    X    X    X    .

gemma-3-1b-it          chat        55/81 cells   <-- heavily truncated
  K=2                                   X    X    X    X    X    X    X    X    X
  K=3,5                                 X    X    X    X    X    X    X    X    .
  K=7,10,15                             X    X    X    X    X    X    .    .    .
  K=20,25,30                            X    X    X    X    .    .    .    .    .

gemma-3-4b-it          chat        76/81 cells
  K=2,3,5,7,10,15                       X    X    X    X    X    X    X    X    X
  K=20                                  X    X    X    X    X    X    X    X    .
  K=25,30                               X    X    X    X    X    X    X    .    .

Qwen2.5-3B (base)      completion  73/81 cells
  K=2,3,5,7,10                          X    X    X    X    X    X    X    X    X
  K=15                                  X    X    X    X    X    X    X    X    .
  K=20,25                               X    X    X    X    X    X    X    .    .
  K=30                                  X    X    X    X    X    X    .    .    .

Qwen3.5-2B             completion  79/81 cells
  K=2,3,5,7,10,15,20                    X    X    X    X    X    X    X    X    X
  K=25,30                               X    X    X    X    X    X    X    X    .

Qwen3.5-4B             completion  50/81 cells   <-- INCOMPLETE RUN, see below
  K=2,3,5,7,10                          X    X    X    X    X    X    X    X    X
  K=15                                  X    X    X    X    X    .    .    .    .
  K=20,25,30                            .    .    .    .    .    .    .    .    .

Qwen3.5-9B             completion  79/81 cells
  K=2,3,5,7,10,15,20                    X    X    X    X    X    X    X    X    X
  K=25,30                               X    X    X    X    X    X    X    X    .
```

### Per-run engine settings

| model | model_id | dtype | max_model_len | gpu_mem_util | max_batched_tok | prompts | runtime |
|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | `Qwen/Qwen2.5-1.5B-Instruct` | float16 | 16384 | 0.95 | 32768 | 15,800 | 687 s |
| Qwen2.5-3B-Instruct | `Qwen/Qwen2.5-3B-Instruct` | float16 | 16384 | 0.92 | 16384 | 15,800 | 1,368 s |
| Qwen2.5-3B | `Qwen/Qwen2.5-3B` | float16 | **8192** | 0.92 | 16384 | 7,600 | 757 s |
| Qwen3.5-2B | `Qwen/Qwen3.5-2B` | bfloat16 | 16384 | 0.92 | 65536 | 14,800 | 1,081 s |
| Qwen3.5-4B | `Qwen/Qwen3.5-4B` | bfloat16 | 16384 | 0.92 | 32768 | **0** | **0.0 s** |
| Qwen3.5-9B | `Qwen/Qwen3.5-9B` | bfloat16 | 16384 | 0.92 | 16384 | 15,800 | 3,837 s |
| gemma-3-1b-it | `google/gemma-3-1b-it` | bfloat16 | **8192** | 0.95 | 32768 | 11,000 | 86 s |
| gemma-3-4b-it | `google/gemma-3-4b-it` | bfloat16 | **8192** | 0.92 | 16384 | 15,200 | 1,090 s |

Note the dtype split (float16 for Qwen2.5, bfloat16 for Qwen3.5/Gemma) and the
8192-token context on three models — that, not model capability, is why
gemma-3-1b-it and Qwen2.5-3B lose their high-K/high-N corners *beyond* the two
pool-bound cells every model loses.

### Two anomalies

**`Qwen3.5-4B` — 50/81 cells, and its metadata says `total_prompts: 0`,
`inference_time_sec: 0.0`.** Every K≥20 row is missing and K=15 stops at N=20,
which is not a context-limit pattern (Qwen3.5-2B and -9B, same context, same
format, completed 79). This looks like an interrupted or resumed run whose
summary counters were never written. **Treat Qwen3.5-4B as partial** — in
particular it contributes nothing to any high-K claim, so it cannot appear in a
scaling comparison against 2B/9B at those cells.

**`gemma-3-1b-it` — 55/81 cells** with an 8192 context. Its coverage is a clean
staircase, consistent with preflight rejection, so this one is explained. But it
means its +0.54 mean gap is measured on an easier subset (no cell beyond
K=15/N=30) than gemma-3-4b-it's +0.46 across 76 cells. **The two Gemma sizes are
not comparable as an in-family scaling pair without restricting to shared cells.**

---

## The cells are not shared — and that changes the numbers

Only **42 of 81** cells were run by all eight open-weight models:
K ∈ {2, 3, 5, 7, 10, 15} × N ∈ {5…100}. Every K ≥ 20 cell is missing from at
least one model (Qwen3.5-4B has none at all).

Mean FVQ−CVQ gap, all cells that model ran vs the 42-cell shared subset:

| model | format | all its cells | **shared 42 only** |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct | chat | +0.46 | **+0.64** |
| Qwen2.5-3B-Instruct | chat | +0.02 | **+0.23** |
| gemma-3-1b-it | chat | +0.54 | **+0.55** |
| gemma-3-4b-it | chat | +0.46 | **+0.31** |
| Qwen2.5-3B *(base)* | completion | **−0.12** | **+0.06** |
| Qwen3.5-2B | completion | +0.07 | **+0.24** |
| Qwen3.5-4B | completion | +0.19 | **+0.24** |
| Qwen3.5-9B | completion | +0.21 | **+0.20** |

**On the shared subset every model is positive**, including Qwen2.5-3B, which is
−0.12 across its own cells. The negative and near-zero means are produced
entirely by high-K cells that only some models reached.

Both readings are true, and they answer different questions:

- **All-cells means confound the effect with coverage.** A model that reached
  K=30 is averaging in reversal-regime cells its neighbours never ran. Do not
  rank models on this.
- **Shared-cell means are comparable but exclude the reversal regime**, which is
  a real and interesting finding in its own right — it just cannot be stated as
  a *cross-model* difference from this data, because coverage and K are
  entangled.

If theme 1 needs one headline number, compute it on the 42 shared cells and say
so. If the reversal claim is being made, make it *within* a model across K, not
between models.

---

## Proprietary — `SEMANTIC_MULTI`, API, early-stopped

All six models: **69 of 70 cells**, uniformly — the complete achievable grid.
K=45/N=50 is **impossible by construction**: `SEMANTIC_MULTI` uses per-category
pools and only **40** of the 46 categories hold ≥50 values, so 45 keys at N=50
cannot be built. Nothing was skipped.

```
                                       N=   1    5   10   15   20   30   50
claude-4.5-haiku, claude-4.5-sonnet,
gpt-4.1, gpt-4.1-mini,
gemini-2.5-flash, gemini-2.5-pro
  K=1,2,5,10,15,20,25,30,40                 X    X    X    X    X    X    X
  K=45                                      X    X    X    X    X    X    .
```

Coverage is identical across models — unlike the open-weight side, there is no
per-model grid confound here.

### Trial counts vary per cell (Wilson early stopping)

| model | min | max | distinct values |
|---|---|---|---|
| claude-4.5-haiku | 40 | 200 | 40, 50, 80, 100, 125, 150, 175, 200 |
| claude-4.5-sonnet | 40 | 200 | 40, 80, 120, 160, 200 |
| gpt-4.1 | 40 | 200 | 40, 80, 120, 160, 200 |
| gpt-4.1-mini | 40 | **175** | 40, 50, 75, 100, 125, 150, 175 |
| gemini-2.5-flash | 40 | 200 | 40, 50, 75, 100, 125, 150, 175, 200 |
| gemini-2.5-pro | 40 | 200 | 40, 80, 160, 200 |

**A cell at n=40 has a Wilson half-width of roughly ±0.15** — five times wider
than a 200-trial cell. Early stopping fires when the CI is already tight, so low
n concentrates on saturated (near-0 or near-1) cells; but any per-cell claim
must carry its own n. The column is in the CSV; use it.

`gpt-4.1-mini` never reaches 200 — it caps at 175, a different stopping schedule
from the other five.

---

## Open-weight `SEMANTIC_MULTI` — sparse, and derived only

The raw open-weight semantic sweep is **not local**
(`v3/results_vllm/semantic_multi/`). What survives, back-extracted from the
paper's committed tables:

| model | cells with FVQ+CVQ accuracy | cells with gap only |
|---|---|---|
| Qwen2.5-3B-Instruct, Qwen3.5-2B, Qwen3.5-4B, Qwen3.5-9B, gemma-3-1b-it, gemma-3-4b-it | K=5, N=30 | K=2/N=10, K=5/N=30, K=10/N=50, K=20/N=50 |
| Qwen2.5-1.5B-Instruct, Qwen2.5-3B | *(none)* | same 4 cells |

So the open-weight semantic evidence is **one cell** with usable accuracies per
model, plus four gap-only points. That is the entire basis for any open-weight
vs proprietary comparison on the semantic task — the proprietary side has 69
cells with full accuracies against the open-weight side's one.

---

## Reading the CSV

Every column above is present per row, so nothing here needs re-deriving:

```python
import pandas as pd
df = pd.read_csv("fvq_cvq.csv")

# what did one model actually run?
df[df.model == "Qwen3.5-4B"][["num_keys","num_updates","condition",
                              "accuracy","n_trials","prompt_format"]]

# avoid the format confound
chat = df[df.prompt_format == "chat_template"]

# only cells every model has (fair scaling comparison)
arb = df[(df.dataset == "arbitrary_single") & df.accuracy.notna()]
shared = set.intersection(*[
    set(map(tuple, g[["num_keys","num_updates"]].values))
    for _, g in arb.groupby("model")])
```
