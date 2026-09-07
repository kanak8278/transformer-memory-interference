# 06 — From-scratch training

The interference signature is not inherited from web-scale pretraining. A
GPT-2-small *architecture* trained from random init on a 51-symbol synthetic
key-value task reproduces it — and reproduces it while being *good* at the task.

| | |
|---|---|
| files | `from_scratch_ivq.csv` (252 rows, per-position grid), `from_scratch_training.csv` (120 rows, the phase transition), `from_scratch_summary.csv` (86 rows, final aggregates) |
| per-model slices | `by_model/` — 1 file (generated; do not edit) |
| model | `gpt2-small` architecture, `variant=scratch_cosine`, random init |
| source run | `synthetic_scratch_training/05_gpt2_scratch_h100/results/h100_cosine/` |
| tiers | 458 `raw`, 0 `derived` |
| chance | **0.10** (10 value symbols, i.i.d. with replacement) |

---

## What was moved, and what was not

**Moved: the `h100_cosine` arm of experiment 05, and nothing else.** That is the
run that reached **iid 0.800 / held-out 0.633**, and it is the only source
behind `present/ONE_PAGER.tex` §7 (both figures: `10_trained_from_scratch_ood.png`
from `joint_knstep.json` via `src/plot_depth_grid.py`, and
`11_trained_from_scratch_heatmap.png` from `final_eval.json` via
`src/plot_results.py`).

Everything below is **deliberately left in place**, not overlooked. Each is a
real, complete run; none of them is presented anywhere.

| left out | what it is | why it was left |
|---|---|---|
| `05/h100_plateau` | complete 30,000-step run, same seed and data, LR held at 6e-4 instead of decayed; iid 0.283 / held-out 0.208 | **The control.** It trained equally long and never escaped the plateau, so it is what separates "the LR decay enabled this" from "we just trained longer." Not presented, so not consolidated — but see the warning below before making the schedule claim in print. |
| `01_baseline_step_ood` (L2, L3) | tiny 2- and 3-layer transformer, d=64; 0.226 / 0.233 iid | Different architecture family, superseded as evidence by 05. |
| `02_gpt2_scratch` | GPT-2-small scratch, 3,000 steps, local | No `final_eval.json` at all — validation only. Superseded by 04. |
| `03_gpt2_pretrained` | GPT-2-small **pretrained**, 0.665 / 0.527 | The pretrained comparator. Worth consolidating if the "from scratch beats pretrained" contrast is ever claimed; it is not claimed in §7. |
| `04_gpt2_scratch_kaggle_t4` | GPT-2-small scratch, 15,000 steps; best val at step **1,500**, declining after | The pre-05 plateau evidence. |
| `06_first_last_only`, `07_gpt2_pretrained_first_last` | code + `setup.md` only | Never launched. No results exist. |

> **If you cite the LR-schedule claim, bring `h100_plateau` back.** It is one
> line in `build/build_06_from_scratch.py` (`ARMS`). Both arms are n=1, so even
> with it the schedule comparison is a single seed per arm.

---

## The result: three phase transitions, in FVQ → CVQ → IVQ order

`from_scratch_training.csv`. Validation accuracy sat at ~0.23 for 13,000 steps,
then broke — but not all at once. The three roles transition **separately**,
in exactly the order this project's whole argument is about:

| train step | lr | train loss | FVQ (first) | CVQ (last) | IVQ (interior) |
|---|---|---|---|---|---|
| 13,000 | 3.96e-4 | 0.979 | 0.304 | 0.293 | 0.203 |
| 14,000 | 3.67e-4 | 0.892 | **0.614** | 0.301 | 0.207 |
| 16,000 | 3.10e-4 | 0.809 | **0.789** | 0.313 | 0.228 |
| 20,000 | 2.00e-4 | 0.635 | 0.883 | **0.448** | 0.355 |
| 24,000 | 1.14e-4 | 0.314 | 0.906 | **0.801** | **0.664** |
| 28,000 | 6.61e-5 | 0.205 | 0.907 | 0.860 | **0.751** |

First-value retrieval breaks out alone at step 14k. Current-value follows ~6k
steps later. Interior counting arrives last. So the FVQ > CVQ > IVQ ordering
seen in every pretrained model is here the **acquisition order** of the
capability, with ~7,000 steps between first and interior.

And the ordering **survives competence**: at 0.80 overall the model is still
0.91 / 0.86 / 0.75. It is not a capability deficit.

### This is not grokking, and the distinction matters

`05_gpt2_scratch_h100/WEIGHTS.md` calls this "the grokked model." In grokking
(Power et al.) train loss reaches ~0 early and validation catches up much
later — memorization, then generalization. Here **train loss and validation
break at the same step** (0.979 → 0.892 as val goes 0.237 → 0.295). Training
loss was plateaued too, not solved. This is escape from a loss plateau, not
memorization giving way to generalization. `synthetic_scratch_training/
FINDINGS.md` independently ruled out memorization on the grounds that training
data is streamed fresh every batch, which is consistent.

---

## The per-position grid

`from_scratch_ivq.csv`, from the step-28,000 checkpoint: 204 trained + 48
held-out (K, N, step) cells, 200 / 500 trials each (64,800 trials total).

Within each (K, N) the curve is U-shaped — the edges are easiest — and adding
keys pushes it down. At K≥10 with long histories it collapses to near chance
across *every* position at once:

```
iid accuracy by query step, N=12   (chance = 0.10; steps 4 and 8 are held out)
  K= 2: 1.00 1.00 1.00   ·  1.00 0.98 1.00   ·  0.99 0.99 0.99 1.00
  K= 4: 1.00 0.98 0.97   ·  0.95 0.94 0.86   ·  0.84 0.85 0.84 0.99
  K= 6: 1.00 0.99 0.98   ·  0.94 0.84 0.78   ·  0.65 0.67 0.63 0.88
  K= 8: 0.99 0.99 0.91   ·  0.72 0.73 0.56   ·  0.50 0.42 0.49 0.72
  K=10: 0.23 0.18 0.15   ·  0.17 0.12 0.15   ·  0.18 0.16 0.17 0.21
  K=12: 0.23 0.14 0.14   ·  0.12 0.21 0.16   ·  0.17 0.12 0.14 0.23
```

### Held-out generalization is partial and position-dependent

**Do not repeat the claim that held-out cells "behave like the trained ones."**
`ONE_PAGER.tex` §7's caption says that; the figure it is attached to
contradicts it. **8 of the 48 held-out cells fall more than 0.25 below their
trained neighbours**, several catastrophically:

| held-out cell | accuracy | trained neighbours (step ±1) | Δ |
|---|---|---|---|
| N=8, step 6, K=2 | 0.19 | 1.00 | **−0.81** |
| N=12, step 8, K=4 | 0.13 | 0.85 | **−0.73** |
| N=6, step 5, K=6 | 0.45 | 0.99 | −0.54 |
| N=12, step 8, K=2 | 0.49 | 1.00 | −0.51 |
| N=12, step 8, K=8 | 0.18 | 0.53 | −0.35 |
| N=12, step 4, K=6 | 0.63 | 0.96 | −0.33 |
| N=8, step 6, K=6 | 0.59 | 0.92 | −0.33 |
| N=10, step 4, K=4 | 0.68 | 0.96 | −0.28 |

Other held-out columns generalize essentially perfectly (all of N=8 step 2, all
of N=10 step 2). The defensible statement is that generalization to unseen
positions is **real but partial, and fails preferentially on deep interior
steps** — which is the same interior weakness the rest of the corpus shows, so
it strengthens rather than weakens the story.

---

## Reading these CSVs

- **`query_type` is `ordinal` on every row, including `position=first/last`.**
  The query is a step *token* (`<S1>`..`<S12>`) — a numeric index. This
  experiment has **no** semantic "the last value of X" arm, so it is not
  comparable to the `semantic` rows in themes 01–04. See the top-level README
  on `condition` vs `query_type`.
- **`prompt_format=token_stream`**, not a natural-language template: the input
  is `<BOS> (Key Value)×K·N <QUERY> Key Step` over a closed 51-symbol
  vocabulary. There is no prompt to quote in `PROMPTS.md`.
- **`dataset=synthetic_kv`** is new to this theme and is *not* the
  `arbitrary_single` / `semantic_multi` pair used elsewhere. Different
  vocabulary, different generator, different task instance. Do not pool it with
  them.
- **`split`**: `train` = (N, step) columns the model was trained on;
  `held_out` = never-trained columns.
- **`held_out_cells` in the source are (N, step) pairs, not (K, N) cells.**
  `(12, 8)` means N=12 / step=8, held out across all six K values — 8 pairs ×
  6 K = the 48 held-out rows. Everywhere else in this folder "cell" means
  (K, N). Each affected row says so in `notes`.
- **`by_duplicate` rows in the summary**: `True` = the target value also occurs
  elsewhere in the queried key's history, so a "most frequent value" shortcut
  can score. `False` is the shortcut-free measure. Prefer `False` when
  precision matters.
- `n_correct` is reconstructed as `acc × n_trials` (the source records only
  `acc`); every value came out integral to within 1e-6.

## QA

`joint_knstep.json` re-aggregated vs `final_eval.json`: **6 values checked, 0
outside the 5-prediction budget (worst 3.0).**

Both files are separate eval passes over the *same* frozen set (both
`build_frozen_eval_set(SEED+2 / SEED+3, …, 200 / 500)`, seed 42, chunk 256) at
the same checkpoint, so pooling one must reproduce the other. They differ by 1
prediction in 40,800 (iid) and 3 in 24,000 (held-out) — 4 flips in 64,800,
0.006%. Same weights and same data leave only non-deterministic float reduction
order under TF32/bf16 flipping near-tie argmaxes, so the check is a budget of
flipped predictions rather than a float epsilon.

## Weights

Not in the repo (~329 MB, gitignored). The step-28,000 checkpoint is on the HF
Hub at `kanak8278/gpt2-small-synthetic-kv-interference` (private). Regenerating
is deterministic — `05_gpt2_scratch_h100/WEIGHTS.md` has the exact command
(~5 h on one H100).
