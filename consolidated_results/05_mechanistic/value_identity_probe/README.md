# Value-identity probe — is the answer *value* decodable, even when wrong?

`../probing.csv` probes **correctness**: a binary "will the model get this
right" readout, which degenerates wherever accuracy sits at 0% or 100%. This
probes **value identity** instead: at the answer position, which of the 50 pool
words is the ground-truth answer? The label comes from the trial definition, not
from the model's output, so it stays defined at 0% and 100% alike.

| | |
|---|---|
| arms | 4 — 2 models × 2 cells, **base only**, one directory each |
| models | Qwen2.5-3B-Instruct, gemma-3-4b-it |
| cells | (K=5, N=10) and (K=10, N=5) |
| task | 50-way classification, **chance 2%** |
| probe | logistic regression on the full residual stream (2,048 / 2,560 dims, no PCA), 5-fold stratified CV, C=0.1 |
| rows | 6,967 layer-rows across arms. All `raw`. |
| design doc | `lora_intervention/experiments/linear_probing/PROBE50_DESIGN.md` |

## Layout — one directory per arm, three subsets per arm

Directory names match the source directories exactly, so no lookup table is
needed to get from a row back to its run.

```
value_identity_probe/
├── Qwen2.5-3B-Instruct_5k_10u/     K=5,  N=10   36 layers, d=2048
├── Qwen2.5-3B-Instruct_10k_5u/     K=10, N=5
├── gemma-3-4b-it_5k_10u/           K=5,  N=10   34 layers, d=2560
├── gemma-3-4b-it_10k_5u/           K=10, N=5
├── summary_all_arms.csv            216 rows — the 4 arms' summaries, stacked
└── behavioral_all_arms.csv          38 rows — the 4 arms' behaviour, stacked
```

Inside every arm directory:

| file | what |
|---|---|
| `probe_all.csv` | **subset `all`** — every trial |
| `probe_correct.csv` | **subset `correct`** — trials the model got right. Confounded with the output head, so this is the confound check, not a result |
| `probe_wrong.csv` | **subset `wrong`** — trials the model got wrong. **The headline**: decodable here means tracked and not emitted |
| `summary.csv` | best layer + best top-1 per (condition, subset, label set) for this arm |
| `behavioral.csv` | accuracy and the 6-bucket output audit per condition |
| `pool.csv` | this arm's 50-word label space (tokenizer-specific — see caveats) |

Row counts differ across the three subset files by design, not by coverage:
`probe_all` and `probe_correct` carry the `expected` label set only, while
`probe_wrong` carries all four label sets, so it is ~4× larger.

| arm | `probe_all` | `probe_correct` | `probe_wrong` |
|---|---|---|---|
| Qwen (5,10) | 444 | 222 | 1,776 |
| Qwen (10,5) | 259 | 185 | 1,036 |
| gemma (5,10) | 420 | 140 | 1,400 |
| gemma (10,5) | 245 | 140 | 700 |

`probe_correct` is short in the gemma arms for a real reason, not a missing run —
see "the 27 cells that are not estimable" below.

## Why the pool is closed and exactly K×N

`generate_values_for_trial` draws `rng.sample(pool, K*N)` without replacement.
Setting `len(pool) == K*N` makes every stream a **permutation of the whole
pool** — all 50 values appear exactly once in every trial.

That removes the presence confound. If only some pool values appeared per trial,
a probe could beat chance by detecting *which values are in context at all*,
which says nothing about retrieval. With a permutation, presence is constant and
carries zero information, so the probe must localise the queried slot.

It also forces the grid: `K*N ≤ 50`, hence only these two small cells. `(5,5)`
was rejected because `K*N = 25` would either shrink the label space to 25-way
(chance 4%, not comparable) or leave 25 values absent per trial (reinstating the
confound).

## Headline — the value is there on trials the model gets wrong

Qwen2.5-3B-Instruct, (5,10), CVQ, **wrong-answer subset**: top-1 **0.455** at
layer 33 against 2% chance. The value is present in the residual stream and the
model emits something else.

Across subsets (`label_set=expected`, CVQ):

| model | cell | subset | n | best layer | rel. depth | top-1 | behavioural acc |
|---|---|---|---|---|---|---|---|
| Qwen2.5-3B | (5,10) | `all` | 3,700 | 32 | 0.89 | 0.736 | 0.662 |
| Qwen2.5-3B | (5,10) | `correct` | 2,442 | 33 | 0.92 | 0.924 | 0.662 |
| Qwen2.5-3B | (5,10) | **`wrong`** | 1,249 | 33 | 0.92 | **0.455** | 0.662 |
| gemma-3-4b | (5,10) | **`wrong`** | 1,262 | 32 | 0.94 | **0.664** | 0.852 |
| gemma-3-4b | (10,5) | **`wrong`** | 1,231 | 31 | 0.91 | **0.735** | 0.920 |

The `correct` subset is confounded with the output head — there, `expected`
equals what the model emitted, so 0.92–0.99 measures the readout, not tracking.
It is reported as the confound check, not as a result. `wrong` is the headline.

**Where it appears is as informative as whether.** Qwen (5,10) CVQ, wrong subset:

```
layer    0     4     8    12    16    20    24    28    32    36
top-1  0.04  0.07  0.05  0.04  0.04  0.03  0.03  0.04  0.40  0.42
MRR    0.13  0.18  0.15  0.14  0.13  0.12  0.12  0.14  0.54  0.59
```

Flat at chance for 28 layers, then a step change at ~L31. Value identity is
**constructed late**, at ~90% relative depth. That lines up with two independent
methods in this theme: logit-lens P(v_last) peaks around L33, and Entropy-Lens
puts Qwen's entropy peak at 89% depth. Three methods, three metrics, same
location.

## Read the controls first — one of them overturns the easy reading

Four label sets are fitted on the **same** activations, so the controls cost no
GPU:

| label set | definition | purpose |
|---|---|---|
| `expected` | `cv[k-1]` | the result |
| `shuffled` | labels permuted across trials | leakage check; must be at chance |
| `cv_first` | `cv[0]` | primacy control |
| `cv_last` | `cv[-1]` | recency control |

**`shuffled` is clean.** 1,228 layer-rows, max top-1 **0.057** against 2%
chance, none above 4× chance. No fold leakage, no memorisation artefact.

**`cv_first` is not.** At every interior slot it beats `expected`:

| model | cell | slot | `expected` | `cv_first` | `cv_last` |
|---|---|---|---|---|---|
| gemma-3-4b | (5,10) | k2 | 0.638 | **0.961** | 0.050 |
| gemma-3-4b | (10,5) | k2 | 0.635 | **0.934** | 0.112 |
| gemma-3-4b | (5,10) | k3 | 0.338 | **0.674** | 0.071 |
| Qwen2.5-3B | (10,5) | k2 | 0.245 | **0.561** | 0.158 |
| Qwen2.5-3B | (5,10) | k2 | 0.207 | **0.537** | 0.064 |
| Qwen2.5-3B | (5,10) | k4 | 0.144 | **0.280** | 0.081 |

So what the residual stream robustly carries at the answer position is the
**first** value, not the queried one — whatever was asked. "Tracked but not
surfaced" holds strongly for the first value and only weakly for interior ones.
The design anticipated a *recency* confound and built `cv_last` for it; the
confound that showed up is *primacy*. State this alongside any claim built on
the interior numbers.

> **Two label sets are degenerate, not controls.** `cv_last` where k = N is the
> same label as `expected` (so CVQ rows read identically by construction), and
> `cv_first` where k = 1 likewise (FVQ and k1 rows). Those rows carry
> `DEGENERATE:` in `notes`. Filter them out before summarising a control.

## Conditions: slot and wording are separate axes

Run labels index the queried slot directly, and two pairs target the *same* slot
with different wording — the ordinal-vs-semantic contrast this repo tracks
everywhere else, available here for free:

| `condition_label` | `slot` | wording | `condition` | `query_type` |
|---|---|---|---|---|
| `FVQ` | 1 | "first" | FVQ | `semantic` |
| `k1` | 1 | "1st" | FVQ | `ordinal` |
| `k2` … `k(N-1)` | 2…N−1 | "2nd" … | IVQ | `ordinal` |
| `kN` | N | "5th" / "10th" | CVQ | `ordinal` |
| `CVQ` | N | "last" | CVQ | `semantic` |

## Sample size, and the 27 cells that are not estimable

T was set **per condition** so the wrong subset lands near 1,250 — the size
needed for 50 classes × 20 examples in a 4/5 training fold. A flat T would have
made any cross-condition gap partly a sample-size artefact. It is `wrong_frac`
that varies, and it varies enormously: gemma (10,5) CVQ needed **15,369 trials**
to yield 1,232 wrong ones, because it is 92% accurate there.

The two subsets then fail at **opposite ends of accuracy**, which is worth
seeing as a single fact:

| subset | empties when | affected |
|---|---|---|
| `wrong` | the model is too **accurate** | 4 rows — gemma FVQ / k1, both cells (n = 10–22) |
| `correct` | the model is too **inaccurate** | 23 rows — interior slots, mostly gemma (n = 0–159) |

All 27 appear in the arm's `summary.csv` (and in `summary_all_arms.csv`) with
`estimable=False` and the reason in `notes`,
so the gap is a row you can see rather than an absence you have to notice. They
contribute no rows to the `probe_*.csv` files. A further 288 individual layer records
inside otherwise-fine fits carry empty metrics plus
`class support < folds` — kept as rows so a layer curve has no silent hole.

## Known gaps

- **The C robustness sweep was never run.** `C_grid` is `[0.1]` in every source
  file, so `global_C` was selected over a single candidate and the design's
  "replot at C ∈ {0.0003 … 1}, parallel curves mean C-independence" figure does
  not exist. The design's own synthetic grid favoured C ≈ 0.003 and found
  regularisation worth ~2.3 points of top-1, so 0.1 is probably not optimal. The
  activations are on disk (17 GB, CPU-only re-fit), so this is a re-fit, not a
  re-run. `C` is a column here and ready for more values.
- **LoRA is absent, and not by oversight.** LoRA scores 1.000 at CVQ and shallow
  k in every cell the closed pool permits, so its wrong subset is empty at any
  T. The cells where LoRA does err need `K*N = 250`, which the closed pool
  forbids. The closed-pool design and a LoRA wrong-subset comparison are
  mutually exclusive — a LoRA arm needs a different design, not more trials.
- **Pools are tokenizer-specific, so top-1 is comparable within a model only.**
  `select_pool` requires each value to be a single token both with and without a
  leading space, and requires that no value be a substring of another (because
  `is_correct` accepts `expected in pred`, which would score "ace" correct for
  "access"). Qwen and gemma therefore get different 50-word pools; both are in
  their arms' `pool.csv`.
- **Scoring agreement is 1.000 in 34 of 38 conditions**, minimum 0.963 — the
  `single_token` argmax and greedy `generate` pick the same answer. The
  single-token shortcut is what halves GPU cost, so this is the check that
  licenses it.

## Sources

`lora_intervention/experiments/linear_probing/results_probe50/<model>_<cell>/` —
`manifest.json` (behaviour, audit, pool) and `probe_fits_*.json` (the fits).
Each arm directory here maps 1:1 onto the source directory of the same name, and
every row repeats its own `source_file`.

**Shards must be resolved, not concatenated.** The wrong-subset fits are spread
over up to three files per cell (`probe_fits_wrong.json` plus `_topup` /
`_missing`), and four (condition, subset) pairs are claimed twice with different
n — Qwen (10,5) CVQ appears at both n=616 and n=1231. The larger is the
completed run; the smaller is an under-powered first pass. The builder keeps the
largest n and prints how many it dropped; affected rows name the superseded
shard in `notes`.

The 17 GB of `reps_*.npy` activations stay in place — they are inputs to the
fit, not results. See `../../PROVENANCE.md` or `../../build/build_05c_value_identity.py`.
