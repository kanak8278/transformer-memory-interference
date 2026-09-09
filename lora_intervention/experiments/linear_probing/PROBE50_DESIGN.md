# Closed-pool value-identity probe — design and configuration

Companion to `probe50_collect.py` (activations) and `probe50_fit.py` (probes).
Distinct from `run_probing.py`, which probes *correctness* and degenerates
wherever the model sits at 0% or 100% accuracy (339 of 868 condition-cells in
the v3 results).

## The question

At the answer position, is the ground-truth answer value linearly decodable
from the residual stream — including on trials where the model emits something
else? A positive result on the wrong-answer subset means the value is tracked
but not surfaced.

## Why the pool is closed and exactly K*N

`generate_values_for_trial` draws `rng.sample(pool, K*N)` without replacement
(`mechanistic_probing_v2/core/dataset_configs.py:430-436`). Setting
`len(pool) == K*N` therefore makes every stream a **permutation of the whole
pool**: all 50 values appear exactly once in every trial.

This is not a convenience. It removes the presence confound. If only some pool
values appeared per trial, a probe could beat chance by detecting *which values
are in context at all*, which says nothing about retrieval. With a permutation,
presence is constant and carries zero information, so the probe must localize
the queried slot.

It also imposes the binding constraint: `K*N <= len(pool)`, and equality for the
permutation property. Hence small cells only.

## Cells

| cell | K*N | pool | classes | chance | slots k |
|---|---|---|---|---|---|
| (5,10) | 50 | 50 | 50 | 2% | 1..10 |
| (10,5) | 50 | 50 | 50 | 2% | 1..5 |

`(10,5)` keeps `K*N = 50`, so the pool stays 50 and top-1 is directly comparable
to `(5,10)`. `(5,5)` was rejected: `K*N = 25` would either shrink the pool to 25
(changing the label space to 25-way, chance 4%, not comparable) or leave 25 of 50
values absent per trial (reinstating the presence confound).

## Conditions

Indexed by queried slot `k` directly, not by depth fraction. `run_probing.py`
computes `k = min(max(round(depth*N), 2), N-1)`, which at N=10 maps the five
depth fractions onto only `{2, 2, 5, 8, 9}` — `IVQ_d10` and `IVQ_d25` collide on
the same slot.

Run order is `CVQ, k2 .. k(N-1), FVQ`. FVQ is last on purpose: it is the least
interesting condition and the most expensive (highest accuracy, so the smallest
wrong subset per trial), and it can be dropped without losing the rest.

| condition | k | query wording |
|---|---|---|
| CVQ | N | "last" |
| k2 .. k(N-1) | 2..N-1 | ordinal ("second", "fifth", ...) |
| FVQ | 1 | "first" |

All conditions share one generator and one shuffle (`shuffle_nc`,
anti-consecutive). `run_probing.py` routes FVQ/CVQ through
`probing_classifier.generate_trial` (plain `rng.shuffle`) and IVQ through
`gen_ivq_trial` (`shuffle_nc`), which confounds any FVQ-vs-IVQ comparison with
the shuffle. Query wording still separates the semantically distinct conditions:
"last" needs no counting, an ordinal does.

## Value pool selection

`select_pool()` applies three filters, then takes a seeded sample:

1. single token **with** a leading space — how values appear after `"category: "`;
2. single token **without** a leading space — how the model's answer starts at a
   fresh generation position. This is what makes `scoring=single_token` exact:
   the argmax of one forward pass is the entire answer, so no separate
   `generate` call is needed and GPU cost halves;
3. **no selected value is a substring of another.** `evaluate_ivq.is_correct`
   accepts `exp in pred`, so in a closed pool of short words a pair like
   `("ace", "access")` would score a wrong answer as correct.

The pool is tokenizer-dependent, so Qwen and gemma get different pools. Compare
within model, not across.

## Labels

Label is the trial's ground-truth answer value, `cv[k-1]`, where `cv` is the
test category's values in stream order. Taken from the trial definition, never
from the model's output, so it is defined at 0% and 100% accuracy alike.

Four label sets are fitted on the *same* activations (so the controls cost zero
GPU):

| label set | purpose |
|---|---|
| `expected` = `cv[k-1]` | the result |
| `shuffled` | control task: permute labels across trials. Must fall to chance, else leakage or overfit |
| `cv_first` = `cv[0]` | recency/primacy control |
| `cv_last` = `cv[-1]` | recency control — **the important one** |

The recency control is what separates "depth-indexed retrieval" from "the model
just tracks the most recent value". If `cv_last` decodes better than `cv[k-1]`
at interior k, the story is recency, not retrieval.

## Subsets

| subset | meaning |
|---|---|
| `wrong` | trials the model got wrong — **headline**. Decodable here = tracked but not emitted |
| `correct` | confounded with the output head (expected == emitted); reported as the confound check |
| `all` | every trial |

## Sample size

Binding constraint is per-class support in the wrong subset, not statistical
power (chance is 2%, so even 300 trials separates 10% from chance).

```
50 classes x 20 examples per class     = 1000  needed in the training fold
5-fold CV, training fold = 4/5         = 0.8
target wrong-subset size               = 1250
T per condition                        = 1250 / wrong_frac
```

`wrong_frac` is **measured by the pilot**, not assumed. The v3 numbers came from
a different setup (2300-value pool, `scoring=generate`) and do not transfer.

T varies by condition precisely so the wrong subset is constant at 1250 across
conditions. A flat T would give FVQ ~238 wrong trials and k=9 ~1200, making any
cross-condition accuracy gap partly a sample-size artifact.

## Probe

```
Pipeline([
    ("scale", StandardScaler()),
    ("lr", LogisticRegression(C=<global>, max_iter=2000, solver="lbfgs")),
])
```

- **Full 2048 dims, no PCA.** PCA was considered purely to cut CPU cost, then
  dropped: measured fit time at this shape (n=1000, d=2048, K=50, correlated
  features with rogue high-norm dims, separable labels) is 0.41 s worst case, so
  the whole sweep is ~12 min serial. PCA also risks discarding a low-variance
  signal direction, which residual streams plausibly have.
- **No `class_weight="balanced"`.** Classes are balanced by construction: the
  answer is uniform over the pool because the stream is a permutation.
- **No `multi_class` argument.** Removed in scikit-learn 1.7+; `lbfgs` does
  multinomial softmax for multiclass automatically.
- `max_iter=2000`: measured `n_iter` is 88 at the strongest regularization, well
  clear of the cap. Assert `n_iter_ < max_iter` anyway.

### On C

`C` is scikit-learn's inverse regularization strength; the objective is
`C * sum(cross_entropy) + 0.5 * ||W||^2`. It matters here because with n=1000 and
d=2048 the data is separable, so a weakly regularized probe reaches 100% training
accuracy by memorizing.

**One global C, identical at every layer and condition.** Tuning C per (layer,
condition) would confound the layer curve — a layer could look better merely for
having drawn a friendlier C — and selecting on the reported folds inflates
accuracy.

Selection rule, fixed in advance:

> C is the value maximizing mean 5-fold CV top-1, averaged over all layers, on
> the CVQ condition of the base (5,10) run. Chosen once, then frozen for every
> layer, condition, cell, and model.

The full layer curve is then replotted at every
`C in {0.0003, 0.001, 0.003, 0.01, 0.1, 1}` as a robustness figure. Parallel
curves mean the result is C-independent.

Grid measured on synthetic data at this exact shape (n=1000, d=2048, K=50,
correlated features, rogue high-norm dims), 5-fold CV on the M5:

| C | cv_top1 | train | \|W\| | n_iter |
|---|---|---|---|---|
| 0.003 | 0.141 | 0.655 | 1.8 | 62 |
| 0.01 | 0.135 | 0.768 | 3.1 | 79 |
| 0.1 | 0.127 | 0.978 | 7.8 | 149 |
| 1 | 0.120 | 1.000 | 15.8 | 188 |
| 100 | 0.119 | 1.000 | 45.7 | 30 |
| `penalty=None` | 0.118 | 1.000 | 82.7 | 21 |

Two readings. Regularization is worth ~2.3 points of top-1 (~20% relative), so
dropping it discards real signal. And the optimum sits at the smallest value
tried, which is why the grid extends below 0.003.

`penalty=None` is not a neutral default: on separable data the MLE does not
exist (the likelihood rises without bound as ||W|| grows), so lbfgs stops
wherever its gradient tolerance is met — note `n_iter=21` at ||W||=82.7, an early
stop along a diverging ray. The fitted weights then depend on `max_iter`/`tol`
rather than on the data.

Caveat on that table: the synthetic labels came from a noisy linear model, so
class support ran min=2/max=45. Real labels are balanced by construction. The
shape of the C curve transfers; the absolute 0.14 does not.

## Metrics

- **top-1** accuracy (chance 1/50 = 2%)
- **rank of the true value** among the 50 scores, and **MRR** — uses the whole
  ordering, far less noisy than top-1 near chance
- 5-fold stratified CV throughout

## Scoring

`single_token`: argmax of the final position's logits from the same forward pass
that yields the activations. Exact because of pool filter 2. The pilot
additionally runs greedy `generate` to (a) verify the two scorings agree and (b)
audit instruction-following.

## Instruction-following audit (pilot only)

Every pool value is present in every stream, so an emitted token outside the pool
was nowhere in context — a genuine instruction violation, not a retrieval error.

| bucket | meaning |
|---|---|
| `exact` | matches expected |
| `wrong_slot_same_key` | a value of the queried key, wrong slot — retrieval error |
| `other_key_value` | a pool value belonging to a different key |
| `off_pool` | not in the pool at all — instruction violation / hallucination |
| `multi_word` | more than one whitespace-separated token — format violation |
| `empty` | nothing emitted |

## Engineering

- **Apple M5, 24 GB.** PyTorch MPS, bfloat16. MLX rejected: it does not expose
  all hidden states in one pass, and hand-rolled hooks are not worth it for a
  job this size.
- **Length-bucketed batching, zero padding.** Trials are grouped by exact token
  length, so batches need no padding. This sidesteps position_ids entirely: a raw
  `model(**batch)` derives positions from an arange, which is wrong for
  left-padded rows. With uniform buckets, `[:, -1, :]` is the true final position
  for every row.
- **One forward pass per trial** (`single_token` scoring), not forward +
  generate.
- **Activations saved as float16** `.npy`, `(n_trials, n_layers, d_model)`,
  144 KB/trial. Every re-fit — 4 C values x 4 label sets x 36 layers x 3 subsets
  — then runs on CPU with the GPU idle.
- **Fixed T, no adaptive stopper.** `collect_activations_adaptive` in
  `run_probing.py` stops on `saturated`/`balanced` with `max_trials=400`; that
  logic serves the correctness probe's class balance and would truncate this run.

## Scope

Qwen2.5-3B-Instruct **base**, cells `(5,10)` then `(10,5)`; then gemma-3-4b-it
**base** with its own tokenizer-specific pool.

LoRA is **deferred**. Its accuracy is 1.000 at CVQ and shallow k for every cell
the closed pool permits (`probe_Qwen2.5-3B-Instruct-lora-v3_5k_10u.json`:
FVQ/CVQ/d10/d25 all 1.000; d50 0.897, d75 0.814, d90 0.667), so the wrong subset
is empty or too thin, at any T. The cells where LoRA errs — e.g. `(25,10)`,
d50 0.669 — need `K*N = 250`, which the closed pool forbids. The closed-pool
design and a LoRA wrong-subset comparison are mutually exclusive.
