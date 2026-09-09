# 07 — Does chain-of-thought rescue the interior?

**Question.** Theme 2 showed base models collapse in the interior of an update
stream once the format hides indices. Is that a *representational* limit, or
just a limit on what one forward pass can compute? Let the model reason first
and see.

| | |
|---|---|
| files | `cot_ivq_nocot.csv` — 270 rows · `cot_ivq_cot_thinking.csv` — 576 rows. All `raw`. |
| per-model slices | `by_model_nocot/`, `by_model_cot_thinking/` — 3 files each (generated; do not edit) |
| models | **3 proprietary** — claude-4.5-{haiku,sonnet,opus} |
| datasets | `semantic_multi` **and** `arbitrary_single` |
| variant | `nocot` \| `cot_thinking` |
| prompt_format | `flat_nolabel` only (indices hidden — the hard condition) |
| trials | 79,680 persisted trial records → 79,552 well-formed, 120 malformed, 8 superseded by a resume |

## One file per arm — and how to pair them back up

The two CSVs **partition** the 846 rows; no row is in both, so concatenating
them needs no de-duplication.

| file | rows | contents |
|---|---|---|
| `cot_ivq_nocot.csv` | 270 | `semantic_multi` only — thinking off |
| `cot_ivq_cot_thinking.csv` | 576 | 270 `semantic_multi` + 306 `arbitrary_single` — thinking on |

The cost of the split: the theme's headline result is a *paired* contrast — same
cell, same stream, arm on vs off — and it now lives in two files. Both are
written with an arm-independent sort order, so the 270 shared rows come out in
the same order in each. Filter the CoT file to `semantic_multi` and they zip
directly:

```python
nocot = pd.read_csv("cot_ivq_nocot.csv")
cot   = pd.read_csv("cot_ivq_cot_thinking.csv").query("dataset == 'semantic_multi'")
# same order, row for row -- verified on every build
delta = cot.accuracy.values - nocot.accuracy.values
```

Join key if you would rather be explicit:
`(model, num_keys, num_updates, position, query_type)` — unique within an arm.
Each build prints the check: **270 keys in both arms, 0 nocot-only, 306
cot_thinking-only, shared rows in identical order: True**. The 306 cot-only rows
are the Arbitrary-Single ladder, which has no no-CoT arm at all.

## The two datasets are not symmetric — read this first

| dataset | arms | grid | rows | in which file |
|---|---|---|---|---|
| `semantic_multi` | **both** `nocot` + `cot_thinking` | K ∈ {5,10} × N ∈ {10,20,50} | 540 | both, 270 each |
| `arbitrary_single` | **`cot_thinking` only** | K = 5, N ∈ {50,100,200,300,400,460} | 306 | `cot_ivq_cot_thinking.csv` only |

So `semantic_multi` is the actual CoT-vs-no-CoT contrast. `arbitrary_single` is
a depth ladder that answers *how deep does CoT hold*, not *does CoT help* — it
has no paired no-CoT arm. Do not report an unqualified "CoT vs no-CoT" number
computed over both datasets.

## Design: the stimulus is byte-identical across arms

The prompt text is the same in both arms, down to the byte (see
`experiments_cloud/cot_ivq_prompts.py`). The only manipulated variable is
whether extended thinking is enabled on the API call:

| | `nocot` | `cot_thinking` |
|---|---|---|
| thinking | off | on, `thinking_budget=3000` |
| `max_tokens` | 64 | 5048 |
| prefill | `<answer>` | — |
| stop sequences | `</answer>` | — |

Seeds come from `blake2b`, not Python's salted `hash`, so both arms see
**identical streams** and the comparison is paired trial-by-trial.

## Headline 1 — on Semantic-Multi, CoT closes the interior gap almost completely

Mean accuracy, ordinal queries, `flat_nolabel`:

| model | arm | FVQ (pos 1) | **IVQ (interior)** | CVQ (pos N, ordinal) |
|---|---|---|---|---|
| claude-4.5-haiku | `nocot` | 1.000 | **0.190** | 0.010 |
| claude-4.5-haiku | `cot_thinking` | 1.000 | **0.969** | 0.956 |
| claude-4.5-sonnet | `nocot` | 0.997 | **0.174** | 0.061 |
| claude-4.5-sonnet | `cot_thinking` | 1.000 | **0.996** | 0.990 |
| claude-4.5-opus | `nocot` | 1.000 | **0.303** | 0.554 |
| claude-4.5-opus | `cot_thinking` | 1.000 | **0.998** | 1.000 |

An interior at 0.17–0.30 goes to 0.97–1.00. Same weights, same prompt, same
stream. **The information was retrievable all along** — what the single forward
pass lacked was the serial computation to count occurrences, not access to the
value. That is a strong constraint on how the theme-2 collapse should be
described: it is a compute limit under a hidden-index format, not evidence that
interior values are unrepresented.

## Headline 2 — but it degrades with depth, and the U-curve comes back

`cot_thinking` on Arbitrary-Single, K=5, mean IVQ accuracy by stream depth:

| model | N=50 | N=100 | N=200 | N=300 | N=400 | N=460 |
|---|---|---|---|---|---|---|
| claude-4.5-haiku | 0.988 | 0.888 | 0.787 | 0.682 | 0.549 | 0.522 |
| claude-4.5-sonnet | 0.996 | 0.990 | 0.961 | 0.841 | 0.760 | 0.723 |
| claude-4.5-opus | 0.998 | 0.994 | 0.980 | 0.910 | 0.802 | 0.756 |

Monotone decay in depth, ordered by model capability throughout. Per position,
opus at N=460:

```
pos    1-8    72    137   201   266   330   395   459   460   last
acc   1.00  0.972  0.784 0.685 0.470 0.300 0.185 0.188 0.770  1.00
```

Two things to notice. The **primacy plateau is perfect** — positions 1–8 stay at
1.00 even 460 updates deep. And the **endpoint effect survives**: position 460
(0.770) sits four times above its neighbour 459 (0.188), and the recency-phrased
`last` query on the same item is 1.00. So CoT flattens the middle of the curve
at moderate depth but does not abolish the shape; push depth far enough and the
U reappears with a wide, sagging floor.

## The ordinal / semantic split — read before using `condition`

Same convention as theme 2. `condition` = *which* value (FVQ/CVQ/IVQ);
`query_type` = *how* it was asked. The final update is queried both ways and the
two diverge sharply in the `nocot` arm:

| model | CVQ ordinal ("the Nth value") | CVQ semantic ("the last value") |
|---|---|---|
| claude-4.5-haiku, `nocot` | **0.010** | **0.734** |
| claude-4.5-sonnet, `nocot` | 0.061 | 0.957 |
| claude-4.5-opus, `nocot` | 0.554 | 0.982 |
| all three, `cot_thinking` | 0.956–1.000 | 0.988–1.000 |

The gap is the counting failure, isolated: identical target item, and the only
difference is whether the model has to arrive at it by index. CoT closes that
gap too. **Never pool the two query types:**

```python
df[(df.condition == "CVQ") & (df.query_type == "ordinal")]
```

## Denominators and convergence — the two traps

**1. `n_trials` counts well-formed responses only.** A response counts if the
answer arrived inside `<answer>...</answer>` tags. Responses with no usable tag
(truncation, or no committed span) are excluded from the denominator and
recorded as `n_malformed=` in `notes` — 120 across the whole corpus. This
exclusion is deliberate upstream: only the thinking arm can exhaust its token
allowance, so charging truncation to the error bucket would bias the comparison
against the arm under test.

**2. Each position retires on its own Wilson half-width.** Within one cell
`n_trials` ranges from 51 to 200. **Do not average accuracy across positions
inside a cell** — that weights an early-retiring easy position the same as one
that ran to `max_trials`. Read `n_trials` per row, or weight by it.

| dataset | arm | rows | well-formed trials | unconverged rows |
|---|---|---|---|---|
| `semantic_multi` | `nocot` | 270 | 31,840 | 74 |
| `semantic_multi` | `cot_thinking` | 270 | 17,343 | 5 |
| `arbitrary_single` | `cot_thinking` | 306 | 30,369 | 63 |

All six `nocot` cells ran to `max_trials=200` without full convergence — that arm
sits at mid accuracies where the half-width closes slowly, so it cost the most
trials for the least certainty. The `cot_thinking` arm converged fast on
Semantic-Multi (saturated cells retire at 56) and slowly at the deep end of the
ladder, which is where its remaining 63 unconverged rows are.

## Output validity

Off-stream — the model named no value in the stream at all — is negligible:
92 `garbage` in 79,552 trials (0.12%), worst cell-arm 0.23%. Nowhere near the
20% exclusion threshold in `../PROVENANCE.md`.

`error_types_json` carries the full four-way taxonomy per row:

| outcome | count | on-stream? |
|---|---|---|
| `correct` | 49,901 | — |
| `in_sequence` | 28,854 | yes — right key, wrong position (a positional-addressing error) |
| `out_of_context` | 705 | yes — a real stream value, but belonging to another key |
| `garbage` | 92 | **no** — counted in `n_offstream` |

`in_sequence` dominating the error mass is itself the finding in the `nocot`
arm: the model reaches the right key's value list and then miscounts within it.

## Caveats

- **Snapshots.** The dated API ids are in `notes` (`model_id=`). Theme 02's
  claude rows come from earlier runs whose CSVs did not preserve a snapshot
  date, so the same canonical `claude-4.5-*` id spans two unrecorded-vs-recorded
  snapshot regimes. Do not treat a theme-02 vs theme-07 difference as purely an
  arm effect.
- **`claude-4.5-opus` appears only here.** No other theme has an opus arm.
- Interior means are over unevenly sampled positions (16–17 probed of N).
- One position was resumed mid-run (sonnet, Arbitrary-Single, K=5 N=400,
  position 6), so `trials.jsonl` there holds 64 records against the
  checkpoint's 56. The builder counts the checkpoint's subset; see
  `../build/build_07_cot_ivq.py:taxonomy`.
- `_figs/ladder_summary.csv` in the source tree is **not** a usable source: no
  `model` column, and a stale haiku-only snapshot. Use this CSV or the
  checkpoints.

## Sources

Both CSVs come from the same 9 runs; the arm is a column in the source, not a
separate experiment.

`experiments_cloud/results/cot_ivq/` and
`experiments_cloud/results/cot_ivq_arbitrary_single/` — per-run
`checkpoint.json` (authoritative for every number) plus `trials.jsonl` (read
only for the output taxonomy and the cross-check). See `../PROVENANCE.md` or
`../build/build_07_cot_ivq.py`.
