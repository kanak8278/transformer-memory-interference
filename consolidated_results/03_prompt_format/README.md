# 03 — Prompt formatting

**Question.** The same key-value updates can be rendered several ways. How much
of the retrieval failure is a property of the model, and how much is a property
of how you wrote the prompt down?

| | |
|---|---|
| file | `format.csv` — 3,222 rows |
| per-model slices | `by_model/` — 11 files (generated; do not edit) |
| models | 6 proprietary + Qwen2.5-3B-Instruct, Qwen3.5-2B, Qwen3.5-4B, Gemma-3-4b-it, SmolLM3-3B |
| datasets | `semantic_multi` (2,192 rows — the format sweep proper), `arbitrary_single` (1,030 rows — the LoRA/block experiments + SmolLM3) |
| tiers | 3,180 `raw`, 42 `derived` |

## The formats

| `prompt_format` | rendering | update indices |
|---|---|---|
| `flat_nolabel` | interleaved `key: value` stream, nothing else | **implicit** — the model must count occurrences |
| `flat_verbose` | same stream, each line labelled with its update number | explicit |
| `block` | grouped `[Update j]` blocks, all keys under each | explicit |
| `landmark` | positional landmarks interleaved into the stream | partial |
| `plain` | the LoRA experiments' unformatted prompt | implicit |
| `completion` / `chat_1024` | completion-style vs chat-template wrapping | n/a |

**`plain` and `flat_nolabel` are not the same template.** Both are unformatted,
but they come from different experiments on different models. Do not merge them.

---

## Headline: formatting moves the gap more than scale does

Mean FVQ / CVQ at K=10, N=50 on **Semantic-Multi**, averaged over models
(`condition=CVQ`, `query_type=semantic`):

| format | open-weight FVQ | open-weight CVQ | **gap** | proprietary FVQ | proprietary CVQ | **gap** |
|---|---|---|---|---|---|---|
| `flat_nolabel` | 0.84 | 0.06 | **+0.78** | 1.00 | 0.74 | **+0.26** |
| `landmark` | 0.85 | 0.20 | **+0.65** | 0.86 | 0.97 | **−0.11** |
| `flat_verbose` | 0.63 | 0.69 | **−0.06** | 1.00 | 0.94 | **+0.06** |
| `block` | 0.94 | 0.94 | **−0.00** | 1.00 | 0.99 | **+0.00** |

Per model, at the same cell (FVQ / CVQ, with the gap in brackets):

| model | flat_nolabel | flat_verbose | block | landmark |
|---|---|---|---|---|
| Qwen2.5-3B-Instruct | 0.49/0.23 (+0.26) | 0.34/0.39 (−0.05) | 0.78/0.92 (−0.14) | 0.71/0.27 (+0.44) |
| Qwen3.5-2B | 0.95/0.01 (**+0.94**) | 0.85/0.93 (−0.08) | 1.00/0.98 (+0.02) | 0.80/0.07 (+0.73) |
| Qwen3.5-4B | 0.99/0.00 (**+0.99**) | 0.93/0.97 (−0.04) | 1.00/1.00 (+0.00) | 0.99/0.47 (+0.52) |
| Gemma-3-4b-it | 0.92/0.01 (**+0.91**) | 0.39/0.47 (−0.08) | 0.98/0.89 (+0.09) | 0.92/0.00 (+0.92) |
| gpt-4.1 | 1.00/0.69 (+0.31) | 1.00/0.96 (+0.04) | 1.00/0.98 (+0.02) | 0.99/0.99 (+0.00) |
| gpt-4.1-mini | 1.00/0.77 (+0.23) | 1.00/0.94 (+0.06) | 0.99/0.99 (+0.00) | 0.78/1.00 (−0.22) |
| claude-4.5-haiku | 1.00/0.43 (+0.56) | 1.00/0.95 (+0.05) | 1.00/0.99 (+0.01) | 0.91/0.90 (+0.01) |
| claude-4.5-sonnet | 0.99/0.80 (+0.20) | 1.00/0.81 (+0.19) | 1.00/1.00 (+0.00) | 0.96/0.99 (−0.04) |
| gemini-2.5-flash | 1.00/0.76 (+0.24) | 1.00/0.98 (+0.02) | 1.00/1.00 (+0.00) | 1.00/0.95 (+0.05) |
| gemini-2.5-pro | 1.00/1.00 (+0.00) | 1.00/1.00 (+0.00) | 1.00/1.00 (+0.00) | 0.54/0.99 (−0.45) |

Three things fall out:

1. **`block` erases the gap for every model tested**, open-weight and
   proprietary alike — 10/10 models within ±0.14, most within ±0.02.
2. **Qwen3.5-4B under `flat_nolabel` scores 0.99 FVQ and 0.00 CVQ** (+0.99).
   A model that can retrieve perfectly at one end and never at the other is not
   capacity-limited; it is format-limited.
3. **`landmark` is not a fix and can invert the gap.** Gemini-2.5-Pro drops to
   0.54 FVQ under landmark (from 1.00 under everything else) — landmarks
   actively interfere with first-value retrieval for some models.

## `flat_verbose` and `block` also fix the interior

The gap-closing above is only the endpoints. Interior accuracy (see
`../02_intermediate_ivq/`) tells the more important story: block formatting
lifts mean IVQ from 0.02–0.03 to **0.79–0.81**, while LoRA — which closes the
endpoint gap just as completely — only reaches 0.20–0.29. Formatting is the
only intervention here that produces general position indexing.

Explicit update indices are also what makes *ordinal* queries answerable at all:
the ordinal/semantic accuracy gap appears under `flat_nolabel` and `landmark`
but vanishes under `block` and `flat_verbose`. Full table in
`../02_intermediate_ivq/README.md`.

## Block vs LoRA, matched conditions (`n=100` per cell)

| model | cell | base plain | base **block** | **+LoRA** plain |
|---|---|---|---|---|
| Qwen2.5-3B-Instruct | K2/N30 | FVQ 0.89 / CVQ 0.41 | 0.90 / 0.96 | 1.00 / 1.00 |
| Qwen2.5-3B-Instruct | K10/N50 | 0.32 / 0.50 | 0.76 / 0.87 | 1.00 / 1.00 |
| Gemma-3-4b-it | K2/N30 | 1.00 / 0.67 | 1.00 / 0.92 | 1.00 / 1.00 |
| Gemma-3-4b-it | K10/N50 | 0.93 / 0.30 | 0.99 / 0.92 | 0.99 / 0.98 |

Two different interventions, the same endpoint outcome, across two model
families. The per-layer logit-lens evidence that they converge on the *same
readout* is in the source files (`.../block_readout_results/`) — only the
behavioural accuracies were extracted here.

## SmolLM3-3B: format × post-training stage (derived, gap only)

Mean FVQ−CVQ gap over 32 (K,N) cells at each post-training checkpoint:

| stage | completion | chat_1024 |
|---|---|---|
| it-mid-training | +0.25 | **−0.00** |
| it-SFT | +0.19 | +0.29 |
| it-soup-APO | +0.19 | +0.20 |
| it-LC-expert | +0.27 | +0.33 |
| final | +0.16 | +0.18 |

The gap is present before SFT under completion format and **absent under chat
format at the same checkpoint** — then chat formatting *acquires* the gap
through SFT. Post-training does not create the gap so much as it makes the chat
path inherit it. Treat this as a lead, not a result: 10 numbers, no CIs, no
per-cell data (raw `training_dynamics_smollm3/` is not local).

## Derived rows (42)

- `paper/figures/tab_format_ci.tex` → the 4 open-weight models × 4 formats at
  K=10, N=50 (32 rows). The raw open-weight sweep
  (`experiments_cloud/results/ucurve_vllm/`) is not local.
- `paper/figures/tab_smollm3_formats.tex` → the 10 SmolLM3 rows above.

**These rows are trustworthy for a specific, checkable reason.** The same
`.tex` table also contains the proprietary models, whose raw data *is* local.
The builder back-extracts those and compares: **48 values checked, 0
mismatched.** The parser reproduces raw exactly where raw exists, which is what
licenses trusting it where raw is gone.

## Caveats

- Open-weight coverage is only 4 models × 4 formats, at a single (K, N) cell.
  Proprietary coverage is 6 models × 4 formats × 6 cells × 16 positions.
- Derived rows are rounded to 2dp and carry a Wilson half-width but no
  per-condition trial count.
- The U-curve and behavioural rows here also appear in
  `../02_intermediate_ivq/ivq.csv`. Do not concatenate without de-duplicating.

## Sources

See `../PROVENANCE.md`, or the docstring of `../build/build_03_formats.py`.
