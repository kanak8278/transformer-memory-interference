# Prompt setups — the exact text, and which models saw which

Every prompt template used anywhere in the consolidated results, verbatim from
source, with the models and configurations each one applies to.

The `prompt_format` column in every CSV keys into this file.

---

## Framing

**Semantic-Multi is the main experiment.** Keys and values are meaningful
(`coffee variety: rural`), multi-token.

**Arbitrary-Single is the supporting experiment.** Values are single arbitrary
tokens from a fixed 2,300-word pool, which the mechanistic work requires — a
single-token answer makes logit lens, probing and head ablation well-defined.
It runs to show the phenomenon is not an artefact of semantic content.

That bridge is measured, not assumed: across 29 (model, K, N) cells where both
datasets have a gap, **r = 0.926 (r² = 0.858), 93% sign agreement, mean |Δ| =
0.114**, with Arbitrary reading **+0.08 hotter** than Semantic. Reproduce with
`build/check_arb_vs_sem.py`.

---

## 1. `chat_template` — open-weight instruct models, Arbitrary-Single

**Models:** Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct, gemma-3-1b-it,
gemma-3-4b-it
**Where:** theme 01 (their 55–79 cells), theme 04 baselines
**Source:** `v3/scripts/experiments/stage1_sweep.py:66,255-276`

System message:

```
You are a precise data extraction tool. Output ONLY a single word - the exact
value requested. No other text, no explanation, no punctuation.
```

User message:

```
Read the following key-value stream. Each key gets updated multiple times.

<category>: <value>
<category>: <value>
...

What was the {first|last} value of <test_category>?
```

Then wrapped with the model's own `apply_chat_template(..., add_generation_prompt=True)`.

- Stream is interleaved with **no two consecutive lines from the same key**
  (`shuffle_no_consecutive`).
- **No update indices anywhere** — the model must count occurrences itself.
- `first` → FVQ, `last` → CVQ. Both **semantic**, never ordinal.
- Greedy, `max_new_tokens=30`.

## 2. `completion_few_shot` — open-weight **base** models, Arbitrary-Single

**Models:** Qwen2.5-3B *(base)*, Qwen3.5-2B, Qwen3.5-4B, Qwen3.5-9B
**Where:** theme 01 (their 50–79 cells)
**Source:** `stage1_sweep.py:279` + `mechanistic_probing_v2/core/dataset_configs.py:746`

No system message and no chat wrapper. Instead, a **fixed two-block few-shot
preamble** the chat models never see:

```
color: red
animal: cat
color: blue
animal: dog
color: green
The first value of color was: red
The last value of color was: green
The first value of animal was: cat
The last value of animal was: dog

fruit: apple
metal: gold
fruit: grape
metal: iron
fruit: lemon
The first value of fruit was: apple
The last value of fruit was: lemon
The first value of metal was: gold
The last value of metal was: iron

<the real stream>
The {first|last} value of <test_category> was:
```

The demos are constant regardless of K and N — they teach format, not scale.

> **⚠ This is the theme-1 confound.** Four models saw §1, four saw §2. The
> preamble, the wrapper and the query phrasing all differ. Given theme 3 shows
> formatting moves the gap by up to 0.94, cross-format model comparisons are not
> clean. It also aligns exactly with the Qwen3.5 line, so "Qwen3.5 differs" and
> "completion format differs" cannot be separated in this data. See
> `01_fvq_cvq/CONFIGURATIONS.md`.

## 3. Proprietary, Semantic-Multi — the main-result prompt

**Models:** all 6 proprietary
**Where:** theme 01 (69 cells each)
**Source:** `experiments_cloud/sweep_semantic.py:162-170`

Single user message, no system message:

```
Read the following key-value stream. Each key gets updated multiple times.

<category>: <value>
...

What was the {first|last} value of <test_category>?
Answer with ONLY the exact value. No explanation.
```

Same body as §1, with the terseness instruction moved inline (no system role)
and no chat-template wrapping — the APIs handle that. **Semantic** queries only.

## 4. LoRA experiments — Arbitrary-Single

**Models:** Qwen2.5-3B-Instruct, gemma-3-4b-it (base and adapted)
**Where:** theme 04
**Source:** `lora_intervention/evaluate.py:147-158`

Byte-identical to §1 — same system prompt, same user text, same chat template.
That is deliberate: the LoRA evaluation compares against the stage-1 sweep as
its baseline (`evaluate.py:54` `DEFAULT_BASELINE`), so the prompt must match.

---

## 5. The four format variants — Semantic-Multi

**Models:** all 6 proprietary (raw); 4 open-weight (derived, K=10/N=50 only)
**Where:** themes 02 and 03
**Source:** `experiments_cloud/ucurve_prompts.py`

All four share the closing line `Answer with ONLY the exact value. No
explanation.` and differ only in how the stream renders and whether update
indices are visible.

### `flat_nolabel` — indices hidden (the default condition)

```
Read the following key-value stream. Each key appears multiple times as it
gets updated. Count each occurrence of a key as one update.

<category>: <value>
...

What was the {ordinal(k)} value of <test_category>?
```

The model must count occurrences itself. **This is where the counting failure
lives.**

### `flat_verbose` — indices inline

```
Read the following key-value stream. Each key is updated multiple times.

<category> (update 3): <value>
...
```

### `block` — grouped by round, indices explicit

```
Read the following key-value stream. Each key is updated multiple times,
grouped by update round.

[Update 1]
  <category>: <value>
  <category>: <value>
[Update 2]
  ...
```

### `landmark` — grouped by round, indices **removed**

```
Read the following key-value stream. Each key is updated multiple times,
grouped by update round. The '---' marker separates consecutive update rounds.

<category>: <value>
<category>: <value>
---
<category>: <value>
...
```

Landmark is block's layout with the round numbers stripped and `---` separators
substituted. That contrast is the cleanest test of what the block format is
actually doing — and landmark fails where block succeeds, which says the win
comes from the **explicit indices**, not the grouping.

### Ordinal vs semantic query

Each format has two query variants against the same target value:

| variant | query line | `query_type` |
|---|---|---|
| numbered position | `What was the {1st,2nd,...} value of X?` | `ordinal` |
| `*_lastquery` | `What was the last value of X?` | `semantic` |

These are not interchangeable. GPT-4.1 under `flat_nolabel` at K=10/N=50 scores
**0.005 ordinal vs 0.69 semantic** on the identical target value. Full analysis
in `02_intermediate_ivq/README.md`.

A fifth template, `prompt_flat_short` (`[U3]:` notation), exists in the source
but does not appear in any committed result.

## 6. Block-vs-LoRA experiments — Arbitrary-Single

**Models:** Qwen2.5-3B-Instruct, gemma-3-4b-it
**Where:** themes 02, 03 (`prompt_format` = `plain` / `block`)
**Source:** `lora_intervention/experiments/behavioral_block_lora_sweep.py:76`,
`block_readout_comparison.py:59-82`, `evaluate_ivq.py:121`

- **`plain`** — §1's template. Endpoints use `first`/`last`; interior positions
  use ordinals (`evaluate_ivq.py:121`: `"last" if k == N else ordinal(k)`).
- **`block`** — `[Update j]` grouping, queried as
  `What was the value of X in Update {k}?` — **always ordinal**, including at
  the endpoints.

> **`plain` ≠ `flat_nolabel`.** Both hide indices, but they are different
> templates from different experiments. Never merge them. Note the dataset no
> longer separates them: `flat_nolabel` was Semantic-Multi only until theme 07
> ran it on Arbitrary-Single too (§7), so filter on `prompt_format`, not on
> `dataset`.

## 7. CoT vs non-CoT IVQ — both datasets

**Models:** claude-4.5-{haiku,sonnet,opus}
**Where:** theme 07
**Source:** `experiments_cloud/cot_ivq_prompts.py`

A variant of §5's `flat_nolabel`, with two changes and one non-change.

**Change 1 — the answer is a tagged span, not a bare line.** Both arms carry
this closing instruction identically:

```
Read the following key-value stream. Each key appears multiple times as it
gets updated. Count each occurrence of a key as one update.

<category>: <value>
...

What was the {ordinal(k)} value of <test_category>?
Respond with only:
<answer>the exact value</answer>
```

The `last`-query variant swaps the question line for `What was the last value
of <test_category>?`, exactly as in §5.

**Change 2 — scoring is exact-span equality, not containment.** §5's scorer
accepts `expected in predicted`, which credits any response that merely
enumerates the stream. Harmless for one-word answers, fatal once reasoning text
is in the response, so this sweep extracts the span inside `<answer>` tags and
compares by equality after normalisation. See "Scoring" below for the contrast.

**Non-change — the prompt is byte-identical across the two arms.** `nocot` and
`cot_thinking` differ only in API parameters: thinking off vs on with
`thinking_budget=3000`, `max_tokens` 64 vs 5048, and `nocot` additionally
prefills `<answer>` with `</answer>` as a stop sequence. Nothing in the stimulus
differs, which is what keeps the arm contrast clean.

**Seeding differs from §5, deliberately.** `cot_ivq_prompts.make_seed` uses
`blake2b`, not `ucurve_prompts.make_seed`, whose tuple contains a string and is
therefore `PYTHONHASHSEED`-salted and not reproducible across processes. So both
arms see identical streams and the comparison is paired trial-by-trial — but
theme 07's streams are **not** the same streams as theme 02's at the same (K, N).

**Prompt caching.** At large N the stream dominates the prompt and is re-sent
once per queried position, so the preamble+stream is sent as a separate content
block with an ephemeral cache breakpoint. Concatenating the two blocks
reproduces the single-string prompt exactly. One caveat recorded in the source:
a block boundary can shift tokenisation by a token or two at the seam, so do not
mix cached and uncached cells inside one comparison.

---

## Quick map

| `prompt_format` | dataset | models | § |
|---|---|---|---|
| `chat_template` | arbitrary_single | 4 open-weight instruct + LoRA models | 1, 4 |
| `completion_few_shot` | arbitrary_single | 4 open-weight base | 2 |
| *(unlabelled)* | semantic_multi | 6 proprietary, theme 01 | 3 |
| `flat_nolabel` | semantic_multi | 6 proprietary + 4 open-weight | 5 |
| `flat_verbose` | semantic_multi | 6 proprietary + 4 open-weight | 5 |
| `block` | semantic_multi | 6 proprietary + 4 open-weight | 5 |
| `landmark` | semantic_multi | 6 proprietary + 4 open-weight | 5 |
| `plain` | arbitrary_single | Qwen2.5-3B-Instruct, gemma-3-4b-it | 6 |
| `block` | arbitrary_single | Qwen2.5-3B-Instruct, gemma-3-4b-it | 6 |
| `completion` / `chat_1024` | arbitrary_single | SmolLM3-3B (derived) | — |
| `flat_nolabel` | semantic_multi | 3 claude-4.5, `nocot` + `cot_thinking`, theme 07 | 7 |
| `flat_nolabel` | arbitrary_single | 3 claude-4.5, `cot_thinking` only, theme 07 | 7 |

## Scoring (§§1–6; theme 07 differs — see below)

`mechanistic_probing_v2/core/evaluation.py:48-76`

```python
pred_lower = predicted.lower().strip()
if exp_lower in pred_lower or pred_lower.startswith(exp_lower):
    return "correct"
```

Case-folded substring-or-prefix match over the whole generation — lenient, so a
model answering in a sentence is not penalised. Outcome taxonomy and its
failure modes: `01_fvq_cvq/VALIDITY.md`.

### Theme 07's scorer — stricter, and five-way

`experiments_cloud/cot_ivq_score.py`

```python
# exact span, not containment
matches = re.findall(r"<answer\s*>(.*?)</\s*answer\s*>", text, re.DOTALL | re.I)
answer = normalize(matches[-1])          # last match: a model that restates commits with the final one
correct = answer == normalize(expected)
```

Two departures, both required once reasoning text is in the response:

1. **No containment.** A CoT response that enumerates the stream contains the
   target value whatever it concluded, so the lenient matcher above would credit
   it. Equality over the tagged span removes that.
2. **Failure modes are named, not silent.** Five outcomes instead of a
   correct/incorrect split:

| outcome | meaning | in `n_offstream`? |
|---|---|---|
| `correct` | exact match after normalisation | — |
| `in_sequence` | a value of the **right** key at the **wrong** position — a positional-addressing error | no |
| `out_of_context` | a real stream value belonging to **another** key — a retrieval error | no |
| `garbage` | no stream value at all | **yes** |
| `no_answer` | no usable `<answer>` span (truncation, or never committed) — excluded from `n_trials` entirely | n/a |

Splitting `in_sequence` from `out_of_context` is the point: the three-way
classifier collapses them and discards the more informative signal.
`no_answer` is kept out of the error bucket because only the thinking arm can
exhaust its token allowance, so charging truncation to it would bias the arm
under test.
