# Validity audit — are FVQ "failures" real retrieval failures?

**The worry.** If the model emits a single token that isn't a stream value — the
key name, an article, a stray word that fits the sentence — a scorer with no
output parsing would mark it wrong and we would read it as a retrieval failure.
How much of the reported failure rate is that?

**Short answer.** The concern is real and large — **21.6% of all FVQ trials are
off-stream** — but it was already instrumented, and it does **not** manufacture
the FVQ>CVQ asymmetry. It does invalidate the absolute accuracies for the small
models.

Reproduce everything below with `python3 ../build/check_output_validity.py`.

> **Outcome:** the seven models at **≥20% off-stream** (rows 1–7 of the table in
> §2) were **dropped from all consolidated results** on 2026-07-22. This document
> is the evidence for that decision and audits the corpus *as originally
> measured* — the tables below therefore still include the excluded models.
> The live exclusion list is `EXCLUDED_MODELS` in `../build/common.py`.

---

## First, two premises to correct

**"We only asked it to generate one token."** No — `max_new_tokens: 30`
(recorded in every sweep's `config.max_new_tokens`). The model writes up to 30
tokens and the scorer searches that whole span.

**"We haven't used any output parsing."** There is parsing, and it is *lenient*,
not exact-match (`mechanistic_probing_v2/core/evaluation.py:48-76`):

```python
pred_lower = predicted.lower().strip()
if exp_lower in pred_lower or pred_lower.startswith(exp_lower):
    return "correct"
```

So "the first value was **age**" scores correct. A model that writes a sentence
instead of a bare word is **not** penalised. There is also a system prompt
demanding a bare value (`stage1_sweep.py:66`).

Every generation is then classified into an outcome taxonomy:

| outcome | meaning | is it a retrieval error? |
|---|---|---|
| `correct` | the expected value appears in the output | — |
| `intermediate_intrusion` | output is some *other* value of the queried key | **yes** |
| `recency_intrusion` (FVQ) / `primacy_intrusion` (CVQ) | output is the value at the opposite end | **yes** |
| `garbage` | output is **not any value in the stream** | **no** — this is your worry |
| `empty` | model generated nothing | **no** |

`garbage` is exactly the "it said the key name / a filler word" case, and it has
been counted all along. It was in the raw JSONs; it was simply not surfaced.
`n_offstream`, `accuracy_onstream` and `error_types_json` are now columns in
`fvq_cvq.csv` (populated for the 1,873 stage-1-sourced rows).

---

## 1. What the models actually emit — 93,700 trials per condition

| outcome | FVQ | CVQ |
|---|---|---|
| `correct` | **56.7%** | 31.5% |
| `intermediate_intrusion` | 19.0% | 41.5% |
| `recency` / `primacy_intrusion` | 2.6% | 5.0% |
| **`garbage`** | **21.6%** | **21.7%** |
| `empty` | 0.1% | 0.2% |

**Roughly one FVQ failure in two is not a retrieval failure at all.** Of the
43.3% of FVQ trials that are wrong, 21.7pp are off-stream and 21.6pp are genuine
intrusions.

But note the second column: **off-stream rate is 21.6% vs 21.7%** — essentially
identical across conditions. It is a common-mode offset, not a confound that
could produce the asymmetry.

## 2. Where it's concentrated — and what happens when you remove it

`acc*` = correct / on-stream trials — retrieval accuracy conditional on the
model having produced *some* candidate value.

| model | off% FVQ | off% CVQ | acc F | acc C | gap | **acc\*F** | **acc\*C** | **gap\*** |
|---|---|---|---|---|---|---|---|---|
| gemma-3-270m-it | **73.2%** | 72.0% | 0.18 | 0.03 | +0.15 | **0.69** | 0.11 | +0.58 |
| pythia-410m | **65.5%** | 61.6% | 0.11 | 0.15 | −0.04 | **0.33** | 0.40 | −0.07 |
| Qwen2.5-0.5B-Instruct | **52.1%** | 51.4% | 0.40 | 0.09 | +0.31 | **0.84** | 0.19 | +0.65 |
| mamba-1.4b-hf | 29.7% | 22.0% | 0.70 | 0.06 | +0.64 | **1.00** | 0.08 | +0.92 |
| stablelm-2-1_6b-chat | 20.0% | 29.3% | 0.71 | 0.05 | +0.66 | 0.88 | 0.07 | +0.81 |
| TinyLlama-1.1B | 19.3% | 23.3% | 0.52 | 0.24 | +0.28 | 0.64 | 0.31 | +0.33 |
| Qwen3.5-2B | 17.5% | 15.0% | 0.46 | 0.39 | +0.07 | 0.56 | 0.46 | +0.10 |
| Qwen3.5-0.8B | 16.6% | 23.9% | 0.59 | 0.26 | +0.33 | 0.71 | 0.34 | +0.37 |
| gemma-3-1b-it | 16.0% | 13.2% | 0.57 | 0.03 | +0.54 | 0.68 | 0.03 | +0.65 |
| Qwen2.5-3B | 15.4% | 9.6% | 0.44 | 0.56 | −0.12 | 0.52 | 0.61 | −0.10 |
| Qwen3.5-4B | 9.5% | 1.7% | 0.83 | 0.64 | +0.19 | 0.92 | 0.65 | +0.27 |
| Qwen2.5-1.5B-Instruct | 8.8% | 6.9% | 0.63 | 0.16 | +0.46 | 0.69 | 0.17 | +0.51 |
| Qwen3.5-9B | 5.1% | 9.0% | 0.85 | 0.64 | +0.21 | 0.90 | 0.70 | +0.20 |
| Qwen2.5-3B-Instruct | 4.5% | 5.0% | 0.53 | 0.50 | +0.02 | 0.55 | 0.53 | +0.02 |
| gemma-3-4b-it | **1.3%** | 5.5% | 0.95 | 0.49 | +0.46 | 0.96 | 0.52 | +0.44 |

Three conclusions:

1. **The gap survives.** Sign is preserved for **15/15 models**, and the
   magnitude *grows* for 12 of them. Off-stream output was suppressing the
   measured gap, not creating it. The headline finding is safe.

2. **Absolute FVQ accuracies for the small models are not usable as stated.**
   gemma-3-270m-it's "0.18 FVQ" is really "0.69, on the 27% of trials where it
   answered the question at all". Reporting 0.18 as first-value retrieval
   conflates instruction-following failure with retrieval failure. Same for
   pythia-410m (0.11 → 0.33) and Qwen2.5-0.5B (0.40 → 0.84).

3. **Off-stream rate tracks model capability almost monotonically** — 73% at
   270M down to 1.3% at 4B. It is essentially a competence-at-the-task-format
   measure. The models the paper leans on (Qwen2.5-3B-Instruct 4.5%,
   gemma-3-4b-it 1.3%, Qwen3.5-9B 5.1%) are clean.

## 3. The opposite risk: false *positives*

The lenient matcher can also score a wrong answer **correct**, because
`classify_error` tests `correct` before any intrusion category. If the expected
value is a substring of another value in the same trial — `age` inside `garage`,
`arm` inside `army`, `fir` inside `firm` — outputting the wrong one is scored
correct. The ARB pool has 58 such collision pairs.

Simulated per-trial exposure:

| N updates | FVQ risk | CVQ risk |
|---|---|---|
| 5 | 0.12% | 0.07% |
| 10 | 0.22% | 0.18% |
| 20 | 0.33% | 0.32% |
| 50 | 0.92% | 0.93% |
| 100 | 1.87% | 2.04% |

Below 1% for most cells, ~2% at N=100, and near-symmetric across conditions —
small relative to the effects being claimed, and not a threat to the gap. Worth
one sentence in a limitations section, not a re-run.

## 4. What we still cannot check

**The sweeps do not persist generations.** `stage1_sweep.py` builds per-trial
records containing `output_raw`, then aggregates to counts and writes only the
counts. So from disk we cannot:

- **break down the 21.6% off-stream** into its subtypes — key name? article?
  refusal? repetition of the prompt? truncation at 30 tokens mid-answer? These
  have different implications: truncation would be a *scoring* bug worth fixing,
  whereas a refusal or a key-name echo is a genuine (if different) failure.
- **audit the 30-token span for extra values.** If a model writes "first was
  **age**, later **garage**", it scores correct while having emitted an intrusion
  too. We cannot bound how often that happens.
- **check whether off-stream outputs cluster on the key name specifically** —
  which is the hypothesis worth testing, since it would suggest the model parsed
  the query but not the stream.

For contrast, `gsm8k_task_eval_*.json` **does** keep per-row `gen` text — so the
infrastructure exists, it just wasn't switched on for the sweeps.

### The cheap fix

Re-run **one cell per model** (say K=5, N=30, 100 trials) with generations
persisted, then tabulate the off-stream outputs. That is ~15 short runs, no
training, and it converts "21.6% garbage" from an unexplained number into a
labelled breakdown. Prioritise the three high-off-stream models
(gemma-3-270m-it, pythia-410m, Qwen2.5-0.5B-Instruct) — they are where the
ambiguity actually bites.

## Recommendation for the paper

- Report `accuracy_onstream` alongside raw accuracy, or **exclude off-stream
  trials from the denominator** and say so. The current numbers understate
  retrieval ability for small models by up to 51pp.
- Alternatively, **drop the three worst models from headline claims** and keep
  them in an appendix — an off-stream rate above 50% means the cell is measuring
  instruction-following, not memory.
- State the matcher explicitly (substring, case-folded, 30-token window) and its
  ~1-2% false-positive exposure at large N.
