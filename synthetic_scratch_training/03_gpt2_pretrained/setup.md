# Experiment 03 — GPT-2 (small), real pretrained weights

Status: implemented, not yet run.

## Why this experiment

Quick check: how does a real pretrained GPT-2 do on this task, as a reference point
against experiment 01 (custom tiny transformer, from scratch) and experiment 02
(real GPT-2-small architecture, from scratch)? Deliberately the crudest possible setup,
per explicit decision: **no embedding resizing, no vocab remapping** — just load
`GPT2LMHeadModel.from_pretrained("gpt2")` as-is (full 50257-token embedding/head,
pretrained weights untouched) and feed our synthetic vocab's token ids (0-50) straight
in as `input_ids`. Those ids are valid indices into GPT-2's embedding table (whatever
real BPE tokens happen to sit at ids 0-50), but carry none of the "key/value/step"
meaning we intend — GPT-2's own token 0-50 have their own pretrained meaning. This is
not an apples-to-apples ablation like experiment 02 (that isolated one variable:
random vs pretrained init at identical architecture); this is a blunt "does pretraining
help at all, even under a semantically-scrambled token mapping" probe.

## Task / data / vocab

Identical to experiments 01/02 — same 51-token vocab (ids reused as raw GPT-2 input
ids), same (K,N) grid, same held-out step-query split, same streaming train / frozen
eval sets. Reuses `data_gen.py`, `grid.py`, `vocab.py`, `eval_utils.py` from
`../01_baseline_step_ood/src/` via path import, same as experiment 02.

## Model

- `transformers.GPT2LMHeadModel.from_pretrained("gpt2")` — real pretrained weights,
  full 50257 vocab_size, no config overrides. Requires network access on first run
  (downloads ~500MB, cached by HF afterward).
- Dropout left at GPT-2's pretrained default (resid/embd/attn_pdrop=0.1) — **not**
  zeroed to match experiment 02, unlike that experiment this isn't a controlled
  ablation, and 0.1 is what the weights were pretrained with. Known confound, not
  fixed, since the point here is "how does the real thing do," not isolating one
  variable.

## Training

Same recipe/loop as experiment 02 (gradient accumulation, WSD schedule, last-two-
position manual loss, checkpointing with pruning, TensorBoard) — only the model
construction differs (`GPT2Pretrained` in `model_gpt2_pretrained.py` vs
`GPT2FromScratch`).

**`--peak_lr` default lowered to 3e-5** (vs experiment 02's 3e-4) — fine-tuning
pretrained weights typically wants a much smaller LR than training from scratch, to
avoid destroying the pretrained representations before the model can adapt them to
this task. Batch size / total_steps otherwise copied from experiment 02's defaults,
same "needs re-profiling" caveat applies (this model is slightly heavier than
experiment 02's, ~124M vs ~85.9M params, due to the full pretrained 50257-row
embedding/head vs experiment 02's 51-row one).

## Changelog
- 2026-07-17: initial implementation, no run yet. Motivated by wanting a real-
  pretrained-model reference point alongside experiments 01/02; see
  `../01_baseline_step_ood/LOG.md` (L3 extension results) for what this is being
  compared against.
