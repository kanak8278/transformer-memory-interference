# Experiment 02 — GPT-2 (small) architecture, from scratch

Status: design settled, code not yet written.

## Why this experiment

Experiment 01 (custom 2-5 layer transformer) showed depth (2 vs 3 layers) making no
difference — both plateau at the same accuracy with the same duplicate-value-shortcut
signature. Candidate causes: (a) shortcut-dominated training, no grokking yet, (b) our
custom model is missing GPT-2-style residual-output init scaling (`1/sqrt(2*n_layers)`),
possibly handicapping deeper stacks specifically, (c) d_model=64 fixed across the whole
depth sweep — width, not depth, may be the real bottleneck. Rather than guess, this
experiment swaps in the real, well-tested GPT-2 architecture at real GPT-2-small scale,
which fixes (b) automatically and tests (c) by using far more width, while keeping the
"trained from scratch, fully controlled" premise of the whole synthetic_scratch_training
line intact — no pretrained weights.

## Task / data / vocab

Identical to experiment 01 — same 51-token vocab, same (K,N) grid, same held-out
step-query split, same streaming train / frozen eval design. Reuses `data_gen.py`,
`grid.py`, `vocab.py`, `eval_utils.py`, `build_eval_sets.py` directly from
`../01_baseline_step_ood/src/` (imported via path, not duplicated) — nothing about the
task changes, only the model.

## Model

- `transformers.GPT2LMHeadModel`, constructed `from_config` — **never
  `from_pretrained`**. Randomly initialized, trained from scratch, exactly like
  experiment 01's custom model.
- `GPT2Config(vocab_size=51)` — every other config value left at HF's default, which
  *is* the real published GPT-2-small architecture: n_positions=1024, n_embd=768,
  n_layer=12, n_head=12 (124M params). Only vocab_size is overridden, since that's
  data-dependent, not an architecture choice — deliberately not touching n_positions
  even though our sequences (≤~294 tokens) don't need the full 1024, to keep as close
  to "real config, unmodified" as possible per the point of this experiment.
- HF's built-in init (`GPT2PreTrainedModel._init_weights`) already applies the
  residual-scaling fix experiment 01 was missing (scales `c_proj` weights by
  `1/sqrt(2*n_layer)`) — this is the whole point of switching implementations rather
  than patching our own.

## Training

Same recipe as experiment 01 where it doesn't depend on model internals: gradient
accumulation over differently-sampled cells, WSD LR schedule, checkpointing,
`torch.manual_seed(seed)` before model construction. Loss computed manually on the
last two positions (VALUE, then EOS) from raw logits — not HF's automatic
label-shifted loss — same as experiment 01, since the K*N context tokens are
unpredictable and shouldn't be trained on.

**Open, to be determined empirically before committing to a full run**: this model is
~700-1000x more parameters than experiment 01's (124M vs ~127-177K). Compute per step
will be far higher — batch size, accum_steps, and total_steps from experiment 01
(256/4/12500) are almost certainly wrong for this scale and need to be re-profiled on
this hardware (Apple Silicon MPS) before launching a real run, the same way experiment
01's timing was profiled before its full sweep.

## Changelog
- 2026-07-17: initial spec, no code yet. Experiment 01's L2/L3 depth-sweep results
  (plateau, duplicate-shortcut signature, missing residual-scaling) are what motivated
  this experiment; see `../01_baseline_step_ood/LOG.md`.
