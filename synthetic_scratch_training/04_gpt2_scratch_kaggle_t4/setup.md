# Experiment 04 — GPT-2 (small) from scratch, on Kaggle T4 x2

Status: design settled, building implementation.

## Why this experiment

Experiment 02 (GPT-2-small from scratch, local Mac, batch=8/accum=4, 3000 steps)
plateaued at ~15% overall val accuracy — barely above chance, in stark contrast to
experiment 03 (same architecture, pretrained weights) hitting 66%+ at the same step
count. Two questions this experiment is built to answer: (1) does from-scratch
GPT-2-small ever catch up with enough training (or does it need architecture/init
changes, not just more steps)? (2) local Mac memory pressure repeatedly capped what
we could test (batch size, sequence length headroom) — Kaggle's dedicated T4 x2 GPUs
remove that constraint, letting us actually find the real ceiling instead of guessing.

Kept as a new experiment (not a resumed/relaunched exp02) because the data loader
changed (see below) — resuming across a data-generator change would silently corrupt
determinism, so this is a clean restart with the improved pipeline.

## What's different from experiment 02

1. **Data loader fix**: `stream_training_batches` now samples `query_step`
   independently per example within a batch, instead of once for the whole batch.
   Sequence length depends only on (K, N) (step/key are always exactly one token
   regardless of value), so this doesn't break padding-free batching — it just gives
   every micro-batch natural first/last/intermediate role mixing internally, instead
   of every micro-batch being homogeneous in role. Own copy of `data_gen.py` (not a
   shared-file edit), so experiments 01/02/03 stay exactly reproducible as run.
2. **DDP across both T4s**: `DistributedDataParallel`, not `DataParallel` — profiled
   both approaches' tradeoffs, DDP chosen for real ~2x scaling (measured: 167.2 ex/s
   vs 80 ex/s single-GPU baseline at batch=32/accum=2, essentially perfect scaling)
   despite more setup complexity (multiprocess spawn, per-rank RNG seeding so ranks
   don't draw duplicate data, `model.no_sync()` during accumulation so gradients only
   all-reduce once per optimizer step).
3. **fp16 mixed precision** (autocast + `GradScaler`, not bf16 — T4 is Turing,
   predates bf16 tensor core support). Profiled 2.5x faster than fp32 at the same
   batch size (400ms vs 1025ms/step @ batch=32), with lower memory too. fp16-from-
   scratch instability is a known risk mainly at much larger model/depth scale than
   this (86M params, 12 layers); `GradScaler` also self-corrects by skipping an
   optimizer step if it detects inf/nan rather than silently corrupting weights.
4. **Batch size**: being profiled directly at the true worst-case sequence length
   (K=12,N=12 -> 294 tokens), not extrapolated from a shorter placeholder — attention
   memory scales ~O(seq^2), so a batch size safe at a shorter sequence isn't
   automatically safe at the real max length. See LOG.md for the actual numbers/choice.
5. **Eval frequency reduced** (every 500-750 steps, not 250) — early steps already
   show the improvement trend, no need to pay full eval cost that often.
6. **Resume support built in from the start** — Kaggle kernels are single push-and-
   run sessions with no persistent state between pushes; a run long enough to be
   interesting (see below) will need to span multiple sessions. Checkpoint dict
   includes everything needed to resume exactly (model, optimizer, scaler, step,
   held_out_cells, data-stream fast-forward point), and the resumed run gets
   uploaded/re-attached via a Kaggle Dataset so a new kernel push can load it.

## Task / data / vocab

Identical to experiments 01/02/03 — same 51-token vocab, same (K,N) grid, same
held-out step-query split. Reuses `vocab.py`, `grid.py`, `eval_utils.py` from
`../01_baseline_step_ood/src/` and `model_gpt2.py` from `../02_gpt2_scratch/src/`
via path import (unchanged, no reason to duplicate). Only `data_gen.py` is a modified
local copy (see above) and `train.py` is new (DDP/fp16/resume support experiments
01/02/03 don't need).

## Open, being decided empirically (see LOG.md)

- Per-GPU micro-batch size (profiling in progress at the real worst-case sequence
  length as of this writing)
- `accum_steps` / total effective batch size
- Total step budget and how many Kaggle sessions it'll take (resume makes this a
  "however many sessions it takes" question rather than a hard ceiling)

## Changelog
- 2026-07-17: initial design, motivated by exp02's local plateau and Kaggle T4 x2
  availability after Colab L4 quota exhaustion and Colab T4 capacity unavailability
  (see [[colab-cli-reference]], [[kaggle-cli-stability]] memories). DDP validated
  working (weight-checksum match across ranks confirmed) before committing to it as
  the batching approach.
