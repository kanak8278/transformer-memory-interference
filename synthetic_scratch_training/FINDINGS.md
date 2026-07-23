# Findings (quick reference)

Results across all 4 experiments so far, and the leading explanation for the
central open question. See each experiment's `setup.md`/`LOG.md` for full
detail — this is a fast-lookup summary.

## Headline numbers (iid_test / heldout_test overall accuracy)

| Experiment | Architecture | Steps | iid_test | heldout_test |
|---|---|---|---|---|
| 01 (L2) | tiny 2-layer, from scratch | 12,500 | 22.6% | 17.7% |
| 01 (L3) | tiny 3-layer, from scratch | 25,000 | 23.3% | 18.1% |
| 02 | GPT-2-small, from scratch (local) | 3,000 | ~15.4%* | — |
| 03 | GPT-2-small, **pretrained** | 3,000 | **66.5%** | **52.7%** |
| 04 | GPT-2-small, from scratch (DDP+fp16, fixed data loader) | 15,000 | 20.9% | 15.7% |

\* exp02 local run's final val checkpoint, not a separate iid_test pass.

## Per-experiment quick points

**exp01 — tiny transformer baseline**
- 2→3 layers, 12.5k→25k steps: only +0.6-0.7pp — doubling depth/steps barely moved the needle.
- `heldout_test` no-duplicate ≈ 10.9%, i.e. random chance (10 possible values) — model has ~zero genuine step-tracking ability once the "most frequent value" shortcut is unavailable.

**exp02 — GPT-2-small from scratch, local**
- Same architecture as exp03, random init instead of pretrained weights.
- Plateaus ~15%, barely above exp01's tiny model — architecture size alone buys nothing without pretraining.

**exp03 — GPT-2-small, real pretrained weights, raw token IDs into GPT-2's untouched 50257-vocab embedding**
- 66.5% / 52.7% by step 3000 — massive, fast improvement vs. from-scratch runs.
- Confirms pretrained weights carry directly-transferable capability for this task.

**exp04 — GPT-2-small from scratch, DDP+fp16 on Kaggle T4x2, redesigned data loader**
- Fixes vs exp02: per-example step sampling (was fixed per batch), effective batch 256 (accum=4), dropout=0.1 (GPT-2's real default), 15k steps, DDP validated ~2x scaling.
- Still plateaus ~17-19% through most of training; WSD cosine decay (last 10% of steps) pulled val accuracy from 0.173→0.190 — decay genuinely helps, but doesn't close the gap to exp03.
- Per-role breakdown at final step: `first=28.5%, last=21.5%, intermediate=15.8%` — model solves the *edges* of the induction problem (leftmost/rightmost occurrence) but not the general "k-th occurrence" case.

## Central open question: why does from-scratch training plateau so far below pretrained fine-tuning?

**Not memorization** — training data is streamed fresh every batch (see `04_gpt2_scratch_kaggle_t4/src/data_gen.py`), so literal example memorization is combinatorially impossible. Training *loss* itself plateaus, not just eval accuracy — this points to gradient descent settling into a genuine local optimum (a cheap, partial strategy), not "memorizing" anything.

**Leading explanation, research-backed:**
- Chan et al. 2022 (NeurIPS) and Reddy 2023/2024 (ICLR) show in-context/induction circuits only reliably emerge from scratch when training data has specific distributional properties: burstiness, large/rare-class vocabularies, and (Reddy) label-count ≫ in-context-exemplar-count. Our task is close to the opposite of all three: only 10 possible values, sampled i.i.d. uniformly, with up to 144 exemplars in context — a small, non-bursty vocabulary relative to context length.
- Cross-check: Zoology/MQAR (Arora et al. 2023) — a near-identical key-value associative recall task — *is* solved near-perfectly by transformers trained from scratch, but at vocab sizes of ~2,000-8,000 tokens, 40-160x larger than ours. Supports "task class is learnable from scratch; our specific vocab/distribution regime is the likely bottleneck," not an architecture or compute-budget limitation.
- Pretrained GPT-2 already has induction heads formed from real (bursty, Zipfian) text — fine-tuning just redirects an existing circuit rather than having to bootstrap one from data that doesn't reward bootstrapping it.

**Ruled out:**
- `n_positions` (1024 vs our actual max length ~294) — unused position-embedding rows never receive gradients or appear in any forward pass; negligible parameter cost (<0.5% of model); doesn't touch the actual mechanism.
- Hardware/compute budget — H100 would help with iteration *speed* (mainly via running more experiments in parallel, since the pipeline is partly CPU-bound on Python data generation and the model is too small to saturate an H100 anyway), not with this specific plateau.

## Untested next steps (ranked by expected leverage, per the research above)
1. Increase value vocabulary substantially (10 → hundreds/thousands) — most directly targets the identified cause, but changes the synthetic task's statistics; needs to be weighed against what the benchmark is meant to measure re: the ACL paper's PI/RI framing.
2. Add burstiness to the training distribution instead of pure i.i.d. sampling.
3. Explicit curriculum — start on easy (low-K, low-N) cells, widen over training.
4. Much longer training as a cheap test of a possible late "grokking" transition (lower expected payoff per the literature, since the issue looks data-driven rather than purely optimization-time-driven).
