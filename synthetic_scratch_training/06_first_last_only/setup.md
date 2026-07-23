# Experiment 06 — First/Last-only training (from scratch)

Status: built + validated (data logic), not yet launched (waiting for exp05 to
confirm plateau before taking a GPU).

## Why this experiment

Every run so far shows the same role signature: edges (first/last) solved better
than interior. exp04 final: first=28.5%, last=21.5%, intermediate=15.8%; exp05 early
evals ~first/last 27% vs intermediate 19%. Hypothesis: the interior/**k-th-occurrence
counting** case is the hard part and may be competing for capacity/gradient and
dragging down the edges. Train on **first + last only** and see whether the edges
climb (toward ~100%) once isolated.

## Design

Identical to exp05 (GPT-2-small from scratch, all best practices: bf16, TF32,
torch.compile, AdamW β=(0.9,0.95) + decoupled WD, grad-clip 1.0, dropout 0, residual
scaled init, prefetch loader) — the **only** change is the training query-step set:

- **Training:** `query_step ∈ {1, n}` only (first-occurrence + last-occurrence).
  `--first-last-only` flag; `data_gen.stream_training_batches(first_last_only=True)`.
  Interior steps (2..n-1) are never trained. (held_out_cells becomes irrelevant since
  it only ever held out interior steps.)
- **Eval, primary:** first/last IID accuracy (the roles we trained) — does isolating
  the edges push them up vs exp05's ~27%?
- **Eval, zero-shot probe:** the entire interior (2..n-1), never trained. Reported
  split into:
  - **seen-token**: query step ∈ {1} ∪ N_VALUES = {1,2,4,6,8,10,12} — the model saw
    this step TOKEN in training, but only in its "last-of-length-s" role; here it must
    reinterpret it as "s-th of a longer n".
  - **novel-token**: query step ∈ {3,5,7,9,11} — never seen at all.

## What each outcome means

- **first/last shoot up (toward ~100%):** edge-retrieval IS learnable from scratch;
  interior counting is the specific bottleneck (competes for capacity). Supports the
  induction-heads-form-for-edges-but-not-general-k-th story.
- **first/last stay ~27% even alone:** the difficulty isn't capacity competition from
  interior — edge retrieval itself is hard from scratch in this distribution.
- **interior zero-shot ≫ chance:** some genuine counting generalization even from
  scratch. **≈ chance (expected):** no general k-th ability, only memorised roles.

## Launch (when a GPU frees)

    source ../env.sh && cd src
    python train.py --gpu <g> --schedule cosine --tag fl_scratch --first-last-only \
      --total-steps 30000 --warmup 600 --batch-size 256 --accum 4

# --accum 4 (eff batch 1024, half of exp05's 2048): "faster updates" -- exp05's
# gradient was very stable/over-averaged, so we take ~2x as many optimizer steps per
# unit data (micro-batch stays 256 to keep the H100 fully utilized). LR kept at 6e-4
# (deliberately aggressive for the smaller batch). Eval cadence unchanged (every 1000
# steps). Watch for loss spikes (grad-clip 1.0 guards).

Reuses vocab/grid/eval_utils/model_gpt2/schedulers/prefetch from exp05; only
`data_gen.py` (first_last_only branch), `prefetch.py` (pass-through) and `train.py`
(flag + eval split) differ. Pairs with exp07 (same, pretrained) for the 2x2.

## Changelog
- 2026-07-22: built, first/last stream validated on CPU (steps {1,2,4,6,8,10,12}, no
  interior leakage). Launch deferred pending exp05 plateau confirmation.
