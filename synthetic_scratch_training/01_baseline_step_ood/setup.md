# Experiment 01 — Baseline Step-Query Generalization

Status: design settled, no code yet.

## Task
- K keys, N sequential updates/key, interleaved across keys (shuffle-merge, per-key order preserved).
- Values: i.i.d. random 0-9, repeats allowed.
- One (key, step) query per context. step=1 → first-value query, step=N → last/current-value query, step 2..N-1 → intermediate-value query.

## Vocab (51 tokens, single-token, no BPE)
- KEY: `<Ka>`-`<Kz>` (26) — pool; active K per example sampled from it, not all 26 at once.
- VALUE: `<V0>`-`<V9>` (10)
- STEP: `<S1>`-`<S12>` (12, sized to max N)
- special: `<BOS>` `<QUERY>` `<EOS>` (3)
- `<PAD>` if batching variable lengths (52 total)

## Sequence format
`<BOS> <Key><Value>... <QUERY> <Key> <Step> <Value> <EOS>`

## Grid
K, N ∈ {2,4,6,8,10,12} → 36 cells. Max length ≈ 294 tokens (K=12,N=12).

## Model (approx defaults, revisit after first runs)
- One shared model trained across all 36 cells (required for the held-out step split to be valid).
- Learned absolute positional embeddings (not RoPE — no length-extrapolation axis this round).
- Depth swept: 2,3,4,5 layers. Width: d_model=64, n_heads=4, d_ff=256, dropout=0.0.

## Held-out step-query split
- First-value query (step=1) and last-value query (step=N): always IID, every cell.
- Held out only from interior steps (2..N-1). 30 interior (N,step) cells total across the grid.
- **25% held out (~7-8 of 30 cells)**, selected via seeded random draw, constrained so every step value stays trained via ≥1 other N.

## Data generation: streaming train, frozen eval
Generation is free (synthetic, tiny vocab) — no reason to cap training data to a fixed corpus and epoch over it.
- **Train: streamed on-the-fly**, deterministic via seeded generator, no fixed count, never repeats an exact example. Only draws from allowed (non-held-out) (K,N,step) combos.
- **Eval: frozen, saved to disk**, generated once:
  - IID val: 50 examples / (K,N,step) triple (allowed combos only) — early-stopping/monitoring.
  - IID test: 200 examples / (K,N,step) triple (allowed combos only).
  - Held-out-step test: 500 examples / held-out (K,N,step) triple — larger, since this is the main result.
- Active keys: uniform from 26-pool. Query key: uniform among active. Query step: uniform among allowed steps (train) or the fixed target (eval). Interleaving: uniform random shuffle-merge.

## Training budget (approx defaults, revisit after first runs)
- Micro-batch 256, **gradient accumulation over 4 differently-sampled cells per optimizer step** (effective batch 1024). Each micro-batch is single-(K,N,step) (no padding needed), but accumulating across several before the optimizer step avoids every update coming from one homogeneous, arbitrarily-easy-or-hard cell. AdamW, peak lr 1e-3, weight decay 0.05 (WD matters more than LR shape for grokking-style late generalization — don't skip it).
- **LR schedule: warmup → stable → decay (WSD), not a fixed-endpoint schedule.** Linear warmup (125 optimizer steps) to peak lr, then hold constant for as long as training continues (re-check at each checkpoint), only decay (cosine/linear to ~10% of peak) over the last ~10% of steps once we've actually decided to stop. A schedule pre-committed to decay-to-zero at a fixed step count would fight the "extend if grokking" plan below.
- **12,500 optimizer steps** default budget (soft — see schedule note above) — scaled down from an initial 50,000 to keep total examples-seen (and wall-clock time) comparable now that each optimizer step covers 4x the data via accumulation. Checkpoint every 500 optimizer steps + best-val.
- Watch for grokking (algorithmic tasks can plateau then jump much later) — extend training (staying in the stable LR phase) if val is still improving near the budget limit, don't stop just because train loss looks flat.
- Logged to TensorBoard (`results/tb/<tag>/`) as well as JSON — train loss (overall + per-cell), val accuracy (overall + by role + per-cell).

## Difficulty imbalance
Flat sampling by default. Log per-cell train/val loss+accuracy. Add curriculum/reweighting only if diagnostics show starvation.

## Duplicate values
Allowed (i.i.d.), not filtered at data-gen. Tag each example with a duplicate-flag (binary: does the target value recur elsewhere in this key's own N-length history); stratify eval accuracy by it.

## Determinism
Global seed = **42**. Drives: streaming train generator, frozen eval-set generation, held-out cell selection, model init, batch order. Frozen eval sets saved to disk once, never regenerated.

## Checkpoints
Every 2,000 steps + best-val, under `checkpoints/`. Each independently reloadable for IID-test and held-out-step eval.

## Open items
1. Width/depth may need adjusting once first runs show under/overfitting.
2. Training step budget (50k) is a starting guess — extend if grokking-like late convergence is suspected.

## Changelog
- 2026-07-16: initial spec, no code yet.
- 2026-07-16: locked approx defaults (held-out %, eval set sizes, model width, seed, training budget) to unblock starting implementation. Switched from fixed-corpus to streaming-train/frozen-eval design.
