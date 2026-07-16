# Experiment 01 — Implementation Log

Running log of implementation decisions, issues found, and run status. `setup.md`
stays the lean spec; this file is the narrative.

## 2026-07-16 — Implementation

Built the full pipeline in `src/`:
- `vocab.py` — 51-token vocab (KEY 26, VALUE 10, STEP 12, BOS/QUERY/EOS).
- `grid.py` — (K,N) grid + held-out (N,step) cell selection.
- `data_gen.py` — example builder (interleaved context + query), streaming train
  generator, frozen eval-set builder.
- `model.py` — custom decoder-only causal transformer (not HF `transformers`; kept
  custom so attention weights are cheap to expose later for mechanistic analysis).
- `train.py` — streaming training loop, WSD LR schedule, checkpointing.
- `eval_utils.py` / `evaluate.py` — scoring + aggregation (overall / per-cell /
  per-step-role / per-step-index / per-duplicate-flag), held-out generalization gap.
- `dump_val_samples.py` — human-readable prediction dumps for manual sanity-checking.
- `build_eval_sets.py` — one-off script that generated and froze `data/{iid_val,
  iid_test,heldout_test}.pt` (seed 42).

**Held-out cell selection verified**: 8 of 30 interior cells held out (~27%, close to
the 25% target after rounding) — `(6,5) (8,2) (8,6) (10,2) (10,4) (10,9) (12,4) (12,8)`.
Every held-out step value confirmed to retain training exposure elsewhere (either as a
last-value-query at N=that step value, or as a non-held-out interior cell at another N).

**Two implementation decisions not spelled out in `setup.md`, resolved while coding:**
1. **Loss is computed on the last two token positions only** (predicting VALUE, then
   EOS) — not the full sequence. The K*N context tokens are i.i.d. random and
   unpredictable; training on them would just add noise to the gradient. This matches
   the "single query per context" design but was an implementation detail setup.md
   didn't pin down explicitly.
2. **Batching is done per-(K,N,step) cell**, not by padding mixed-length sequences.
   Since sequence length is a deterministic function of (K,N) only, one cell per batch
   gives exact-length batches for free — no PAD token needed anywhere in this
   experiment (it's still reserved in the vocab for later experiments that might mix
   lengths in a batch).

**Sanity checks before the real run:**
- Manually traced one generated example end-to-end (interleaving, query resolution,
  duplicate-flag) against the decoded token string — correct.
- Ran two throwaway smoke trainings (60 steps, then 40 steps) to exercise the whole
  loop (data → model → loss → checkpoint → val-eval → readable dump → evaluate.py) —
  no crashes, loss decreased sensibly (3.56→1.58 over 60 steps).
- Timed 200 steps at the real batch size (256): ~61ms/step steady-state on MPS →
  ~50-55 min per depth for the full 50k-step budget, ~3.5-4h for the full 2/3/4/5-layer
  sweep.

**Status: full depth sweep launched in background** (`run_depth_sweep.sh`, nohup'd,
logs to `results/depth_sweep.log`), training L2→L3→L4→L5 sequentially, each followed
by `evaluate.py` (iid_test + heldout_test) and a readable held-out-sample dump.
Monitoring via a persistent log-tail watch for per-depth completion and any
crash/error signatures.

**Correction after launch (user caught this):** the first launch used `accum_steps=4`
default but left `total_steps=50000` unchanged, which would have made each depth run
~4x longer than planned (~14-15h total instead of ~3.5-4h), since each optimizer step
now covers 4x the data. Killed and relaunched with `total_steps=12500`,
`warmup_steps=125`, `ckpt_every=500` (all scaled down by the same 4x) so total
examples-seen and wall-clock time match the original plan. Also added TensorBoard
logging (`results/tb/<tag>/`) alongside the JSON logs, per user request — chose
TensorBoard over W&B since it's fully local/no-auth, which matters for an unattended
background run.

## 2026-07-16 (later) — fixed `by_step` role contamination

User caught (via questioning the reported numbers, not a crash) that `eval_utils.
aggregate_results`'s `by_step` breakdown pooled examples by raw step-index value
regardless of role. Since the N grid ({2,4,6,8,10,12}) is all even, any even step
value coincides with a valid N — e.g. step=4 pools N=4 (where step=4 *is* the last/CVQ
query) together with N=6,8,10,12 (where step=4 is genuinely intermediate). Odd steps
(3,5,7,9,11) never collide with an N and were already clean.

This wasn't a training-time bug — data generation and loss are unaffected, every
example is generated and trained with the correct role/target. It was purely an
eval-aggregation artifact. But it directly corrupted the headline held-out-generalization
comparison in `evaluate.py` (`generalization_gap_by_step`, which reads `by_step`): 4 of
the 6 held-out step values (2,4,6,8, out of {2,4,5,6,8,9}) are even, so their
"trained_elsewhere_acc" comparison was mixing in last-value-query accuracy from the
N=step case.

**Fix**: `by_step` now excludes `is_first`/`is_last` examples before grouping — a clean
intermediate-only curve, consistent with what `by_role["intermediate"]` already
measures in aggregate. Verified against the live L2 checkpoint (step 4500): step=1 and
step=12 correctly disappeared from `by_step` (always first/last respectively at
N_max=12), and even-step sample counts shrank to just their genuinely-intermediate N
contributions (e.g. step=4: n=900→600) while odd steps were unchanged. Safe to edit
without disrupting the running L2 training — that process doesn't read `by_step` during
its periodic val checks (only `by_role`), and already has the old module loaded in
memory regardless; L3/L4/L5 (launched as fresh subprocesses by `run_depth_sweep.sh`)
will pick up the fix automatically.

## 2026-07-16 (later) — L2 (2-layer) results

Best checkpoint: step 9500/12500 (essentially plateaued 10k-12.5k, fluctuating 21.8-22.6%
overall val acc — training did not improve meaningfully in the last ~25% of the budget
at this depth).

**iid_test overall 22.6% vs heldout_test overall 17.7%** — a real ~5pp generalization
gap on step queries never seen in training. `by_role` (iid_test): first 29.3%, last
27.8%, intermediate 19.4% — first/last now roughly level, intermediate clearly behind,
consistent with periodic val checks throughout training.

**Headline finding: duplicate-value stratification shows the model is mostly running on
a shortcut, not genuine step-tracking, at this depth:**

| split | duplicate=True | duplicate=False |
|---|---|---|
| iid_test | 26.7% | 18.3% |
| heldout_test | 22.7% | **10.9%** |

heldout_test + no-duplicate is 10.9% — essentially the ~10% random-chance floor (10
possible VALUE tokens). On genuinely novel step queries where the "most frequent value
for this key" shortcut isn't available, the 2-layer model shows ~zero real step-tracking
ability. This is exactly the confound flagged in `setup.md` ("Duplicate values") — now
confirmed as a real, large effect rather than a theoretical risk. Worth checking whether
this shrinks with depth (L3/L4/L5) — if it doesn't, that's evidence 2-5 layers may not
be enough capacity for genuine per-key counting, not that the task is unlearnable in
principle.

`generalization_gap_by_step` (trained-elsewhere vs held-out, per step value) is mixed
and mostly within noise at this depth/accuracy level (n per step is only a few hundred)
— not drawing conclusions from it yet, revisit once accuracy is higher / more depths in.

## 2026-07-16 (later still) — L2≈L3 investigation, resume fix, extending L3

At matching step counts (9000), L2 and L3 tracked nearly identically (22.4% vs 22.6%
overall, same duplicate-shortcut signature: ~27% on duplicate-value examples, ~17.6-
17.8% on no-duplicate). Verified this is not an "n_layers silently ignored" bug —
checkpoints confirmed structurally different (L2: 2 blocks/127,104 params, L3: 3
blocks/177,088 params). Leading hypothesis: both models found the same cheap
"most-frequent-value" shortcut early and neither has discovered genuine step-counting
yet; L2's own LR-decay phase (steps 11250-12500, its last 10%) showed no jump either
(22.55→21.8→22.18→22.6→22.4→22.55, just noise) — so within the original 12,500-step
budget, decay alone didn't unlock anything. Candidate causes discussed: (a) shortcut-
dominated training / no grokking transition yet, (b) missing GPT-2/nanoGPT-style
residual-output init scaling (`1/sqrt(2*n_layers)`) possibly handicapping deeper models
specifically, (c) d_model=64 fixed across the whole depth sweep — width, not depth,
may be the actual bottleneck. **Decided: pause before L4/L5, don't jump straight to a
bigger architecture (e.g. 10 layers) without first isolating the cause** — un-diagnosed,
a bigger model could show the same flat result for the same reason and just cost more
compute to learn nothing new. Killed `run_depth_sweep.sh` (the driver) so L4/L5 don't
auto-launch; left L3's own training process running to finish its current budget.

**Fixed the resume-stream bug found in the earlier code review**: `train.py` was
rebuilding `stream_training_batches` from the base seed on `--resume`, silently
replaying already-seen data rather than continuing with new examples. Fix: on resume,
fast-forward the stream generator by replaying (and discarding) exactly
`(start_step-1) * accum_steps` micro-batches before the main loop, so the RNG lands in
the same state a truly-uninterrupted run would have reached. **Verified correct**: a
simulated 15-micro-batch continuous run vs. a checkpoint-at-10-then-resume run produced
byte-identical (k,n,step) sequences for steps 11-15. Also added `torch.manual_seed
(args.seed)` before model construction (previously unseeded, contradicting setup.md's
determinism claim) — harmless for resumed runs (weights load from checkpoint,
overwriting fresh init) so doesn't affect the extension below; matters for future fresh
runs (L2/L3 re-runs, or eventually L4/L5).

**Plan**: let L3's original 12,500-step run finish naturally, then automatically (via
`extend_l3.sh`, backgrounded) resume from its final checkpoint with `--total_steps
25000` (doubled) using the now-fixed resume path — genuinely new data, not a replay.
Per the WSD schedule design, resuming with a larger `total_steps` correctly pushes the
decay phase out (new decay_start = step 22500) and returns the LR to the "stable" peak
phase immediately, exactly as intended for an "extend if grokking suspected" scenario —
expect an LR jump back to 1e-3 right after resuming, not a smooth continuation; this is
expected/correct, not a bug. Explicitly NOT touching the residual-scaling init question
or L4/L5 yet — that's still open pending this result.

## 2026-07-17 — L3 extension (25,000 steps) results: more training doesn't help

Extension finished cleanly (fixed resume-stream picked up genuinely new data —
fast-forwarded 50,000 micro-batches at resume, confirmed in the log). Best checkpoint
step 24000/25000.

**iid_test 23.29% vs heldout_test 18.09%** (was 22.6%/17.7% at the original 12,500-step
plateau) — a **~0.6-0.7pp bump**, well inside the noise band the original run already
showed step-to-step (21.8-22.6%). `by_role` (iid): first 29.4%, last 28.7%, intermediate
20.1% (was 19.4%). `by_duplicate`: iid True/False 27.4%/18.9% (was 26.7%/18.3%); heldout
True/False 22.9%/**11.4%** (was 22.7%/10.9%).

**Conclusion: doubling the training budget gave a small, uniform improvement across
every slice, not a qualitative jump.** heldout+no-duplicate is still ~11%, essentially
unchanged from the ~10% chance floor. This is evidence against "just needs more
steps/grokking" and consistent with the shortcut-learning hypothesis: the model found
the frequency heuristic early and 2x more steps at the same LR schedule / same width
just polishes it slightly rather than unlocking real step-counting.

`generalization_gap_by_step` (intermediate-only, post role-contamination-fix) remains
noisy and inconsistent in direction (e.g. step 8: trained-elsewhere 20.3% vs held-out
14.2%; step 6: trained-elsewhere 17.2% vs held-out 18.7% — gap flips sign) — still not
drawing conclusions from it, n per step is only a few hundred.

**Decision going forward**: not spending more compute on "train longer" for this
architecture. Moving to experiment 02 (GPT-2-small-from-scratch) to test whether width
and/or GPT-2's residual-output init scaling change the qualitative picture, per the plan
already in `../02_gpt2_scratch/setup.md`. L4/L5 of the original custom-model depth sweep
remain on hold pending that result.

### Open watch-items once results land
- Does per-step accuracy (first/intermediate/last) show the expected shape, or is
  intermediate-step recall near-chance regardless of depth (would suggest 2-5 layers
  genuinely can't do the counting mechanism, not just needs more steps)?
- Held-out generalization gap (`generalization_gap_by_step` in `final_eval.json`):
  does held-out accuracy track "trained elsewhere" accuracy per step value, or lag
  badly — and does that gap shrink with depth?
- Duplicate-flag stratification: is accuracy on no-duplicate examples meaningfully
  higher than on duplicate-heavy ones (evidence of a shortcut) or similar (evidence of
  genuine tracking)?
- Any depth/cell showing much worse loss than others (difficulty-imbalance check —
  `per_cell` losses logged every `log_every` steps in `results/<tag>/train_log.json`).
