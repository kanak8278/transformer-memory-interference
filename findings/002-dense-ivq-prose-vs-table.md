# Finding 002 — Dense-IVQ prose contradicts the paper's own table

- **Severity:** 🟠 High (internal contradiction visible on the same page)
- **Status:** Open — prose fix (data is fine and honestly tabulated)
- **Related task:** C1.2 (verify), Group B (edit)
- **Verified:** 2026-07-11, against `dense_ivq_ivq_20260523_101418.json`

## Paper claim (three places, all overstated)

1. §5.2 close (main.tex ~L462, Fig dense_ivq caption):
   > "Base collapses on intermediate positions; LoRA **holds near ceiling at
   > every position**."
2. App dense_ivq (main.tex ~L1193–1195):
   > "the LoRA-tuned model **recovers near-ceiling accuracy at every position
   > queried**."
3. Table dense_ivq caption (main.tex ~L1219):
   > "the LoRA-tuned model **holds near ceiling**."

Also §5.2 uses this to argue: "simultaneous recovery across positions, not just
the last, rules out a pure recency-amplification account."

## Actual data

`tab_dense_ivq.tex` (the paper's own table) honestly prints LoRA
"Interm. min–max":

| Cell | LoRA interm min–max |
|---|---|
| (5,10) | 70%–100% |
| (10,20) | 5%–100% |
| (10,30) | **0%**–100% |
| (15,30) | **0%**–90% |
| (20,30) | **0%**–85% |
| (25,50) | **0%**–70% |

Per-position detail (source JSON), e.g. LoRA at (10,30), N=30, positions queried
1–20: `1:1.0, 2:1.0, 3:1.0, 4:1.0, 5:0.9, 6:0.45 … 15:0.2, 16:0.1, 17:0.05,
18:0.0, 19:0.0, 20:0.0`. The adapter is near-ceiling for early positions and
**decays to 0.0 at late-intermediate positions in long streams**.

## How they differ

The prose says "every position" is near-ceiling; the table beneath it shows the
LoRA at **0%** for some intermediate positions in **4 of 6 cells**. A reviewer
reading the caption over the table catches this immediately. (The table and the
figure are honest; only the prose overstates.)

Nuance for the recency-amplification argument: LoRA *does* recover many
intermediate positions the base model fails (early-to-mid), so "not just the
last" is partly supported — but "every position" is false.

## Root cause

LoRA was trained on N ≤ 20. At high stream length, late-but-not-last positions
(indices ~15–20 inside an N=30 stream) collapse — plausibly the adapter
conflates high position indices with "give me the last." This is the same
generalization-decay the E1 extrapolation experiment is designed to map; it is
already visible in-repo.

## Resolution

Soften all three statements, e.g.:
> "LoRA recovers the early-to-mid intermediate positions the base model fails,
> with accuracy decaying at late positions in the longest streams (Table X)."

Keep the "not purely recency" argument but scope it to the recovered positions.
Consider citing this decay as motivation for E1. Also add trial count (n=20/pos,
Finding 004).
