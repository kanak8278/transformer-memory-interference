# E1 — LoRA extrapolation frontier: findings

**Run:** 2026-07-14, Colab L4, HF backend, greedy. Base vs main Qwen2.5-3B LoRA
adapter (trained on K≤10, N≤20). Grid = multiples of the training maxima,
emanating from the training corner (K=10, N=20). 100 trials/cell with Wilson
early-stop (HW≤0.07). Data: `results_multiples.jsonl` (140 rows).

## Headline

The LoRA recovery **splits into two separable capabilities with different
generalization**:

1. **Endpoint retrieval (first & last value) fully generalizes to 5× training
   load in BOTH dimensions.**
2. **Intermediate-position retrieval does NOT extrapolate**: it is bounded to
   the training stream-length (N) and decays with query depth. Growing N breaks
   it; growing K (at fixed N) barely touches it.

This sharply qualifies the paper's §5.2 claim that IVQ training instilled
general position-indexed retrieval "ruling out a pure recency-amplification
account." What the adapter actually learned is robust **first/last extraction**
that generalizes far, plus **shallow, N-bounded** position indexing that does
not.

## Endpoints (FVQ / CVQ) — LoRA holds everywhere, base fails everywhere

N-scan (K=10 fixed):

| N (×) | base FVQ/CVQ | LoRA FVQ/CVQ |
|---|---|---|
| 20 (1×) | 0.39 / 0.48 | 1.00 / 0.96 |
| 40 (2×) | 0.32 / 0.57 | 1.00 / 1.00 |
| 60 (3×) | 0.40 / 0.56 | 1.00 / 0.98 |
| 80 (4×) | 0.24 / 0.61 | 1.00 / 0.96 |
| 100 (5×) | 0.04 / 0.52 | 1.00 / 1.00 |

K-scan (N=20 fixed):

| K (×) | base FVQ/CVQ | LoRA FVQ/CVQ |
|---|---|---|
| 10 (1×) | 0.39 / 0.48 | 1.00 / 0.96 |
| 20 (2×) | 0.29 / 0.54 | 1.00 / 0.98 |
| 30 (3×) | 0.23 / 0.55 | 1.00 / 0.98 |
| 40 (4×) | 0.13 / 0.50 | 1.00 / 0.95 |
| 46 (~5×) | 0.07 / 0.60 | 1.00 / 0.96 |

Base is in the **reversal regime** at K=10 (CVQ > FVQ, both poor); FVQ collapses
with load (→0.04 at 5× N, →0.07 at ~5× K) while CVQ sits ~0.5–0.6. LoRA restores
FVQ to 1.00 and CVQ to 0.95–1.00 across the entire 1–5× envelope in both axes.

## Intermediate positions (IVQ, mean over 5 relative depths)

| cell | base | LoRA | LoRA by depth [10/25/50/75/90 %] |
|---|---|---|---|
| N=20 (1×) | 0.06 | 0.56 | [1.0, 0.79, 0.44, 0.20, 0.35] |
| N=40 (2×) | 0.03 | 0.20 | [0.63, 0.33, 0, 0, 0.06] |
| N=60 (3×) | 0.01 | 0.13 | [0.56, 0, 0, 0, 0.09] |
| N=80 (4×) | 0.00 | 0.08 | [0.34, 0, 0, 0, 0.04] |
| N=100 (5×) | 0.00 | 0.07 | [0.25, 0, 0, 0, 0.09] |
| K=20 (2×), N=20 | 0.03 | 0.42 | [1.0, 0.65, 0.18, 0.06, 0.23] |
| K=30 (3×), N=20 | 0.02 | 0.37 | [0.98, 0.62, 0.07, 0.02, 0.16] |
| K=40 (4×), N=20 | 0.04 | 0.37 | [1.0, 0.52, 0.04, 0.04, 0.25] |
| K=46 (~5×), N=20 | 0.04 | 0.36 | [0.96, 0.60, 0, 0, 0.25] |

Two clean regularities:
- **N is the killer, not K.** Holding N=20 (training length) and pushing K to
  ~5×, LoRA intermediates only sag (0.56→0.36). Pushing N past training with K
  fixed collapses them (0.56→0.07). So the adapter's positional addressing is
  bounded by the stream *length* it was trained on, not the number of keys.
- **Depth decay.** Even within training N, accuracy falls monotonically with
  query depth (≈1.0 at 10% depth → ≈0.2 at 75%). Beyond training N, only the
  shallowest positions retain any signal.

## Interpretation / paper impact

- The "latent capability" the LoRA surfaces is **first/last extraction**, which
  is genuinely robust (generalizes to 5× in K and N). This strengthens the
  latency claim for the FVQ/CVQ result the paper centers on.
- But the IVQ-based argument in §5.2 (that the recovery is general position
  indexing, not recency amplification) is **only true within the training N**.
  Recommend softening: the adapter learns robust endpoint retrieval plus
  N-bounded shallow position indexing; true arbitrary-position retrieval does
  not extrapolate.
- The N-vs-K asymmetry (length bounds positional addressing, key-count does not)
  is a new, mechanistically suggestive result — consistent with a
  retrieval/separation-bound story (more update positions to disambiguate at
  higher N; first/last stay uniquely anchored). Candidate new figure for the
  AAAI version.

## Caveats
- Single model (Qwen2.5-3B), single adapter, K=10 for N-scan / N=20 for K-scan.
- Related earlier smoke (K=2, N=1000) showed CVQ also eventually collapses far
  out (0.25) — so even endpoint recovery has a frontier, just well beyond 5×.
- HF backend (not vLLM); greedy so deterministic and comparable to paper.
