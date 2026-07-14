# Behavioral Block-vs-LoRA sweep — "two roads, one readout; but only one road indexes positions"

**Run:** 2026-07-14, Colab L4, Qwen2.5-3B, ARB single-token. 3 conditions
(base_plain / base_block / lora_plain) × 8 cells; dense cells sweep every
position (alternate on N≥50). Method identical to the naive dense-IVQ study
(seeds, `shuffle_nc`, ordinal phrasing, `is_correct`); Wilson 95% CI with
early-stop (min 30, cap 200, HW≤0.07). Behavioral scoring = multi-token greedy +
prefix/contains match. Data: `results.json` (per-position, all 24 conditions).

## 1. Endpoints (CVQ / last-value): both fixes converge

| cell | base | block | lora |
|---|---|---|---|
| K5/N20 | 0.42 | 0.95 | 1.00 |
| K10/N25 | 0.57 | 0.90 | 1.00 |
| K10/N50 | 0.51 | 0.89 | 0.98 |
| K5/N50 | 0.44 | 0.97 | 1.00 |
| K5/N75 | 0.49 | 0.94 | 0.98 |
| K2/N30 | 0.41 | 0.95 | 1.00 |
| K20/N20 (reversal) | 0.58 | 1.00 | 0.98 |
| K10/N100 | 0.56 | 0.90 | 1.00 |

Base fails last-value everywhere (0.41–0.58); **both Block and LoRA restore it**
(0.89–1.00) across every cell and both regimes (primacy + reversal). FVQ is high
in all conditions (base already gets first-value).

## 2. Interior positions: the roads diverge

Mean accuracy over interior positions (1 < k < N), dense cells:

| cell (×train N) | base | block | lora |
|---|---|---|---|
| K5/N20 (1×) | 0.05 | **0.93** | **0.54** |
| K10/N25 (1.25×) | 0.04 | **0.85** | **0.31** |
| K5/N50 (2.5×) | 0.02 | **0.83** | **0.19** |
| K10/N50 (2.5×) | 0.02 | **0.72** | **0.12** |
| K5/N75 (3.75×) | 0.02 | **0.73** | **0.11** |

## 3. Per-position shape (the real signature — means hide it)

Accuracy by position depth, K10/N50 (N=50):

| depth | base | block | lora |
|---|---|---|---|
| pos 1 (first) | 0.34 | 0.78 | 1.00 |
| ~10% | 0.02 | 0.44 | 0.43 |
| ~25% | 0.01 | 0.74 | 0.09 |
| ~50% | 0.00 | 0.73 | 0.00 |
| ~75% | 0.01 | 0.81 | 0.00 |
| ~90% | 0.04 | 0.82 | 0.15 |
| pos N (last) | 0.51 | 0.89 | 0.98 |

Three distinct signatures:
- **base = U-shape**: only first + last have signal; the entire middle is ~0.
- **Block = flat-high** across all depths (~0.7–0.9) → genuine per-position
  addressing (the `[Update k]` markers).
- **LoRA = boundary-anchored + interior decay**: near-perfect at the ends
  (first 1.00, last 0.98) and shallow offsets, collapses in the deep middle
  (0.00 at 25–75% depth), tiny recency uptick at ~90%.

LoRA's interior collapse **sharpens with N** (interior widens vs. its reach):

| depth | K5/N20 | K10/N25 | K10/N50 |
|---|---|---|---|
| ~25% | 0.82 | 0.50 | 0.09 |
| ~50% | 0.45 | 0.14 | 0.00 |
| ~75% | 0.27 | 0.06 | 0.00 |

## 4. Interpretation

- **Endpoints:** base fails; Block and LoRA both fix (convergent), across
  primacy and reversal regimes.
- **Interior:** base can't do it at all; **Block installs genuine general
  position-indexing** (flat across positions, holds as N grows); **LoRA only
  amplified end-anchored retrieval** — it reads near the boundaries and shallow
  offsets but has no true interior addressing, so the deep middle collapses and
  worsens with stream length.

This behaviorally **explains E1**: LoRA's intermediate retrieval is N-bounded and
decaying because it never learned position indexing — it learned to grab
first/last robustly. Block and LoRA are "two roads" that converge at the
endpoints but diverge completely in the interior. The **per-position curve** (not
the interior mean) is the figure: U-shape (base) vs flat (Block) vs
boundary-anchored-decay (LoRA).

## Caveats
- Single model (Qwen2.5-3B), ARB single-token, behavioral (no mechanistics).
- Operational note: the run took ~5–6 h on L4; the dense high-N cells dominate
  (Block cells slow because mid-accuracy positions run near the 200-trial cap).
- The mechanistic "convergent readout" result (logit-lens) is in
  `block_readout_results/`; this behavioral sweep is its behavioral counterpart
  and adds the interior-divergence finding.
