# Gemma-3-4b behavioral 3-way sweep — cross-family replication of Qwen

**Run:** 2026-07-15, Colab L4, `google/gemma-3-4b-it` ± main Gemma adapter.
base_plain / base_block / lora_plain × 8 cells; dense cells sweep every position,
alternate on N≥50, endpoints cells probe only pos 1 and N. Same
seeds/dataset/metric as the Qwen sweep (`../../behavioral_block_lora_results/`),
Wilson early-stop (min30/cap200, HW≤0.07), multi-token greedy + prefix/contains
match. Length-aware micro-batching (fix for K10/N100 OOM). All 8×3 = 24
conditions complete. Data: `results.json`.

## 1. Endpoints (FVQ=first / CVQ=last): both fixes converge
| cell | mode | base F/C | block F/C | lora F/C |
|---|---|---|---|---|
| K5N20  | dense | 1.00/0.72 | 1.00/0.95 | 1.00/1.00 |
| K10N25 | dense | 0.98/0.44 | 1.00/0.98 | 1.00/1.00 |
| K10N50 | dense | 0.90/0.32 | 0.98/0.95 | 1.00/0.97 |
| K5N50  | alternate | 0.95/0.58 | 1.00/1.00 | 1.00/1.00 |
| K5N75  | alternate | 0.97/0.43 | 0.97/0.95 | 1.00/1.00 |
| K2N30  | endpoints | 1.00/0.61 | 1.00/0.94 | 1.00/1.00 |
| K20N20 | endpoints | 0.87/0.22 | 0.91/0.94 | 1.00/1.00 |
| K10N100| endpoints | 0.91/0.10 | 0.97/0.94 | 1.00/0.94 |

Base CVQ collapses with load (0.72→0.10) while FVQ stays high; **both Block and
LoRA restore last-value (0.94–1.00) at every cell**, including the reversal
regime (K20N20). Mirrors Qwen exactly.

## 2. Interior mean (1<k<N) — the divergence
| cell | mode | base | block | lora |
|---|---|---|---|---|
| K5N20  | dense (1×N) | 0.05 | 0.97 | 0.40 |
| K10N25 | dense (1.25×) | 0.02 | 0.93 | 0.24 |
| K10N50 | dense (2.5×) | 0.01 | 0.66 | 0.11 |
| K5N50  | alternate (2.5×) | 0.02 | 0.91 | 0.14 |
| K5N75  | alternate (3.75×) | 0.01 | 0.80 | 0.10 |

- **base ≈ 0** (U-shape: only endpoints carry signal).
- **Block flat-high (0.66–0.97)** → genuine general position indexing, holds as N grows.
- **LoRA boundary-anchored, decays with N (0.40→0.10)** → amplified end-anchored
  retrieval, no true interior addressing.

## 3. Per-position signature — K10N50 (the money figure)
| depth | base_plain | base_block | lora_plain |
|---|---|---|---|
| first  | 0.90 | 0.98 | 1.00 |
| ~10%   | 0.00 | 0.78 | 0.31 |
| ~25%   | 0.00 | 0.82 | 0.09 |
| ~50%   | 0.00 | 0.44 | 0.00 |
| ~75%   | 0.00 | 0.60 | 0.00 |
| ~90%   | 0.00 | 0.80 | 0.06 |
| last   | 0.32 | 0.95 | 0.97 |

Three distinct shapes: **base U**, **Block flat**, **LoRA boundary-decay** — the
headline figure, identical to Qwen.

## Cross-family verdict
Qwen K10N50 interior means were base ~0.02 / Block ~0.72 / LoRA ~0.12; Gemma-3-4b
gives **0.01 / 0.66 / 0.11** — near-identical. "Endpoints converge, interior
diverges; Block installs position indexing, LoRA only amplifies end-anchored
retrieval (N-bounded, decaying)" holds across model families. Behaviorally
explains Gemma E1 (LoRA interior N-bounded). The per-position curve is the figure.

## Caveats / notes
- Endpoint-mode cells (K2N30, K20N20, K10N100) have no interior positions by
  design — they confirm endpoint convergence across regimes (incl. reversal).
- Run recovered from two mid-run issues (torchao 0.10.0 crash on adapter load;
  K10/N100 CUDA OOM at fixed batch-16) — resumed from per-cell checkpoints, no
  data lost. See `../logs/main_run*.log`.
