# Gemma-3-4b E1 extrapolation — cross-family replication of Qwen

**Run:** 2026-07-15, Colab L4, `google/gemma-3-4b-it` ± main Gemma adapter
(r16/α32, attn-only q/k/v/o, trained to 1.42 ep on the same K/N grid as Qwen:
K≤10, N≤20). HF backend, sdpa, bf16, ARB single-token, Wilson early-stop
(min30/cap100, HW≤0.07). Data: `results.jsonl` (70 base + 70 lora rows).
Same script/grid/metric as the Qwen E1 (`../e1_results/`).

## Result: endpoints generalize to ~5×, interior is N-bounded & decaying

### Scan A — N-scan (K=10 fixed, N = 1×..5× of N_train=20)
| cell | FVQ base/lora | CVQ base/lora | IVQ-mean base/lora |
|---|---|---|---|
| K10/N20  | 1.00/1.00 | 0.52/0.98 | 0.10/0.49 |
| K10/N40  | 0.87/1.00 | 0.32/0.98 | 0.00/0.23 |
| K10/N60  | 0.90/1.00 | 0.19/1.00 | 0.00/0.08 |
| K10/N80  | 0.81/1.00 | 0.16/1.00 | 0.00/0.08 |
| K10/N100 | 0.88/1.00 | 0.11/1.00 | 0.00/0.05 |

### Scan B — K-scan (N=20 fixed, K = 1×..~5× of K_train=10; 46 = category cap)
| cell | FVQ base/lora | CVQ base/lora | IVQ-mean base/lora |
|---|---|---|---|
| K10/N20 | 1.00/1.00 | 0.52/0.98 | 0.10/0.49 |
| K20/N20 | 0.90/1.00 | 0.13/0.98 | 0.05/0.39 |
| K30/N20 | 0.83/0.98 | 0.12/0.95 | 0.04/0.36 |
| K40/N20 | 0.77/1.00 | 0.00/0.93 | 0.05/0.34 |
| K46/N20 | 0.88/0.98 | 0.04/0.94 | 0.03/0.36 |

## Reading
- **Base current-value suppression, load-dependent:** base CVQ collapses as load
  grows (N: 0.52→0.11; K: 0.52→0.04) while FVQ stays high — the paper's core
  asymmetry, reproduced in Gemma's base model.
- **LoRA endpoints generalize to 5× in BOTH axes:** LoRA CVQ 0.93–1.00 at every
  cell incl. N=100 (5×N) and K=46 (~5×K); FVQ ~1.00. The fix is robust far
  beyond the training grid.
- **Interior is N-bounded and decays:** LoRA IVQ-mean 0.49→0.05 as N grows, but
  stays ~0.34–0.39 across the K-scan (N fixed). **N is the killer, not K** — LoRA
  surfaced robust *endpoint* retrieval, not general position indexing.

## Cross-family verdict
Matches Qwen (Qwen LoRA CVQ 0.96–1.00, interior 0.56→0.07; Gemma LoRA CVQ
0.93–1.00, interior 0.49→0.05). The extrapolation frontier — endpoints
generalize, intermediates are N-bounded/decaying — is **architecture-general**,
not Qwen-specific. Supports the paper's "surfaced endpoint retrieval, not
position indexing" claim across model families.
