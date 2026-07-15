# Gemma-3-4b Block-vs-LoRA readout — cross-family replication

**Run:** 2026-07-15, Colab L4, `google/gemma-3-4b-it` ± main Gemma adapter.
Logit-lens P(v_last) per layer (final-norm→lm_head, skip fold-in at last layer)
+ behavioral acc, 3 conditions (base_plain / base_block / lora_plain), 100 trials.
Gemma-3-4b = 34 layers (readout at L33). Same script as Qwen (`../block_readout_results/`).
Data: `results_K2N30.json`, `results_K10N50.json`.

## CVQ P(v_last) — both fixes converge at a shared late readout

### K2/N30 (low-key: Gemma base only mildly suppressed)
| condition | behav_acc | P(v_last) final | peak |
|---|---|---|---|
| base_plain | 0.67 | 0.67 | L33 |
| base_block | 0.92 | 0.93 | L33 |
| lora_plain | 1.00 | 1.00 | L33 |

### K10/N50 (genuine failure cell — the clean test)
| condition | behav_acc | P(v_last) final | peak |
|---|---|---|---|
| base_plain | 0.30 | 0.29 | L33 |
| base_block | 0.92 | 0.92 | L33 |
| lora_plain | 0.98 | 0.98 | L33 |

Trajectory (K10/N50): first ~20 layers ~0 for all three; base weakly rises to
0.29, while Block/LoRA rise to 0.92/0.98 — **all peaking at the same layer L33.**

## Reading
- **Convergent readout, divergent routing:** base_block and lora_plain produce
  near-identical high P(v_last) at the SAME late readout locus (L33/34), while
  base_plain stays suppressed — cleanest at the K10/N50 failure cell (base 0.29).
- **Title mechanism holds cross-family:** base *tracks* v_last (nonzero, rises to
  0.29) but *suppresses* it; both interventions release the same readout via
  different upstream triggers (Block markers vs LoRA weights).
- **Cell dependence:** at low-key K2/N30 Gemma's base is only mildly suppressed
  (0.67) — Gemma is more current-value-robust when keys are few; K10/N50 is the
  genuine-failure cell (consistent with E1: Gemma base CVQ collapses with load).

## Cross-family verdict
Matches Qwen (base ~0.35–0.50 → Block & LoRA ~0.98 at L35). Readout layer differs
(Gemma L33/34 vs Qwen L35/36, both near-final) but the pattern is identical:
**convergent readout at a shared late locus from a failing base.** Architecture-
general support for "tracked but suppressed" + "two roads, one readout."
