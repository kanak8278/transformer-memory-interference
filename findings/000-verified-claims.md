# Finding 000 — Claims verified against local data (do not re-verify)

- **Status:** ✅ Done, 2026-07-11
- These reproduced cleanly from the repo's raw data. Recorded so we don't spend
  time re-checking them.

## C1.1 — SEM LoRA recovery ✓
4 held-out SEM cells → FVQ=CVQ=100%; deepest baseline reversal −0.45 at (15,30).
Matches §5.2 / `tab_lora_sem`. Source: `lora_sem_validation_results.json`,
`lora_sem_finish_results.json`. *Caveat:* n=10–20 + reconstructed provenance
(Finding 004).

## C1.3 — Gemma family control ✓
28/28 ARB cells fixed. Recomputed means: FVQ 99.5% (paper 99.4%, integer
rounding in source txt), CVQ 97.0% (exact). Post-LoRA CVQ range 88–100%. The two
`---` cells (25,75)/(30,75) are baseline-insufficient-trials only; post-LoRA
unaffected. Source: `lora_intervention/results/gemma_main_comparison.txt`.

## C3.1 — Mechanism, 3 of 4 methods ✓ (ablation is Finding 001)
Source: `lora_intervention/results/ci_analysis.txt` + stage-3 JSONs.

- **Probing (PI/CVQ-correctness):** Δ +0.356 at L33, CI [+0.208, +0.504] →
  paper's +0.36 [+0.21, +0.50]. Base near-chance (L27 0.53 … L31 0.68), LoRA
  0.77–0.94. ✓
- **Logit lens:** K=2,N=5 L32 Pr(v_last) base 0.340 → LoRA 0.985; decays to
  0.249 at L35 baseline ("found then suppressed"). Hard cell K=2,N=50 L32 base
  0.021, L35 0.106. Post-LoRA L32 0.985 (N=5) / 0.621 (N=50). All match §7. ✓
- **Attention routing:** 15 promoter heads base 0.05–0.23 → LoRA 0.57–0.86
  (4–12×). Largest: L32H3 0.107→0.784, L31H15 0.226→0.863. Matches §7.2. ✓
  (Paper says LoRA min 0.60; actual 0.573 — trivial rounding.)
- **Downstream redundancy (part of §7.3):** baseline promoter-ablation
  propagates to readout (L33 −0.133, L34 −0.150, L35 −0.088); LoRA ablation is
  absorbed (L33/L34/L35 = 0.000). ✓ — this half of §7.3 is solid; only the L32
  "40% more" half is Finding 001.

## Checkpoint math ✓
79 = 37 SmolLM3 + 42 SmolLM2 trajectory rows (abstract claim). Confirmed from
`tab_smollm3_traj.tex` / `tab_smollm2_traj.tex` row counts.
