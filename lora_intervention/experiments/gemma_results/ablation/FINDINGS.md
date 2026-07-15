# Gemma-3-4b stage-3C ablation — cross-family amplification confirmed

**Run:** 2026-07-15, Colab L4, `google/gemma-3-4b-it` ± main Gemma adapter.
HF forward-hooks (no TransformerLens), bf16, sdpa. Ablate the 8 Gemma promoter
heads (top-8 FVQ−CVQ from `head_analysis_report.md`: **L23H3, L23H1, L23H0,
L23H6, L23H7, L17H5, L17H0, L29H4**) — the SAME set in base and LoRA (paired),
zeroing each head's slice of the `o_proj` input (`head_dim=256`, GQA-correct).
n=200, deterministic paired seeds (`stable_seed`), bootstrap 95% CIs (n_boot=2000).
Logit-lens P(v_last) at CVQ position, readout layers L30–L33 (Gemma readout=L33).
Data: `results.json` (per-trial). Script: `hf_ablation_stage3_gemma.py`.

## Precheck (behavioral, 60 trials)
| cell | base FVQ | base CVQ |
|---|---|---|
| K2/N30 (mild) | 0.98 | **0.63** |
| K10/N50 (failure) | 0.93 | **0.25** |

Base fails current-value at the load cell (K10/N50 CVQ 0.25); K2/N30 is the mild
contrast. LoRA restores both (see behavioral/block results).

## Result: LoRA amplification confirmed at every layer, both cells
Δ = ablated − normal P(v_last); paired = (LoRA − base) per trial.

### K2/N30 (mild)
| layer | base Δ | lora Δ | **paired (LoRA−base)** |
|---|---|---|---|
| L30 | −0.212 | −0.586 | **−0.374 [−0.434, −0.313]** |
| L32 | −0.373 | −0.801 | **−0.427 [−0.497, −0.361]** |
| L33 (readout) | −0.295 | −0.824 | **−0.529 [−0.601, −0.458]** |

### K10/N50 (genuine failure — cleanest)
| layer | base Δ | lora Δ | **paired (LoRA−base)** |
|---|---|---|---|
| L30 | −0.073 | −0.684 | **−0.611 [−0.650, −0.566]** |
| L32 | −0.096 | −0.871 | **−0.775 [−0.824, −0.721]** |
| L33 (readout) | −0.189 | −0.927 | **−0.738 [−0.794, −0.677]** |

**Every paired CI excludes zero** → ablating the promoter heads reduces P(v_last)
significantly more under LoRA than base, at both cells and all readout layers.
This is the paper's amplification claim, confirmed in a second model family.

## Reading
- **K10/N50 is the cleanest evidence:** base normal P(v_last) is low (0.11–0.21,
  base fails CVQ) and ablation barely changes it (Δ −0.07 to −0.19), while LoRA's
  v_last (0.74–0.98) is **almost entirely head-driven** — ablation removes 0.68–0.93.
  The L23-cluster heads causally produce the v_last signal LoRA installs and base
  lacks.
- **Amplification is even larger than Qwen:** Qwen paired Δ was −0.082 (K2/N5) /
  −0.230 (K2/N50) at L32; Gemma is −0.427 (K2/N30) / −0.775 (K10/N50). Gemma's
  promoters carry a larger share of LoRA's readout.
- **Upstream cluster feeds the late readout:** Gemma's promoters are mid-depth
  (L23), well upstream of the L33 readout, yet ablating them collapses the L33
  v_last readout under LoRA (−0.93). Different topology from Qwen (late L30–33),
  same causal role.

## Cross-family verdict
Confirms the §7 amplification mechanism cross-family: LoRA amplifies the causal
contribution of pre-existing promoter heads to the current-value readout.
Direction + significance match Qwen (Finding 001 resolution); the Gemma effect is
stronger. Completes all 4 session-new experiments as two-family results.

## Note on head topology (new cross-family observation)
Qwen promoters: L30–L33 (late, at the readout). Gemma promoters: L23 (mid-depth),
upstream of the L33 readout. So the *readout locus* converges late in both
families, but the *promoter cluster* sits at different depths — the mechanism is
the same (amplify promoter→readout causal path), its spatial layout is not.
