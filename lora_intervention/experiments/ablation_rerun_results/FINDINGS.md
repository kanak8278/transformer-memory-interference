# Stage-3C ablation re-run (n=200) — resolves Finding 001

**Run:** 2026-07-15, Colab L4, Qwen2.5-3B ± main LoRA, ARB single-token,
HF forward-hooks (no TransformerLens). Ablate the 8 promoter heads
(L32H3, L32H7, L31H12, L31H15, L32H10, L30H11, L32H0, L32H14) — the SAME set in
base and LoRA (paired), at two cells. n=200, deterministic paired seeds
(`stable_seed`), per-trial export → bootstrap 95% CIs (n_boot=2000, seed 17).
Ablation = zero each head's slice of the o_proj input; logit-lens =
final_norm→lm_head P(v_last) (same as run_logit_lens_lora_hf.py). Data:
`results.json` (per-trial).

## Pre-check: the ablation cell (K2/N5) is a near-success regime
Behavioral base accuracy (60 trials):
| cell | base FVQ | base CVQ |
|---|---|---|
| K2/N5 (paper's ablation cell) | 0.85 | **0.77** |
| K2/N50 | 0.90 | **0.40** |

At **K2/N5 the base model mostly gets CVQ right (0.77)** — the entire §7 mechanism
was probed at a barely-failing cell. K2/N50 is a genuine failure (0.40). This
motivated re-running at both.

## Result: LoRA amplification confirmed (resolves Finding 001)

L32 P(v_last) ablation Δ = ablated − normal (n=200; paired = LoRA−base per trial):

| cell | base Δ | LoRA Δ | paired (LoRA−base) |
|---|---|---|---|
| K2/N5 | −0.065 [−0.101, −0.032] | −0.147 [−0.182, −0.114] | **−0.082 [−0.108, −0.056]** |
| K2/N50 | +0.017 [+0.004, +0.032] | −0.213 [−0.250, −0.179] | **−0.230 [−0.267, −0.196]** |

- **Both paired CIs exclude zero** → ablating the promoter heads reduces L32
  P(v_last) **significantly more in LoRA than in base**. This is the paper's
  directional claim ("post-LoRA the heads carry more causal effect") — now
  supported at n=200 with pinned seeds, at BOTH a mild and a clear failure cell.
- The **local n=50 data** (which Finding 001 flagged as contradicting, base
  −0.377 > LoRA −0.278) was **unreliable** (small n, unpinned seeds); the proper
  re-run reverses it.
- **K2/N50 is the cleanest evidence:** base has essentially no v_last to remove
  (normal 0.02 — it fails CVQ), while LoRA's v_last (normal 0.38) is largely
  head-driven (Δ −0.21). The promoter heads causally produce the v_last signal
  that LoRA installs and base lacks.

Normal (unablated) L32 P(v_last): base 0.30 (K2/N5) / 0.02 (K2/N50); LoRA 0.39 /
0.38. Base normal at K2/N5 (0.30) matches the paper's TL logit-lens (0.34).

## Magnitude caveat
Absolute Δ magnitudes are smaller than the paper's stated −0.28/−0.40; the
HF-hook ablation removes less of the signal than the paper's TL ablation
(the normal baselines match, so it's the ablation operation, not the lens). The
**direction and significance** are what resolve Finding 001; for the paper,
report these n=200 numbers (or state the effect qualitatively) rather than the
old −0.28/−0.40.

## Downstream (L33–35) — redundancy is cell-dependent
- K2/N5 (easy): base ablation propagates to readout (L35 Δ −0.094), LoRA absorbs
  it (L35 Δ −0.007) → the "downstream redundancy" claim holds here.
- K2/N50 (hard): LoRA ablation *does* propagate (L35 Δ −0.110) — at the failure
  cell the promoters are the sole v_last source, so removing them hits the
  readout. Redundancy is a low-load phenomenon.

## Bottom line for §7.3
Keep the amplification claim (now confirmed at n=200), but (a) use these numbers,
(b) note the ablation cell is near-success and cite the K2/N50 result as the
genuine-failure confirmation, (c) scope the downstream-redundancy claim to low
load.
