# Finding 001 — Stage-3C L32 ablation: paper claim contradicted by local data

- **Severity:** 🔴 Critical (affects a headline causal claim in §7)
- **Status:** ✅ **RESOLVED 2026-07-15** via n=200 HF-hook re-run (pinned seeds,
  per-trial CIs), K2/N5 + K2/N50. **The paper's amplification DIRECTION is
  confirmed** (LoRA ablation Δ > base, paired CI excludes zero at both cells);
  the local n=50 that contradicted it was noise (small n, unpinned seeds). See
  `lora_intervention/experiments/ablation_rerun_results/FINDINGS.md`.
  Action left for the paper: swap the old −0.28/−0.40 magnitudes for the n=200
  numbers (paired K2/N5 −0.082 [−0.108,−0.056], K2/N50 −0.230 [−0.267,−0.196]),
  note K2/N5 is a near-success cell (base CVQ 0.77) and cite K2/N50 as the
  genuine-failure confirmation, and scope the downstream-redundancy claim to low
  load. Details below are the original diagnosis.
- **Related task:** C3.1 (verify), C3.4 (re-obtain mechanism files)
- **Verified:** 2026-07-11 (diagnosis); 2026-07-15 (resolved via re-run)

## Paper claim

§7.3 "Two LoRA changes: amplification and downstream redundancy" (main.tex
~L604–610):

> "ablation drops L32 Pr(v_last) by **−0.28** in baseline (95% CI [−0.34, −0.22],
> n=200) and by **−0.40** in LoRA (95% CI [−0.46, −0.33]). The paired difference
> is **+0.12** (95% CI [+0.02, +0.21], excludes zero) — post-LoRA the same eight
> heads carry **∼40% more** causal effect at L32."

Table `figures/tab_ablation_fourway.tex` (rows 3–4, L32 column):
> +LoRA + 8 promoter heads: **−0.396**
> Baseline + 8 promoter heads (control): **−0.277**

Narrative: LoRA *amplifies* the L32 causal effect (LoRA magnitude > baseline).

## Actual data (local)

Source JSONs (n=50, `trials: 50` in config), v_last row, Δ = ablated − normal at L32:

| Condition | L32 Δ | source |
|---|---|---|
| LoRA + 8 promoter heads | **−0.278** | `v3/scripts/experiments/results/Qwen2.5-3B-Instruct-LoRA-promoters/stage3_causal_20260525_112134.json` |
| Baseline + 8 promoter heads | **−0.377** | `v3/scripts/experiments/results/Qwen2.5-3B-Instruct-baseline-promoters/stage3_causal_20260525_122807.json` |

The source analysis `lora_intervention/results/stage3_four_way_comparison.txt`
(also 50 trials) matches exactly (−0.278 / −0.377) and concluded only:
> "The L32 Δ in baseline (−0.38) is comparable to the L32 Δ in LoRA (−0.28) …
> these heads ALSO carry v_last in baseline" (i.e., **pre-existence**, not
> amplification-at-L32).

## How they differ

1. **Ordering is flipped.** Local n=50: baseline (−0.377) > LoRA (−0.278) in
   magnitude. Paper: LoRA (−0.40) > baseline (−0.28). The paper's paired
   difference +0.12 ("LoRA carries ~40% more") is *reversed* in the local data
   (there LoRA carries ~26% LESS at L32).
2. **Suspected label swap.** Paper's baseline value −0.277 is numerically
   ~identical to the local LoRA value −0.278.
3. **Missing data.** Paper labels these rows n=200; the local JSONs are n=50.
   The n=200 run is not in the repo. `ci_analysis.txt` explicitly states the
   per-trial arrays needed for the ablation CIs "are not preserved," so the
   paper's [−0.34,−0.22] / [−0.46,−0.33] / [+0.02,+0.21] CIs come from a re-run
   we don't have.

## What is NOT at risk (verified from the same local data)

- Heads pre-exist as **causal partial v_last carriers**: baseline ablation
  −0.377 is large → confirmed.
- **Downstream redundancy is real:** baseline ablation propagates
  (L33 −0.133, L34 −0.150, L35 −0.088); LoRA ablation is absorbed
  (L33/L34/L35 = 0.000). This is the §7.3 redundancy claim, and it holds.
- Attention-routing amplification (4–12×) is independently verified (Finding 000).

So the broad "amplify pre-existing promoters + install downstream redundancy"
story survives. The single fragile sentence is "post-LoRA the 8 heads carry
~40% MORE causal effect at L32 (paired +0.12)."

## Root-cause hypothesis

Between the n=50 and (claimed) n=200 runs, either (a) the effect genuinely
flipped ordering — meaning it sits near the noise floor and is n-sensitive
(the four-way txt warns seeds weren't paired, `PYTHONHASHSEED` unpinned), or
(b) a transcription/label error when the table was hand-written ("Hand-written
from … stage3_four_way_comparison.txt" per the table comment).

## Resolution (pick one)

1. **Recover** the n=200 per-trial stage-3 promoter/baseline ablation data from
   the GPU box, confirm −0.28/−0.40 and regenerate CIs.
2. **Re-run** `v3/scripts/experiments/stage3_causal.py` for both conditions at
   n=200 with `PYTHONHASHSEED` pinned and per-trial export enabled (~30 min per
   the source txt), then recompute paired Δ + bootstrap CIs.
3. If neither reproduces LoRA > baseline at L32, **rewrite the claim** to what
   the data supports: heads are pre-existing causal carriers of comparable L32
   magnitude in both models, and LoRA's novel contribution is the *downstream
   redundancy* (which is robustly verified), not L32 gain.

**Blocker note:** do not build new mechanism experiments assuming +0.12 holds.
