# New Results + Next Steps (session summary, 2026-07-15)

Branch: **aaai-prep** (all work committed & pushed; head `9274167`). No Colab VMs
running. Read this + `AAAI_PREP_PLAN.md` + `findings/` to resume.

Context: expanding "Tracked but Suppressed" for AAAI. This session ran 4 GPU
experiments (Colab L4, Qwen2.5-3B, HF backend) and resolved the one integrity
blocker. Operational notes (Colab teardowns kill the keep-alive daemon; HF Xet
download needs a retry-loop; monitor via `colab_fetch.py`, never re-`exec`) are in
the global memory `colab-cli-stability` and `experiments/E1_COLAB_SETUP.md`.

---

## The 4 new results (all support the paper's narrative)

### 1. E1 — LoRA extrapolation frontier
`lora_intervention/experiments/e1_results/` (`results_multiples.jsonl`, `E1_FINDINGS.md`)
- Grid = multiples of training maxima (K≤10, N≤20), from the (K10,N20) corner.
- **Endpoints (FVQ/CVQ) generalize to 5× in both K and N** (LoRA CVQ 0.96–1.00).
- **Intermediate positions are N-bounded and decay** (mean 0.56→0.07 as N grows);
  **N is the killer, not K**.
- Verdict: LoRA surfaced robust *endpoint* retrieval, not general position
  indexing.

### 2. Block-vs-LoRA readout (mechanistic, logit-lens)
`lora_intervention/experiments/block_readout_results/` (`results_K2N30.json`,
`results_K10N50.json`, `FINDINGS.md`)
- 3 conditions (base_plain / base_block / base+LoRA), K2/N30 and K10/N50.
- **Convergent readout, divergent routing**: base_block and lora_plain produce
  near-identical per-layer P(v_last) trajectories (both → ~0.98 at L35) while
  base_plain stays suppressed (~0.35–0.50). Routing differs (block_locus).
- Ties to the title: base *tracks* v_last but *suppresses* it; both fixes release
  the same readout via different upstream triggers.
- Note: cross-format activation patching is ill-defined (prompts not aligned) —
  logit-lens is the right evidence; don't add patching here.

### 3. Behavioral Block-vs-LoRA sweep (24 conditions)
`lora_intervention/experiments/behavioral_block_lora_results/` (`results.json`
per-position, `FINDINGS.md`)
- **Endpoints converge** (base CVQ 0.41–0.58 → Block 0.89–1.00, LoRA 0.98–1.00),
  all 8 cells, both primacy + reversal regimes.
- **Interior diverges** (the money result): per-position shape is
  **base = U** (ends only), **Block = flat-high** (general position indexing),
  **LoRA = boundary-anchored, decaying into the deep middle and worsening with N**.
- Behaviorally explains E1. Headline figure = the per-position curve (not the mean).

### 4. Stage-3C ablation re-run — **resolves Finding 001**
`lora_intervention/experiments/ablation_rerun_results/` (`results.json` per-trial,
`FINDINGS.md`); `findings/001` marked ✅ resolved.
- n=200, pinned paired seeds, HF-hook ablation of the 8 promoter heads, K2/N5 + K2/N50.
- **SUPPORTS the paper's amplification claim (direction + significance):** ablating
  hurts LoRA more than base — paired (LoRA−base) L32: K2/N5 **−0.082 [−0.108,−0.056]**,
  K2/N50 **−0.230 [−0.267,−0.196]** (both CIs exclude zero).
- The local n=50 that contradicted it was **noise**. K2/N50 is the cleaner proof
  (base has no v_last; LoRA's is head-driven).
- Pre-check surfaced: **K2/N5 is a near-success cell (base CVQ 0.77)**; K2/N50 fails
  (0.40).
- **Does NOT reproduce the exact −0.28/−0.40 magnitudes** (HF-hook ablation gentler
  than TL; baselines match). → paper must swap in the n=200 numbers.

---

## Findings ledger (`findings/`)
- **001** ablation contradiction — ✅ RESOLVED (see above).
- **002** dense-IVQ prose overstates ("near-ceiling at every position") vs table — OPEN (prose fix; now also backed by result #3/#1).
- **003** LoRA held-out table shows 16 of 28 cells; "16/16" cross-family should be 28/28 — OPEN (script + caption fix; data fine, min 92.5%).
- **004** trial counts small/undocumented (SEM 10–20, dense-IVQ 20, old ablation 50-vs-200) — OPEN (caption fixes).
- **005** leftover "RI" in 4 appendix spots (code/JSON keep RI/PI keys) — OPEN (text fix + release naming note).
- **000** verified claims (don't re-check).

---

## NEXT STEPS (prioritized)

### A. Paper drafting from the 4 new results (highest value — do while fresh)
1. **§ figure + paragraph: interior divergence** (result #3) — the per-position
   curve (base U / Block flat / LoRA boundary-decay) is the headline. Pair with
   E1 (#1) as its extrapolation counterpart.
2. **§7 update: "two roads, one readout"** (result #2) — reframe the softened
   "operationally distinct" into "convergent readout, divergent routing."
3. **§7.3 rewrite** (result #4): swap −0.28/−0.40 → n=200 paired numbers; note
   K2/N5 is near-success and cite K2/N50; scope downstream-redundancy to low load.

### B. Mechanical text fixes (findings 002–005) — batch pass on `main.tex`
- 002 soften dense-IVQ prose; 003 fix 16→28 cell labels + `tab_lora_cross_family`
  "16/16"→"28/28" (rerun `generate_lora_tables.py` or re-caption); 004 add trial
  counts to captions; 005 purge 4 "RI" mentions + add RI=FVQ/PI=CVQ release note.

### C. Optional strengthening (GPU, only if time)
- **Gemma cross-family** for E1 + block (behavioral + mechanistic) — DEFERRED by the
  HF-Xet download blocker; run off-Colab or when Xet is healthy (see finding notes).
- **MLP-LoRA full eval** (App L is a 5-cell preview) — retrain (~100 steps) + full
  28-cell eval; adapter not local.

### D. AAAI port (Group D in the plan) — last
- Confirm AAAI-27 CFP (deadline, page limit, template, reproducibility checklist).
- ACL→AAAI template conversion (the EMNLP desk-reject was formatting — triple-check).
- Page-budget compression; reframe intro/discussion for agents audience; decide on
  citing arXiv preprint (AAAI has no ACL-style bar).

### Recommended order
A (draft while results are fresh) → B (quick, mechanical) → D (port) ; C only if
time/GPU allow. Finding 001 (the integrity blocker) is already closed.
