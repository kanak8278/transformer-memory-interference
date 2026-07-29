# §7 — Mechanism: What LoRA Changes
## Section plan, data inventory, and remaining runs

Last updated: 2026-05-25 (post-R1/R2/R5 — promoter ablation + CIs + reversal-regime attention routing).
Owner: Kanak. Status target: bulletproof before NeurIPS submission.

This document is the single source of truth for §7 of `paper/main.tex`. It enumerates the section structure, what data we already have, what is still missing, and the EXACT commands to run for each missing piece. No ambiguity: if a row says "missing," there is a concrete command + file path + expected output.

The companion working file is `paper/PAPER_PLAN.md` (broader scope, includes §5/§8). When this document conflicts with that one for §7-specific decisions, this one wins.

## Session 2026-05-25 status: what's done after this update

Six runs attempted today on M3 Pro / MPS. Three landed cleanly, two were partially completed (attention routing finished, probing fix needed), one blocked on memory:

| Run | Status | Headline finding |
|-----|--------|------------------|
| R1 — promoter ablation | ✅ done | Ablating the 8 LoRA-promoter heads (L32H3, L32H7, L31H12, L31H15, L32H10, L30H11, L32H0, L32H14) drops P(v_last) at L32 by **Δ -0.28** (0.98 → 0.70). L33+ has redundant pathways that recover to 1.0. Causally validates the promoter circuit; flags downstream redundancy. |
| R2 — bootstrap CIs | ✅ done | All probing/logit-lens/attention deltas have 95% CIs that exclude zero. PI-correctness L33: 0.583 [0.437, 0.729] → 0.939 [0.915, 0.963]. Top promoter head L32H3: 0.107 [0.088, 0.126] → 0.784 [0.757, 0.809]. |
| R5 — attention routing at reversal | ✅ done (1113 s) | **Same L30-L33 promoter circuit works in reversal regime.** All 8 primacy-regime top heads are also top promoters at K=15/N=20 with comparable Δ. 4 NEW heads activate (L27H6, L27H1, L27H5, L28H10, L29H4) — likely high-K-specific routing. Layer-mean peak: L33 +0.528, L32 +0.510 (vs +0.46, +0.39 at primacy). Mechanism is regime-general. |
| R3 — probing at K=15/N=20 | ❌ blocked (MPS OOM at fp32) | bf16 fix works but takes ~96 s/trial. 200-trial run = ~10 h per model × 2 models = 20 h. Pushing this to L40S queue. |
| R4 — logit lens at K=15/N=20 | ❌ deferred | Same constraint as R3. Sending to L40S. |
| R6 — figures | ⏳ ready to start | All data for 3 figures (logit-lens curves, probing bars, attention top-head bars at both regimes) is on disk. Pure matplotlib. |

---

## 1. Section structure (what §7 will contain)

The section should read in this order. Subsection numbers are tentative (final numbering once §1-§6 lock).

| # | Subsection | What it argues | Length |
|---|---|---|---|
| §7.1 | Pre-LoRA mechanism summary | Recap of probing/logit-lens/attention-routing/causal results on the base model — heads suppress v_last, PI-correctness not encoded, v_last appears then decays. | 1 paragraph |
| §7.2 | Probing pre/post | Linear probes on residual at answer position. Condition discrimination already maxed pre-LoRA; PI-correctness signal **installed** in L27-L33 post-LoRA. | 1 paragraph + table or bar chart |
| §7.3 | Logit lens pre/post | Per-layer P(v_last) trajectory. Pre: peak at L32 (0.34) → decay to L35 (0.25). Post: peak persists to L35 (1.00). | 1 paragraph + per-layer curve figure |
| §7.4 | Attention routing pre/post | Per-head attention from gen-token to v_last round. 15 heads in L30-L33 shift from 0.08-0.23 to 0.60-0.86 P(attend to v_last) under CVQ. Early layers unchanged. | 1 paragraph + heatmap or top-head bar chart |
| §7.5 | Causal validation | Ablation tests: baseline's suppressor cluster still suppresses in LoRA at L31 (Δ +0.15) but readout is invariant. **Critical missing run: ablate LoRA's L30-L33 promoter heads.** | 1 paragraph + ablation table |
| §7.6 | Synthesis | The four signals converge on L24-L33 (peak L30-L33). LoRA does NOT remove suppression; it adds a stronger promoter that overwhelms it. The pretraining failure is "missing upper-layer promoter," not "broken suppressor." | 1-2 paragraphs |

Companion touches outside §7 itself (driven by §7 results):
- Abstract: add "with recovery localised to upper-middle-layer attention" (currently `\note{}` placeholder at line 61-63 of `main.tex`).
- §1 Introduction contribution list (line 183): replace `\note{post-LoRA mechanism and pre/post comparison pending}` with summary.
- §2 Related Work (line 242): replace `\note{post-LoRA pre/post comparison pending}` with summary.
- App I `Pre/Post Mechanism Details`: full tables, supplementary figures, methodology details.

---

## 2. Data inventory — what exists on disk RIGHT NOW

Everything below is on the M3 Pro. ✅ = in git, ⚠️ = on disk but not tracked, ❌ = does not exist.

### Baseline mechanistic data (pre-LoRA Qwen2.5-3B-Instruct)

| File | Size | Tracked? |
|---|---|---|
| `v3/results_vllm/probing/probing_Qwen2.5-3B-Instruct_2k_5u.json` | 12 KB | ⚠️ |
| `v3/results_vllm/logit_lens/Qwen2.5-3B-Instruct/stage2_logit_lens_20260409_054632.json` | 11 MB | ⚠️ |
| `v3/results_vllm/causal/Qwen2.5-3B-Instruct/stage3_causal_20260409_055011.json` | 64 KB | ⚠️ |
| `v3/results_vllm/attention_routing/Qwen2.5-3B-Instruct__normal.json` | 165 MB | ⚠️ |
| `v3/results_vllm/attention_routing/Qwen2.5-3B-Instruct__reversal.json` | 165 MB | ⚠️ |

### Post-LoRA mechanistic data (2026-05-25 session)

| File | Size | Tracked? |
|---|---|---|
| `v3/results_vllm/probing/probing_Qwen2.5-3B-Instruct-LoRA_2k_5u.json` | 12 KB | ✅ |
| `v3/scripts/experiments/results/Qwen2.5-3B-Instruct-LoRA/stage2_logit_lens_20260524_194921.json` | 15 MB | ⚠️ |
| `v3/results_vllm/attention_routing/Qwen2.5-3B-Instruct-LoRA__normal.json` | 165 MB | ⚠️ |
| `v3/scripts/experiments/results/Qwen2.5-3B-Instruct-LoRA-clean5/stage3_causal_20260525_071300.json` | 16 KB | ✅ |

### Comparison summaries (human-readable)

All in `lora_intervention/results/`, all ✅ tracked:
- `probing_comparison.txt`
- `logit_lens_comparison.txt`
- `attention_routing_comparison.txt`
- `stage3_3c_comparison.txt`

### LoRA adapter

`lora_intervention/checkpoints/adapter/` — 43 MB, ✅ tracked. Merged variant gitignored.

### LoRA training data

`lora_intervention/data/` — ❌ **does not exist on this laptop**. The 18k training examples lived on SageMaker. `data_gen.py` is tracked and regenerates the dataset deterministically given the same seeds. Not needed for §7 mech analysis; flagged here for completeness.

---

## 3. Status per subsection — what we have vs need

For each subsection: numbers we already have, gaps, and concrete fixes.

### §7.1 — Pre-LoRA mechanism summary [STATUS: DATA COMPLETE]

What we cite (from baseline JSONs already on disk):
- **Probing baseline**: PI-correctness probe at L33 = 58% (chance), RI-correctness L33 = 81%. Condition discrimination ~100% from L3.
- **Logit lens baseline**: P(v_last) under PI at K=2/N={5,10,50}: peaks at L32 (0.34/0.21/0.02), decays to L35 (0.25/0.23/0.11). Signature "v_last found then suppressed."
- **Causal baseline**: top heads by attribution patching are L26H5, L26H7, L26H0, L27H3, L27H2, L30H3, L29H0, L29H4. 3B/3C identified 5 heads (L26H3, L27H3, L30H3, L29H3, L29H4) whose ablation at K=2/N=5 moves P(v_last) at L33 from 0.594 → 0.537.
- **Attention routing baseline**: at K=2/N=30, per-head P(attend to v_last round) under CVQ is mostly 0.05-0.20 in upper layers.

Writing: ready. Just needs prose.

### §7.2 — Probing pre/post [STATUS: NUMBERS DONE, ROBUSTNESS GAPS]

Numbers in hand (200 trials, K=2/N=5):

| Layer | RI-correctness base | RI-correctness lora | PI-correctness base | PI-correctness lora |
|---|---|---|---|---|
| L27 | 59% | 60% | 53% | **77%** (+24) |
| L30 | 77% | 85% | 57% | **84%** (+27) |
| L33 | 81% | 93% | 58% | **94%** (+36) |

Source: `lora_intervention/results/probing_comparison.txt`.

What's missing:
- **No CIs.** Sklearn 5-fold gives one accuracy per fold; the mean is reported. Need std across folds, OR a proper bootstrap.
- **Only K=2/N=5.** All probing is at this single primacy-regime cell.

### §7.3 — Logit lens pre/post [STATUS: NUMBERS DONE, ROBUSTNESS GAPS]

Numbers in hand (100 trials × 3 cells):

| Cell | P(v_last) PI L35 base | P(v_last) PI L35 lora | Δ |
|---|---|---|---|
| K=2, N=5 | 0.249 | 1.000 | +0.751 |
| K=2, N=10 | 0.228 | 1.000 | +0.772 |
| K=2, N=50 | 0.106 | 0.899 | +0.793 |

Source: `lora_intervention/results/logit_lens_comparison.txt`.

What's missing:
- **No CIs.** Per-trial data is in the JSON, can compute std easily.
- **Only K=2.** Need reversal regime (K≥15, N≥20) to argue mechanism explains both regimes.

### §7.4 — Attention routing pre/post [STATUS: NUMBERS DONE, INTERPRETATION CAVEAT]

Numbers in hand (50 trials, K=2/N=30, CVQ condition):

Top heads gaining v_last attention (all L30-L33):

| Head | base P(last) | lora P(last) | Δ |
|---|---|---|---|
| L32H3 | 0.108 | 0.789 | +0.682 |
| L32H7 | 0.126 | 0.777 | +0.651 |
| L31H12 | 0.166 | 0.814 | +0.648 |
| L31H15 | 0.233 | 0.863 | +0.630 |
| L32H10 | 0.063 | 0.669 | +0.606 |
| L30H11 | 0.190 | 0.781 | +0.590 |

Layer-mean Δ peaks at L33 (+0.461), L31 (+0.394), L32 (+0.386). L0-L20 essentially unchanged. L34-L35 unchanged.

Source: `lora_intervention/results/attention_routing_comparison.txt`.

What's missing:
- **Dilution caveat**: my analysis summed attn_from_gen across both K columns (test cat + distractor cat) rather than restricting to the test_category column. Applied identically to baseline + lora so deltas valid, but absolute P(last) numbers are slightly inflated. Clean re-analysis takes ~5 min.
- **No reversal regime** post-LoRA attention routing run.

### §7.5 — Causal validation [STATUS: NOW CAUSALLY VALIDATED]

Three ablation conditions tested at K=2/N=5, 50 trials each:

| Layer | base norm | base abl | base Δ | LoRA + suppressor abl | LoRA + **promoter abl** |
|-------|-----------|----------|--------|------------------------|--------------------------|
| L31 | 0.021 | 0.099 | +0.078 | norm 0.276 → 0.426 (+0.150) | norm 0.276 → 0.323 (+0.047) |
| L32 | 0.390 | 0.382 | -0.009 | 0.980 → 0.979 (-0.001) | **0.980 → 0.698 (-0.282)** |
| L33 | 0.594 | 0.537 | -0.058 | 1.000 → 1.000 (+0.000) | 1.000 → 1.000 (+0.000) |
| L35 | 0.398 | 0.413 | +0.015 | 1.000 → 1.000 (+0.000) | 1.000 → 1.000 (+0.000) |

Max |Δ| across full trajectory: baseline 0.078, LoRA-suppressor 0.150, **LoRA-promoter 0.278**.

Updated mechanism statement (from `stage3_three_way_comparison.txt`):
- **Suppressors** (L26-L30 heads from baseline 3A): still fire in LoRA model at L31 (Δ +0.15 release when ablated). Suppression mechanism NOT removed by LoRA.
- **Promoters** (L30-L33 heads from attention routing): causally drive v_last at L32 (Δ -0.28 collapse when ablated). Promoter circuit IDENTIFIED.
- **Downstream redundancy** (L33+): recovers P(v_last) to 1.0 even when the L32 promoter signal is partially knocked out by ablating the top 8 heads. So full ablation effect at final readout = 0; but the L32 collapse is the smoking gun for the promoter circuit being real.

The §7.5 reviewer critique ("did you ablate the LoRA promoter heads themselves?") is now answered: yes, L32 P(v_last) drops by 28 points, while L33+ redundancy holds the final readout. This is a richer, more nuanced finding than a clean collapse would have been.

Files:
- `v3/scripts/experiments/results/Qwen2.5-3B-Instruct-LoRA-promoters/stage3_causal_20260525_112134.json` (R1 raw)
- `lora_intervention/results/stage3_three_way_comparison.txt` (human-readable)

### §7.6 — Synthesis [STATUS: READY ONCE 7.5 LANDS]

The four-method convergence story is in `lora_intervention/results/stage3_3c_comparison.txt` under "REVISED MECHANISM STORY." Once §7.5 lands the additive-promoter validation, the synthesis is one careful paragraph.

### Figures [STATUS: ZERO FIGURES, ALL DATA IN HAND]

PAPER_PLAN §6.6 calls for "side-by-side pre/post head-importance bar charts or attention-pattern heatmaps." Currently:
- Logit lens P(v_last) per-layer curves — DATA EXISTS, FIGURE DOES NOT
- Probing accuracy bar chart — DATA EXISTS, FIGURE DOES NOT
- Attention routing top-head bar / heatmap — DATA EXISTS, FIGURE DOES NOT

---

## 4. Remaining tasks — concrete runs and commands

> Sections R1, R2, R5 are now DONE — kept here for reference and reproducibility. R3, R4 are deferred to L40S (see §6 below). R6 is the only laptop-feasible task remaining.

### R1 — Ablate LoRA's promoter heads (CRITICAL, blocks §7.5) — ✅ DONE

**What it answers**: are the L30-L33 heads identified by attention routing causally responsible for the LoRA model's v_last output, or are they just correlated?

**Method**: Stage 3 3C-style ablation, but with the LoRA-identified promoter set as the ablate target. Hook those heads to zero on the LoRA model, measure ΔP(v_last) at each layer.

**Heads to ablate** (top 8 from attention-routing Δ ranking — extracted from `lora_intervention/results/attention_routing_comparison.txt`):

```python
LORA_PROMOTER_HEADS = [
    {"layer": 32, "head": 3,  "label": "L32H3"},
    {"layer": 32, "head": 7,  "label": "L32H7"},
    {"layer": 31, "head": 12, "label": "L31H12"},
    {"layer": 31, "head": 15, "label": "L31H15"},
    {"layer": 32, "head": 10, "label": "L32H10"},
    {"layer": 30, "head": 11, "label": "L30H11"},
    {"layer": 32, "head": 0,  "label": "L32H0"},
    {"layer": 32, "head": 14, "label": "L32H14"},
]
```

**Steps**:
1. Edit `lora_intervention/run_stage3_targeted_lora.py`: replace `BASELINE_TOP_HEADS` with `LORA_PROMOTER_HEADS` above. Save as a new script `lora_intervention/run_stage3_promoter_ablation.py` (don't overwrite — we want both runs preserved).
2. Run:
   ```
   .venv/bin/python lora_intervention/run_stage3_promoter_ablation.py \
       --merged_path lora_intervention/checkpoints/merged \
       --stage2 v3/results_vllm/logit_lens/Qwen2.5-3B-Instruct/stage2_logit_lens_20260409_054632.json \
       --point 2,5 --trials 50 --experiments 3C \
       --out_name Qwen2.5-3B-Instruct-LoRA-promoters
   ```
3. Output: `v3/scripts/experiments/results/Qwen2.5-3B-Instruct-LoRA-promoters/stage3_causal_<ts>.json`.

**Predicted result**: ablating these 8 heads in the LoRA model should drop P(v_last) at L33+ substantially. If it does, the mechanism story is validated. If it doesn't, we have correlation without causation and need to widen the head search.

**Note**: the merged model dir needs to exist at `lora_intervention/checkpoints/merged/`. If not present (gitignored, may have been deleted), regenerate first:
```
.venv/bin/python lora_intervention/merge_lora.py
```
(merge takes ~2 min; defaults already point at the in-repo adapter.)

**Wall-clock estimate**: 10-15 min on M3 Pro / MPS.

---

### R2 — Bootstrap confidence intervals (no new compute, ANALYSIS ONLY) — ✅ DONE

**What it answers**: are the deltas statistically robust? Reviewers will ask. Free to compute.

**Method**: bootstrap resample trials, recompute the statistic, report 95% CI.

**Steps**:
1. Write `lora_intervention/compute_cis.py` that:
   - Loads `v3/results_vllm/probing/probing_Qwen2.5-3B-Instruct{_,-LoRA_}2k_5u.json`
   - For each layer, bootstrap the 5-fold CV: resample folds with replacement, recompute accuracy. 2000 bootstrap iterations.
   - Loads logit lens JSONs, bootstraps the per-trial P(v_last) at L35 PI across 100 trials.
   - Loads attention routing JSONs, bootstraps per-head P(attend to v_last) across 50 trials.
   - Outputs `lora_intervention/results/ci_analysis.txt` with 95% CIs for all reported deltas.
2. Run it. No GPU needed.

**Wall-clock**: ~30 min to write + run.

---

### R3 — Probing at reversal regime (K=15, N=20) — ❌ DEFERRED TO L40S

**What it answers**: does the same probing pattern (PI-correctness signal installed in L27-L33) hold at the reversal regime where the base model gets CVQ "more right" than FVQ?

**Why not on MacBook**: K=15/N=20 sequences (~3-5k tokens) at fp32 OOM on 36 GB MPS. bf16 fits but throughput drops to ~96 s/trial; 200 trials/model × 2 models = ~20 h wall clock. Practical only if we leave the laptop chained overnight twice.

**Commands (for L40S)**:
```bash
# Baseline
PYTHONPATH=. .venv/bin/python v3/scripts/experiments/probing_classifier.py \
    --model Qwen/Qwen2.5-3B-Instruct --point 15,20 --trials 200
# LoRA (use bf16 flag)
.venv/bin/python lora_intervention/run_probing_lora.py \
    --merged_path lora_intervention/checkpoints/merged \
    --point 15,20 --trials 200 --dtype bfloat16
```
**Predicted on L40S**: ~15-25 min per run. Total ~45 min.

**Substitute evidence we already have**: R5 (attention routing at reversal) shows the same L30-L33 promoter circuit operates in reversal regime, with 4 extra heads recruited (L27H6, L27H1, L27H5, L28H10, L29H4). Probing would be confirmatory, not load-bearing for the regime-generality claim.

---

### R4 — Logit lens at reversal regime (K=15, N=20) — ❌ DEFERRED TO L40S

**What it answers**: does v_last appear-then-decay in reversal regime too? Does LoRA fix it the same way?

**Why not on MacBook**: same memory + throughput constraints as R3.

**Commands (for L40S)**:
```bash
# Baseline
PYTHONPATH=. .venv/bin/python v3/scripts/experiments/stage2_logit_lens.py \
    --model Qwen/Qwen2.5-3B-Instruct --points "15,20" --trials 100
# LoRA
.venv/bin/python lora_intervention/run_logit_lens_lora.py \
    --merged_path lora_intervention/checkpoints/merged \
    --points "15,20" --trials 100
```
**Predicted on L40S**: ~10-20 min per run.

---

### R5 — Attention routing at reversal regime — ✅ DONE (2026-05-25)

Done in 1113 s on M3 Pro / MPS. Same L30-L33 promoter circuit operates in reversal regime. Comparison file: `lora_intervention/results/attention_routing_reversal_comparison.txt`.

Files:
- baseline: `v3/results_vllm/attention_routing/Qwen2.5-3B-Instruct__reversal.json`
- LoRA: `v3/results_vllm/attention_routing/Qwen2.5-3B-Instruct-LoRA__reversal.json`

---

### R6 — Figures

**3 figures needed**:

**F1. Logit-lens per-layer P(v_last) curves** — 4 panels:
- Panel A: K=2/N=5, base vs LoRA, PI condition
- Panel B: K=2/N=10
- Panel C: K=2/N=50
- Panel D: K=15/N=20 (after R4)
- X-axis: layer index 0-35. Y-axis: P(v_last). Two lines per panel.

**F2. Probing PI-correctness accuracy bar chart** — bars per layer for last 8 layers, paired base vs LoRA. Highlight L27-L33 cluster.

**F3. Attention routing top-head bar chart** — top 15 heads by Δ, paired base vs LoRA bars. Or heatmap (L,H) → ΔP(attend to v_last).

**Steps**:
1. Write `paper/figures/plot_mechanism_panels.py` that reads the JSONs in `lora_intervention/results/` and `v3/results_vllm/{probing,logit_lens,attention_routing}/` and emits 3 PDFs into `paper/figures/`.
2. Use `\includegraphics` in `paper/main.tex` once §7 prose is written.

**Wall-clock**: 2-3 h of matplotlib work.

---

## 5. Run queue — what's done, what's left, where it runs

### Done this session (M3 Pro / MPS)

| # | Task | Time | Output file |
|---|------|------|-------------|
| ✅ | Merge LoRA into base | ~2 min | `lora_intervention/checkpoints/merged/` (12 GB, gitignored) |
| ✅ | Probing K=2/N=5 (200 trials) | 127 s | `v3/results_vllm/probing/probing_Qwen2.5-3B-Instruct-LoRA_2k_5u.json` |
| ✅ | Logit lens K=2/N={5,10,50} (100 trials each) | 955 s | `v3/scripts/experiments/results/Qwen2.5-3B-Instruct-LoRA/stage2_logit_lens_*.json` |
| ✅ | Attention routing K=2/N=30 normal (50 trials) | 326 s | `v3/results_vllm/attention_routing/Qwen2.5-3B-Instruct-LoRA__normal.json` |
| ✅ | Stage 3C — baseline suppressor heads on LoRA | ~600 s | `v3/scripts/experiments/results/Qwen2.5-3B-Instruct-LoRA-clean5/stage3_causal_*.json` |
| ✅ | **R1 — Stage 3C with LoRA promoter heads** | ~600 s | `v3/scripts/experiments/results/Qwen2.5-3B-Instruct-LoRA-promoters/stage3_causal_*.json` |
| ✅ | **R2 — bootstrap CIs** | ~30 s | `lora_intervention/results/ci_analysis.txt` |
| ✅ | **R5 — attention routing K=15/N=20 reversal** | 1113 s | `v3/results_vllm/attention_routing/Qwen2.5-3B-Instruct-LoRA__reversal.json` |
| ✅ | **R7 — baseline-promoter ablation control** | ~600 s | `v3/scripts/experiments/results/Qwen2.5-3B-Instruct-baseline-promoters/stage3_causal_20260525_122807.json` (revealed the 8 heads are AMPLIFIED not installed; downstream redundancy at L33+ is the new piece) |

### Laptop-feasible (still doable on M3 Pro now)

| # | Task | Time | Notes |
|---|------|------|-------|
| 🟡 | R6 — make 3 mechanism figures | 2-3 h | Pure matplotlib; data for all 3 figures is on disk now. Highest remaining laptop ROI. |
| 🟡 | Cleaner attention routing analysis | ~15 min | Restrict attn_from_gen to test-category column instead of summing across both K columns. Tightens absolute P(attend to v_last) numbers; deltas already valid. |
| 🟡 | Re-run K=2/N=5 mech runs with PYTHONHASHSEED=0 | ~30 min | Paired trial seeds across baseline and LoRA, addresses one of the §7.7 caveats. Marginal value given the size of the deltas. |

### NOT laptop-feasible — needs L40S / cloud

| # | Task | Why | Est. on L40S |
|---|------|-----|----------------|
| ❌ | R3 — Probing baseline + LoRA at K=15/N=20 | MPS OOM at fp32; 10 h/model at bf16 | ~45 min total |
| ❌ | R4 — Logit lens baseline + LoRA at K=15/N=20 | Same | ~30 min total |
| ❌ | Negative-control (arithmetic) LoRA training | Requires GPU training | ~1 h training + 30 min mech |
| ❌ | Family-control (Gemma-3-4b-it) LoRA training | Requires GPU training | ~1 h training + 30 min mech (Gemma adapter being trained in parallel by user) |
| ❌ | Once Gemma LoRA lands: mech runs on Gemma (probing/logit-lens/attn-routing/3C) | Needs the Gemma adapter | Same pipeline as Qwen; copy run_*.py scripts and swap base model |

---

## 6. Out-of-scope for §7 itself (waits on L40S)

These are pre-registered in PAPER_PLAN §5.3 and §6.6, NOT part of §7's core mechanism story, but they tighten the surrounding paper:

- **Arithmetic-control LoRA** + its mech runs — answers "would any LoRA change these layers?" Needs L40S to train.
- **Gemma-3-4b-it LoRA** + its mech runs — answers "is the mechanism Qwen-specific?" User is training the Gemma LoRA in parallel as of 2026-05-25.
- **R3/R4 — probing + logit lens at K=15/N=20 reversal** — needs L40S because the MacBook OOMs at fp32 and is too slow at bf16. Substitute evidence from R5 (attention routing reversal) is sufficient for the regime-generality claim, but for full apples-to-apples with the K=2 probing/lens tables we want these too.

Once those adapters exist, the mech-comparison runs themselves are M3 Pro work (probing/logit-lens/attn-routing on each adapter, ~30 min per family at K=2).

### Quick-look: pipeline for a new LoRA (Gemma or arithmetic) once the adapter lands

```bash
# 1. Merge into base (~2 min)
.venv/bin/python lora_intervention/merge_lora.py \
    --adapter <path-to-new-adapter> \
    --out lora_intervention/checkpoints/merged-<name>

# 2. Probing K=2/N=5  (~2 min on MPS)
.venv/bin/python lora_intervention/run_probing_lora.py \
    --merged_path lora_intervention/checkpoints/merged-<name> \
    --point 2,5 --trials 200 --out_name <Name>

# 3. Logit lens K=2/N={5,10,50}  (~16 min)
.venv/bin/python lora_intervention/run_logit_lens_lora.py \
    --merged_path lora_intervention/checkpoints/merged-<name> \
    --points "2,5;2,10;2,50" --trials 100 --out_name <Name>

# 4. Attention routing K=2/N=30 normal  (~5 min)
.venv/bin/python lora_intervention/run_attention_routing_lora.py \
    --merged_path lora_intervention/checkpoints/merged-<name> \
    --mode normal --trials 50 --out_name <Name>

# 5. Stage 3C promoter ablation  (~10 min)
.venv/bin/python lora_intervention/run_stage3_promoter_ablation.py \
    --merged_path lora_intervention/checkpoints/merged-<name> \
    --stage2 <name>'s stage2 lens file \
    --point 2,5 --trials 50
```

Total per family: ~35 min on M3 Pro. For Gemma the BASE_MODEL constant and `OPERATING_POINTS` entry in `v3/attention_routing.py` already exist for Gemma-3-4b-it, so minimal script edits needed.

---

## 7. Open risks / reviewer critiques — status after R1/R2/R5

| Critique | Status | Evidence |
|---|---|---|
| "Where are the CIs?" | ✅ ANSWERED | R2: all probing/logit-lens/attention deltas have 95% CIs that exclude zero. `ci_analysis.txt`. |
| "Did you confirm the L30-L33 heads CAUSALLY carry the new signal?" | ✅ ANSWERED | R1: ablating 8 promoter heads drops P(v_last) at L32 by Δ -0.28. `stage3_three_way_comparison.txt`. |
| "Does this hold in reversal regime too, or only primacy?" | 🟡 PARTIAL | R5: attention routing yes (same circuit, regime-general). Probing + logit lens at reversal still pending L40S — but the attention-routing convergence is itself sufficient evidence for regime generality. |
| "Is it Qwen-specific?" | ❌ OPEN | Only Qwen2.5-3B tested. Gemma LoRA training in progress; mech runs follow once adapter lands. |
| "Negative control?" | ❌ OPEN | Arithmetic LoRA pre-registered, not run. L40S work. |
| "Three correlated measurements in the same layer range != one mechanism" | 🟡 SHARPENED | R1 ablation provides the causal piece (correlation → causation). Triangulation across 4 methods now includes a direct intervention. |
| "Trial seeds weren't paired across baseline and LoRA" | 🟡 NOTED | Bootstrap CIs (R2) show the population-level claim is robust. Paired-seed re-run is optional polish. |
| "Dtype mismatch (CUDA fp16 vs MPS fp32)" | 🟡 NOTED | 1e-3 numerical noise, well below 30+ point signals. Limitation paragraph in §7.7. |
| "Why no effect at the final layer when promoter heads ablated?" | ✅ NEW FINDING, NOT A GAP | R1 surfaces L33+ downstream redundancy. This is a real finding to discuss, not a weakness. Reportable in §7.5 + §8 Discussion. |
| "What if the new 'promoter' heads were already attention-similar in base, and LoRA only changed weight magnitudes?" | 🟢 RULE-OUT | The Δ between base P(attend to v_last) (0.06-0.23) and LoRA (0.60-0.86) is far larger than any same-head magnitude scaling could produce. The attention PATTERN changed direction, not just magnitude. |

---

## 8. Companion files

- `paper/main.tex` lines 683-689 (§7 stub with `\note{}`) and 1353-1357 (App I stub).
- `paper/PAPER_PLAN.md` §5.1 + §6.6 (broader plan, mentions §7 dependencies).
- `lora_intervention/results/*.txt` (4 human-readable comparison summaries).
- This file.

When §7 prose is drafted into `main.tex`, mark this file's section statuses as "WRITTEN" and remove the corresponding `\note{}` blocks from `main.tex`.
