# Paper Plan — Narrative B

Living document. Update as items resolve.
Last updated: 2026-05-23.

## 1. Headline finding (confirmed)

> **18 trillion pretraining tokens leave the FVQ-CVQ asymmetry intact in
> Qwen2.5-3B-Instruct. 18,000 targeted LoRA examples close it completely
> across held-out cells, datasets (ARB → SEM), and ~10× context-length
> extrapolation.**

Citation for the 18T number: Qwen Team, *Qwen2.5 Technical Report*,
arXiv:2412.15115, 2025. The technical report states the Qwen2.5 series
(all sizes) is trained on 18T tokens.

## 2. Narrative — confirmed direction

**Narrative B**: the paper is about a measurable, recoverable
skill-coverage gap in pretraining. The FVQ-CVQ asymmetry is the
instrument; the LoRA recovery is the proof that the capacity is
architectural and the gap is curricular.

What this displaces:
- "Characterize a regime-dependent failure" is no longer the
  contribution; it's the setup.
- The reversal-predictor blocker dissolves: both regimes are unlearned
  retrieval heuristics; LoRA fixes both.

What needs to land in §1 and the abstract:
- The magnitude gap (1B : 1 ratio of pretraining tokens to LoRA examples).
- Two independent recoveries (format + LoRA) → robust evidence the skill
  is latent.
- The mechanism story (pre vs post LoRA) connects behavior to
  representation across the intervention.

## 3. Open decisions

- [ ] **Title.** Candidates:
  - *The Latent Capacity: Pretraining Doesn't Develop Sequential Update
    Tracking, but 18k Examples Do*
  - *What Pretraining Doesn't Learn: Recovering Update Tracking with
    Targeted LoRA*
  - *Eighteen Trillion Tokens vs Eighteen Thousand: A Coverage Gap in
    LLM Pretraining*
  - Decide after §1/§5/§7 are drafted.
- [ ] **Venue / format.** ACL (8 pages) vs NeurIPS (9 pages + unlimited
  appendix). Current `acl.sty` is ACL two-column. If NeurIPS: switch to
  one-column template; figures need re-rendering at new widths.
- [ ] **Whether to also LoRA-tune Gemma-3-4b-it** (recommended; see §5).
- [ ] **Whether to spend on the post-LoRA mechanistic re-run** (highly
  recommended; see §5).

## 4. Final section outline

| § | Title | Status |
|---|---|---|
| 1 | Introduction — what pretraining doesn't learn | rewrite |
| 2 | Related work | minor edit |
| 3 | Task and setup | minor edit |
| 4 | Pretraining fails at update tracking | small edit (acknowledge reversal as part of failure, not as separate issue) |
| 5 | Two recoveries: format and LoRA | **new** (combines existing §6 format material + LoRA writeup) |
| 6 | When does pretraining fail? (training dynamics) | drafted; re-titled |
| 7 | Mechanism: what LoRA changes | **needs writing once post-LoRA mech runs are done** |
| 8 | Discussion | needs writing |
|   | Limitations / Ethics / Acknowledgments | needs expansion |
| App A | Dataset details | done |
| App B | Model list | done |
| App C | Full open-weight grid | done |
| App D | Proprietary grid | done |
| App E | ARB replication | done |
| App F | Format intervention data | done |
| App G | Training dynamics data | done |
| App H | LoRA training details (config, data gen, controls) | **new** — needs writing |
| App I | Pre/post mechanism details | **new** — needs experiments + writing |

## 5. Gap-closing experiments — priority order

These are the experiments that take the paper from "publishable" to
"bulletproof at top venue." Each closes a reviewer critique we cannot
otherwise answer.

### 5.1 Pre/Post LoRA mechanistic re-run (HIGH priority)

Why: converts §7 from black-box ("LoRA fixes it") into a mechanistic
story ("LoRA changes attention head X from primacy-encoding to
position-aware"). The single highest-value missing experiment.

Tasks:
- [ ] Re-run probing classifier on Qwen2.5-3B-Instruct + LoRA on a held-out
  cell (e.g., K=10, N=20). Compare to existing baseline probing result
  (v3/results/.../probing_*.json).
- [ ] Re-run logit lens (v_first vs v_last appearance across layers) on
  the LoRA-tuned model.
- [ ] Re-run attention routing analysis on the LoRA-tuned model. Identify
  whether the 11 discriminating heads (L27-L32) change function, get
  amplified, or get replaced.
- [ ] Causal patching / ablation on the LoRA-tuned model. Which components
  carry the new tracking signal?

Source data needed:
- LoRA adapter (have it: `/tmp/lora_intervention/checkpoints/main/final`)
- Baseline mechanistic results (have them in v3/results)

Estimated time: 1-2 days of focused GPU work.

### 5.2 Family control: LoRA on Gemma-3-4b-it (HIGH priority)

Why: rules out "Qwen-specific" critique. Gemma-3-4b-it shows clean
primacy regime in our data and no reversal — if LoRA closes the gap
there too, the recovery is architecture-agnostic.

Tasks:
- [ ] Generate train/val data using same script as Qwen run.
- [ ] LoRA train Gemma-3-4b-it with same config (rank 16, attention-only,
  2 epochs, lr 2e-4).
- [ ] Evaluate on the same 28 held-out ARB cells.
- [ ] Spot-check SEM OOD on 2-3 cells.

Estimated time: 1 day on a single GPU.

### 5.3 Negative-control LoRA (MEDIUM priority)

Why: rules out "any LoRA training would close it" critique. Pre-registered
in PLAN.md Decision 7 but not run.

Tasks:
- [ ] Generate arithmetic word-problem training data (~2k examples).
- [ ] LoRA train Qwen2.5-3B-Instruct with arithmetic data, same hyperparameters.
- [ ] Evaluate on the failure grid. Expectation: gap does not close.

Estimated time: 0.5 day.

### 5.4 Few-shot baseline (LOW priority — nice-to-have)

Why: distinguishes "task is just unfamiliar" from "needs gradient signal."

Tasks:
- [ ] Construct 5/10/20-shot in-context demonstrations.
- [ ] Evaluate base model on the same 28 held-out cells with each shot
  count. Compare to LoRA.

Estimated time: 0.5 day.

## 6. Writing tasks — section by section

### 6.1 §1 Introduction (rewrite)

- [ ] Open with the magnitude gap (18T pretraining tokens vs 18k LoRA
  examples).
- [ ] Frame: LLMs deployed in stateful settings → multi-turn, agents,
  RAG. Sequential update tracking is required.
- [ ] Empirical finding: 20 models systematically fail in load-dependent
  regimes (primacy + reversal both = failure of state tracking).
- [ ] Intervention finding: format + LoRA recover the skill independently.
- [ ] Mechanism finding (once §7 lands): what changed across the
  intervention.
- [ ] Replace contribution list with 4 contributions matching the new
  outline.

### 6.2 §3 Task and setup (clean-up)

- [ ] Standardise terminology: **dataset** = ARB / SEM / NAR;
  **format** = plain / labeled / block / landmark. Drop the §3.2
  "Stream Formats" subsection in main text; move format definitions to
  §5 where they're used.
- [ ] Refresh model count phrasing: 14 open-weight + 6 proprietary = 20
  models (currently mixed up in places).
- [ ] Remove or replace the SUTI acronym (defined once, never reused).

### 6.3 §4 Pretraining fails at update tracking (small edit)

- [ ] Reframe paragraph headers: instead of "Shape of the failure" +
  "Cross-model breadth," use "Both first-value and current-value retrieval
  fail" or similar.
- [ ] Add one paragraph acknowledging the reversal regime explicitly
  (CVQ > FVQ at high load in mid-capability models) as another failure
  mode of state tracking, not a separate phenomenon.
- [ ] Drop or soften "canonical asymmetric pattern" phrasing (line 429 of
  current main.tex).

### 6.4 §5 Two recoveries (new — most important section)

- [ ] §5.1 Format intervention. Adapt existing Appendix F prose. Lead
  with the headline number: open-weight CVQ goes from ≤ 0.23 (Plain) to
  ≥ 0.89 (Block) at K=10, N=50 on SEM.
- [ ] §5.2 LoRA intervention. Write fresh. Cover:
  - Setup: 18k examples, K ∈ {2,3,5,10} × N ∈ {5,10,15,20}, 35 train
    categories, attention-only rank-16 LoRA, 2 epochs.
  - Held-out evaluation: 28 cells (K up to 30, N up to 75), 11 held-out
    categories, ~10× context-length extrapolation.
  - Headline ARB result: 28/28 cells fixed, RI 5-52% → 93-100%, PI
    38-62% → 92-100%.
  - SEM OOD result: 4 SEM cells, all fixed, FVQ/CVQ both 100%.
  - Dense-IVQ result: position-2 to position-N accuracy 10-25% → 70-100%
    (kills "amplified recency head" critique).
  - Two independent fixes converge → skill is present-but-unused.
- [ ] Figure: pre/post LoRA gap heatmap on the (K, N) grid.

### 6.5 §6 When does pretraining fail? (re-title existing §7.1)

- [ ] Keep current training-dynamics figure (SmolLM2 + SmolLM3 with
  stage bands and two regime lines).
- [ ] Reframe prose around: "the failure exists from the earliest
  pretraining checkpoint and survives every documented training stage.
  This rules out alignment, SFT, expert specialisation, and APO as the
  source. The gap is in pretraining."

### 6.6 §7 Mechanism: what LoRA changes (NEW — gated on 5.1)

- [ ] Pre-LoRA mechanism summary (from existing data): discriminating
  attention heads in L27-L32 of Qwen2.5-3B-Instruct, primacy/recency
  boost magnitudes predict regime, probing shows RI encoded but PI at
  chance.
- [ ] Post-LoRA mechanism: results from the gap-closing experiments
  (probing, logit lens, attention routing).
- [ ] Direct comparison: what changed, what stayed the same.
- [ ] One figure: side-by-side pre/post head-importance bar charts or
  attention-pattern heatmaps.

### 6.7 §8 Discussion (new)

- [ ] Implication 1: scaling alone may not close this. SmolLM trajectory
  data and the model-size variation in §4 both argue against trivial
  scaling fixes.
- [ ] Implication 2: pretraining curriculum lacks signal for
  distinguishing past-binding from current-binding. LM loss rewards
  locally plausible continuations; both readings are plausible.
- [ ] Implication 3: methodology — characterise → recover → mechanism
  is a general recipe for studying capability gaps.
- [ ] Open questions: which other skills have similar gaps? How to find
  them without lab-by-lab manual probing?

### 6.8 Limitations expansion

- [ ] $K$ vs context-length confound (already noted).
- [ ] LoRA recovery tested on 1-2 model families.
- [ ] Mechanism story currently for Qwen2.5-3B; may differ across
  families.
- [ ] Did not separate "missing data" from "missing supervision signal"
  — both could explain the pretraining failure.
- [ ] No tests on real-world stateful tasks (multi-turn dialogue, agent
  rollouts).

## 7. Verifications and sanity checks

- [ ] **Numbers consistency sweep.** Currently main.tex has "20 models",
  "14 open-weight", "15 models", "twelve models" in different places. Do
  one consistency pass after §1 rewrite.
- [ ] **References sweep.** li2025state, allal2025smollm2, smollm3,
  qwen2.5 cite all need to be present in references.bib.
- [ ] **Strip all `\note{}` blocks before submission.** Currently 4-5
  remaining.
- [ ] **Auto-generator runs clean.** `paper/figures/generate_main_results.py`
  must produce all figures + tables without errors. Already verified
  end-to-end.
- [ ] **Reproducibility.** Decide what training/eval scripts go in repo
  vs supplementary. Currently lora_intervention scripts are extracted at
  `/tmp/`; need to be committed to repo for camera-ready.
- [ ] **Compile sweep.** After every major edit, recompile and check
  page count + warning count.

## 8. Things explicitly NOT in scope

- Architectural ablations (Mamba/SSM comparison) — dropped earlier.
- Full positional curve as a standalone section — folded into §5
  (format intervention demonstrates the same point).
- A universal $(K, N)$ predictor for reversal — does not exist in our
  data (best pooled AUC for K is 0.669). The narrative-B framing makes
  this moot.

## 9. Provisional timeline

Assuming all gap-closing experiments are run:

- Days 1-2: §5.1 + §5.2 mechanistic re-run (Section 5.1).
- Day 3: Gemma-3-4b-it LoRA + arithmetic control (Section 5.2, 5.3).
- Day 4: Update all generators, regenerate figures/tables.
- Days 5-7: Write §5 (Two Recoveries) and §1 rewrite.
- Days 8-9: Write §7 (Mechanism) once data lands.
- Day 10: Write §8 (Discussion), Limitations expansion.
- Day 11: Title + abstract + consistency pass.
- Day 12: External read + revisions.

Total: ~2 weeks of focused work after gap-closing experiments land.
