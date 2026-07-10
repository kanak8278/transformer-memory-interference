# AAAI Submission Preparation Plan

**Paper:** "Tracked but Suppressed: How LLMs Fail Current-Value Retrieval"
(`paper/main.tex`, 23 pp compiled, currently in ACL review format)

**History:** Submitted to EMNLP 2026 / ARR May 2026 cycle → **desk-rejected for
formatting**. No dual-submission conflict; the paper is free to go to AAAI.

**Goal:** Fix all content-level issues first (consistency, naming, verification,
optional strengthening experiments), then port to AAAI format last.

**Created:** 2026-07-11 (branch `aaai-prep`).

---

## 1. Asset Inventory (audited 2026-07-11)

### 1.1 Canonical data locations

Only these directories hold results the paper actually uses. Everything else
(`results/`, `v3/results/`, `datasets_v0/`, `mechanistic_probing_v2/`, `draft/`)
is legacy from the earlier PI/RI-era project and is **not** relevant to this
paper.

| Location | Contents | Status |
|---|---|---|
| `v3/results_vllm/` | Behavioral sweeps (ARB), mechanism (probing / logit lens / attention routing) | **Partial** locally (see 1.3) |
| `lora_intervention/results/` | LoRA evals (Qwen 28-cell, Gemma, GSM8K control), block-locus, dense-IVQ, CI analyses | **Complete** locally (24 files) |
| `experiments_cloud/results/` | Proprietary sweeps (2 CSVs); open-weight format ucurve | CSVs present; `ucurve_vllm/` **missing** |
| `v3/scripts/experiments/results/` | Stage-3 causal ablation JSONs (Qwen promoters, baseline) | Present (3 dirs) |
| `paper/figures/` | 45 committed `.tex`/`.pdf`/`.png` artifacts — the paper compiles standalone | **Complete** |

### 1.2 Scripts — all present, everything re-runnable

- Experiment runners: `v3/scripts/experiments/` (`stage1_sweep.py`,
  `stage2_logit_lens.py`, `stage3_causal.py`, `probing_classifier.py`,
  `training_dynamics.py`, …)
- LoRA pipeline: `lora_intervention/` (`train.py`, `data_gen.py`, `evaluate.py`,
  `evaluate_ivq.py`, `merge_lora.py`, routing/logit-lens/probing/stage-3
  runners, CI scripts). `train.py` supports MLP-only LoRA via
  `--target_modules gate_proj,up_proj,down_proj`.
- Cloud/format sweeps: `experiments_cloud/` (`ucurve_sweep.py`,
  `ucurve_vllm.py`, `sweep_semantic.py`, …)
- Paper generators: `paper/scripts/` (each table/figure has a generator with
  documented input paths)

### 1.3 Raw results MISSING locally (existed on L40S / SageMaker nodes)

Not blocking for submission — the derived `.tex`/`.pdf` artifacts are committed.
Re-run on demand only if a table/figure must be regenerated from raw data.

| Missing path | Feeds | Re-run cost |
|---|---|---|
| `v3/results_vllm/semantic_multi/` | Main table (open-weight rows), Fig 2, `tab_full_sem`, Gemma SEM baseline | High (full sweep, 11 models) |
| `v3/results_vllm/training_dynamics/` (SmolLM2) | Training-dynamics fig + `tab_smollm2_traj` | High (42 checkpoints) |
| `v3/results_vllm/training_dynamics_smollm3/` | Training-dynamics fig + `tab_smollm3_traj` + formats table | High (37 checkpoints) |
| `experiments_cloud/results/ucurve_vllm/` | Format table (open-weight rows), both multimodel-IVQ figures | Moderate |
| `v3/results_vllm/probing/probing_Qwen2.5-3B-Instruct_2k_5u.json` (base) | Mechanism fig 5a | Cheap (~1–3 GPU-h) |
| `v3/results_vllm/logit_lens/Qwen2.5-3B-Instruct/` (base) | Mechanism fig 5a | Cheap |
| `v3/results_vllm/attention_routing/Qwen2.5-3B-Instruct{,-LoRA}__normal.json` | Mechanism fig 5b | Cheap |
| `v3/scripts/experiments/results/Qwen2.5-3B-Instruct-LoRA/stage2_logit_lens_*.json` | Mechanism fig 5a | Cheap |

**Action if GPU box still alive:** one-time rsync of these 8 paths makes the
repo self-contained (also wanted for the reproducibility archive).

### 1.4 LoRA adapters — inventory (answers "are they present / documented?")

Three adapters are present locally under `lora_intervention/checkpoints/`,
each with `adapter_config.json` + `run_config.json`:

| Adapter | Base model | Size | Role in paper | Training grid documented? |
|---|---|---|---|---|
| `adapter/` (run `main`) | Qwen2.5-3B-Instruct | 43 MB | §5.2 main LoRA | **Not in `run_config.json`** — grid lives in code (`data_gen.py`: `TRAIN_KEYS=[2,3,5,10]`, `TRAIN_UPDATES=[5,10,15,20]`) and paper App G Table 10 |
| `gemma_adapter/` (run `gemma_main`) | Gemma-3-4b-it | 71 MB | §5.2 family control | Same grid, regenerated data; `run_config.json` records early-stop at step 400 / 1.42 epochs with reason |
| `qwen_arith_adapter/` (run `qwen_arith_control`) | Qwen2.5-3B-Instruct | 43 MB | App G negative control | GSM8K (no K,N grid); well-documented `run_config.json` incl. deviation note |

Gaps found:
- `run_config.json` records hyperparameters + train-file *paths* but **not the
  K,N training grid itself**; the referenced `lora_intervention/data/*.jsonl`
  files are **not local** (regenerable deterministically via `data_gen.py`).
- `README.md` in each adapter dir is the **unfilled PEFT boilerplate template**
  — must be written properly before public release (promised in Ethics section).
- **MLP-only LoRA adapter (App L) is NOT local** — exists only on the GPU box
  (or needs retraining: 100 steps, cheap).

---

## 2. Verified facts (checked against raw data this session)

1. **The 16-vs-28 held-out cell "inconsistency" is a display bug, not a data
   problem.** `lora_intervention/results/main_eval_20260523_055309.json`
   contains all 28 held-out ARB cells; minimum post-LoRA accuracy across all
   28 = **92.5%**, so the main-text claim "93–100% on the 28 held-out cells"
   holds. The paper tables show only a hard-coded 16-cell `DISPLAY_CELLS`
   subset (`paper/scripts/generate_lora_tables.py`), and captions +
   cross-family summary then wrongly say "16 held-out cells" / "16/16".
2. **Main LoRA eval trial counts are 40–80 per cell with early stopping**
   (`stopped_early` field), not the 100–200 a reader would assume. Currently
   undocumented in the paper.
3. Checkpoint math checks out: 37 SmolLM3 rows + 42 SmolLM2 rows = **79**
   checkpoints as claimed in the abstract.
4. `\note{}` macro defined in `main.tex` but no live uses remain.

---

## 3. Task Groups

### Group A — Repo & data consolidation (checking / exploring)

- [ ] **A1. Remote data decision:** confirm whether the L40S / SageMaker
      storage still exists. If yes → rsync the 8 missing paths (§1.3).
      If no → accept re-run-on-demand policy (scripts all present).
- [ ] **A2. Write `DATA_MAP.md`:** table → generator script → raw data file
      lineage for every table/figure in the paper (the audit in §1 is the
      starting point). Locks in which folders are canonical.
- [ ] **A3. Mark legacy dirs:** add a note (or move under `legacy/`) for
      `results/`, `v3/results/`, `datasets_v0/`, `mechanistic_probing_v2/`,
      `draft/` — old PI/RI-era experiments, not used by this paper.
- [ ] **A4. Update stale `v3/results_vllm/EXPERIMENT_SUMMARY.md`** (claims 118
      files / 8 experiment types; the folder now has 5 subdirs).
- [ ] **A5. Anonymization sweep** (carried from EMNLP checklist): grep repo for
      author name/email; note `main_eval_*.json` and `run_config.json` contain
      `/home/sagemaker-user/...` paths — strip in the release archive.

### Group B — Renames / updates (data fully present; no re-runs; pure edits)

- [ ] **B1. 16→28 cell fix:** extend `DISPLAY_CELLS` in
      `generate_lora_tables.py` to all 28 (appendix full table), or re-caption
      compact tables as "16 of 28 shown"; change "16/16" → "28/28" in the
      cross-family summary table (`main.tex` ~L1653). Regenerate tables.
- [ ] **B2. RI/PI → FVQ/CVQ terminology purge** in `main.tex`: leftover "RI"
      at L1599, L1784, L1800, L1810. Code/JSONs keep `RI`/`PI` keys (renaming
      would break every generator); instead add a naming-map note to the
      release README.
- [ ] **B3. Document adaptive trial counts** (40–80/cell, early stopping) in
      the LoRA table captions or App G.
- [ ] **B4. Harmonize model counts** (11 main open-weight / 14 incl. auxiliary
      / 20 total) across abstract, intro, §3.
- [ ] **B5. CVQ wording:** add one main-text sentence in §3 that the CVQ
      prompt literally asks for the "last value" (currently disclosed only in
      App B).
- [ ] **B6. Fill in the three adapter `README.md`s** (currently PEFT
      boilerplate) with base model, K,N training grid, data recipe, license —
      needed for the promised public release.
- [ ] **B7. Record the K,N training grid inside each adapter's
      `run_config.json`** (or a sidecar `training_grid.json`) so the adapter
      artifact is self-describing.

### Group C — Experiments

#### C1. Behavioral

- [ ] **C1.1 Verify SEM LoRA claim** ("all 4 held-out SEM cells → 100%")
      against `lora_sem_validation_results.json` + `lora_sem_finish_results.json`
      (local; no GPU needed).
- [ ] **C1.2 Verify dense-IVQ claim** ("<5% within two positions of stream
      start") against `lora_intervention/results/dense_ivq_*.json` (local).
- [ ] **C1.3 Verify Gemma baseline `---` cells** (insufficient-trial cells in
      `tab_lora_gemma_arb`) — confirm which cells and that the 28/28 claim is
      unaffected (local: `gemma_main_comparison.txt`).
- [ ] **C1.4 (post-A1) Regenerate main + format tables end-to-end** as a
      consistency check once/if raw data is synced.
- [ ] **C1.5 (decision) Real-world bridge experiment** for the AAAI agents
      audience (e.g., tool-use variable-tracking probe). New work; optional;
      defensible to skip via Limitations.
- [ ] **C1.6 Training-dynamics data:** nothing scientifically open (79-checkpoint
      story complete); only sync/verify per A1. Re-run only if challenged.

#### C2. Training / LoRA fine-tuning tasks

- [ ] **C2.1 MLP-only LoRA — finish the job.** Paper reports a "5-cell
      preview" (App L) and declares the full 28-cell + SEM OOD eval open.
      Adapter not local → either sync from GPU box or retrain (100 steps,
      cheap). Then run the full held-out eval. Highest value-per-GPU-hour of
      all open experiments.
- [ ] **C2.2 Regenerate LoRA training data locally** (`data_gen.py`,
      deterministic seeds) so `lora_intervention/data/*.jsonl` exists for the
      release archive.
- [ ] **C2.3 (decision) Any new fine-tune variants?** e.g., rank ablation or
      smaller-data LoRA ("how few examples suffice?") — nice-to-have,
      strengthens the "latent capability" story, not required.
- [ ] **C2.4 Package adapters for release** (HF-style cards from B6/B7;
      Qwen Research License for Qwen-based, Gemma ToU for Gemma-based).

#### C3. Mechanistic interpretability

- [ ] **C3.1 Verify mechanism CI numbers** (probing Δ +0.36, ablation −0.28 /
      −0.40, paired diffs) against the local stage-3 causal JSONs +
      `ci_analysis*.txt`.
- [ ] **C3.2 (decision, GPU) Gemma paired causal ablation** — currently "left
      to future work"; closes the paper's most attackable gap (causal evidence
      on one family only). Most valuable, most work.
- [ ] **C3.3 (decision, GPU) Mechanism at a harder cell** (logit lens +
      routing at K=10, N=50) — defuses "you probed where the model succeeds"
      (all causal work is at K=2, N=5).
- [ ] **C3.4 (post-A1) Re-obtain the 4 missing Qwen mechanism files** (base
      probing, base logit lens, base+LoRA routing, LoRA logit lens) by sync or
      cheap re-run, so Fig 5 is regenerable.

### Group D — AAAI porting (LAST; blocked on A–C decisions)

- [ ] **D1. Confirm AAAI-27 CFP:** exact deadlines (~early Aug abstract /
      mid-Aug paper, verify), page limit (~7 + refs), template kit, and
      supplementary-material rules.
- [ ] **D2. Template conversion** ACL → AAAI (`aaai` style, author kit rules:
      no negative vspace, caption placement, etc. — the thing that killed the
      EMNLP submission was formatting; triple-check this time).
- [ ] **D3. Page-budget compression** of main body; decide what moves to
      appendix/supplementary given AAAI reviewers aren't obliged to read it —
      the 28-cell headline numbers must be self-contained in the main body.
- [ ] **D4. AAAI reproducibility checklist** (reuse material from
      `paper/EMNLP_SUBMISSION_CHECKLIST.md`).
- [ ] **D5. Reframe for general-AI audience:** tilt intro/discussion toward
      agents & stateful systems; trim ACL-style related-work taxonomy.
- [ ] **D6. Preprint citation decision:** AAAI has no ACL-style prohibition —
      decide whether to cite arXiv 2603.00270 (the earlier behavioral-only
      version) in Related Work.
- [ ] **D7. Anonymized code/data archive** (from A5 + C2.2 + C2.4).

---

## 4. Open decisions (need discussion before scheduling)

1. **Remote data:** does the L40S / SageMaker storage still exist? (gates A1,
   C1.4, C3.4)
2. **GPU budget & timeline:** which of C2.1 / C3.2 / C3.3 are in scope before
   the AAAI deadline? Suggested priority by review-risk reduction per
   GPU-hour: **C2.1 (MLP-LoRA full eval) → C3.3 (harder-cell mechanism) →
   C3.2 (Gemma ablation)**.
3. **C1.5 bridge experiment:** do it or defend synthetic-only scope?
4. **D6 preprint citation:** cite or not.

## 5. Suggested execution order

1. Group B (mechanical, no dependencies) + C1.1–C1.3 + C3.1 (local
   verification) — can start immediately.
2. Group A (A1 decision first; A2–A5 in parallel with B).
3. Group C GPU work per decision #2.
4. Group D porting once content is frozen.
