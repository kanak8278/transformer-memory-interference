# Repo Map — What's Relevant for the Current Paper

**Current paper:** "Tracked but Suppressed: How LLMs Fail Current-Value Retrieval"
(`paper/main.tex`), being prepared for AAAI (see `AAAI_PREP_PLAN.md`).

This repo contains **two generations of work**. As of 2026-07-29 they're
physically separated: dead PI/RI-era code and superseded docs live under
`legacy/`, loose reading material lives under `reference/`, everything else
at root is active.

> Note: the current paper's *code and raw JSONs* still use `RI`/`PI` key names
> internally (RI→FVQ, PI→CVQ in paper terminology).

---

## ✅ RELEVANT — the current paper lives here

| Folder | What it is |
|---|---|
| `paper/` | LaTeX source, 45 committed figure/table artifacts (`figures/`), table/figure generator scripts (`scripts/`), EMNLP checklist |
| `lora_intervention/` | Complete LoRA pipeline: `train.py`, `data_gen.py`, evaluators, mechanism runners (routing/logit-lens/probing/stage-3), **adapters** (`checkpoints/`: main Qwen 43MB, Gemma 71MB, arith control 43MB), **results** (`results/`), plus `experiments/bigcell_ivq/` (E1 big-cell IVQ scan, run 2026-07-22/23 on a local H100 — see its `GOAL.md`) |
| `v3/` | The current experimental generation — see breakdown below |
| `experiments_cloud/` | Proprietary-model sweeps + format ucurve scripts; museum narrative sweeps (`museum_ivq_haiku.py`, `museum_endpoint_sweep.py`, `plot_museum_ivq.py`) |
| `synthetic_scratch_training/` | From-scratch GPT-2 training on the synthetic KV-recall task (7 numbered experiments, 01-07). Best checkpoint uploaded to HF (`kanak8278/gpt2-small-synthetic-kv-interference`, private) — see `05_gpt2_scratch_h100/WEIGHTS.md`. Self-contained: no cross-imports with any other top-level folder. |
| `narrative_generator/` | 5 domains (`atc/`, `dota2/`, `icu/`, `museum/`, `wildlife/`) — **only `museum/` is actively used** (recent addition, feeds `experiments_cloud/museum_*` scripts). The other 4 are dead but the package can't be split apart without code edits (see live-dependency note below) — left physically whole. |
| `present/` | Presentation assets for the current paper: figures, a one-pager (`ONE_PAGER.tex`), `openreview.md` |
| `scripts/analysis/` | Error-type classification code (RI/PI error patterns) + `ERROR_ANALYSIS_FINDINGS.md` — kept active, still referenced/valued for the current work even though the underlying data predates the FVQ/CVQ pivot |
| `findings/` | Verified-claims writeups tied directly to `AAAI_PREP_PLAN.md` §2 (e.g. `001-ablation-l32-contradiction.md`) |
| Root planning docs | `AAAI_PREP_PLAN.md`, `EXPERIMENTS.md`, `EXPERIMENT_STATUS.md`, `NEW_RESULTS_AND_NEXT_STEPS.md` — current, keep reading these first |
| `lora_sem_*.py`/`*_results.json`, `build_results_notebook.py` (root) | Loose but active — local SEM-validation data referenced in `AAAI_PREP_PLAN.md` §5b |

### Inside `v3/` (mostly current; a few reference-only subdirs)

| Subdir | Status |
|---|---|
| `v3/results_vllm/` | **Canonical results** for the paper: `arbitrary_single/` (15 models), `attention_routing/`, `logit_lens/`, `probing/` (partial — some Qwen files GPU-box only; see AAAI_PREP_PLAN §1.3), `RUN_LOG.md` |
| `v3/scripts/experiments/` | **Canonical experiment runners** (`stage1_sweep.py`, `stage2_logit_lens.py`, `stage3_causal.py`, `probing_classifier.py`, `training_dynamics.py`) + `results/` (stage-3 causal JSONs). Also contains older narrative-domain scripts (`narrative_api_new_domains.py` etc.) not part of the canonical list — not yet audited for use, left in place. |
| `v3/PLAN.md`, `v3/WORK_LOG.md` | Theory framing + run history — useful context |
| `v3/paper/`, `v3/presentations/`, `v3/observations/`, `v3/plots/`, `v3/figures/`, `v3/docs/` | Drafts/notes from earlier iterations — reference only |

### Live dependencies (do not move, regardless of how "legacy" the parent looks)

| Path | Why |
|---|---|
| `models/` | Imported directly by `experiments_cloud/*.py` (sweep_semantic, sweep_arbitrary, remedy_sweep, ucurve_sweep, museum_ivq_haiku, test_openai_tr) and `v3/scripts/experiments/*.py` (narrative_api_experiment, narrative_api_new_domains, remedy_experiment) |
| `mechanistic_probing_v2/core/` | `dataset_configs.py` (ARB shared 2,300-word pool + 46 category-keys), `model_loader.py`, `evaluation.py`, `inference.py` — imported from `lora_intervention/*.py`, `v3/*.py`, `v3/scripts/experiments/*.py`, `experiments_cloud/*.py`. The rest of `mechanistic_probing_v2/` (experiments, figures, notebooks, docs) is archived under `legacy/mechanistic_probing_v2/`. |
| `narrative_generator/` (whole package) | `__init__.py` eagerly imports all 5 domains; `museum/generator.py` imports `..base`; several `v3/scripts/*narrative*.py` files import the sibling domains directly. Splitting `museum/` out requires code edits first — not done. |

---

## 📦 `legacy/` — dead PI/RI-era code and superseded docs (moved 2026-07-29)

Everything here is unreferenced by any active code (verified via import/path
grep before moving) and kept for history, not active development:

`results/`, `data/`, `datasets_v0/`, `prompts/`, `scripts/core/`,
`scripts/shell/`, `notebooks/`, `draft/`, `docs/` (old paper drafts, ICML
templates — minus `ERROR_ANALYSIS_FINDINGS.md`, which moved to
`scripts/analysis/`), `mechanistic_probing_v2/{experiments,figures,
notebooks,*.md}` (its `core/` stays live at the original path), and the
superseded top-level planning docs from the earlier NeurIPS-pivot era:
`CLAUDE.md`, `NEURIPS_EXPANSION_PLAN.md`, `MECHANISTIC_PROBING_DESIGN.md`,
`COMPETITIVE_LANDSCAPE.md`, `program.md`, `program2.md`.

## 📚 `reference/` — reading material, not code

`55_Confidently_Wrong_Silent_Me.pdf`, `EIML-Review.pdf`,
`LLM___Retrospective_interference.pdf`, `RESEARCH_PROPOSAL.html`,
`gemini_conversation`.

---

## Quick-start pointers for a new session

- Task plan & groups: `AAAI_PREP_PLAN.md`
- Which raw data backs which table/figure: generator headers in
  `paper/scripts/*.py` (each lists its input paths)
- Missing-vs-present raw data: `AAAI_PREP_PLAN.md` §1.3
- Adapter inventory & training grid: `AAAI_PREP_PLAN.md` §1.4
  (grid: `TRAIN_KEYS=[2,3,5,10]`, `TRAIN_UPDATES=[5,10,15,20]` in
  `lora_intervention/data_gen.py`)
- Big-cell IVQ scan (endpoints hold, interior collapses at large K,N):
  `lora_intervention/experiments/bigcell_ivq/GOAL.md`
