# Repo Map — What's Relevant for the Current Paper

**Current paper:** "Tracked but Suppressed: How LLMs Fail Current-Value Retrieval"
(`paper/main.tex`), being prepared for AAAI (see `AAAI_PREP_PLAN.md`).

This repo contains **two generations of work**:
1. **Legacy (PI/RI era):** the original ACL paper "Transformers Remember First,
   Forget Last" — API-based behavioral study across 39 models. Most top-level
   folders belong to this.
2. **Current (FVQ/CVQ era):** the v3 rework — vLLM behavioral sweeps, LoRA
   interventions, training dynamics, mechanistic analysis. Only four folder
   trees matter.

> Note: the current paper's *code and raw JSONs* still use `RI`/`PI` key names
> internally (RI→FVQ, PI→CVQ in paper terminology).

---

## ✅ RELEVANT — the current paper lives here

| Folder | What it is |
|---|---|
| `paper/` | LaTeX source, 45 committed figure/table artifacts (`figures/`), table/figure generator scripts (`scripts/`), EMNLP checklist |
| `lora_intervention/` | Complete LoRA pipeline: `train.py`, `data_gen.py`, evaluators, mechanism runners (routing/logit-lens/probing/stage-3), **adapters** (`checkpoints/`: main Qwen 43MB, Gemma 71MB, arith control 43MB), **results** (`results/`: 24 files, all local) |
| `v3/` | The current experimental generation — see breakdown below |
| `experiments_cloud/` | Proprietary-model sweeps + format ucurve scripts; `results/` has the 2 proprietary CSVs (open-weight `ucurve_vllm/` results are NOT local — GPU box only) |

### Inside `v3/` (mixed — only some subdirs current)

| Subdir | Status |
|---|---|
| `v3/results_vllm/` | **Canonical results** for the paper: `arbitrary_single/` (15 models), `attention_routing/`, `logit_lens/`, `probing/` (partial — some Qwen files GPU-box only; see AAAI_PREP_PLAN §1.3), `RUN_LOG.md` |
| `v3/scripts/experiments/` | **Canonical experiment runners** (`stage1_sweep.py`, `stage2_logit_lens.py`, `stage3_causal.py`, `probing_classifier.py`, `training_dynamics.py`) + `results/` (stage-3 causal JSONs) |
| `v3/PLAN.md`, `v3/WORK_LOG.md` | Theory framing + run history — useful context |
| `v3/results/` | ❌ Older non-vLLM runs (superseded by `results_vllm/`) — do not use |
| `v3/paper/`, `v3/presentations/`, `v3/observations/`, `v3/plots/`, `v3/figures/`, `v3/docs/` | Drafts/notes from earlier iterations — reference only |

---

## ❌ IRRELEVANT — legacy PI/RI-era project (do not explore for paper work)

| Folder | What it was |
|---|---|
| `results/` (top-level) | Original ACL paper API results (bedrock/claude/gpt/gemini raw runs) |
| `data/`, `datasets_v0/`, `prompts/`, `models/`, `scripts/` (top-level) | Original paper's datasets, jinja prompts, API model wrappers, analysis scripts |
| `mechanistic_probing_v2/` | Mostly superseded pre-v3 probing — **BUT `core/dataset_configs.py` is a LIVE dependency**: `lora_intervention/data_gen.py` imports `get_eligible_categories`/`generate_values_for_trial` from it (ARB shared 2,300-word pool + 46 category-keys live here). Do not delete this module. |
| `narrative_generator/` | Dota2/ATC/ICU narrative variant — abandoned (see paper §Limitations "Open questions") |
| `notebooks/` | Old reversal-analysis notebooks |
| `draft/`, `docs/` | Old paper drafts / ICML-era notes |
| `web/` | Interactive showcase page (branch `web-showcase`) — not paper content |

---

## Quick-start pointers for a new session

- Task plan & groups: `AAAI_PREP_PLAN.md`
- Which raw data backs which table/figure: generator headers in
  `paper/scripts/*.py` (each lists its input paths)
- Missing-vs-present raw data: `AAAI_PREP_PLAN.md` §1.3
- Adapter inventory & training grid: `AAAI_PREP_PLAN.md` §1.4
  (grid: `TRAIN_KEYS=[2,3,5,10]`, `TRAIN_UPDATES=[5,10,15,20]` in
  `lora_intervention/data_gen.py`)
