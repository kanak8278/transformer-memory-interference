# `paper/scripts/` — figure & table generators

All scripts in this directory read from raw data elsewhere in the repo
(`v3/results_vllm/`, `experiments_cloud/results/`,
`lora_intervention/results/`, `lora_sem_*_results.json` at the repo
root) and emit reproducible LaTeX tables and PDF figures into
`paper/figures/`.

## Run all generators

From the project root:

```bash
.venv/bin/python paper/scripts/generate_main_results.py
.venv/bin/python paper/scripts/generate_lora_tables.py
.venv/bin/python paper/scripts/generate_dense_ivq_figure.py
```

Or in one line:

```bash
for s in paper/scripts/generate_*.py; do
  .venv/bin/python "$s" || break
done
```

Each script is idempotent. Re-running overwrites any prior outputs in
`paper/figures/` without touching `main.tex`.

## What lives where

| Script | Inputs | Outputs (in `paper/figures/`) |
|---|---|---|
| `generate_main_results.py` | `v3/results_vllm/{semantic_multi,arbitrary_single}/`, `experiments_cloud/results/proprietary_semantic_multi.csv`, `experiments_cloud/results/ucurve_*`, `v3/results_vllm/training_dynamics{,_smollm3}/` | `fig2_main_results.pdf`, `tab_main.tex`, `tab_full_sem.tex`, `tab_full_arb.tex`, `tab_full_sem_prop.tex`, `tab_format_gap.tex`, `tab_format_cvq.tex`, `fig3_training_dynamics.pdf`, `tab_smollm2_traj.tex`, `tab_smollm3_traj.tex`, `tab_smollm3_formats.tex` |
| `generate_lora_tables.py` | `lora_intervention/results/main_comparison.txt`, `lora_intervention/results/main_eval_*.json`, `lora_sem_validation_results.json`, `lora_sem_finish_results.json` | `tab_lora_arb.tex`, `tab_lora_sem.tex` |
| `generate_dense_ivq_figure.py` | `lora_intervention/results/dense_ivq_ivq_*.json` | `fig_dense_ivq.pdf` |

## Dependencies

Python ≥ 3.10, with the project virtualenv (`.venv/`). Required
packages: `matplotlib`, `numpy`. No exotic deps.

## Naming convention

- `tab_*.tex` — `\begin{tabular}{...}` … `\end{tabular}` body only,
  designed for `\input{}` inside a `table`/`table*` float in `main.tex`.
- `fig_*.pdf` — single-PDF figure, designed for `\includegraphics{}`.
- `*.png` companions are previews for quick inspection (not loaded by
  the paper).
