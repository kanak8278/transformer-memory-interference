# Interactive showcase — *Tracked but Suppressed: How LLMs Fail Current-Value Retrieval*

A single-page, static, scrollytelling companion to the paper. All figures are
**precomputed** measurements (no model runs in the browser). Every number is
sourced from `paper/` (the current submission) — FVQ/CVQ terminology, 20 models,
the §7 mechanism table.

## Run locally

```bash
# from repo root
.venv/bin/python -m http.server 8765 --directory web
# open http://localhost:8765
```

## Sections

| # | Section | Data file | Paper source |
|---|---------|-----------|--------------|
| 1 | Feel the bug (multi-key stream replay) | `data/examples.json` (← `demo_runs.json`) | real Qwen2.5-3B runs on the semantic task |
| 2 | Almost every model lands above the line | `data/models_scatter.json` | `paper/figures/tab_main.tex` + `semantic_multi/` |
| 3 | Where it fails (position curve) | `data/position_curve.json` | dense-IVQ (`tab_dense_ivq.tex`) |
| 4 | The more updates, the worse (CVQ vs N) | `data/load.json` | `semantic_multi/` cells, K=5 |
| 5 | The answer is there, then erased (logit lens) | `data/logit_lens.json` | `tab_mechanism.tex` + `tab_gemma_logit_lens.tex` |
| 6 | What the model knows about itself (probing) | `data/probing.json` | `tab_mechanism.tex` (probes) |
| 7 | A small fine-tune closes the gap | `data/intervention.json` | `tab_lora_sem.tex` + `tab_mechanism.tex` |

§1's real model runs are generated separately (one-time, on MPS) via `build_data.py`'s companion:

```bash
.venv/bin/python web/build_demo_data.py   # writes data/demo_runs.json
```

## Regenerate the data

After any change to the underlying results, rebuild the static JSON:

```bash
.venv/bin/python web/build_data.py
```

`build_data.py` reads the vetted result files and the paper's published values.
It mirrors `paper/scripts/generate_main_results.py`'s loaders so the page and the
paper never drift. Internal JSON keys are `RI`/`PI` (the code's names); the UI
displays the paper's **FVQ** (first-value) / **CVQ** (current/last-value).

## Deploy

It's fully static — drop `web/` on Cloudflare Pages, GitHub Pages, or Netlify.
`paper.pdf` is a copy of `paper/main.pdf`; refresh it on submission updates.

## Notes / honesty

- Scatter shows **19 of the paper's 20** models — Pythia-410M lacks a comparable
  K=5,N=30 cell with ≥30 trials. Mamba is excluded (it's an SSM *control*).
- §3/§4 curves are full-resolution per-layer; the paper quotes specific layers
  (e.g. L32). They match exactly at every sampled layer.
