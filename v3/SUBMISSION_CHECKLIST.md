# NeurIPS Submission Checklist

## Content Checklist

### Section 1: Introduction
- [x] Opening hook (PI > RI universality)
- [x] What's known vs what's new
- [x] 4 contribution bullets
- [x] Figure 1 reference (4-panel teaser)
- [ ] Final polish for clarity and flow

### Section 2: Related Work
- [x] Position bias papers (Liu, Hsieh, Chowdhury, Wu)
- [x] Attention sink (Barbero 2024, 2025)
- [x] Softmax limitations (Velickovic 2025)
- [x] SSM analysis (Wang 2025)
- [x] Mechanistic interp (Conmy, Nanda)
- [x] Representation collapse (Pasten 2025)
- [ ] LaTeX formatting

### Section 3: Theory (Three-Force Model)
- [x] Formal setup and definitions
- [x] Proposition 1: Primacy in transformers
- [x] Proposition 2: Primacy in SSMs
- [x] Proposition 3: Positional discrimination
- [x] Proposition 4: Exponential PI(N) decay
- [x] Informal unified theorem
- [ ] Tighten proof sketches (currently hand-wavy in places)
- [ ] LaTeX formatting with proper math

### Section 4: Behavioral Results
- [x] Table 2: Full results with Wilson CIs (LaTeX ready)
- [x] Figure 2: PI vs N (9 models)
- [x] PI(N) exponential fits
- [x] Error position analysis (3 modes)
- [x] Narrative transfer (+18% gap)
- [ ] Missing: 200-trial data for TinyLlama, StableLM
- [ ] Missing: Narrative with 100+ trials for significance

### Section 5: Mechanistic Evidence
- [x] Logit lens (4 models)
- [x] Probing classifiers (3 models)
- [x] Causal analysis (3 models)
- [x] Jacobian at init (2 architectures)
- [ ] Note: probing condition probe detects query word (acknowledge)
- [ ] Note: causal effects are small/distributed (frame correctly)

### Section 6: Component Elimination
- [x] Cross-architecture table (7 components)
- [x] Bidirectional control (preliminary)
- [ ] Missing: Flan-T5-large for cleaner bidirectional control

### Section 7: Discussion
- [x] Implications draft
- [x] Limitations listed
- [x] Future work identified
- [ ] Polish

## Figures Checklist (Main Paper)

| # | Content | File | Status | Quality |
|---|---|---|---|---|
| 1 | 4-panel teaser | paper_main_figure.png | Done | Good |
| 2 | PI vs N (9 models + CIs) | cross_model_pi_vs_n.png | Done | Good |
| 3 | Error positions (6 models) | cross_model_error_positions.png | Done | OK |
| 4 | Logit lens (4 models) | cross_model_logit_lens.png | Done | OK (small text) |
| 5 | Probing (3 models) | probing_results.png | Done | Good |
| 6 | Jacobian comparison | jacobian_comparison.png | Done | Good |

## Appendix Figures

| # | Content | File | Status |
|---|---|---|---|
| A1 | Scaling law fitted | scaling_law_fitted.png | Done |
| A2 | Error position model | error_position_model.png | Done |
| A3 | Per-model regime maps | figures/{model}/fig1*.png | Done |
| A4 | Per-model logit lens | figures/{model}/fig5*.png | Done |

## Tables Checklist

| # | Content | Status |
|---|---|---|
| 1 | Model inventory | Done |
| 2 | Behavioral summary (LaTeX) | Done |
| 3 | Logit lens metrics | Done |
| 4 | Causal patching results | Done |
| 5 | Component elimination | Done |
| 6 | Scaling law fits | Done |

## Data Quality

| Model | Trials | Wilson CI | Sufficient? |
|---|---|---|---|
| Qwen 1.5B | 200 | ±7% | YES |
| Qwen 3B-Base | 200 | ±7% | YES |
| Qwen 3B-Inst | 200 | ±7% | YES |
| Qwen 0.5B | 50 | ±14% | BORDERLINE |
| Gemma 1B | 50 | ±14% | BORDERLINE |
| TinyLlama | 30 | ±18% | NO |
| StableLM | 30 | ±18% | NO |
| Pythia | 50 | ±14% | BORDERLINE |
| Mamba 1.4B | 30→200 (running) | ±7% | PENDING |

## Before Submission

- [ ] All figures at 300 DPI, consistent font/style
- [ ] LaTeX compiles cleanly
- [ ] Abstract ≤ 200 words
- [ ] Main paper ≤ 9 pages
- [ ] All references in BibTeX
- [ ] Supplementary materials organized
- [ ] Code repository prepared
- [ ] Ethics statement (if needed)
- [ ] Reproducibility checklist
