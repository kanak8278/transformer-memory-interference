# Project: Transformer Memory Interference — NeurIPS Expansion

## What This Is

An ACL paper ("Transformers Remember First, Forget Last: Dual-Process Interference in LLMs") being expanded for NeurIPS submission. The paper shows that LLMs exhibit PI > RI (proactive interference dominates retroactive interference) — the opposite of human memory — across 39 models. The NeurIPS version needs theory and mechanistic evidence to explain *why*.

## Core Finding

When LLMs see conflicting key-value updates in context:
- **Retroactive Interference (RI)**: recall first value despite later updates → models are GOOD at this
- **Proactive Interference (PI)**: recall last value despite earlier entries → models are BAD at this
- All 39 tested models show PI > RI (Cohen's d = 1.73)
- RI resistance scales with model size (R²=0.49), PI does not (R²=0.06)
- RI and PI are uncorrelated (R²=0.044) → dual-process hypothesis

## What We're Building (NeurIPS Expansion)

### Goal
Add mechanistic theory and empirical evidence explaining WHY transformers show PI > RI.

### Three Pillars

1. **Formal Theory** (Hopfield/associative memory model)
   - Prove PI > RI is a necessary consequence of causal attention
   - Show RI resistance scales with d_model (capacity), PI is bounded by attention sharpness
   - See `NEURIPS_EXPANSION_PLAN.md` for mathematical framework

2. **Mechanistic Probing** (current active work)
   - Using SmolLM2-135M-Instruct (and 360M, 1.7B) to probe internal representations
   - Logit lens, attention patterns, activation patching, probing classifiers
   - See `mechanistic_probing/EXPERIMENT_LOG.md` for raw lab notebook (every detail, every attempt)
   - See `mechanistic_probing/FINDINGS.md` for curated paper-ready results and narrative
   - See `MECHANISTIC_PROBING_DESIGN.md` for full experimental design

3. **Architectural Controls** (planned)
   - Test state-space models (Mamba/RWKV) which lack attention
   - Prediction: different interference profile → confirms attention-specificity

### Key Result So Far
**PI > RI appears even at 135M parameters.** SmolLM2-135M-Instruct shows the same qualitative pattern as GPT-5 and Claude-4.5-opus. The asymmetry is architectural, not emergent with scale.

## Project Structure

```
├── CLAUDE.md                          # This file
├── NEURIPS_EXPANSION_PLAN.md          # Full expansion plan with 5 directions
├── MECHANISTIC_PROBING_DESIGN.md      # Detailed probing pipeline design
├── mechanistic_probing/
│   ├── EXPERIMENT_LOG.md              # Raw lab notebook — every detail, attempt, dead end
│   ├── FINDINGS.md                    # Curated paper-ready results and narrative
│   ├── experiments/
│   │   └── 01_behavioral_screen.py    # Finds model's interference limits
│   ├── analysis/                      # Logit lens, attention, patching scripts
│   ├── results/                       # Output JSON files
│   └── notebooks/                     # Visualization
├── data/                              # Original paper datasets
├── datasets/                          # Original paper data loaders
├── models/                            # Original paper model wrappers (API-based)
├── prompts/                           # Jinja templates for original experiments
├── results/                           # Original paper results
└── docs/                              # Paper LaTeX source
```

## Technical Setup

- **Virtualenv:** `.venv/` (created with `uv`, Python 3.10.14)
- **Activate:** `source .venv/bin/activate`
- **Run experiments:** `.venv/bin/python mechanistic_probing/experiments/<script>.py`
- **Key deps:** torch 2.10.0 (MPS), transformers 5.2.0, huggingface_hub
- **Device:** Apple Silicon MPS GPU

## Models

| Model | Params | Use |
|-------|--------|-----|
| SmolLM2-135M-Instruct | 135M | Primary probing target (fast iteration) |
| SmolLM2-360M-Instruct | 360M | Size scaling validation |
| SmolLM2-1.7B-Instruct | 1.7B | Larger scale validation |
| SmolLM2-135M-intermediate-checkpoints | 135M × 9 | Training dynamics analysis |

## Conventions

- All experiment scripts go in `mechanistic_probing/experiments/` with numbered prefixes (01_, 02_, ...)
- Every experiment gets logged in `mechanistic_probing/EXPERIMENT_LOG.md` with date, design, results, and observations
- Results saved as JSON in `mechanistic_probing/results/`
- Always use `.venv/bin/python` to run (no global installs)
- Discovery notes and surprising findings go in `EXPERIMENT_LOG.md` under "Surprising findings"
