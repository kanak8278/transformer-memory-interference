# Experiment Summary — All vLLM Results

Generated: 2026-04-09

## Overview
118 JSON result files across 8 experiment types, 11 models, 2 datasets.

## Stage 1: Behavioral Sweep (22 files)
**11 models × 2 datasets (ARBITRARY_SINGLE + SEMANTIC_MULTI)**

| Model | Size | ARBI Gap | SEM Gap |
|-------|------|----------|---------|
| Qwen 0.5B-Inst | 0.5B | +31.0% | +39.7% |
| Qwen 1.5B-Inst | 1.5B | +46.4% | +32.4% |
| Qwen 3B-Inst | 3B | +2.5% | +4.8% |
| Qwen 3B-Base | 3B | -11.8% | -15.1% |
| Gemma 270m | 270M | +15.3% | +6.4% |
| Gemma 1b | 1B | +54.3% | +51.1% |
| Gemma 4b | 4B | +45.9% | +32.0% |
| TinyLlama | 1.1B | +28.2% | +11.7% |
| StableLM | 1.6B | +65.8% | +65.8% |
| Pythia | 410M | -3.8% | -27.1% |
| Mamba | 1.4B | +64.0% | N/A |

**Key finding**: PI > RI in 8/11 models. Base models without instruction tuning show reversed asymmetry.

## Stage 2: Logit Lens (8 files)
**6 clean + 2 NaN (Gemma 270m, 4b)**

Qwen 0.5B/1.5B/3B-Inst/3B-Base, Gemma 1b, Pythia — all show Pattern B:
P(v_last) rises at ~85% depth then gets outcompeted in final layers.

## Stage 3: Causal Analysis (6 files)
**Qwen x4, Gemma 1b, Pythia**

3A Attribution + 3B Targeted Patching + 3C Ablation Logit Lens.
Finding: Distributed mechanism, no single bottleneck head.

## Training Dynamics (16 files)
**SmolLM2-1.7B: 16 checkpoints from 125K to final**

PI > RI gap present from step 125K (+17%), peaks at 625K (+44%), fluctuates, settles at +25% (final).
Both RI and PI improve during training; RI improves more consistently.

## Probing Classifiers (6 files)
**Qwen x4, Gemma 1b, Pythia (200 trials each at 2k_5u)**

Condition discrimination: 81-100% (models clearly encode RI vs PI).
RI correct probe: 51-83% (above chance).
PI correct probe: 0-73% (varies by model, often near chance).

## Jacobian at Init (14 files)
**7 models × (untrained + pretrained)**

Untrained models show primacy bias (Qwen 0.5B: 2.1× ratio).
Pretrained models show recency dominance (training builds recency > primacy).
All models show U-shape (first + last positions have high influence).

## Analysis (6 files)
- Scaling law fits (PI = exp decay, R² = 0.83-0.97 for most models)
- Error positions (3 failure modes: near-last, primacy fallback, diffuse)
- Cross-model scaling (RI~size r=0.88, PI~size r=0.73)

## File Locations
```
v3/results_vllm/
├── arbitrary_single/     # 11 models
├── semantic_multi/       # 11 models
├── logit_lens/           # 8 models
├── causal/               # 6 models
├── training_dynamics/    # 16 SmolLM2 checkpoints
├── probing/              # 6 models
├── jacobian/             # 14 files (7 models × 2)
├── analysis/             # 6 analysis files
├── summary/              # 4 consolidated files
├── RUN_LOG.md            # Full run history
└── EXPERIMENT_SUMMARY.md # This file
```
