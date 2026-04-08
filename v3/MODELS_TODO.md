# Models: What We Have and What We Need to Run

## Current State

| Model | Arch | Params | Max trials | N values (2k) | Garbage | Usable |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | Transformer GQA | 0.5B | 50 | N=2..50 (10 pts) | 29% | ✅ |
| Qwen2.5-1.5B-Instruct | Transformer GQA | 1.5B | 200 | N=5,10,20 (3 pts) | 0% | ✅ |
| Qwen2.5-3B (Base) | Transformer GQA | 3B | 200 | N=1..50 (11 pts) | 20% | ✅ |
| Qwen2.5-3B-Instruct | Transformer GQA | 3B | 200 | N=5,10,20 (3 pts) | 1% | ✅ |
| TinyLlama-1.1B | Transformer Llama | 1.1B | 200 | N=5,7,10 (3 pts) | 2% | ✅ |
| StableLM-1.6B | Transformer | 1.6B | 200 | N=5,7,10 (3 pts) | 2% | ✅ |
| Gemma-3-1B-it | Transformer MHA | 1B | 50 | N=5..50 (9 pts) | 14% | ✅ |
| Pythia-410M | Transformer GPT-NeoX | 410M | 57 | N=5..50 (10 pts) | 48% | ⚠️ |
| Mamba-1.4B | SSM | 1.4B | 200 | N=5,7,10 (3 pts) | 36% | ✅ |
| Mamba-130M | SSM | 130M | 50 | — | 87% | ❌ unusable |
| RWKV-430M | RWKV | 430M | 30 | — | 99% | ❌ unusable |

---

## What We Need to Run (Priority Order)

### Priority 1 — Critical for scaling law claim (need ≥6 N values)

These models currently have only 3 data points (N=5,10,20). We cannot fit or claim
any curve with 3 points. Need to add N=3,7,15,30,50 at minimum.

| Model | Current N values | Need to add | Estimated time |
|---|---|---|---|
| **Qwen2.5-1.5B-Instruct** | 5,10,20 | 3,7,15,30,50 | ~2h (200 trials each) |
| **Qwen2.5-3B-Instruct** | 5,10,20 | 3,7,15,30,50 | ~3h (200 trials each) |
| **TinyLlama-1.1B** | 5,7,10 | 3,15,20,30 | ~1h (model is fast) |
| **StableLM-1.6B** | 5,7,10 | 3,15,20,30 | ~1.5h |
| **Mamba-1.4B** | 5,7,10 | 3,15,20 | ~3h (slow sequential) |

**Run command for each:**
```bash
.venv/bin/python v3/scripts/experiments/stage1_sweep.py \
    --model <model_name> \
    --key-levels 2 \
    --update-levels 3 7 15 30 50 \
    --trials 200
```

---

### Priority 2 — Statistical rigor (need 200 trials)

These models have <100 trials per cell. Direction is clear but CIs are wide (±14-18%).

| Model | Current trials | Target | Priority |
|---|---|---|---|
| **Gemma-3-1B-it** | 50 | 200 | HIGH — only cross-arch MHA model |
| **Qwen2.5-0.5B** | 50 | 200 | LOW — noisiest model, 29% garbage |
| **Pythia-410M** | 50-57 | 200 | LOW — 48% garbage, nearly unusable |

---

### Priority 3 — New models (not yet tested)

These would strengthen the cross-architecture claim:

| Model | Arch | Params | Why useful | Effort |
|---|---|---|---|---|
| **Falcon-Mamba-7B-Instruct** | SSM (Mamba) | 7B | Larger, instruction-tuned SSM — would give clean SSM data | High (API or large GPU) |
| **Qwen2.5-7B-Instruct** | Transformer | 7B | Extends size range upward | Medium |
| **Mistral-7B-Instruct** | Transformer | 7B | Different architecture family | Medium |
| **Phi-3.5-mini** | Transformer | 3.8B | Microsoft architecture | Medium |

---

### NOT Worth Running

| Model | Reason |
|---|---|
| Mamba-130M | 87% garbage — model too small to follow format |
| RWKV-430M | 99% garbage — completely unusable |

---

## Summary: Minimum Runs for a Credible Paper

To make the scaling law claim across models (need ≥6 N values per model):

1. Add N=3,7,15,30,50 to: Qwen 1.5B, TinyLlama, StableLM (~4 models × 5 cells × 200 trials)
2. Add N=3,15,20 to Mamba 1.4B
3. Bump Gemma to 200 trials

**Total estimated GPU time: ~12 hours on MPS (sequential)**
**Or: ~2 hours if run on A100 cloud GPU**
