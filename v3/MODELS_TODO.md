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

### Priority 3 — Gemma size series (to match our Qwen series)

We have Qwen at 0.5B, 1.5B, 3B. We should have Gemma at comparable sizes.
All Gemma 3 sizes have Gemma Scope 2 SAEs — meaning we can do SAE analysis across
the full Gemma scaling series, unique in the MI literature.

**Models to add (all MPS-feasible, all have SAEs):**

| Model | Params | SAE | Status | Run |
|---|---|---|---|---|
| google/gemma-3-270m-it | 270M | ✅ gemma-scope-2-270m-it | ❌ Not run | Needed |
| google/gemma-3-1b-it | 1B | ✅ gemma-scope-2-1b-it | ✅ Exists (50 trials) | Needs 200t + more N |
| google/gemma-3-4b-it | 4B | ✅ gemma-scope-2-4b-it | ❌ Not run | Needed |

**Skip 12B and 27B** — too large for MPS, not worth cloud cost for this paper.

**Run commands:**
```bash
# Gemma 270M
.venv/bin/python v3/scripts/experiments/stage1_sweep.py \
    --model google/gemma-3-270m-it \
    --key-levels 2 3 5 --update-levels 3 5 7 10 15 20 30 --trials 100

# Gemma 4B
.venv/bin/python v3/scripts/experiments/stage1_sweep.py \
    --model google/gemma-3-4b-it \
    --key-levels 2 3 5 --update-levels 3 5 7 10 15 20 30 --trials 100
```

---

### Priority 4 — Other new models

| Model | Arch | Params | Why useful |
|---|---|---|---|
| **Falcon-Mamba-7B-Instruct** | SSM (Mamba) | 7B | Instruction-tuned SSM, clean data, no garbage |
| **Qwen2.5-7B-Instruct** | Transformer GQA | 7B | Extends Qwen series upward |

---

### NOT Worth Running

| Model | Reason |
|---|---|
| Mamba-130M | 87% garbage — model too small to follow format |
| RWKV-430M | 99% garbage — completely unusable |
| Gemma-3-12B, 27B | Too large for MPS, not needed for paper |

---

## Final Model List for Paper

| Model | Size | Arch | Data status |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 0.5B | Transformer GQA | ✅ (50 trials, needs more N) |
| Qwen2.5-1.5B-Instruct | 1.5B | Transformer GQA | ✅ (200 trials, needs more N) |
| Qwen2.5-3B-Base | 3B | Transformer GQA | ✅ (200 trials, 11 N values) |
| Qwen2.5-3B-Instruct | 3B | Transformer GQA | ✅ (200 trials, needs more N) |
| **Gemma-3-270M-it** | 270M | Transformer MHA | ❌ need to run |
| **Gemma-3-1B-it** | 1B | Transformer MHA | ⚠️ (50 trials, needs more) |
| **Gemma-3-4B-it** | 4B | Transformer MHA | ❌ need to run |
| TinyLlama-1.1B | 1.1B | Transformer Llama | ✅ (200 trials, needs more N) |
| StableLM-1.6B | 1.6B | Transformer | ✅ (200 trials, needs more N) |
| Pythia-410M | 410M | Transformer GPT-NeoX | ⚠️ (50 trials, 48% garbage) |
| Mamba-1.4B | 1.4B | SSM | ✅ (200 trials, needs more N) |

**Total: 11 models, 2 architecture families (Transformer + SSM), 2 model families with scaling series (Qwen 4 sizes, Gemma 3 sizes)**
