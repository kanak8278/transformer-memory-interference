# Experiment 05 — GPT-2 (small) from scratch on H100, best-practice recipe

Status: implementation built, environment being provisioned, not yet run.

## Why this experiment

exp02 (local) and exp04 (Kaggle T4x2, 15k steps) both plateaued ~15-21% on the
from-scratch task while exp03 (pretrained) hit 66%. Before accepting the FINDINGS.md
conclusion that this is a data/distribution limitation, this experiment does two
things the earlier runs did not:

1. **Rules out code faults decisively** via `src/sanity_check.py` — a single-batch
   overfit test plus alignment/weight-tying/residual-init assertions. If a 124M
   model can drive one fixed batch to ~0 loss / 100% acc, the model+loss+data path
   is provably correct and the plateau is a genuine learning problem, not a bug.
2. **Applies GPT-2-from-scratch best practices** (nanoGPT + GPT-2 paper), several of
   which exp04 got wrong or omitted, on H100 hardware that lets us use a large batch
   and train fast.

## Hardware / environment

- 2x NVIDIA H100 PCIe (80GB each), 64 CPUs, 503GB RAM. Driver CUDA 12.2.
- Single-GPU per run by design (GPT-2-small is tiny; one H100 holds a big batch and
  avoids DDP complexity). We run the two schedule arms in parallel, one per GPU.
- All Python deps in a project-local `.venv`; all caches/model downloads under a
  shared cache root at the raka6003 level (`$HOME` is nearly full). See `env.sh` and
  the block appended to `~/.bashrc`. torch must be a CUDA-12.x build (cu12x wheel) to
  match the 12.2 driver — a userspace wheel choice, no system CUDA change.

## What's different from experiment 04 (best practices incorporated)

Source for the recipe: Karpathy nanoGPT (`train.py`, `model.py`), GPT-2 paper §2.3,
PyTorch AMP/compile docs. Full citations were gathered via web research (see LOG once
runs start).

| Knob | exp04 (T4) | exp05 (H100) | Why |
|---|---|---|---|
| precision | fp16 + GradScaler | **bf16, no scaler** | H100 has bf16; fp16-from-scratch is an instability risk. |
| matmul | fp32 | **TF32 (`set_float32_matmul_precision("high")`)** | ~faster fp32 matmuls, H100. |
| compile | none | **`torch.compile`** | ~1.85x/iter per nanoGPT. |
| dropout | 0.1 | **0.0** | Best practice for from-scratch pretraining; dropout is for small-data finetune. |
| AdamW betas | (0.9, 0.999) default | **(0.9, 0.95)** | GPT-2/nanoGPT value. |
| weight decay | 0.05, all params | **0.1 on `dim>=2` only** | nanoGPT decoupled param groups; no WD on bias/LayerNorm/1D. |
| grad clip | none | **1.0 (global norm)** | nanoGPT; prevents loss spikes. |
| peak LR | 3e-4 | **6e-4 → 6e-5** | nanoGPT GPT-2 small. |
| data loader | synchronous inline | **background prefetch process** | exp04 left the GPU idle ~18-50ms/step on pure-Python data-gen; prefetch overlaps it. |
| effective batch | 256 seq | **~2048 seq (256 micro x 8 accum)** | Bigger, less noisy gradients; H100 has the memory. |
| steps | 15000 | **30000 (configurable)** | H100 is fast; also tests the "does it ever catch up / grok" question. |

Confirmed NOT bugs (asserted, not assumed): HF `GPT2LMHeadModel` ties `wte`/`lm_head`
and applies the `0.02/sqrt(2*n_layer)` residual (`c_proj`) init; exp04's grad-accum
loss scaling (divide by accum before backward) was already correct.

## LR schedule arms (the run comparison)

Two arms, same seed / model / budget, differ only in the LR schedule:

- **`cosine`** (control): linear warmup → cosine decay to min_lr. The validated
  LLM-training default.
- **`sgdr`** (oscillating): `CosineAnnealingWarmRestarts` — warmup once, then cosine
  to min_lr over period T_0, jump back to peak, repeat with period x t_mult.
- **`plateau`** (oscillating, reactive): hold peak after warmup; drop LR (cosine ramp
  down) only after val accuracy stalls for `patience` evals, ride back up when it
  improves. Matches "lr keeps going up and down if the model is not improving."

**Research caveat (important):** warm restarts / cyclical LR are NOT standard for
large-LLM pretraining and there is evidence they can hurt it — essentially all LLM
pretraining uses a single warmup+cosine. SGDR/CLR were validated on vision CNNs.
We run the oscillating arms anyway because this is a small algorithmic / grokking-
regime task (very different from 300B-token pretraining), where a periodic LR kick
plausibly *could* help escape the plateau — and we measure it directly against the
cosine control rather than assuming.

## Task / data / vocab

Identical to experiments 01-04 — same 51-token vocab, same (K,N) grid, same held-out
step-query split, same seed=42. Reuses `vocab.py`, `grid.py`, `eval_utils.py` from
exp01 and the per-example-step `data_gen.py` from exp04 (+ a `pack_split` helper).

## Files

- `src/model_gpt2.py` — GPT-2-small from scratch, dropout=0, `configure_optimizers`.
- `src/schedulers.py` — cosine / sgdr / plateau schedules.
- `src/prefetch.py` — background-process prefetching train loader.
- `src/sanity_check.py` — the decisive debug (overfit + assertions).
- `src/train.py` — single-GPU best-practice training loop (bf16/compile/clip/...).

## Open items
1. Confirm the cu12x torch wheel runs on the 12.2 driver (`torch.cuda.is_available()`).
2. Run sanity_check first; only launch training if the overfit test passes.
3. Tune batch/accum to the H100 memory once profiled at worst-case seq_len=294.
4. Step budget 30k is a starting point; extend if val is still climbing (grokking watch).

## Changelog
- 2026-07-22: initial design + implementation; env provisioning (uv, project-local
  venv, shared raka6003 cache root).
