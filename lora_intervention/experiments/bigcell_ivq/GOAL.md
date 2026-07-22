# Big-cell IVQ extension — goal & design

## Question

The main LoRA adapter was trained on **small configs** (K ∈ {2,3,5,10},
N ∈ {5,10,15,20}). At **large configs** (more keys, deeper update chains),
how does the fine-tuned model behave at the **endpoints** (first / last value)
versus the **interior** (the k-th value in the middle)?

We already know, from the existing data:
- **Endpoints generalize** out of the training grid — LoRA holds CVQ ~0.95–1.00
  on held-out cells (test_id) and out to 5× extrapolation (E1).
- **Interior does not** — LoRA interior is ~0.5 in-config and collapses to
  ~0.1–0.2 one axis out, driven mostly by N.

**What's missing:** direct per-position (endpoint + interior) measurement for
the **main adapter** at the big cells K ∈ {15,20,25,30}. The interior there was
never measured for this adapter (only endpoints, in test_id). This experiment
fills that gap and confirms the "endpoints hold, interior stays dead at large
K,N" story with real numbers instead of inference.

## What it runs

- **Dataset:** `ARBITRARY_SINGLE` (single-token — the set the adapter trained
  on; keeps this comparable to training and to the mechanistic work).
- **Grid (scan C):** K ∈ {15,20,25,30} × N ∈ {10,15,20,30,50} = **20 cells**
  (N=75 dropped — too slow / OOM-prone on a 16 GB T4).
- **Positions per cell:** FVQ (first), CVQ (last), + 3 interior depths
  (10/50/90 %).
- **Conditions:** base **vs** the main LoRA adapter, paired (identical seeds).
- **Trials:** Wilson early-stop, floor 15 / cap 30 (coarse CIs, ±~0.15 — enough
  for the qualitative pattern, not publication error bars).
- **Runner:** `lora_intervention/experiments/e1_extrapolation_frontier.py`
  `--scans C` (HF backend, resumable, greedy).

## Models

| model | base | adapter | notes |
|---|---|---|---|
| Qwen2.5-3B-Instruct | `Qwen/Qwen2.5-3B-Instruct` | `checkpoints/adapter` | works; primary result |
| gemma-3-4b-it | `google/gemma-3-4b-it` | `checkpoints/gemma_adapter` | gated + a live FVQ=0 generation bug (see below) |

## How to run (Kaggle)

`kaggle_run.py` is the self-contained kernel. It clones this repo on the VM and
calls the runner. See its header for the exact steps. Essentials:

```bash
# edit MODEL = "qwen" | "gemma" at the top of kaggle_run.py, then:
cd lora_intervention/experiments/bigcell_ivq
kaggle kernels push -p . --accelerator NvidiaTeslaT4     # T4 is REQUIRED
kaggle kernels status  kanakraj/<qwen|gemma>-bigcell-ivq
kaggle kernels output  kanakraj/<qwen|gemma>-bigcell-ivq -p ./out
```

`--accelerator NvidiaTeslaT4` is mandatory: `enable_gpu` alone lets Kaggle
assign a **P100 (sm_60)**, which the preinstalled PyTorch no longer supports —
the first CUDA op dies with `no kernel image is available`. `"T4"` is silently
ignored; the exact enum `NvidiaTeslaT4` is required.

Gemma is gated: add an **HF token as a Kaggle secret named `HF_TOKEN`**
(Add-ons → Secrets). Never commit the token.

Fold results into the consolidated set with
`consolidated_results/build/build_04_lora.py` (extend `build_lora_ivq()` to read
the new `results.jsonl`), producing a scan-C block in `04_lora/lora_ivq.csv`.

## Status (2026-07-23)

- **Qwen:** partial — an OOM (`micro_batch` too large) cut the last Kaggle run to
  4 cells, base only. `micro_batch` is now a CLI flag (set to 4 here) so a
  re-run should complete. **Re-run pending.**
- **Gemma:** blocked on a **generation bug** — base FVQ = 0.000 (every
  first-value answer wrong), independent of GPU and attention implementation
  (seen under both sdpa on Colab and eager on Kaggle). Almost certainly a
  transformers-version regression in the gemma-3 chat/generation path. Needs a
  focused diagnostic that **dumps raw generations** (the runner discards them) to
  see what it emits — then pin transformers or fix the template. The kernel's
  smoke guard aborts in minutes so this never burns a full run.

## Known constraints (hard-won)

- Kaggle T4 ≈ Colab T4 in speed (no speedup); ~170–270 s per condition at the
  big cells. Full 20-cell base+LoRA is several hours.
- 16 GB T4 OOMs at `micro_batch` 6 on N ≥ 30 streams → use 2–4.
- Colab won't hold two concurrent T4s (2nd gets reclaimed).
- Kaggle exposes logs/output only after the kernel finishes (no live view).
