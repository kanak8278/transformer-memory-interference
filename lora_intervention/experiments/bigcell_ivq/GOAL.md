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
- **Trials:** Wilson early-stop, floor 30 / cap 200, half-width threshold 0.07
  (canonical `evaluate.py` setup — restored now that this runs on a dedicated
  H100 instead of a shared Kaggle T4).
- **Runner:** `lora_intervention/experiments/e1_extrapolation_frontier.py`
  `--scans C` (HF backend, resumable, greedy).

## Models

| model | base | adapter | notes |
|---|---|---|---|
| Qwen2.5-3B-Instruct | `Qwen/Qwen2.5-3B-Instruct` | `checkpoints/adapter` | done |
| gemma-3-4b-it | `google/gemma-3-4b-it` | `checkpoints/gemma_adapter` | gated (HF_API_KEY); done — see Status for the loader fix required |

## How to run (local, dedicated GPU)

Runs directly on a local CUDA GPU now (no more Kaggle/Colab — those existed
only because earlier runs had no dedicated GPU). Isolated `uv` venv at repo
root (`torch==2.5.1+cu121`, `transformers`, `peft`, `accelerate` — cu121 chosen
to match this box's driver, not cu126). `HF_API_KEY` for the gated Gemma
weights lives in `lora_intervention/.env` (never commit it).

Use the supervisor scripts (retry-on-crash, resumable, logs to
`out/{qwen,gemma}/{supervisor.log,run.log,results.jsonl}`):

```bash
lora_intervention/experiments/bigcell_ivq/run_qwen.sh   # micro-batch 32, sdpa
lora_intervention/experiments/bigcell_ivq/run_gemma.sh  # micro-batch 16, sdpa
```

Both pin `CUDA_VISIBLE_DEVICES=1` (adjust to whichever GPU is free), source
`HF_API_KEY` from `lora_intervention/.env` into `HUGGING_FACE_HUB_TOKEN`, and
redirect `HF_HOME` to `.hf_cache/` on the data partition — the home partition's
HF cache (`~/.cache/huggingface`) is often near-full on shared boxes and too
small for the ~9 GB Gemma download.

Fold results into the consolidated set with
`consolidated_results/build/build_04_lora.py` (extend `build_lora_ivq()` to read
the new `results.jsonl`), producing a scan-C block in `04_lora/lora_ivq.csv`.

## Status (2026-07-22, local H100 run) — BOTH MODELS DONE

- **Qwen: DONE.** Full scan C (base+LoRA, 20 cells x 5 conditions, n=200 Wilson,
  floor 30 / half-width 0.07) completed on a dedicated H100 in ~42 min, no
  crashes. LoRA: FVQ mean 1.000, CVQ mean 0.984 (endpoints hold far past
  training); IVQ@0.50 mean 0.246, IVQ@0.90 mean 0.196 (interior stays
  collapsed at large K,N). Base: FVQ mean 0.247, CVQ mean 0.536.

- **Gemma: DONE, after fixing two real bugs in the runner (not env/hardware).**
  Base FVQ=0.000 was NOT a transformers-version/attention-backend regression:
  1. `_model_class_for()` forced `Gemma3ForCausalLM` (text-only head), but
     `google/gemma-3-4b-it`'s Hub checkpoint is the multimodal one (language
     weights live under `language_model.*`). Loading it into the text-only
     class silently drops every real weight (confirmed via a raw-generation
     diagnostic: full MISSING/UNEXPECTED report on load, garbage
     newline-only output). `AutoModelForCausalLM` already resolves
     `Gemma3Config` to `Gemma3ForConditionalGeneration` correctly in
     transformers 5.14 -- switched to that.
  2. That alone wasn't enough: `gemma_adapter/`'s checkpoint was trained with
     *flat* `model.layers.N.*` keys (a `Gemma3ForCausalLM`-shaped model), which
     don't match `Gemma3ForConditionalGeneration`'s nested
     `model.language_model.layers.N.*` -- attaching the adapter there reports
     100% missing keys (LoRA deltas silently never applied; "lora" would have
     just been the unpatched base model). Fixed by transplanting the
     correctly-loaded `language_model` + `lm_head` into a bare
     `Gemma3ForCausalLM` shell so the adapter's flat keys line up -- verified
     zero missing/unexpected keys on attach.
  3. Bonus: the original "eager attention required for gemma-3" belief was
     itself a symptom of bug #1 (garbage under both sdpa and eager, since
     base weights were random either way). With real weights, `sdpa` gives
     identical output to `eager` at ~1/3 the memory and is faster --
     switched Gemma to `sdpa`, which also sidesteps an OOM eager's fp32
     O(seq^2) softmax hit on the biggest cell (K=30,N=50, ~8.5K tokens).

  Full scan C (base+LoRA, n=200 Wilson) completed in ~48 min, no crashes.
  LoRA: FVQ mean 0.999, CVQ mean 0.976 (endpoints hold); IVQ@0.50 mean 0.247,
  IVQ@0.90 mean 0.283 (interior collapses) -- confirms the hypothesis
  cross-family, matching Qwen's pattern closely. Base is interestingly
  inverted vs Qwen: FVQ mean 0.892 (strong), CVQ mean 0.214 (weak) -- Qwen
  base is the opposite (FVQ weak, CVQ strong). Worth a note in the paper if
  this scan-C data gets used: the two base models fail in different
  directions, but LoRA converges both to the same endpoint-holds pattern.

## Known constraints (hard-won)

- On an 80GB H100 with `sdpa`, `micro_batch` only matters up to `BATCH_TRIALS`
  (10) — the runner never passes more than 10 prompts to `generate()` per
  call, so anything >=10 behaves identically. Qwen tuned to 32, Gemma to 16
  (both far under the ~40-70GB peaks measured; real per-10-trial batches use
  less). Full 20-cell base+LoRA: ~42 min (Qwen), ~48 min (Gemma).
- `python-multipart`/HF Hub downloads over a corporate TLS-intercepting proxy
  (Zscaler) need `SSL_CERT_FILE`/`REQUESTS_CA_BUNDLE`/`CURL_CA_BUNDLE` pointed
  at the system CA bundle (`/etc/pki/tls/certs/ca-bundle.crt` on this box) —
  `uv` needs `--system-certs` for the same reason.
- Historical (Kaggle/Colab T4, no longer used): T4 ≈ 170-270s/condition at
  big cells; 16GB OOMs at `micro_batch` 6 on N>=30; Colab won't hold two
  concurrent T4s; Kaggle only exposes logs after the kernel finishes.
