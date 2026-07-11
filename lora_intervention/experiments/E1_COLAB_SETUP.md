# E1 on Colab (L4) — working recipe

Validated 2026-07-11 via a smoke run (base+LoRA, cells (2,30) & (2,1000)).

## Environment gotchas (discovered during smoke)
1. **Do NOT use vLLM on Colab.** Prebuilt vLLM wheels expect CUDA 13
   (`libcudart.so.13`); Colab ships CUDA 12 → `ImportError` on import. The
   runner defaults to `--backend hf` (transformers). Greedy decoding is
   deterministic, so results match what vLLM would give.
2. **Uninstall `torchao`.** Colab ships `torchao 0.10.0`; the current `peft`
   (0.19.1) raises `ImportError: incompatible torchao ... only >0.16.0` during
   adapter injection, even though our LoRA is not quantized. Removing torchao
   makes peft's guard skip it. `pip uninstall -y torchao`.
3. HF backend needs only `peft` + `transformers` + `torch` (torch/transformers
   preinstalled on Colab). No vLLM install required.

## Recipe
```bash
# provision (isolated config so it never touches other projects' session state)
colab --config /tmp/e1_colab.json new -s e1 --gpu L4

# clone public repo on VM (via a single exec running git clone)
#   git clone --depth 1 -b aaai-prep https://github.com/kanak8278/transformer-memory-interference.git /content/repo

# deps: ensure peft, remove torchao (one exec)
#   pip install -q peft ; pip uninstall -y torchao

# launch DETACHED (idempotent pgrep guard), log to file:
#   cd /content/repo && python lora_intervention/experiments/e1_extrapolation_frontier.py \
#       --out-dir <OUT> --scans A --backend hf
# monitor via: colab download <OUT>/run.log  (never a second exec)
# stop when done: colab stop -s e1
```

## Timing (L4, HF backend, observed at smoke n=8)
- N=1000 (~10K-token stream): ~30 s per condition at n=8; runs without OOM.
- Full run cost dominated by high-N cells (500/750/1000) × 100 trials, but
  Wilson early-stopping collapses saturated (acc≈0 or ≈1) conditions to ~40
  trials. Rough Scan A estimate: ~1.5–3 L4-hours for base+LoRA.

## Durability
- Runner is resumable: skips completed (model,scan,K,N,condition) rows in
  results.jsonl. For the full run, either mount Drive (interactive, per-VM) and
  point --out-dir there, or write to /content and `colab download` results.jsonl
  periodically so a VM loss costs at most the un-downloaded tail.
