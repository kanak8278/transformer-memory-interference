# Entropy-Lens experiment — reproducible run recipe

Entropy-Lens (arXiv:2502.16570) on the KV-interference task, comparably across
**base** (Qwen2.5-3B-Instruct), **lora** (+ main adapter), and **scratch** (the
from-scratch GPT-2-small trained only on the synthetic task).

## What is computed
Per trial, one forward pass. At the **answer position**, apply a logit lens at
every layer: `softmax(unembed(final_norm(resid_post_L)))` over the full vocab →
Shannon entropy (nats). Stored as raw nats **and** normalised `H / log(vocab)`.
Correctness is scored for free from the same forward (single-token argmax at the
answer position). Profiles are split into all / correct / wrong trials.

Answer position: Qwen = last prompt token; scratch = the step-token position
(index -2 of the EOS-trimmed sequence), matching `eval_utils.evaluate_split`.

## Comparability (identical config across all three models)
- **Grid** (`GRID` in `run_entropy_lens.py`): K∈{2,4,6,8,10,12} × N∈{4,6,8,10,12}
  = 30 cells — the scratch model's native range, runnable by all three.
- **Conditions**: FVQ (first), CVQ (last), IVQ_d25/d50/d75 (intermediate at
  relative depths 0.25/0.50/0.75; interior step = clip(round(d·N),2,N-1), same
  rule as linear_probing).
- **Trials**: 150 per (cell, condition).
- Cross-model axes: y = normalised entropy `H/log(V)` (Qwen V=151936 vs scratch
  V=51); x = relative depth `layer/n_layers` (Qwen 36L vs scratch 12L).
- Qwen sees text ARBITRARY_SINGLE prompts (linear_probing trial generators);
  scratch sees its own 51-symbol integer prompts (`data_gen.build_example`).
  Different tokenizations, same task/cell/query-type/relative-depth.

## Determinism / reproducibility
- Trial seeds are `hash((nk,nu,condition,t_idx,tag)) % 2**31`. Python randomises
  str/tuple hashing per process, so **`PYTHONHASHSEED=0` is required** for the
  seeds (hence the trials) to be identical across reruns. All commands below set
  it. (Same caveat applies to linear_probing.)
- Greedy/argmax only, no sampling.

## Environment (every run)
```
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
export HF_HOME=/rnd_ai_datasets1/projects/raka6003/.hf_home
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
export PYTHONHASHSEED=0
```

## Commands
```
PY=.venv/bin/python
EL=lora_intervention/experiments/entropy_lens
# both Qwen models in parallel on the two GPUs, then the tiny scratch model:
$PY $EL/run_entropy_lens.py --model base    --device cuda:0    # -> results/entropy_base.json
$PY $EL/run_entropy_lens.py --model lora    --device cuda:1    # -> results/entropy_lora.json
$PY $EL/run_entropy_lens.py --model scratch --device cuda:0    # -> results/entropy_scratch.json
$PY $EL/analyze_entropy.py                                     # -> profiles/confusion PNG + summary
```
Or run the whole thing (waits for linear probing to free the GPUs first):
```
bash $EL/run_entropy_suite.sh   # logs to results/, prints "[suite] SUITE COMPLETE"
```

## Model sources
- base: `Qwen/Qwen2.5-3B-Instruct`
- lora: base + `lora_intervention/checkpoints/adapter` (PEFT, no merge)
- scratch: `synthetic_scratch_training/05_gpt2_scratch_h100/checkpoints/h100_cosine/best.pt`
  (GPT-2-small, 12L/768d, vocab 51, step 28000, val_acc 0.797)

## Outputs
`results/entropy_{base,lora,scratch}.json`, `entropy_profiles.png`,
`entropy_confusion.png`, `entropy_summary.json`.
