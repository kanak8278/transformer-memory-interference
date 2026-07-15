# Experiments — master inventory

One-line description + what was measured, then models / configurations / datasets,
for every experiment in this repo. Grouped by phase. Datasets live in
`mechanistic_probing_v2/core/data/` (ARBITRARY_SINGLE, ARBITRARY_MULTI,
SEMANTIC_SINGLE, SEMANTIC_MULTI); original-study data in `data/` + `results/`.

Naming: **FVQ** = first-value query (old "RI"), **CVQ** = current/last-value query
(old "PI"), **IVQ** = intermediate k-th-value query.

---

## A. Original interference study (ACL paper foundation)

**A1. PI vs RI behavioral screen** — probe conflicting key–value updates; recall
first value (RI/FVQ) vs last value (PI/CVQ).
- *Measured:* accuracy, PI vs RI gap (Cohen's d), scaling vs model size.
- *Models:* 39 total — API (Claude, GPT, Gemini families) + AWS Bedrock models.
- *Config:* varying #keys / #updates; narrative and key-value formats.
- *Data:* `data/interleaved_dataset*.json`, `data/narrative_interference/`;
  raw results in `results/raw_pi_*`, `results/raw_ri_*`, `results/ri_vs_pi_comparison/`.

---

## B. LoRA / Block interventions — training

**B1. LoRA adapter training (main)** — fine-tune attention (q/k/v/o) to fix
current-value retrieval.
- *Measured:* train/eval loss + token accuracy.
- *Models:* Qwen2.5-3B-Instruct (`checkpoints/adapter`); google/gemma-3-4b-it
  (`checkpoints/gemma_adapter`).
- *Config:* LoRA r=16, α=32, attn-only q/k/v/o, dropout 0.05, lr 2e-4; trained on
  grid K∈{2,3,5,10} × N∈{5,10,15,20}, condition mix FVQ 40% / CVQ 40% / IVQ 20%.
  Qwen 2 epochs; Gemma 1.42 ep (stopped early). bf16.
- *Data:* ARBITRARY_SINGLE (18k train / 2k val / 600 test-ID / 2200 test-OOD).

**B2. Arithmetic-control adapter** — control LoRA trained on an unrelated task.
- *Measured:* whether a non-interference LoRA changes CVQ (specificity control).
- *Models:* Qwen2.5-3B-Instruct (`checkpoints/qwen_arith_adapter`).
- *Config:* same LoRA hyperparams (r16/α32, q/k/v/o).
- *Data:* arithmetic task data (`data_gen_arithmetic.py`).

---

## C. Mechanistic pipeline (open-weight models)

**C1. Stage-1 behavioral sweep** — accuracy across the K×N grid for FVQ/CVQ/IVQ
(establishes each model's interference profile).
- *Measured:* per-condition accuracy vs #keys, #updates.
- *Models (15):* Qwen2.5 {0.5B, 1.5B, 3B, 3B-Instruct}; Qwen3.5 {0.8B, 2B, 4B,
  9B}; Gemma-3 {270m, 1b, 4b}-it; **architecture controls** mamba-1.4b-hf,
  pythia-410m, stablelm-2-1.6b-chat, TinyLlama-1.1B-Chat.
- *Config:* ARB single-token; K×N sweep. *Data:* ARBITRARY_SINGLE.
- *Results:* `v3/results_vllm/arbitrary_single/<model>/stage1_sweep_*.json`.

**C2. Stage-2 logit lens** — per-layer P(v_last)/P(v_first) via final-norm→lm_head.
- *Measured:* layer at which the current/first value becomes readable.
- *Models:* Qwen2.5-3B-Instruct (base + LoRA); gemma-3-4b-it (base + LoRA).
- *Config:* CVQ & FVQ prompts; per-layer readout. *Data:* ARBITRARY_SINGLE.
- *Results:* `v3/results_vllm/logit_lens/`; `lora_intervention/results/logit_lens_comparison.txt`.

**C3. Attention-routing head analysis** — rank heads by FVQ−CVQ attention
differential (promoter/suppressor head discovery).
- *Measured:* per-head FVQ−CVQ, primacy/recency, argmax-round.
- *Models:* Qwen2.5-3B-Instruct (normal, reversal, K2N10, K5N20); gemma-3-4b-it
  (normal, K2N30, K5N15).
- *Config:* HF `output_attentions`; per (layer,head) scoring. *Data:* ARBITRARY_SINGLE.
- *Results:* `v3/results_vllm/attention_routing/head_analysis/head_analysis_report.md`;
  plots in `v3/plots/attention_routing/`.

**C4. Stage-3 causal ablation** — zero promoter/suppressor heads, measure ΔP(v_last).
- *Measured:* causal contribution of heads to the v_last readout (base vs LoRA).
- *Models:* Qwen2.5-3B-Instruct (baseline-promoters, LoRA-promoters, LoRA-clean5);
  gemma-3-4b-it (pre-existing suppressor-head run).
- *Config:* head-set ablation; logit-lens readout. *Data:* ARBITRARY_SINGLE.
- *Results:* `v3/scripts/experiments/results/Qwen2.5-3B-Instruct-*promoters*/stage3_causal_*.json`.

**C5. CVQ-correctness probing** — linear probe on residual stream for whether CVQ
will be answered correctly, per layer.
- *Measured:* probe accuracy per layer (base vs LoRA).
- *Models:* Qwen2.5-3B-Instruct-LoRA; gemma-3-4b-it (base + LoRA).
- *Config:* 2 keys / 5 updates; 5-fold CV. *Data:* ARBITRARY_SINGLE.
- *Results:* `v3/results_vllm/probing/probing_*_2k_5u.json`.

---

## D. LoRA-intervention evaluations

**D1. Main held-out eval** — does the trained LoRA fix CVQ on held-out K×N cells?
- *Measured:* FVQ/CVQ/IVQ accuracy, base vs LoRA, ID + OOD cells.
- *Models:* Qwen2.5-3B-Instruct ±adapter; gemma-3-4b-it ±adapter.
- *Config:* 28-cell held-out grid; SEM + Wilson CIs. *Data:* ARBITRARY_SINGLE.
- *Results:* `lora_intervention/results/main_eval_*.json`, `gemma_main_eval_*.json`.

**D2. Dense-IVQ eval** — CVQ/IVQ accuracy at every interior position for a cell.
- *Measured:* per-position accuracy (base vs LoRA).
- *Models:* Qwen2.5-3B-Instruct ±adapter.
- *Config:* dense position sweep. *Data:* ARBITRARY_SINGLE.
- *Results:* `lora_intervention/results/dense_ivq_*.json`.

**D3. GSM8K capability control** — does the interference LoRA hurt general ability?
- *Measured:* GSM8K task accuracy (base vs LoRA vs arith-control).
- *Models:* Qwen2.5-3B-Instruct (base, +adapter, +arith-control).
- *Data:* GSM8K. *Results:* `lora_intervention/results/gsm8k_task_eval_*.json`.

**D4. Attention-routing base-vs-LoRA comparison** — how LoRA shifts head routing.
- *Measured:* per-head routing deltas (normal + reversal).
- *Models:* Qwen2.5-3B-Instruct base vs LoRA.
- *Results:* `lora_intervention/results/attention_routing_*comparison*.txt`.

---

## E. Session-new experiments (cross-family: Qwen2.5-3B-Instruct + gemma-3-4b-it)

All use the main adapters, ARB single-token, Wilson early-stop where behavioral.
Qwen results in `lora_intervention/experiments/<exp>_results/`; Gemma in
`lora_intervention/experiments/gemma_results/<exp>/`.

**E1. Extrapolation frontier** — push the LoRA beyond its training grid; where does
it still generalize?
- *Measured:* FVQ/CVQ/IVQ accuracy at multiples (1–5×) of training K,N (base vs LoRA).
- *Models:* Qwen2.5-3B-Instruct; gemma-3-4b-it (±adapter).
- *Config:* N-scan (K=10, N=20→100), K-scan (N=20, K=10→46); IVQ depths
  {0.1,0.25,0.5,0.75,0.9}; min30/cap100 trials, Wilson HW≤0.07. *Data:* ARBITRARY_SINGLE.

**E2. Block-vs-LoRA readout comparison** — do the Block-format fix and the LoRA fix
converge on the same internal readout?
- *Measured:* per-layer P(v_last) (logit-lens) + CVQ-correctness probe, 3 conditions
  (base_plain / base_block / lora_plain).
- *Models:* Qwen2.5-3B-Instruct; gemma-3-4b-it.
- *Config:* cells K2/N30 + K10/N50, 100 trials. Probe degenerate for lora_plain
  (≈100% correct → single class). *Data:* ARBITRARY_SINGLE.

**E3. Behavioral 3-way per-position sweep** — base vs Block vs LoRA accuracy at every
position (the interior-divergence signature).
- *Measured:* per-position accuracy; endpoint (FVQ/CVQ) + interior means.
- *Models:* Qwen2.5-3B-Instruct; gemma-3-4b-it.
- *Config:* 8 cells (dense {5×20,10×25,10×50}, alternate {5×50,5×75}, endpoints
  {2×30,20×20,10×100}); 3 conditions; min30/cap200, Wilson HW≤0.07. *Data:* ARBITRARY_SINGLE.

**E4. Stage-3C ablation (n=200, paired)** — ablate promoter heads, base vs LoRA, at
n=200 with paired seeds (resolves Finding 001; cross-family amplification test).
- *Measured:* ΔP(v_last) (ablated−normal) per readout layer; paired (LoRA−base) Δ with
  bootstrap 95% CIs.
- *Models:* Qwen2.5-3B-Instruct (heads L30–33 cluster; cells K2/N5 + K2/N50);
  gemma-3-4b-it (heads L23/L17/L29 cluster; cells K2/N30 + K10/N50).
- *Config:* HF forward-hook ablation of 8 promoter heads (top-8 FVQ−CVQ per model),
  n=200 paired seeds, n_boot=2000; readout layers Qwen L31–35 / Gemma L30–33.
  *Data:* ARBITRARY_SINGLE.

---

## Coverage summary

| experiment | Qwen2.5-3B-Inst | gemma-3-4b-it | other models |
|---|---|---|---|
| A1 PI/RI screen | — | — | 39 API/Bedrock |
| C1 behavioral sweep | ✅ | ✅ (+1b,270m) | 11 more incl. mamba/pythia |
| C2 logit lens | ✅ | ✅ | — |
| C3 head routing | ✅ | ✅ | — |
| C4 causal ablation | ✅ | ✅ (suppressor) | — |
| C5 probing | ✅ | ✅ | — |
| D1–D4 LoRA evals | ✅ | ✅ (D1) | — |
| E1 extrapolation | ✅ | ✅ | — |
| E2 block-vs-LoRA readout | ✅ | ✅ | — |
| E3 behavioral 3-way | ✅ | ✅ | — |
| E4 stage-3C ablation | ✅ | ✅ | — |

All four session-new experiments (E1–E4) are two-family (Qwen + Gemma). See
`lora_intervention/experiments/gemma_results/GEMMA_CROSS_FAMILY.md` for the Gemma
cross-family synthesis and `NEW_RESULTS_AND_NEXT_STEPS.md` for status.
