# Gemma-3-4b cross-family replication — summary

**Date:** 2026-07-15. **Model:** `google/gemma-3-4b-it` (34 layers) ± the main
Gemma adapter (LoRA r16/α32, attn-only q/k/v/o, trained 1.42 ep on the same K/N
grid as Qwen: K≤10, N≤20). **Hardware:** Colab L4, HF backend, bf16, sdpa.
Purpose: run the same 3 experiments we ran for Qwen2.5-3B and check the paper's
mechanism claims hold across model families.

The original paper pipeline (ARB behavioral, logit-lens, probing, attention-
routing) was already done for gemma-3-4b (in `v3/results_vllm/`, from SageMaker).
This adds the three session-new experiments. Ablation (#4) deferred (needs
Gemma-specific promoter-head discovery). Data + per-experiment findings:
`e1/`, `block/`, `behavioral/`.

## Result: all three replicate Qwen closely

### 1. E1 extrapolation (`e1/FINDINGS.md`)
- **Endpoints (FVQ/CVQ) generalize to ~5× in both K and N** (LoRA CVQ 0.93–1.00
  at N=100 and K=46).
- **Interior N-bounded and decaying** (LoRA IVQ mean 0.49→0.05 as N grows; ~0.34
  across the K-scan). **N is the killer, not K.**
- Base current-value suppression is load-dependent (CVQ 0.52→0.04).
- Qwen: LoRA CVQ 0.96–1.00, interior 0.56→0.07. **Gemma: 0.93–1.00, 0.49→0.05.**

### 2. Block-vs-LoRA readout (`block/FINDINGS.md`)
- **Convergent readout, divergent routing:** at the K10/N50 failure cell,
  base_plain P(v_last)=0.29, base_block=0.92, lora_plain=0.98 — all peaking at
  the **same late layer L33/34**.
- "Tracked but suppressed": base tracks v_last (nonzero, rises) but suppresses it;
  both fixes release the same readout via different upstream triggers.
- Qwen: base ~0.4 → fixes ~0.98 at L35/36. Same pattern, one layer earlier.

### 3. Behavioral per-position sweep (`behavioral/FINDINGS.md`)
- **Endpoints converge** across all 8 cells + both regimes (base CVQ 0.72→0.10 →
  Block/LoRA 0.94–1.00).
- **Interior diverges — the headline:** base = U-shape (~0.01), Block = flat-high
  (0.66–0.97, genuine position indexing), LoRA = boundary-anchored decay
  (0.40→0.10, worsens with N).
- K10/N50 per-position: base 0.01 / Block 0.66 / LoRA 0.11 interior mean vs Qwen
  0.02 / 0.72 / 0.12. **Near-identical.**

## Bottom line
The paper's three session-new results are **architecture-general**, not
Qwen-specific:
1. LoRA surfaces robust *endpoint* retrieval that extrapolates to ~5×, not general
   position indexing (interior N-bounded, decaying).
2. Base tracks-but-suppresses v_last; Block and LoRA converge on the same late
   readout via different routing.
3. Block installs true position indexing; LoRA only amplifies end-anchored
   retrieval. Endpoints converge, interior diverges.

Readout layer differs (Gemma L33/34 vs Qwen L35/36, both near-final); everything
else matches quantitatively. Strong second-family evidence for the paper.

## Operational notes (for reproduction)
- Download: `google/gemma-3-*` is Xet-backed but downloads fine on Colab with a
  modern stack (`transformers==4.56.2`, `huggingface_hub>=0.34.0`, `hf_transfer`,
  `HF_XET_HIGH_PERFORMANCE=1`). The old "Xet blocks Colab" note is stale.
- Gemma-3 loads via `Gemma3ForCausalLM` (text head; `AutoModelForCausalLM` pulls
  the VLM). Chat template folds the system role into the first user turn.
- VM image ships torchao 0.10.0 → crashes on adapter load; force `torchao>=0.16.0`.
- Long prompts (K10/N100) OOM at fixed batch-16 on a 23GB L4 → length-aware
  micro-batching. Both fixes committed; runs are per-cell resumable.
