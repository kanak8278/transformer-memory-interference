# Gemma-3-4b-it Mechanistic Probing — Scope and Plan

Status: planning. Behavioral evaluation in §5.2 of PAPER_PLAN.md is the
required deliverable. Mechanistic probing is a follow-up, NOT in §5.2's
scope.

## What §5.2 actually requires (paper_plan.md)

- Generate train/val data ✓ done (shared with Qwen)
- LoRA-train Gemma-3-4b-it, same config ⏳ in progress
- Evaluate on 28 held-out ARB cells (behavioral RI/PI accuracy)
- Spot-check SEM OOD on 2-3 cells

**Mechanistic probing on Gemma is NOT in §5.2.** Only Qwen's mechanistic
re-run (§5.1) is committed in the paper plan. Doing Gemma mechanism is
exploratory bonus work — useful if results land but not required to claim
"architecture-agnostic recovery."

## Why Qwen-derived mechanism doesn't carry to Gemma

| Item | Qwen2.5-3B-Instruct | Gemma-3-4b-it | Status |
|---|---|---|---|
| n_layers | 36 | 34 | different |
| n_heads | 16 (Q) | 8 (Q) — grouped-query | fundamentally different |
| Identified suppressor heads | L26H3, L27H3, L29H3, L29H4, L30H3 | unknown | **cannot reuse indices** |
| Post-LoRA promoter heads | L30-L33 cluster | unknown | **cannot reuse indices** |
| Stage 2 logit lens results | committed JSON | not run | **need from-scratch on base Gemma** |
| Stage 3 attribution targets | committed JSON | not run | **need from-scratch** |

**Bottom line**: every Qwen-specific index in the existing scripts becomes
meaningless for Gemma. The pipeline (probe classifier, logit lens, attention
routing, ablation) is transferable; the numbers are not.

## What's blocking mechanistic probing on Gemma right now

1. **`transformer_lens` not installed.** All four mechanistic scripts
   (`run_probing_lora.py`, `run_logit_lens_lora.py`,
   `run_attention_routing_lora.py`, `run_stage3_targeted_lora.py`) require
   it. Installing it is straightforward but may bring its own version
   pinning conflicts on this SageMaker env.
2. **TL Gemma-3 support uncertain.** TL supports Gemma 1/2 but Gemma-3
   architecture (released March 2025) needs TL >= ~2.0. Must verify before
   committing to the TL-based path.
3. **No Gemma base mechanistic baseline data.** Existing scripts compare
   post-LoRA against committed Qwen baseline JSONs. For Gemma we'd need
   to run base-Gemma probing/logit-lens FIRST, then merged-LoRA Gemma.
   Doubles the compute budget.

## Recommended sequencing

**Phase 1 (in scope for §5.2):**
1. Install vLLM
2. Run `evaluate.py --model google/gemma-3-4b-it --adapter none` →
   Gemma BASE behavioral baseline on held-out grid (~30 min on L4 + vLLM)
3. Run `evaluate.py --model google/gemma-3-4b-it --adapter <gemma_main>`
   → Gemma + LoRA behavioral on same grid (~30 min)
4. Compare: gap closed? regime shift? — produces the §5.2 table.

If the gap closes on Gemma too → §5.2 claim is substantiated. Ship it.

**Phase 2 (optional bonus, time permitting):**
5. Install transformer_lens, verify Gemma-3 support
6. Adapt scripts: parameterize `BASE_MODEL`, replace hard-coded layer/head
   sets with config files
7. Re-discover Gemma mechanism from scratch:
   - Stage 1: logit lens on base Gemma → find late-layer suppression layers
   - Stage 2: attribution patching on base Gemma → identify Qwen-equivalent
     suppressor heads
   - Stage 3: ablate, compare to post-LoRA Gemma
8. Side-by-side §7 comparison: "Qwen built promoters in L30-L33; Gemma
   built promoters in L*?*-L*?*"

Phase 2 is roughly **1-2 days of compute + 1 day of script adaptation**.

## Scripts that need adaptation (Phase 2 only)

All four `run_*_lora.py` scripts share the same pattern:

- Hard-coded `BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"` → CLI arg `--base`
- `AutoModelForCausalLM.from_pretrained(...)` → dispatch to
  `Gemma3ForCausalLM` for Gemma-3 (same pattern as the `_model_class_for`
  helper added to `evaluate.py` and `merge_lora.py`)
- `HookedTransformer.from_pretrained(BASE_MODEL, hf_model=..., ...)` →
  needs TL Gemma-3 support; may need to pass `model_name` differently
- `run_stage3_targeted_lora.py` BASELINE_TOP_HEADS list → must be
  replaced with Gemma-base's actual top heads from a fresh 3A run

These edits are mechanical once the design above is locked in.
