# LoRA Adapters — *Transformers Remember First, Forget Last: Dual-Process Interference in LLMs*

Provenance manifest for the LoRA intervention adapters used in the paper. Every field below is
read from each adapter's `run_config.json` / `adapter_config.json` and the training/eval scripts in
`lora_intervention/` — not inferred. Where a directory name is misleading, it's flagged.

All adapters share the same LoRA hyperparameters unless noted: **rank 16, α 32, dropout 0.05,
lr 2e-4, bf16, sdpa attention**.

| Dir | Base model | What it is | Target modules | Trained on | Result | Adapter size |
|---|---|---|---|---|---|---|
| `adapter/` | Qwen2.5-3B-Instruct | **MAIN intervention** (§5.1) | attention: q,k,v,o | ARB `data/train.jsonl` (18k), 2 epochs | fixes **28/28** cells (RI~100%, PI~95–100%) | 28 MB |
| `qwen_arith_adapter/` | Qwen2.5-3B-Instruct | **§5.3 negative control** (NOT a main adapter) | attention: q,k,v,o | GSM8K main (6,726 train), stopped at step 100 | fixes **0/5** cells; GSM8K acc 16%→69.6% | 28 MB |
| `qwen_mlp_adapter/` | Qwen2.5-3B-Instruct | **§7 critique-#7 control** (MLP-only) | MLP: gate,up,down (attn FROZEN) | same ARB data as `adapter/`, 100 steps | fixes **5/5** cells (RI 0.96–1.0, PI 0.94–0.98) | 86 MB |
| `gemma_adapter/` | google/gemma-3-4b-it | **§5.2 family control** (main task, diff arch) | attention: q,k,v,o | same ARB `data/train.jsonl`, 400 steps (early-stopped) | eval token acc 0.984 | 34 MB |

---

## Details

### `adapter/` — the main intervention LoRA  ⭐
- **run_name:** `main` · base `Qwen/Qwen2.5-3B-Instruct`
- The §5.1 task-specific adapter. Attention-only (q,k,v,o). 2 epochs on the ARB interference task.
- This is the adapter that **closes the PI/RI gap** (28/28 cells fixed at high trial count) and is the
  subject of the §7 mechanism story (L30–L33 attention promoter heads).
- A merged-into-full-weights variant (`lora_intervention/checkpoints/merged/`, ~12 GB) is built from
  THIS adapter via `merge_lora.py` (its `--adapter` default points here). That merged dir is gitignored
  and lives outside this folder; logit-lens / transformer_lens scripts load it because they can't load PEFT.

### `qwen_arith_adapter/` — §5.3 negative control  ⚠️ name is misleading
- **run_name:** `qwen_arith_control` · purpose: *"does arithmetic-domain training close the FVQ/CVQ gap?"*
- Despite living next to the main adapter and sharing its exact config, this is a **control, not an
  intervention.** Trained on full GSM8K (with chain-of-thought, a deviation from PLAN.md Decision 7's
  single-token spec — made deliberately to give a *stronger* control).
- Stopped early at step 100/212: loss plateaued, model demonstrably learned arithmetic (69.6% vs 16%
  baseline GSM8K acc) yet **fixed 0/5 interference cells** → arithmetic competence does not transfer.

### `qwen_mlp_adapter/` — §7 circularity control
- **run_name:** `qwen_mlp_lora` · purpose: rebut the "amplification via attention is structurally
  circular" critique (attention-only LoRA can only reweight existing features; can't install new ones).
- **MLP-only** (gate/up/down); attention is frozen. ~1.5× more trainable params than the attention
  adapter (22.6M vs 14.9M) because intermediate_size > hidden_size.
- **Also fixes the gap (5/5 cells).** Implication recorded in run_config: *the capacity is in the
  network; either pathway suffices.* The §7 "amplification not installation" claim must be reframed —
  the L30–L33 attention promoter heads are **a** way to express the task, not the only way.

### `gemma_adapter/` — §5.2 cross-family control
- **run_name:** `gemma_main` · base `google/gemma-3-4b-it`.
- Same ARB interference task as the main Qwen adapter, different model family → shows the intervention
  isn't Qwen-specific. Early-stopped at step 400/564 (eval loss still improving at 0.046; stopped to
  save compute on an L4). Trained on an NVIDIA L4 (23 GB), wall time 3h44m.

---

## Quick mental model

- **One real intervention:** `adapter/` (Qwen) — replicated cross-family by `gemma_adapter/`.
- **Two controls that probe *why* it works:** `qwen_arith_adapter/` (wrong domain → fails) and
  `qwen_mlp_adapter/` (different pathway → also works).
- **`merged/`** = `adapter/` baked into full weights, for non-PEFT tooling. Not in this folder (12 GB,
  gitignored); regenerate with `python lora_intervention/merge_lora.py --out .../merged`.

> Note: directory names are referenced by scripts (`merge_lora.py` defaults to
> `.../transformers-remember-first-forget-last/adapter`). They were **not** renamed when moved here —
> renaming would break those references. Update the script paths first if you want clearer names.
