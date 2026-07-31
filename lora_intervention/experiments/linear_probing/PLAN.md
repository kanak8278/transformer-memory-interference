# Linear probing — plan (base vs LoRA, per-layer correctness decodability)

Status: pilot build. Single model first (Qwen2.5-3B-Instruct, base vs main
adapter). Extend to gemma-3-4b-it only after the pilot's cell grid and probe
design are validated.

This is a fresh, self-contained experiment (own folder, own script) rather
than a patch to the old `run_probing_lora_hf.py` / `probing_classifier.py`
pipeline — those only ever ran FVQ/CVQ at a single degenerate K2/N5 point
(flagged "DO NOT USE" in `EXPERIMENT_STATUS.md`). We reuse their trial
generation code (prompts are fixed, unchanged) but build new
activation-collection + probe-training + orchestration code that supports
FVQ/CVQ/IVQ across a real K,N grid.

---

## 1. Question

Is CVQ/FVQ/IVQ correctness *linearly decodable* from the residual stream at
each layer, and does that change base → LoRA? This extends the paper's
"tracked but suppressed" mechanism story (E2 block-readout, E4 ablation) with
a systematic per-layer, per-condition, per-load view instead of two spot-check
cells.

## 2. Dataset & prompts — UNCHANGED from existing mechanistic pipeline

- **Dataset: ARBITRARY_SINGLE only.** Same as every other mechanistic
  experiment in this repo (C2 logit lens, C3 routing, C4 ablation, C5
  probing, E1–E4). Values are verified single-token (`verify_single_token`),
  which is what makes exact-match correctness labeling and clean per-token
  readouts possible. SEMANTIC_MULTI (multi-token) is explicitly not used here
  for the same reason it isn't used anywhere else in the mechanism pipeline.
- **Prompts are reused verbatim, not redesigned:**
  - FVQ/CVQ: `generate_trial()` from `v3/scripts/experiments/probing_classifier.py`
    (system prompt "You are a precise data extraction tool...", user turn
    "Read the following key-value stream... What was the {first/last} value
    of {category}?", `format_for_chat` chat-templated). Internally still
    called RI (=FVQ) / PI (=CVQ) in code, matching repo-wide convention.
  - IVQ: `make_prompt()` from `lora_intervention/evaluate_ivq.py` (same
    system prompt, ordinal wording "What was the {2nd/3rd/...} value of
    {category}?", anti-consecutive-category shuffle `shuffle_nc`,
    `apply_chat_template` directly).
  - These two use slightly different internal shuffle logic (plain
    `rng.shuffle` for FVQ/CVQ vs. `shuffle_nc` — no two consecutive stream
    items share a category — for IVQ). This is *not* harmonized: each
    condition keeps exactly the generation logic its original script used.
    Only the metadata each trial returns is extended (candidate value list,
    category) so our new script can log the same bookkeeping run_probing_lora_hf.py
    already logs.

## 3. Models / conditions

| State | What |
|---|---|
| `base` | `Qwen/Qwen2.5-3B-Instruct`, unmodified |
| `lora` | Same base + `lora_intervention/checkpoints/adapter` (main adapter, r16/α32, q/k/v/o), loaded via PEFT directly — **no merge step**. `merge_and_unload()` was only ever needed by scripts requiring a plain HF dir (e.g. for TransformerLens); we call the model with the PEFT wrapper active, which behaves identically for a forward pass and skips writing a ~6GB merged checkpoint to disk. |

Gemma-3-4b-it (± gemma adapter) is the planned second model, deferred until
the pilot is validated (see §7).

## 4. K,N cells — mixed train + held-out, reusing an existing convention

Reusing `evaluate_ivq.py`'s `EVAL_CELLS` grid (already used for the
dense-IVQ base-vs-LoRA comparison, so its numbers are cross-checkable against
`lora_intervention/results/dense_ivq_ivq_*.json`):

| Split | Cells | Purpose |
|---|---|---|
| `train` | (5,10), (10,20) | Inside the LoRA training grid (`TRAIN_KEYS=[2,3,5,10]`×`TRAIN_UPDATES=[5,10,15,20]`) — did training itself change decodability here? |
| `held_moderate` | (10,30), (15,30) | Just past the training grid |
| `held_hard` | (20,30), (25,50) | Well past the training grid, from the paper's canonical 28-cell held-out set |

Full pilot (post-smoke-test) additionally adds (2,5) — the retired K2/N5
probing point, kept for continuity with the old runs — and the remaining
cells of the 28-cell held-out grid
(`K∈{7,10,15,20,25,30}×N∈{10,15,20,30,50,75}`, 28 of 36 combos, from
`lora_intervention/results/main_eval_20260523_055309.json`) as budget allows.

## 5. Query types per cell

- **FVQ** — first value of the queried key.
- **CVQ** — last/current value of the queried key.
- **IVQ, depth-stratified** — interior position at 5 relative depths of the
  stream length N: {10%, 25%, 50%, 75%, 90%}, position
  `k = clip(round(depth * N), 2, N-1)`. Stratified rather than pooled: E1
  already showed IVQ accuracy varies steeply by depth (≈1.0 at 10% depth →
  ≈0.2 at 75% depth for the main Qwen adapter at N=20), so pooling would
  average away exactly the structure we want to see.

Full sub-condition list per cell: `FVQ, CVQ, IVQ_d10, IVQ_d25, IVQ_d50,
IVQ_d75, IVQ_d90` (7 total).

## 6. What "correct" means (ground truth, computed once per trial)

We generate the trial, so we already know the expected value string
deterministically. The model greedy-decodes one token (values are verified
single-token); `correct = decoded_text.strip().lower() == expected.lower()`
— exact string match, no embedding similarity, no fuzzy matching. This
matches `run_probing_lora_hf.py`'s existing convention (stricter than
`evaluate.py`'s multi-token-safe lenient match — appropriate here since
ARBITRARY_SINGLE guarantees single-token values and we only decode one
token).

This label is computed **before** the probe ever runs; the probe only ever
sees `(residual_vector, correct_bit)` pairs. It has no access to the actual
words involved.

**Descoped for this pilot:** an earlier idea to also probe "which value-index
does the model's answer match" as a proxy for whether the *correct* position
is internally represented independent of output — on inspection this
collapses to be identical to the correctness label whenever candidate values
are unique (which they always are here), so it adds no information over the
correctness probe. A genuine independent test of "tracked but suppressed"
(is the correct value's logit rising internally even when the argmax output
is wrong) is what the existing logit-lens experiments (C2, E2) already do —
not duplicated here. If we want a probe-level version of this later, it
needs a different construction (e.g. probe trained to predict expected_idx
directly as a multi-class target, independent of the model's own output),
noted as a possible future extension, not built now.

## 7. Probe training

Independent logistic-regression probe **per layer**, per (condition, cell,
state) — not one probe pooled across cells or layers. Rationale: pooling
across cells risks the probe learning "which cell is this" (residual-stream
scale differs a lot across K,N) as a shortcut for correctness, and it would
erase the K,N-dependence that's the actual object of study.

- **Classifier:** `Pipeline(StandardScaler(), LogisticRegression(C=0.1,
  max_iter=500, solver="lbfgs", class_weight="balanced"))`.
  Two deliberate deviations from the old `probing_classifier.py` precedent
  (flagging, not silently changing): added `StandardScaler` (old script fed
  raw activations directly; standardizing stabilizes LR across layers/models
  with very different activation scales) and `class_weight="balanced"`
  (correctness labels are often skewed — e.g. LoRA at 90% CVQ gives a 90/10
  split — unweighted LR would just learn the majority class).
- **Evaluation:** k-fold cross-validation (stratified where possible),
  `k = min(5, min class count)`, so small/skewed smoke-test samples degrade
  gracefully instead of erroring.
- **Majority-class baseline reported alongside every probe accuracy** — a
  probe "beating" the majority baseline is the actual signal; matching it is
  not informative.
- **Degenerate handling:** if a (condition, cell, state, layer) has only one
  class present, skip LogisticRegression, record `"degenerate": true` +
  majority baseline. This is an expected, informative outcome at several
  cells (e.g. LoRA CVQ at (5,10) is likely all-correct), not a bug.

## 8. Output

One JSON per (state, cell) under `results/`, e.g.
`probe_Qwen2.5-3B-Instruct-base_5k_10u.json`:

```json
{
  "state": "base | lora",
  "base_model": "Qwen/Qwen2.5-3B-Instruct",
  "adapter_path": "... | null",
  "cell": {"keys": 5, "updates": 10},
  "n_layers": 36,
  "conditions": {
    "FVQ": {
      "behavioral_accuracy": 0.95, "n": 20,
      "layers": {
        "0": {"cv_accuracy": 0.55, "majority_baseline": 0.90,
              "degenerate": false, "n_pos": 18, "n_neg": 2},
        "...": "..."
      }
    },
    "CVQ": {...},
    "IVQ_d10": {"depth": 0.10, "...": "..."},
    "IVQ_d25": {...}, "IVQ_d50": {...}, "IVQ_d75": {...}, "IVQ_d90": {...}
  },
  "elapsed_sec": ..., "device": "cuda", "dtype": "bfloat16"
}
```

`behavioral_accuracy` per condition is a free cross-check against existing
tables (`main_eval_*.json`, `dense_ivq_ivq_*.json`) — if it doesn't roughly
match at shared cells, the trial-generation code has a bug, not the probe.

## 9. Smoke test (before the full run)

Purpose: validate the pipeline end-to-end on a tiny slice before committing
GPU time to the full grid.

- **Cells:** all 6 of `evaluate_ivq.py`'s `EVAL_CELLS` — (5,10), (10,20),
  (10,30), (15,30), (20,30), (25,50) — chosen specifically because
  `dense_ivq_ivq_20260523_101418.json` already has base/LoRA IVQ numbers at
  these exact cells to cross-check against.
- **Conditions:** FVQ, CVQ, IVQ_d50 only (skip the other 4 IVQ depths for
  the smoke test — full depth stratification is a full-run concern).
- **States:** base and lora (both — need to exercise the PEFT-load path too).
- **Trials:** 20 per (condition, cell, state) — fixed, no Wilson early-stop
  logic yet.
- **Checks:**
  1. Spot-print a few raw trials (prompt, injected values, expected answer)
     per condition — eyeball correctness of the ground truth.
  2. Hidden-state shapes are right (36 layers, d_model=2048 for Qwen2.5-3B).
  3. Behavioral accuracy per (condition, cell, state) roughly matches
     `dense_ivq_ivq_*.json` / `main_eval_*.json` at shared cells.
  4. Degenerate-cell detection fires without crashing where expected (likely
     LoRA CVQ/FVQ at in-training cells).
  5. Output JSON is well-formed and loads cleanly.
  6. Record seconds/trial to project full-grid runtime.

Not attempting to draw conclusions from n=20 CV numbers — this stage is only
about pipeline correctness.

## 10. Sequencing after the pilot

1. Smoke test (this file, §9) → fix anything broken.
2. Scale trials to Wilson-capped 200/condition/cell/state, keep the 6-cell
   grid, add all 5 IVQ depths → first real pilot results on Qwen2.5-3B.
3. Look at layer-trajectory plots together; adjust the cell grid if needed
   (e.g. degenerate cells found in smoke test may need trimming or
   replacing).
4. Expand cells to the full 31 (3 in-training + 28 held-out).
5. Port to gemma-3-4b-it (± gemma adapter) — same script, `--base
   google/gemma-3-4b-it --adapter .../gemma_adapter`, dispatch to
   `Gemma3ForCausalLM`.
