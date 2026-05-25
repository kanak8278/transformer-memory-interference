# Experiment: Baseline Promoter-Head Ablation Control

**Status:** TODO (~15 minutes of compute on MPS)
**Author intent:** Close the missing cell in the §7 cross-method
convergence table (`probing / logit-lens / attention-routing / Stage 3C`).

---

## 1. Why we run this

We have causal ablation evidence that the 8 LoRA-installed *promoter heads* in
L30–L33 are necessary for the post-LoRA P(v_last) recovery
(see `results/stage3_three_way_comparison.txt`, Condition 3:
ablating those 8 heads in the LoRA model drops P(v_last) at L32
from **0.98 → 0.70**, Δ = −0.28).

What we **don't** have is the **baseline-model control** for the same
ablation. The exact question:

> When we ablate the **same 8 head IDs** in the **frozen baseline model**
> (no LoRA), does P(v_last) at L32 change?

The expected answer is **no, near-zero effect** — because in baseline these
heads attend to v_last only weakly
(P(attend to v_last) ≈ 0.05–0.23 from
`results/attention_routing_comparison.txt`). They are dormant pre-LoRA;
LoRA *installs* their v_last-attending function.

**Reviewer question this addresses:**
> *"How do you know those heads are LoRA-installed promoters rather than
> heads that were already v_last carriers and LoRA just amplified them?"*

If baseline-ablation has ~0 effect → the heads gained their function from
LoRA, not just amplification.
If baseline-ablation has a large effect → mechanism claim has to be
re-framed as amplification, not installation.

---

## 2. What to run, concretely

### Model

- **Base model identifier (HuggingFace):** `Qwen/Qwen2.5-3B-Instruct`
- **No LoRA adapter applied.** Run the frozen pretrained model end-to-end.
- 36 transformer layers (L0–L35), 16 attention heads per layer, hidden
  dim 2048, head dim 128.

### Heads to ablate (exact list)

The same 8 heads identified as LoRA-installed promoters via attention
routing
(source: `results/attention_routing_comparison.txt`,
top 8 by Δ P(attend to v_last round) under CVQ):

| # | Layer | Head | Pre-LoRA P(attend v_last) | Post-LoRA P(attend v_last) | Δ |
|---|-------|------|---------------------------|-----------------------------|------|
| 1 | 32    | 3    | 0.108                     | 0.789                       | +0.682 |
| 2 | 32    | 7    | 0.126                     | 0.777                       | +0.651 |
| 3 | 31    | 12   | 0.166                     | 0.814                       | +0.648 |
| 4 | 31    | 15   | 0.233                     | 0.863                       | +0.630 |
| 5 | 32    | 10   | 0.063                     | 0.669                       | +0.606 |
| 6 | 30    | 11   | 0.190                     | 0.781                       | +0.590 |
| 7 | 32    | 0    | 0.085                     | 0.675                       | +0.590 |
| 8 | 32    | 14   | 0.077                     | 0.661                       | +0.584 |

Hard-code these in the script as `PROMOTER_HEADS` (identical list to the
LoRA promoter-ablation run).

### Ablation method

**Zero out the attention output for these 8 heads at every forward pass.**

Concretely, in HookedTransformer terms: install a hook on each
(layer, head) pair in `PROMOTER_HEADS` that sets the per-head output
contribution to zero. The exact mechanism matches what the existing
`run_stage3_promoter_ablation.py` does (i.e. patch the head's output to
the zero vector at the residual-stream write).

**Do NOT modify base attention weights**; this is a *zero-ablation hook*,
not a weight edit.

### Eval data / operating point

- **Dataset:** Arbitrary-Single (`ARBITRARY_SINGLE` in
  `mechanistic_probing_v2.core.dataset_configs`)
- **Operating point:** K = 2, N = 5
  - Matches the existing Stage 3C runs so values are directly
    comparable
  - Use `--point 2,5`
- **Trials:** 50
  - Matches Conditions 1–3; same trial budget so noise is comparable
- **Condition:** PI ("last value" query). The probe trial logic uses the
  same seeded interleaved sequence builder as `evaluate.py`.

### Measurement

For each trial:

1. Run the **baseline model with the 8-head ablation hook installed**
   through the full prompt
2. Apply the model's own logit lens (residual stream at every layer L
   projected through `W_U`) at the answer-token position
3. Record per-layer `P(v_last)` at layers L0, L10, L20, L25, L28,
   L30, L31, L32, L33, L34, L35 (same layer set as the other Stage 3C
   reports)

Average over the 50 trials.

### Comparison condition (already exists)

Compare against:

- **Baseline normal** (no ablation) — already in
  `v3/results_vllm/causal/Qwen2.5-3B-Instruct/stage3_causal_20260409_055011.json`,
  under PI normal trajectory. Use as the reference baseline trajectory.

So the new row in the Stage 3C table will be:

| Layer | BASE normal | **BASE ablated (NEW)** | BASE Δ |
|-------|-------------|------------------------|--------|

You compute Δ = ablated − normal per layer; the load-bearing cell is
**L32 Δ**, which we predict will be near zero.

---

## 3. How to run it

### Option A — Fork the existing script

`run_stage3_promoter_ablation.py` already ablates these exact heads, but
loads the **merged LoRA model**. Fork it to load the frozen base model
instead:

```bash
cp lora_intervention/run_stage3_promoter_ablation.py \
   lora_intervention/run_stage3_baseline_promoter_ablation.py
```

Then in the new script:

1. **Replace the merged-model loader** with the standard base-model
   loader:

   ```python
   # OLD: model = load_merged_into_tl(args.merged_path, device, dtype)
   # NEW:
   tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
   model = HookedTransformer.from_pretrained(
       BASE_MODEL,
       device=device,
       dtype=dtype,
       fold_ln=True,
       center_writing_weights=True,
       center_unembed=True,
   )
   model.eval()
   ```

2. **Keep `PROMOTER_HEADS` unchanged** (same 8 heads from §2 above).

3. **Change `--out_name` default** to e.g. `Qwen2.5-3B-Instruct-baseline-promoters`
   so the JSON doesn't overwrite the LoRA-ablation output.

4. **`--stage2`** — point at the **baseline** stage2 logit-lens JSON, not
   the LoRA one:

   ```
   --stage2 v3/results_vllm/logit_lens/Qwen2.5-3B-Instruct/stage2_logit_lens_20260409_054632.json
   ```

   (Same path the original Stage 3C baseline run used; needed so the
   `analyze_stage2` helper finds the per-trial anchor data.)

### Run command

```bash
.venv/bin/python lora_intervention/run_stage3_baseline_promoter_ablation.py \
    --stage2 v3/results_vllm/logit_lens/Qwen2.5-3B-Instruct/stage2_logit_lens_20260409_054632.json \
    --point 2,5 \
    --trials 50 \
    --experiments 3C \
    --out_name Qwen2.5-3B-Instruct-baseline-promoters
```

Note: `--experiments 3C` skips the 3A/3B paths (we only need 3C — the
logit-lens-under-ablation trajectory).

### Expected runtime

~5–10 minutes on M3 Pro / MPS / bf16 (50 trials × ~5–10 s/trial).

---

## 4. Outputs

### JSON file

Save to: `v3/results_vllm/causal/Qwen2.5-3B-Instruct-baseline-promoters/`
(or wherever the existing `save_results` helper writes by default;
match the LoRA-promoter run's directory pattern.)

The JSON should contain a Stage 3C per-trial-averaged trajectory: at each
of the recorded layers, `{"P_v_last_normal": ..., "P_v_last_ablated": ...,
"delta": ...}`.

### Per-layer summary (manual table)

Fill in this table by hand from the JSON output once the run completes:

```
P(v_last) trajectory — BASELINE model + 8 LoRA-promoter heads ablated
K=2, N=5, 50 trials, PI condition

Layer    BASE normal    BASE ablated    BASE Δ
L 0      0.0000          ___            ___
L10      0.0000          ___            ___
L20      0.0000          ___            ___
L25      0.0000          ___            ___
L28      0.0000          ___            ___
L30      0.0000          ___            ___
L31      0.0207          ___            ___
L32      0.3902          ___            ___    ← key cell (predicted ≈ 0)
L33      0.5942          ___            ___
L34      0.5038          ___            ___
L35      0.3977          ___            ___
```

(Baseline-normal values come from the existing
`results/stage3_3c_comparison.txt` table.)

---

## 5. How to write up the result

### Update `results/stage3_three_way_comparison.txt`

Append a fourth condition:

```
4. BASELINE model + 8 LoRA-promoter heads (control)
   (Tests whether the heads were already v_last carriers pre-LoRA, or
   whether LoRA installed their function.)
```

with the corresponding trajectory table.

### Update the §7 convergence table (in `paper/main.tex` once §7 is drafted)

Fill the row:

| Method | Baseline | +LoRA | Δ |
|---|---|---|---|
| Stage 3C ablation effect at L32 | **___** (from new run) | −0.28 | ___ |

Predicted: baseline cell ≈ 0, Δ ≈ −0.28.

### Interpretation for the prose

If the predicted result holds (baseline Δ at L32 ≈ 0):

> *"Ablating the 8 promoter heads in the **baseline** model produces no
> measurable change in P(v_last) at L32 (Δ ≈ 0), confirming that these
> heads gained their v_last-attending function during LoRA training rather
> than being amplified from pre-existing function."*

If baseline Δ is non-trivial (say ≥ 0.05):

> *"Ablating the 8 promoter heads in the baseline model produces a
> small reduction in P(v_last) at L32 (Δ = X), indicating that these
> heads carried partial v_last attention pre-LoRA. The post-LoRA effect
> (Δ = −0.28) is therefore more accurately described as **amplification**
> of a partial pre-existing circuit rather than installation of a new one.
> We re-frame the §7 mechanism claim accordingly."*

---

## 6. Sanity checks before declaring the run complete

- [ ] The 8 heads listed in §2 are exactly the heads installed as hooks
- [ ] `trust_remote_code=True` is on (Qwen tokenizer needs this)
- [ ] Random seed matches the baseline normal run (50 trials, same seeded
  prompts so paired comparison is meaningful)
- [ ] Output JSON contains per-layer arrays, not just aggregates
- [ ] L32 ablated value is reported (not just the change)

---

## 7. If something looks wrong

- **Baseline ablated L32 P(v_last) is dramatically higher than normal
  (e.g. > 0.6)** → likely the hook installed the wrong heads or the wrong
  model (would mean we're seeing release-from-suppression). Check the
  hook layer/head indices against `PROMOTER_HEADS`.
- **Trajectory is identical to LoRA-ablated trajectory** → script is
  still loading the merged model. Verify `BASE_MODEL` is passed without
  `--merged_path`.
- **All zeros across all layers** → tokenizer mismatch or eval data not
  loading. Check `--stage2` path resolves and produces valid trial
  anchors.

---

## 8. Cross-references

- LoRA-side counterpart (already run): `run_stage3_promoter_ablation.py`
- Three-way comparison summary:
  `results/stage3_three_way_comparison.txt`
- Attention-routing source of the 8 heads:
  `results/attention_routing_comparison.txt`
- §7 plan that depends on this cell:
  `paper/PAPER_PLAN.md` §7b (optional / nice-to-have)
