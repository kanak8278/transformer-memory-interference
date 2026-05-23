# LoRA Intervention Experiment

## Scientific Question

Is the PI > RI asymmetry (and the regime D reversal) caused by training data
coverage, or is it a structural property of the architecture?

The few-shot experiment (May 2026) showed that 5 in-context examples partially
help at moderate load but fail to close the gap at high load. LoRA is the
graduated test: if targeted gradient signal on exactly this task format fixes
the asymmetry across held-out cells and datasets, the failure is data-driven.
If it doesn't, a structural floor remains.

**Pre-registered outcomes and interpretations:**

| Result | Interpretation |
|---|---|
| Both RI and PI high on held-out cells (K≥15, N≥30) | Training data hypothesis confirmed — model can learn this, pretraining didn't cover it |
| Held-out cells generalize, SEMANTIC_MULTI also improves | Strong — general state-tracking learned, not format memorization |
| Training cells fixed, held-out cells don't generalize | Cell-level memorization — negative result, argues against data hypothesis |
| Gap narrows but high-load floor persists (K≥20, N≥50) | Mixed — data component at moderate load, structural floor at extreme load |
| CVQ rises but FVQ drops | Shortcut learned (model outputs v_last always) — invalidates the run |
| Stage 3: same heads strengthened post-FT | LoRA calibrated existing mechanism — supports calibration story |
| Stage 3: new heads, original heads weakened | LoRA built a different mechanism — report both |

---

## Decision Log

Every decision is justified against the specific baseline data for
Qwen2.5-3B-Instruct (ARBITRARY_SINGLE, stage1_sweep_20260409_000134.json).

### Decision 1: Model — Qwen2.5-3B-Instruct

**Why this model, not others:**
- Only model with Stage 3 causal results (L24H1/L27H3/L23H0 identified,
  ablating 5 heads drops P(v_last) by 54%). Post-FT Stage 3 re-run directly
  tests what LoRA changed mechanistically.
- Full baseline sweep done (arbitrary_single + semantic_multi).
- Shows both regime C (RI > PI) and regime D (PI > RI reversal) — necessary
  to test whether LoRA fixes both the gap and the reversal.
- 3B is fast: ~1h per LoRA run on L40S.

Gemma-3-4b-it: Stage 3 failed (bfloat16 + raw residual issue). Can't do
post-FT mechanistic tracing. Skip for now; add as family-control follow-up
if budget allows.

### Decision 2: Training Dataset — ARBITRARY_SINGLE only

**Why ARBITRARY_SINGLE:**
- Single-token values → the label for every training example is one token.
  Clean cross-entropy loss, no multi-token generation artifacts.
- 2300-word pool × 46 categories → enough diversity to avoid memorization.
- Matches the dataset used in all existing mechanistic work (Stage 2, Stage 3).
  Pre/post comparison is apples-to-apples.

**Why not SEMANTIC_MULTI for training:**
- Multi-token values make the loss noisier (partial credit across tokens).
- Introduces real-world semantic associations (ruby IS a gemstone) that could
  confound whether the model learned state tracking or semantic knowledge.
- Reserved as pure OOD test — if LoRA on ARBITRARY_SINGLE generalizes to
  SEMANTIC_MULTI, the claim is strong. Training on both would muddy this.

### Decision 3: Category Split (35 train / 11 held-out)

**Train categories (35):**
visual art, tools, landform, musical instrument, gemstone, fabric, tree species,
cheese variety, architectural style, cloud formation, bird species, culinary herb,
flower species, wine variety, dance style, pasta shape, literary genre, cooking
method, mathematical concept, weather phenomenon, ocean current, mineral type,
coffee variety, telescope type, martial art, sea creature, psychology term,
chemical element, dinosaur genus, programming language, ancient civilization,
bridge type, photography technique, boat type, cartoon character

**Held-out categories (11):**
stadium name, surgical procedure, constellation, spice blend, guitar type,
hat style, painting medium, volcano name, fruit variety, sword type, board game

**Why category holdout is a WEAK test:**
ARBITRARY_SINGLE values are random English words with no semantic link to their
category. "gemstone: hammer" and "fruit variety: hammer" are structurally
identical to the model. Category generalization here tests surface-form
variation only. The primary generalization test is cell holdout (below).
Report category holdout as a sanity check, not the headline result.

### Decision 4: Cell Split — the critical decision

**Training cells (16 cells):**

| K | N values |
|---|---|
| 2 | 5, 10, 15, 20 |
| 3 | 5, 10, 15, 20 |
| 5 | 5, 10, 15, 20 |
| 10 | 5, 10, 15, 20 |

Context sizes at training cells: 50–730 tokens (stream only).

Note: K=10, N=15 (RI=49%, PI=52%) and K=10, N=20 (RI=44%, PI=52%) are
already regime D. They are INTENTIONALLY included — they are exactly the
failure mode LoRA is supposed to fix. The gradient signal is clean: accuracy
is ~50%, failures are intermediate intrusions (not garbage), so the model is
doing something coherent and can be corrected.

**Held-out test cells (no overlap by construction):**

| K | N values | Notes |
|---|---|---|
| 7 | 30, 50, 75 | K=7 not in training at any N |
| 10 | 30, 50, 75 | N≥30 only; training was N≤20 |
| 15 | 15, 20, 30, 50, 75 | K=15 not in training |
| 20 | 15, 20, 30, 50, 75 | K=20 not in training |
| 25 | 10, 15, 20, 30, 50, 75 | K=25 not in training |
| 30 | 10, 15, 20, 30, 50, 75 | K=30 not in training |

Context sizes at test cells: 1000–7000+ tokens. The model is trained on
50-730 token contexts and tested on 1000-7000 token contexts. This is the
genuine generalization gap. If LoRA still helps here, the learning is
about task structure, not context-length memorization.

**Why these specific training cells (not harder ones):**
Training on easy-to-moderate cells serves two purposes:
1. The model can actually learn from them (high enough accuracy for meaningful gradients).
2. Tests generalization from low-load to high-load — the mechanistically interesting direction.
If LoRA only works when trained on hard cells, it learned those cells, not the task.

### Decision 5: Training Data Composition

**40% FVQ + 40% CVQ + 20% intermediate k-th value**

| Query type | Fraction | Format |
|---|---|---|
| FVQ — "What was the first value of X?" | 40% | Matches RI evaluation condition |
| CVQ — "What was the last value of X?" | 40% | Matches PI evaluation condition |
| Intermediate — "What was the k-th value of X?" | 20% | k uniform from 2 to N-2 |

**Why 20% intermediate (not 0%):**
The training dynamics experiment showed a brief window at step 625K where RI
suddenly became better than PI — the only such window in SmolLM2's entire
training. This suggests the model transiently developed something like general
state-tracking ability, which was then overwritten by stronger recency signal.

Including intermediate k-th queries in LoRA training provides the signal the
model never got during pretraining: "track ALL N values, not just endpoints."
This prevents the shortcut: LoRA could pass FVQ/CVQ training by just amplifying
the recency-attending heads (L24H1/L27H3/L23H0) for CVQ. Intermediate queries
break that path — you cannot answer "what was the 4th value?" with a recency
heuristic.

**Why 20% (not more):**
- 80% of training data directly matches evaluation → strong alignment
- Intermediate queries at small N (N=5 gives k ∈ {2,3}) are very limited;
  the intermediate signal comes primarily from larger N training cells
- If 20% isn't enough, we can increase in a follow-up run

**Total: 20,000 unique training examples**
Distribution: ~1,250 per training cell across 16 cells.
Split: 18,000 train / 2,000 validation (10% hold-out from training cells only).

### Decision 6: LoRA Configuration

```python
rank          = 16
alpha         = 32        # 2× rank — standard effective learning rate scaling
target        = ["q_proj", "k_proj", "v_proj", "o_proj"]  # attention only
dropout       = 0.05
lr            = 2e-4      # cosine decay, 100-step linear warmup
epochs        = 2
batch_size    = 8
grad_accum    = 8         # effective batch = 64
max_seq_len   = 2048      # covers all training cells with overhead
dtype         = bfloat16
optimizer     = AdamW (paged, 8-bit)
```

**Why attention-only (no MLP):**
Stage 3 causal identified the interference mechanism in attention heads
(L24H1/L27H3/L23H0). Keeping LoRA to attention matrices makes the post-FT
Stage 3 re-run directly interpretable: we are intervening on the same
component we mechanistically characterized. If attention-only LoRA doesn't
converge, add MLP gates in a second run.

**Why rank 16:**
Standard for a task-specific intervention on a 3B model. Gives enough capacity
to learn a new retrieval pattern without overfitting. Rank 8 if we see
over-fitting on validation set during training; rank 32 if rank 16 doesn't
converge on training cells.

### Decision 7: Control Condition

Run a second LoRA on **arithmetic word problems** (~2,000 examples, same
format: single-token numeric answers). Evaluate on the failure grid.

- If the PI/RI gap closes → our evaluation is contaminated (something about
  LoRA training in general fixes it, not the specific task data).
- If the gap doesn't close → closure in the main experiment is task-specific.

This takes 30 min to generate data and 1h to train. Non-negotiable.

### Decision 8: Optional Comparison Condition

**Endpoint-only LoRA:** same setup but 50% FVQ + 50% CVQ, no intermediate.

Directly tests whether intermediate supervision matters. Compare:
- Condition A (with intermediate): 40/40/20
- Condition B (endpoint-only): 50/50/0

If Condition A generalizes better to held-out cells than Condition B, the
intermediate supervision was the key ingredient — directly explains the
training dynamics cliff (model needed general tracking signal, not just
endpoint signal).

Run this AFTER the main experiment if time/budget allows.

---

## Dataset Files

Generated by `data_gen.py`. All files in `lora_intervention/data/`:

```
train.jsonl         18,000 examples (35 train categories, training cells)
val.jsonl            2,000 examples (35 train categories, training cells, different seeds)
test_id.jsonl       held-out categories, test cells, ARBITRARY_SINGLE
test_ood.jsonl      SEMANTIC_MULTI, all categories, test cells
```

Each example (JSONL record):
```json
{
  "prompt": "<full chat-formatted prompt including system message>",
  "label": "ruby",
  "condition": "FVQ",
  "num_keys": 5,
  "num_updates": 10,
  "query_position": 0,
  "seed": 1234567,
  "category": "gemstone",
  "categories_in_trial": ["gemstone", "tools"]
}
```

---

## Evaluation Protocol

**Run baseline evaluation first** on the test grid using the unmodified
Qwen2.5-3B-Instruct (via vLLM, same as existing stage1 results). This is
already done; use `stage1_sweep_20260409_000134.json` as the reference.

**Post-FT evaluation** uses `evaluate.py`, which mirrors stage1_sweep.py:
- 200 trials per cell per condition (RI, PI)
- Wilson CI early stopping (threshold 0.05, min 50 trials)
- Same prompt format as baseline

**Report per cell:**
- RI accuracy (pre vs post)
- PI accuracy (pre vs post)
- Gap = RI − PI (pre vs post)
- Regime classification change (A/B/C/D)

**Success criterion (pre-registered):**
A cell is "fixed" if:
1. Both RI accuracy ≥ 0.65 AND PI accuracy ≥ 0.65 (genuine state tracking)
2. Gap reduced by ≥ 0.15 compared to baseline

Both conditions required. A cell where PI goes from 55% to 80% but RI drops
from 45% to 15% is NOT fixed — it's a shortcut.

---

## Post-FT Mechanistic Probe

Re-run Stage 3 causal (`v3/scripts/experiments/stage3_causal.py`) on the
LoRA-FT model at the same operating point: K=2, N=30.

Compare to baseline Stage 3 results:
- Baseline: L24H1 (+0.0134), L27H3 (+0.0073), L23H0 (+0.0056) promote v_last
- Post-FT: do these heads show higher/lower attribution?
- Are new heads active?

This is ~2 hours of compute and directly answers whether LoRA calibrated
the existing mechanism or built a new one.

---

## Timeline

| Step | Script | Est. time |
|---|---|---|
| Generate training data | `data_gen.py` | 20 min |
| LoRA training (main) | `train.py` | ~2h |
| LoRA training (control: arithmetic) | `train.py --control` | ~1h |
| Post-FT evaluation | `evaluate.py` | ~4h |
| Post-FT Stage 3 | `stage3_causal.py` | ~2h |
| Optional: endpoint-only comparison | `train.py --no-intermediate` | ~2h + 4h eval |

Total for minimum experiment: ~9h. Full experiment with optional condition: ~17h.

---

## File Structure

```
lora_intervention/
├── PLAN.md            ← this file
├── data_gen.py        ← generate train/val/test JSONL
├── train.py           ← LoRA training (PEFT + TRL SFTTrainer)
├── evaluate.py        ← post-FT behavioral evaluation on test grid
├── data/
│   ├── train.jsonl
│   ├── val.jsonl
│   ├── test_id.jsonl
│   └── test_ood.jsonl
├── checkpoints/       ← LoRA adapter weights (saved every epoch)
└── results/           ← evaluation JSONs, comparison tables
```
