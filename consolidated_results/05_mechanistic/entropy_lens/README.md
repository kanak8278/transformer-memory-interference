# Entropy-Lens — base vs LoRA vs from-scratch

Per-layer logit-lens entropy at the answer position, run identically across
three arms. Entropy-Lens is arXiv:2502.16570; this applies it to the
KV-interference task.

Per trial: one forward pass, then at the answer position apply a logit lens at
every layer — `softmax(unembed(final_norm(resid_post_L)))` over the full vocab
→ Shannon entropy. Stored as raw nats **and** normalised `H/log(V)`.
Correctness is scored **free from the same forward pass** (single-token argmax),
so every profile splits into all / correct / wrong.

| | |
|---|---|
| arms | `qwen_base`, `qwen_lora`, `gpt2_scratch` |
| models | Qwen2.5-3B-Instruct (±adapter), GPT-2-small from scratch — **no Gemma** |
| grid | K ∈ {2,4,6,8,10,12} × N ∈ {4,6,8,10,12} = 30 cells |
| conditions | FVQ, CVQ, IVQ_d25, IVQ_d50, IVQ_d75 |
| trials | 150 per (cell, condition) → **22,500 scored trials per arm**, 67,500 total |
| run | Sat 1 Aug 2026, 20:49–21:07 EDT; base 788 s, lora 971 s, scratch 99 s |
| source | `lora_intervention/experiments/entropy_lens/results/entropy_{base,lora,scratch}.json` |
| tier | all `raw` |

```
entropy_lens/
├── summary.csv          15 rows — the cross-arm comparison (3 arms × 5 conditions)
├── qwen_base/
│   ├── profiles.csv     16,200 — per-layer entropy: cell × condition × subset × layer
│   ├── confusion.csv       150 — output-layer entropy, correct vs wrong, per cell
│   └── behavioral.csv      150 — the free correctness labels, in the SHARED schema
├── qwen_lora/           (same three; profiles 13,140)
└── gpt2_scratch/        (same three; profiles 4,824)
```

**Subfolders are by *arm*, not by model** — `base` and `lora` are the same
Qwen2.5-3B-Instruct with and without the adapter, so a by-model split would
collide. The `model` and `variant` columns carry the distinction on every row.

`behavioral.csv` uses the **shared tidy-long schema** (same columns as
`01_fvq_cvq/fvq_cvq.csv` etc.) precisely so it can be concatenated with the
behavioural themes. The other three files use their own columns.

---

## Read this before using the numbers

### 1. The grid does not overlap theme 04 — at all

It was chosen as **the from-scratch model's native training range**, the only
range all three arms can run.

```
this grid:                K ∈ {2,4,6,8,10,12}          × N ∈ {4,6,8,10,12}
04_lora/lora_ivq.csv:     K ∈ {2,5,10,15,20,25,30,40,46} × N ∈ {10,15,…,100}
cells in common:          0
```

Not "few" — zero. So the Qwen rows here are the **small-K / small-N regime**,
which appears nowhere else in this folder. **Do not read LoRA's 0.81–0.93
interior accuracy here as contradicting the 0.20–0.29 quoted in theme 04.**
Different regime, not a disagreement — theme 04's interior numbers come from
cells like K=10, N=50.

### 2. `query_type` is not uniform, and differs across arms

| arm | FVQ / CVQ | IVQ_d25/50/75 |
|---|---|---|
| `qwen_base`, `qwen_lora` | **semantic** — "What was the *first*/*last* value of X?" | **ordinal** — "What was the *6th* value of X?" |
| `gpt2_scratch` | **ordinal** — a step token `<S1>`..`<S12>` | **ordinal** |

So the **IVQ comparisons are like-for-like across arms; the FVQ/CVQ ones are
not.** Base Qwen's FVQ 0.727 is a semantic-anchor query; the scratch model's
0.896 is a numeric-index query — not the same question. This folder's own
headline finding is that the two phrasings can differ by 138× on the same
target, so `query_type` is set per condition per arm and must not be pooled.

> `entropy_lens/RUN.md` states the arms are matched on "the same task, cell,
> query-type and relative depth." The query-type part is **not** correct for the
> endpoints. It does not affect the entropy findings below (those are geometry
> and calibration), only the cross-arm reading of the behavioural table.

### 3. Cross-arm axes are normalised, necessarily

36-layer / 151,936-vocab Qwen against 12-layer / 51-vocab GPT-2 only compare on
`entropy_norm` (`H/log V`) and `rel_depth` (`layer/n_layers`, in (0,1]). Use
those columns, not `entropy_nats` / `layer`, for anything cross-arm. Layer
indices run 1..n_layers; the embedding output (`hidden_states[0]`) is not lensed.

### 4. Some `wrong` subsets are empty or tiny

`profiles.csv` simply omits a subset with n=0 — LoRA was perfect on 85 of its
150 (cell, condition) combinations and the scratch model on 48. In
`confusion.csv`, `confusion_gap` is blank when either subset is empty, and
`notes` warns when `n_wrong < 20`. **LoRA's CVQ gap rests on n_wrong = 1.**
Always read `n_wrong` before quoting a gap.

---

## What it found

Verified against the raw JSONs; these reproduce
`lora_intervention/experiments/entropy_lens/FINDINGS.md` exactly.

### Two qualitatively different depth geometries

**Qwen (base and lora): expand → prune.** Entropy climbs to a peak of
**0.755–0.785 at relative depth 0.89** (layer ~32/36), then collapses to ~0 at
the output. Expansion amplitude +0.17–0.20, pruning drop ~0.73–0.77. The
canonical large-LM dynamic.

**From-scratch GPT-2: flat → cliff.** Entropy sits at ~0.57 essentially
unchanged across layers 1–11 — expansion amplitude **+0.007 to +0.053, i.e.
none** — then drops in a single step at the final layer, and commits less far
(output entropy 0.070–0.152 vs Qwen's 0.002–0.058). The task-specialised model
does not use depth to progressively resolve uncertainty; it defers all
commitment to the last layer.

### LoRA does not change the geometry

Same peak height, same peak depth (0.89), same expand-prune shape as base. It
only prunes harder at the very end: output entropy **0.002** vs base
0.023–0.058. So the adapter's effect is **not** in token-space entropy
dynamics — consistent with linear probing, where LoRA changes *what is
represented*.

### "Confidently wrong" vs genuinely confused

Output-layer normalised entropy, wrong − correct:

| arm | FVQ | CVQ | IVQ_d25 | IVQ_d50 | IVQ_d75 |
|---|---|---|---|---|---|
| `qwen_base` | +0.035 | +0.014 | +0.009 | +0.009 | +0.005 |
| `qwen_lora` | +0.048 (n=8) | +0.058 (n=1) | +0.058 (n=14) | +0.039 (n=331) | +0.042 (n=859) |
| `gpt2_scratch` | **+0.486** | **+0.373** | **+0.474** | **+0.418** | **+0.279** |

Base Qwen commits with ~0.05 entropy whether it is right or wrong — a clean
confidently-wrong signature. The from-scratch model's uncertainty **tracks its
correctness** (0.02–0.08 when right, 0.36–0.51 when wrong). LoRA is marginally
better calibrated than base, but only its IVQ_d50/d75 cells have enough wrong
trials to say so.

### The behavioural table, which no write-up covers

Pooled over the 30 cells, 4,500 trials per (arm, condition):

| arm | FVQ | CVQ | IVQ_d25 | IVQ_d50 | IVQ_d75 |
|---|---|---|---|---|---|
| `qwen_base` | 0.727 | 0.206 | 0.307 | 0.179 | **0.098** |
| `qwen_lora` | 0.998 | 1.000 | 0.997 | 0.926 | 0.809 |
| `gpt2_scratch` | 0.896 | 0.832 | 0.872 | 0.837 | 0.734 |

- **Base's trough is late, not central.** IVQ_d75 (0.098) < IVQ_d50 (0.179) <
  IVQ_d25 (0.307). Accuracy degrades monotonically with depth into the chain and
  recovers only at the exact endpoint — an asymmetric U.
- **The from-scratch model is nearly flat across conditions** (0.734–0.896)
  where base spans 0.098–0.727. On this grid the tiny task-specialised model has
  roughly 8× base's interior accuracy.
- Remember §2 before comparing the FVQ/CVQ columns across arms.

---

## QA

**Scratch arm vs `06_from_scratch/from_scratch_ivq.csv`: 150 cells, mean
|diff| 0.0156, median 0.0083, max 0.1100, 149/150 within 0.10.**

The `gpt2_scratch` arm re-measures the *same* model as theme 06 (the grokked
h100_cosine step-28,000 checkpoint) on the *same* 30 cells — but with a
different seed formula, n=150 instead of 200/500, and the checkpoint loaded from
the HF Hub rather than the local `.pt`. It is therefore an **independent
replication**, and the two agree to under two percentage points on average. This
is the only cross-experiment replication of theme 06 that exists.

## Coverage note: no Gemma

Within theme 05 this puts Entropy-Lens alongside `attention_routing.csv`, which
is also Qwen-only; `logit_lens`, `probing` and `causal_ablation` all have Gemma.
A Gemma arm is a **flag change, not new code** — `run_entropy_lens.py` takes
`--base` / `--adapter` and `lora_intervention/checkpoints/gemma_adapter`
exists — and would cost roughly 30 minutes of GPU time given the base and lora
arms took 788 s and 971 s each.

## Reproducing

`lora_intervention/experiments/entropy_lens/RUN.md` has the exact commands.
Two things it is right to stress: **`PYTHONHASHSEED=0` is required** (trial seeds
are `hash((nk,nu,cond,t,tag)) % 2**31`, and Python randomises tuple hashing per
process), and everything is greedy/argmax with no sampling.

Rebuild these CSVs with `cd build && python3 build_05b_entropy_lens.py`.
`profiles.csv` dominates the ~6 MB footprint; the three source JSONs are 2.4 MB.
