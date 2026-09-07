# Key–value configuration

What the keys and values actually *are* in each dataset, how a trial is
constructed, and why the (K, N) grids stop where they stop.

Source of truth: `mechanistic_probing_v2/core/dataset_configs.py` — note this is
a live dependency, not legacy code, despite living in a superseded folder.

---

## The 2×2

| | single-token values | multi-token values |
|---|---|---|
| **no key↔value relation** | `ARBITRARY_SINGLE` | `ARBITRARY_MULTI` |
| **real key↔value relation** | `SEMANTIC_SINGLE` | `SEMANTIC_MULTI` |

Only two are used in the consolidated results:

- **`SEMANTIC_MULTI` — the main experiment.** `gemstone: alexandrite`. The value
  really is a member of the category.
- **`ARBITRARY_SINGLE` — the supporting experiment.** `gemstone: above`. The
  value is a random English word with no relation to the key. Single-token,
  which is what makes logit lens / probing / head ablation well-defined.

`ARBITRARY_MULTI` appears only in the 2-model March 2026 proprietary sweep (not
ingested). `SEMANTIC_SINGLE` is unused here.

---

## `ARBITRARY_SINGLE` — 46 keys, one **shared** pool of 2,300 words

| | |
|---|---|
| category names (keys) | 46 — `visual art`, `tools`, `landform`, `musical instrument`, `gemstone`, `fabric`, … |
| value pool | **2,300 single-token English words, shared across all categories** |
| pool type | `shared` |
| example values | `above`, `absorb`, `accent`, `accept`, `access`, `accord`, `accuse`, `ace`, `acre`, `across` |

Per trial: sample K categories, then draw **K × N values from the single shared
pool without replacement** and slice them into K blocks of N.

Two consequences that make this the right control:

1. **No value can be linked to a key by meaning.** `gemstone: above` is
   arbitrary, so the model cannot shortcut retrieval via semantic association —
   it must track position.
2. **No value repeats anywhere in a trial**, across keys or within one. Any
   intrusion is therefore unambiguously attributable to one source position.

### The pool is what caps the grid

Because sampling is without replacement from a 2,300-word pool:

> **K × N ≤ 2300**

Of the 81 cells in the sweep grid (K∈{2,3,5,7,10,15,20,25,30} × N∈{5,7,10,15,20,30,50,75,100}),
exactly **two are impossible by construction**: K=25/N=100 (2,500) and
K=30/N=100 (3,000). Maximum achievable is **79**.

That predicts the observed coverage exactly:

| model | cells | missing | pool-bound | context/other |
|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | **79** | 2 | 2 | **0** |
| Qwen2.5-3B-Instruct | **79** | 2 | 2 | **0** |
| Qwen3.5-2B | **79** | 2 | 2 | **0** |
| Qwen3.5-9B | **79** | 2 | 2 | **0** |
| gemma-3-4b-it | 76 | 5 | 2 | 3 |
| Qwen2.5-3B *(base)* | 73 | 8 | 2 | 6 |
| gemma-3-1b-it | 55 | 26 | 2 | 24 |
| Qwen3.5-4B | 50 | 31 | 2 | 29 |

**Four models ran the complete achievable grid** — their "79/81" is not a
truncation, it is the ceiling. Only the bottom four lost cells to context limits
(gemma-3-1b-it and Qwen2.5-3B run 8192-token contexts) or, for Qwen3.5-4B, to an
interrupted run.

## `SEMANTIC_MULTI` — 46 keys, **per-category** pools

| | |
|---|---|
| categories (keys) | 46 |
| total values | 2,403 |
| pool per category | **45–70 values** (min `chemical element` = 45, max `tools` = 70) |
| pool type | `per_category` |
| example | `ancient civilization: akkadian, amorite, anasazi, assyrian, athenian, axumite, aztec, …` |
| origin | the ACL paper's `interleaved_dataset_meaningful.json` |

Per trial: sample K categories, then draw N values **from that category's own
pool**. Values are category-appropriate, so `gemstone` only ever takes gemstone
names.

### Here the cap is per-category depth, not a global pool

> **need K categories each holding ≥ N values**

| N | categories with ≥ N values |
|---|---|
| 1–30 | **46** (all) |
| 50 | **40** |
| ≥75 | **0** |

So N is hard-capped at 50 — which is why the Semantic grid tops out at N=50
while Arbitrary reaches N=100. And of the 70 cells in the proprietary grid
(K∈{1,2,5,10,15,20,25,30,40,45} × N∈{1,5,10,15,20,30,50}), exactly one is
impossible: **K=45/N=50** needs 45 categories with ≥50 values and only 40 exist.

Observed: all six proprietary models ran **69/70**, missing exactly K=45/N=50.
Complete runs, no truncation.

---

## Trial construction (identical for both datasets)

1. Sample K categories, seeded — `rng.sample(eligible, K)`.
2. Draw N values per category (shared pool vs per-category pool, as above).
3. Build K×N `category: value` items, then **interleave with
   `shuffle_no_consecutive`** — a random order with the constraint that no two
   adjacent lines share a key. This prevents a key's updates from clustering.
4. Query one category, `cats[seed % K]`.
5. Ground truth is that category's **first** (FVQ) or **last** (CVQ) value *in
   stream order*, not in generation order.

Seeds are deterministic functions of `(K, N, condition, trial_idx)`, so base and
adapted models see byte-identical prompts — paired comparisons throughout.

## Load axes

- **K = num_keys** — how many competing keys share the context. Raises
  *interference*.
- **N = num_updates** — how many times each key is overwritten. Raises
  *chain depth*: the distance the model must traverse to reach the first value,
  and the number of stale candidates competing with the last one.

Total context items = K × N, so K=30/N=75 is 2,250 lines.

---

## LoRA train / held-out split — by **category**, not by cell

`lora_intervention/data_gen.py:56-72`

| | count | examples |
|---|---|---|
| `TRAIN_CATEGORIES` | **35** | visual art, tools, landform, musical instrument, gemstone, fabric, … |
| `HELD_OUT_CATEGORIES` | **11** | stadium name, surgical procedure, constellation, spice blend, guitar type, hat style, painting medium, volcano name, fruit variety, sword type, board game |

The 46 categories partition cleanly, 35 + 11, with **no overlap**. So LoRA
evaluation is held out along two independent axes at once:

1. **Unseen categories** — the 11 above never appear in training.
2. **Unseen load** — training used K∈{2,3,5,10} × N∈{5,10,15,20}; evaluation
   starts at K=7/N=30 and runs to K=30/N=75.

That is what makes "fixes 28/28 held-out cells" a generalisation claim rather
than a memorisation one. Worth stating explicitly in the paper — the category
split is currently only visible in code.

Note the arithmetic control adapter (`qwen_arith_adapter`) trained on GSM8K and
so has no category split at all.

---

## Why this matters for reading the results

- **The Arbitrary/Semantic difference is not cosmetic.** Arbitrary removes the
  semantic route to the answer entirely. That the gap survives it — r = 0.926
  against Semantic across 29 matched cells (`build/check_arb_vs_sem.py`) — is
  the evidence that the failure is positional, not lexical.
- **Missing cells are almost never failures.** In the four complete runs, every
  absent cell is mathematically impossible. Do not read a gap in the grid as a
  model breaking down.
- **Grid ceilings differ by dataset**, so "N up to 100" (Arbitrary) and "N up to
  50" (Semantic) are pool properties, not design choices — relevant if the paper
  compares depth-scaling across the two.
