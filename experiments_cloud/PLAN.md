# Plan: Cloud Behavioral Sweep Experiments

## Context
The existing behavioral sweep (`mechanistic_probing_v2/experiments/behavioral_sweep_base_models.py`)
runs PI/RI interference experiments on local HuggingFace models. We need equivalent experiments
for cloud models (Claude, GPT, Gemini, Bedrock — 39+ models).

Two separate scripts because the dataset types have fundamentally different max-update limits:
- SEMANTIC_MULTI: ~50 updates max (real category members, 45-70 values per category)
- ARBITRARY_MULTI: 300 updates max (synthetic Prefix+Number, pool of 500)

---

## Folder Structure

```
experiments_cloud/
├── PLAN.md
├── sweep_semantic.py        # SEMANTIC_MULTI, UPDATE_LEVELS capped at 50
├── sweep_arbitrary.py       # ARBITRARY_MULTI, UPDATE_LEVELS up to 300
└── results/
    ├── semantic/
    │   ├── claude-haiku/
    │   │   ├── sweep_partial.json           ← checkpoint, overwritten every 3 cells
    │   │   ├── sweep_full_partial.json      ← same + full trial details
    │   │   ├── sweep_20260301_142301.json   ← final summary (timestamped)
    │   │   └── sweep_full_20260301_142301.json
    │   ├── gpt-4o/
    │   └── bedrock-llama-3.3-70b/
    └── arbitrary/
        └── {model}/
            └── ...
```

---

## Reused Components (no new code needed)

| Module | What we use |
|--------|-------------|
| `models/model_factory.py` | `create_model(model_name)` — routes to correct provider |
| `models/base_model.py` | `last_input_tokens`, `last_output_tokens`, `last_error` |
| `mechanistic_probing_v2/core/dataset_configs.py` | `get_eligible_categories()`, `generate_values_for_trial()`, `build_interleaved_sequence()` |
| `mechanistic_probing_v2/core/evaluation.py` | `classify_error()`, `bootstrap_ci()` |

---

## Model Tiers

Models are assigned to tiers by parameter count (open models) or capability naming
(closed models: nano < mini < standard < opus/pro). Tiered grids reduce API cost
by skipping cells that are trivially easy for large models and running dense coverage
where each tier actually shows interesting failure behaviour.

Final graph shows union of all tiers on shared axes; missing points interpolated.
Reasoning models noted in paper appendix.

### Small (< 10B / nano / flash-lite)
```
gpt-5-nano, gemini-2.5-flash-lite,
bedrock-llama-3.2-1b, bedrock-llama-3.2-3b, bedrock-mistral-7b
```

### Medium (10B–70B / mini / haiku / flash / standard-small)
```
gpt-5-mini, o3-mini, o4-mini, claude-4.5-haiku, gemini-2.5-flash,
bedrock-llama-3.3-70b, bedrock-mistral-8x7b
```

### Large + Reasoning (> 70B / opus / pro / o-series)
```
gpt-5, claude-4.5-sonnet, claude-4.5-opus*, gemini-2.5-pro*,
bedrock-llama-3.1-405b, bedrock-mistral-large, bedrock-deepseek-r1*, o3*
```
*reasoning mode enabled

**Total: 20 models** (6 small, 6 medium, 8 large — 4 reasoning)

---

## Tiered Grids

Models run their tier's grid. Final graph interpolates across the union of all levels.

### Key Levels by Tier
```python
KEY_SMALL  = [2, 3, 5, 7, 10, 12, 15, 20, 25, 30, 45]   # dense low-mid (they fail ~10-20 keys)
KEY_MEDIUM = [2, 5, 10, 15, 20, 25, 30, 40, 45]           # skip very low, mid-high
KEY_LARGE  = [2, 10, 15, 20, 25, 30, 35, 37, 40, 42, 45]  # dense high end (fail ~30-45)
```

### Update Levels by Tier (ARBITRARY_MULTI only — SEMANTIC capped at 50)
```python
UPD_SMALL  = [1, 3, 5, 10, 15, 20, 30, 40, 50, 100, 200, 300]          # dense everywhere
UPD_MEDIUM = [1, 5, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300]         # skip very low
UPD_LARGE  = [1, 20, 50, 75, 100, 125, 150, 175, 200, 225, 275, 300]    # dense high end
```

### Update Levels for SEMANTIC_MULTI (same tiers, capped at pool max ~50)
```python
UPD_SMALL_SEM  = [1, 3, 5, 10, 15, 20, 30, 40, 50]
UPD_MEDIUM_SEM = [1, 5, 10, 15, 20, 30, 50]
UPD_LARGE_SEM  = [1, 10, 20, 30, 40, 50]
```

### Union axes (for final graph)
```python
KEY_UNION = [2, 3, 5, 7, 10, 12, 15, 20, 25, 30, 35, 37, 40, 42, 45]
UPD_UNION = [1, 3, 5, 10, 15, 20, 30, 40, 50, 70, 75, 100, 125, 150, 175, 200, 225, 275, 300]
```
Points not run for a model are linearly interpolated from its nearest neighbours.

---

## Trial Design: Fully Random Sampling (True Independence)

Each trial is **fully independent** — every aspect is freshly randomised per trial:
categories, values, sequence ordering, and test category. We do NOT iterate over
categories systematically. The goal is one aggregate RI/PI number per cell, not
per-category curves.

**What "fresh" means per trial:**
- Randomly sample `num_keys` categories from the eligible pool
- Randomly pick 1 of those as the test category
- Randomly sample `num_updates` values per category from their value pools
- Randomly shuffle the interleaving (no consecutive same-category constraint)
- Seed = `hash((num_keys, num_updates, trial_idx)) % 2**31`

Two draws for the same cell `(5 keys, 5 updates)` use **different category sets,
different values, different test categories, different orderings** — truly independent.

**Trial loop per cell (num_keys, num_updates):**
```
for trial_idx in range(trials_per_cell):          # 200 trials
    seed = hash((num_keys, num_updates, trial_idx)) % 2**31
    categories    = random sample of num_keys from eligible pool
    test_category = random pick from categories
    values        = fresh sample per category from value pool
    sequence      = shuffle_no_consecutive(all items)

    for condition in ["RI", "PI"]:                # same sequence, 2 questions
        build prompt (stream + single question about test_category)
        response = model.generate(prompt)          ← 1 API call
        classify_error() + get_error_detail()
        sleep(delay)
```

**Total API calls per cell** = `trials_per_cell × 2 conditions` = 400 calls per cell

**Independent observations per cell per condition** = `trials_per_cell` = 200

Each observation has its own unique sequence, categories, and test category.
Zero within-trial correlation. With 200 trials: CI ±7% at worst case p=0.5.

---

## Prompt Format (one API call, one key)

```
Read the following key-value stream. Each key gets updated multiple times.

visual art: impressionism
tools: hammer
gemstone: emerald
visual art: baroque
tools: screwdriver
gemstone: onyx
...

What was the {first/last} value of gemstone?
Answer with ONLY the exact value. No explanation.
```

---

## Error Classification + Position Detail

Reuses `classify_error()` from `evaluation.py`. Additionally records the exact
position of the wrong answer for intrusion errors:

```python
def get_error_detail(predicted, all_values):
    """
    Returns the index and relative position of the predicted value
    within all_values for this test category.
    all_values = values in sequence order for test_category only.
    relative_pos: 0.0 = first occurrence, 1.0 = last occurrence.
    """
    pred_lower = predicted.lower().strip()
    for idx, val in enumerate(all_values):
        if val.lower() in pred_lower or pred_lower.startswith(val.lower()):
            return {
                "predicted_idx": idx,
                "predicted_relative_pos": round(idx / max(len(all_values) - 1, 1), 4),
            }
    return {"predicted_idx": None, "predicted_relative_pos": None}
```

Error types from `classify_error()`:
- `correct`
- `primacy_intrusion` — PI task, gave first value instead of last
- `recency_intrusion` — RI task, gave last value instead of first
- `intermediate_intrusion` — gave some middle value (position tracked above)
- `garbage` — value not in sequence (hallucination)

---

## Per-Trial Result Record (stored in sweep_full_*.json)

```json
{
  "trial_idx": 3,
  "seed": 1847263,
  "condition": "RI",
  "test_category": "gemstone",
  "categories_in_sequence": ["gemstone", "tools", "visual art", "dance", "landform"],
  "num_keys": 5,
  "num_updates": 10,
  "expected": "ruby",
  "predicted": "jade",
  "correct": false,
  "error_type": "intermediate_intrusion",
  "error_detail": {
    "predicted_idx": 2,
    "predicted_relative_pos": 0.2222
  },
  "token_usage": {
    "input_tokens": 312,
    "output_tokens": 3,
    "was_truncated": false,
    "error": null
  }
}
```

---

## Sequential Convergence (Early Stopping Within a Cell)

Rather than always running all 200 trials, each cell stops early once the
estimate has converged. This concentrates the trial budget on uncertain cells
(near the inflection point) and saves 60-80% on trivial ceiling/floor cells.

**Mechanism:** trials are submitted in batches of `workers` (40). After each
batch, Wilson CI half-width is checked for both RI and PI. Stop when both
are <= CI_THRESHOLD (7%).

```
Batch 1 (40 trials)  → check CI → converged? stop : continue
Batch 2 (80 trials)  → check CI → converged? stop : continue
...
Batch 5 (200 trials) → final (max)
```

**Wilson score interval half-width** (handles p near 0/1 correctly):
```
hw = z * sqrt(p(1-p)/n + z²/(4n²)) / (1 + z²/n)    where z=1.96 for 95% CI
```

**Why ±7%?**

| True accuracy | Stops after | Trials used | Saved vs 200 |
|---------------|-------------|-------------|--------------|
| p=0.99 (ceiling) | Batch 1 | 40 | 80% |
| p=0.95 (easy) | Batch 2 | 80 | 60% |
| p=0.80 (moderate) | Batch 3 | 120 | 40% |
| p=0.50 (inflection) | Never | 200 | 0% |
| p=0.20 (degraded) | Batch 3 | 120 | 40% |
| p=0.05 (near-floor) | Batch 2 | 80 | 60% |
| p=0.01 (floor) | Batch 1 | 40 | 80% |

±5% would never stop at p=0.5 (good) but barely improves savings at ceiling vs ±7%.
±10% stops at p=0.5 after 80 trials — loses precision exactly where needed.
±7% is the sweet spot: full budget for uncertain cells, 60-80% savings elsewhere.

**MIN_TRIALS_CHECK = DEFAULT_WORKERS (40):** first check only happens after
the first full batch. This is natural — the batch IS the check interval.
No point checking mid-batch since all workers are already in-flight.

**Resume note:** resuming from checkpoint automatically applies early stopping
to all remaining (not-yet-completed) cells. Already-completed cells in the
checkpoint are not re-run — they used full 200 trials which is valid, just
slightly more than necessary. New cells will benefit from early stopping.

**Paper justification (for Appendix):**
"Trials continued until the 95% Wilson score CI half-width fell below ±7%
for both RI and PI, or until 200 trials were completed, whichever came first.
This concentrates the trial budget on cells near the accuracy inflection point
while stopping early on ceiling and floor cells."

---

## Cell Stats (stored in sweep_*.json summary)

```json
"5_10": {
  "num_keys": 5,
  "num_updates": 10,
  "regime": "B",
  "elapsed_sec": 45.2,
  "n_trials": 80,
  "n_observations": 80,
  "stopped_early": true,
  "max_trials": 200,
  "stats": {
    "RI": {
      "accuracy": 0.85,
      "ci_lower": 0.78,
      "ci_upper": 0.91,
      "n": 50,
      "error_types": {
        "correct": 43,
        "recency_intrusion": 4,
        "intermediate_intrusion": 2,
        "garbage": 1
      }
    },
    "PI": { "...": "..." }
  }
}
```

---

## CLI Args (identical in both scripts except defaults)

```
--model          Model name via create_model() (default: claude-haiku)
--tier           Grid tier: very_small | small | medium | large (default: small)
--trials         Max trials per cell (default: 200, stops early via CI check)
--workers        Concurrent trials per cell = batch size for CI check (default: 40)
--save-dir       Output directory (default: experiments_cloud/results/semantic or arbitrary)
--resume         Path to sweep_partial.json to resume from
--key-levels     Space-separated ints, overrides tier default
--update-levels  Space-separated ints, overrides tier default
```

---

## Early Stopping (saturation)

For each fixed key count `k`, after running each update level `n`, check whether
accuracy has collapsed to near-zero. If so, skip remaining update levels for `k`.

**Three-zone rule (derived from Clopper-Pearson exact binomial bounds, n=200 trials):**

```
acc <= 1.5%  → near-zero zone  : increment zero_count
1.5% < acc < 6%  → neutral zone: do nothing (count unchanged)
acc >= 6%    → recovery zone   : reset zero_count to 0
```

Stop when `zero_count >= 3` for **either** RI or PI.

**Tracking per (key_count, condition):**
```python
sat = {"zero_count": 0}

if acc <= NEAR_ZERO:     sat["zero_count"] += 1   # near-zero: count up
elif acc >= RECOVERY:    sat["zero_count"] = 0    # recovered: reset
# else neutral zone:     do nothing

if sat["zero_count"] >= 3:
    stop remaining update levels for this key count
```

**Statistical derivation of thresholds (for Appendix):**

Thresholds are grounded in Clopper-Pearson exact binomial confidence intervals
with n=200 trials and 95% confidence level.

*Near-zero threshold (1.5%):*
Using the exact upper bound formula `CP_upper(k, n) = Beta(1-α/2, k+1, n-k)`,
with n=200 and α=0.05: observing ≤3 correct out of 200 (≤1.5%) gives
CI_upper ≤ 4.3%. This means we are 95% confident the true accuracy is ≤5%,
i.e., the model is statistically indistinguishable from chance/zero.

*Recovery threshold (6%):*
Using the exact lower bound `CP_lower(k, n) = Beta(α/2, k, n-k+1)`,
with n=200: observing ≥12 correct out of 200 (≥6%) gives CI_lower ≥ 3.1%.
This means we are 95% confident the true accuracy is >3% — the model has
genuinely recovered above the noise floor and is not saturated.

*Neutral zone (1.5%–6%):*
The gap between thresholds is intentional. Values in this range are ambiguous —
too high to confidently call zero, too low to confidently call recovery.
Leaving the counter unchanged prevents noise from falsely triggering either
increment or reset.

```
Verified values (Clopper-Pearson, n=200, 95% CI):
  observe 3/200 = 1.5%  → CI = [0.3%, 4.3%]   upper < 5% ✓  near-zero
  observe 4/200 = 2.0%  → CI = [0.5%, 5.0%]   upper = 5%    boundary
  observe 12/200 = 6.0% → CI = [3.1%, 10.1%]  lower > 3% ✓  recovery
  observe 11/200 = 5.5% → CI = [2.8%, 9.6%]   lower < 3%    not yet recovered
```

---

## Save Behaviour

- **Every 3 cells**: overwrite `sweep_partial.json` + `sweep_full_partial.json` (checkpoint)
- **On resume**: reads partial file, skips completed cells by key `"{num_keys}_{num_updates}"`
- **Final**: timestamped `sweep_{ts}.json` + `sweep_full_{ts}.json` (never overwritten)

---

## Notes for Future Mechanistic Study

**Position control experiment (TODO — not part of this sweep):**
Test whether the first value appearing at position 0 in the sequence and the last value
appearing at position N-1 (bookending the sequence) changes RI/PI accuracy vs. random
interleaving. Hypothesis: position explains part of the primacy bias. This is a clean
follow-up to separate interference from positional salience effects.

---

## Verification

```bash
# Quick smoke test: 2 key levels, 3 update levels, 2 trials, claude-haiku
cd /Users/kanak.raj/workspace/hobby/research_work_ri
.venv/bin/python experiments_cloud/sweep_semantic.py \
  --model claude-haiku --trials 2 \
  --key-levels 2 3 --update-levels 1 3 5

# Check structure
ls experiments_cloud/results/semantic/claude-haiku/
python -m json.tool experiments_cloud/results/semantic/claude-haiku/sweep_partial.json | head -80
```
