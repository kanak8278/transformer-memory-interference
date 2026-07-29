# Multi-Token Validation: Quick Implementation Guide

**Goal:** Run Exp 26 (per-head knockout on SEMANTIC_MULTI) to validate single-token findings generalize

**Status:** Ready to implement
**Time:** ~1 hour code + ~1 hour compute

---

## **Quick Checklist**

- [ ] Copy `experiments/25a_per_head_knockout.py` → `experiments/26_per_head_knockout_multitoken.py`
- [ ] Change dataset: `"SEMANTIC_SINGLE"` → `"SEMANTIC_MULTI"`
- [ ] Change output filename to include "_multitoken"
- [ ] Test on 10 heads (quick sanity check)
- [ ] Run full sweep on 1.5B at Point B (1k, 3u)
- [ ] Compare rankings with Exp 25a results
- [ ] Create comparison plot

---

## **Code Diff (Minimal Changes)**

### **Change 1: Import & Config**
```python
# BEFORE
DATASET_TYPE = "SEMANTIC_SINGLE"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
NUM_KEYS = 1000
NUM_UPDATES = 3  # Point B for 1.5B

# AFTER
DATASET_TYPE = "SEMANTIC_MULTI"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
NUM_KEYS = 1000
NUM_UPDATES = 3  # Point B for 1.5B
```

### **Change 2: Output Path**
```python
# BEFORE
out_path = results_dir / f"per_head_knockout.json"

# AFTER
out_path = results_dir / f"per_head_knockout_multitoken.json"
```

### **Change 3: Trial Generation**
No change needed! The `generate_trial()` function already handles SEMANTIC_MULTI:

```python
# This just works:
trial = generate_trial(
    dataset_type="SEMANTIC_MULTI",  # Changed here
    num_keys=num_keys,
    num_updates=num_updates,
    condition=condition,
    seed=seed,
    use_chat_format=True,
    tokenizer=tokenizer,
)
```

**That's it.** The dataset_configs module handles the rest.

---

## **Run Command**

```bash
cd /Users/kanak.raj/workspace/hobby/research_work_ri

# Quick test (10 heads, 20 trials per condition)
.venv/bin/python mechanistic_probing_v2/experiments/26_per_head_knockout_multitoken.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --dataset SEMANTIC_MULTI \
    --num_keys 1000 \
    --num_updates 3 \
    --trials 20 \
    --gpu 0 \
    --test

# Full run (336 heads, 100 trials per condition)
.venv/bin/python mechanistic_probing_v2/experiments/26_per_head_knockout_multitoken.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --dataset SEMANTIC_MULTI \
    --num_keys 1000 \
    --num_updates 3 \
    --trials 100 \
    --gpu 0
```

---

## **Expected Output**

### **Console Output**
```
Loading Qwen/Qwen2.5-1.5B-Instruct...
Dataset: SEMANTIC_MULTI (46 categories, 2403 values)
Trials per condition: 100
Baseline PI accuracy: 0.17 (should match Exp 25a on SEMANTIC_SINGLE)

Running per-head knockout (336 heads × 2 conditions × 100 trials = 67,200 forward passes)...

Progress:
  L0: 14 heads, avg causal_effect = +1.8
  L1: 12 heads, avg causal_effect = +0.3
  ...
  L8: 12 heads, avg causal_effect = +4.2  ← L8H3 should be high
  ...
  L27: 12 heads, avg causal_effect = +0.1

Top 5 primacy heads:
  1. L8H3: +8.1Δ (PI: 17% → 91%)  ← Compare to Exp 25a
  2. L0H7: +3.4Δ
  3. L0H3: +3.0Δ
  4. L0H11: +2.8Δ
  5. L0H1: +2.7Δ

Correlation with SEMANTIC_SINGLE rankings: r = 0.78 (p < 0.001)
Top-10 overlap: 80%

Results saved: results/Qwen2.5-1.5B-Instruct/1k_3u/per_head_knockout_multitoken.json
```

---

## **Comparison: Single-Token vs Multi-Token**

After running Exp 26, create this quick comparison:

```python
import json
import numpy as np
from scipy.stats import spearmanr

# Load both
single = json.load(open("results/Qwen2.5-1.5B-Instruct/1k_3u/per_head_knockout.json"))
multi = json.load(open("results/Qwen2.5-1.5B-Instruct/1k_3u/per_head_knockout_multitoken.json"))

# Get rankings
single_ranked = sorted(single["heads"].items(), key=lambda x: x[1]["causal_effect"], reverse=True)
multi_ranked = sorted(multi["heads"].items(), key=lambda x: x[1]["causal_effect"], reverse=True)

# Print comparison
print("\n" + "="*70)
print("SEMANTIC_SINGLE vs SEMANTIC_MULTI: Per-Head Knockout Comparison")
print("="*70)

print("\nTop 10 Primacy Heads:")
print(f"{'Rank':<5} {'SEMANTIC_SINGLE':<30} {'SEMANTIC_MULTI':<30}")
print("-" * 70)

for i in range(10):
    single_head, single_data = single_ranked[i]
    multi_head, multi_data = multi_ranked[i]

    single_str = f"{single_head}: {single_data['causal_effect']:.2f}Δ"
    multi_str = f"{multi_head}: {multi_data['causal_effect']:.2f}Δ"

    match = "✓" if single_head == multi_head else "✗"
    print(f"{i+1:<5} {single_str:<30} {multi_str:<30} {match}")

# Correlation
single_ids = [h[0] for h in single_ranked[:30]]
multi_ids = [h[0] for h in multi_ranked[:30]]

single_ranks = {h: i for i, h in enumerate(single_ids)}
multi_ranks = {h: i for i, h in enumerate(multi_ids)}

common = set(single_ranks.keys()) & set(multi_ranks.keys())
s_ranks = [single_ranks[h] for h in common]
m_ranks = [multi_ranks[h] for h in common]

r, p = spearmanr(s_ranks, m_ranks)

print(f"\n{'='*70}")
print(f"Spearman correlation (top-30): r = {r:.3f}, p = {p:.2e}")
print(f"Top-10 overlap: {len(set(single_ids[:10]) & set(multi_ids[:10]))}/10 heads")
print(f"Effect size ratio: {multi_ranked[0][1]['causal_effect'] / single_ranked[0][1]['causal_effect']:.2f}x")
print(f"{'='*70}\n")

# Validation
if r > 0.7:
    print("✓ PASS: Strong correlation between single and multi-token findings")
else:
    print("✗ FAIL: Weak correlation - investigate differences")
```

**Save this output to:** `results/Qwen2.5-1.5B-Instruct/1k_3u/multitoken_validation_summary.txt`

---

## **Create Comparison Plot**

```python
import matplotlib.pyplot as plt
import json

# Load both
single = json.load(open("results/Qwen2.5-1.5B-Instruct/1k_3u/per_head_knockout.json"))
multi = json.load(open("results/Qwen2.5-1.5B-Instruct/1k_3u/per_head_knockout_multitoken.json"))

# Extract top 15 heads from each
single_ranked = sorted(single["heads"].items(), key=lambda x: x[1]["causal_effect"], reverse=True)[:15]
multi_ranked = sorted(multi["heads"].items(), key=lambda x: x[1]["causal_effect"], reverse=True)[:15]

# Create side-by-side bar charts
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Single-token
heads_s = [f"L{h[1]['layer']}H{h[1]['head']}" for h in single_ranked]
effects_s = [h[1]['causal_effect'] for h in single_ranked]
colors_s = ['red' if 'L8H3' in h else 'green' if h[0]=='L' and h[1]=='0' else 'steelblue' for h in heads_s]

ax1.barh(range(len(heads_s)), effects_s, color=colors_s)
ax1.set_yticks(range(len(heads_s)))
ax1.set_yticklabels(heads_s)
ax1.set_xlabel('Causal Effect (Δlogit_diff)')
ax1.set_title('SEMANTIC_SINGLE\n(Single-Token Values)')
ax1.invert_yaxis()

# Multi-token
heads_m = [f"L{h[1]['layer']}H{h[1]['head']}" for h in multi_ranked]
effects_m = [h[1]['causal_effect'] for h in multi_ranked]
colors_m = ['red' if 'L8H3' in h else 'green' if h[0]=='L' and h[1]=='0' else 'steelblue' for h in heads_m]

ax2.barh(range(len(heads_m)), effects_m, color=colors_m)
ax2.set_yticks(range(len(heads_m)))
ax2.set_yticklabels(heads_m)
ax2.set_xlabel('Causal Effect (Δlogit_diff)')
ax2.set_title('SEMANTIC_MULTI\n(Multi-Token Values)')
ax2.invert_yaxis()

# Match x-axis scales for easy comparison
max_effect = max(max(effects_s), max(effects_m))
ax1.set_xlim(0, max_effect * 1.1)
ax2.set_xlim(0, max_effect * 1.1)

plt.suptitle('Per-Head Knockout: Validation Across Token Counts\nQwen 1.5B, Point B (1k keys, 3 updates)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/Qwen2.5-1.5B-Instruct/1k_3u/multitoken_validation_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: multitoken_validation_comparison.png")
```

---

## **Success Criteria Checklist**

After running, verify:

- [ ] Baseline PI accuracy ~17% (same as Exp 25a)
- [ ] L8H3 in top-3 (ideally #1)
- [ ] L0 heads in top-10
- [ ] Spearman r > 0.7
- [ ] Top-10 overlap > 70%
- [ ] Effect magnitudes similar (within 2x)

**If all pass:** → "Findings generalize to multi-token, validate paper claims"
**If any fail:** → Debug: Is dataset different? Are categories too different?

---

## **What to Do If Results Differ**

### **Scenario 1: Different heads rank top**
- Check: Do multi-token values tokenize differently than expected?
  ```python
  from transformers import AutoTokenizer
  tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

  # Check tokenization of sample values
  for cat in ["gemstone", "bird", "fruit"]:
    from mechanistic_probing_v2.core.dataset_configs import get_value_pool
    values = get_value_pool("SEMANTIC_MULTI", category=cat)[:3]
    for v in values:
      tokens = tokenizer.tokenize(v)
      print(f"{cat}: {v} → {tokens} ({len(tokens)} tokens)")
  ```
- Issue: If values tokenize to 3+ tokens, mechanism might be different
- Action: Filter to 2-token values only, rerun

### **Scenario 2: Causal effects much smaller**
- Check: Is baseline accuracy different?
  ```python
  # Compare Exp 25a baseline (SEMANTIC_SINGLE)
  # vs Exp 26 baseline (SEMANTIC_MULTI)
  # If very different, dataset difficulty differs
  ```
- Issue: Multi-token values might be harder for model
- Action: Report in paper, explain dataset difference

### **Scenario 3: Spearman r < 0.5**
- Check: Are we comparing same operating point?
  ```python
  # Point B for 1.5B is (1k, 3u), NOT (1k, 5u)
  # Verify both experiments at same point
  ```
- Issue: Operating point mismatch
- Action: Re-match difficulty between single/multi-token

---

## **When to Run**

**Option A (This Week):** Run Exp 26 in parallel with Exp 25a on 3B
- Both are knockout experiments
- Can't run simultaneously on same GPU
- Use different GPUs or stagger time

**Option B (Next Week):** Run after Exp 25a on 3B completes
- Lower priority (appendix validation, not critical path)
- Can focus on Figure creation in parallel

**Recommendation:** Option A (parallelize on different GPU)

---

## **Files to Update When Done**

1. **mechanistic_probing_v2/QWEN_1.5B_EXPERIMENT_LOG.md**
   - Add section: "Exp 26: Multi-Token Validation (SEMANTIC_MULTI)"
   - Add results table

2. **docs/FIGURES_DRAFT.md**
   - Update Appendix Figure A1 section with actual results

3. **mechanistic_probing_v2/core/dataset_configs.py**
   - Should already work, no changes needed
   - But verify SEMANTIC_MULTI loads correctly

---

## **Summary**

| Step | Time | Command |
|------|------|---------|
| 1. Copy Exp 25a script | 5 min | `cp experiments/25a_*.py experiments/26_per_head_knockout_multitoken.py` |
| 2. Change 3 lines | 5 min | DATASET_TYPE, output filename |
| 3. Test on 10 heads | 10 min | `--test --trials 20` |
| 4. Run full sweep | 1 hour | `--trials 100` |
| 5. Compare + plot | 30 min | Run comparison script + create figure |
| **Total** | **1.5 hours** | **Ready for paper** |

---

## **Questions?**

If dataset_configs doesn't work as expected, check:
- `mechanistic_probing_v2/core/data/semantic_multi.json` exists?
- Categories load correctly with `get_eligible_categories("SEMANTIC_MULTI", min_values=3)`?
- Trials generate without errors?

All should work since dataset_configs is already integrated. This is truly a 3-line copy-paste experiment.

