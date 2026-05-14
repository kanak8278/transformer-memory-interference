#!/usr/bin/env python3
"""
Deep garbage analysis:
For every garbage trial, classify what the model actually output:
  1. IN_POOL       — the output IS a valid word from the 2300-word pool,
                     just not the test category's value (model found a real word, wrong slot)
  2. IS_KEY        — the output is one of the 46 category names (model output a key, not a value)
  3. IS_SUBWORD    — the output is a partial match / prefix of a pool word
  4. HALLUCINATION — not in pool, not a key name (pure pretraining hallucination)
  5. MULTI_TOKEN   — model output multiple words / a phrase

Also checks: for IN_POOL garbage, was the word semantically plausible for the key
             (indicates model used pretraining category-to-value associations)?

Covers all models with trial data in arbitrary_single.
"""

import json, sys, os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from mechanistic_probing_v2.core.dataset_configs import _load_arbitrary_single_pool, load_dataset_config

BASE = Path(__file__).parent
os.chdir(BASE)

# ─── load dataset ─────────────────────────────────────────────────────────────
POOL = set(w.lower() for w in _load_arbitrary_single_pool())
CFG  = load_dataset_config('ARBITRARY_SINGLE')
CATEGORIES = set(c.lower() for c in CFG['categories'])
CATEGORY_WORDS = set()                  # individual words inside category names
for cat in CFG['categories']:
    for w in cat.lower().split():
        CATEGORY_WORDS.add(w)

print(f"Pool size: {len(POOL)} words")
print(f"Category names: {len(CATEGORIES)}")
print(f"Category name tokens: {len(CATEGORY_WORDS)}")
print()


# ─── classify a single garbage output ─────────────────────────────────────────
def classify_garbage(output_raw, predicted):
    raw  = output_raw.strip().lower()
    pred = predicted.strip().lower() if predicted else raw

    tokens = raw.split()

    if len(tokens) > 2:
        return "MULTI_TOKEN"

    # Check the predicted word (first token, what the eval code parsed)
    word = pred

    if word in POOL:
        return "IN_POOL"

    if word in CATEGORIES:
        return "IS_KEY"

    # Any word in the output that is a category name token?
    for tok in tokens:
        if tok in CATEGORY_WORDS and tok not in {'art', 'type', 'style', 'breed',
                                                   'variety', 'form', 'name', 'term'}:
            return "IS_KEY_FRAGMENT"

    # Partial match: output starts with a pool word?
    for w in POOL:
        if word.startswith(w) or w.startswith(word):
            return "IS_SUBWORD"

    return "HALLUCINATION"


# ─── load trials ──────────────────────────────────────────────────────────────
def load_trials(model):
    folder = BASE / 'arbitrary_single' / model
    if not folder.exists(): return None
    # prefer largest file with 'trials' in name (most complete)
    candidates = [f for f in folder.iterdir()
                  if 'trials' in f.name and 'checkpoint' not in f.name
                  and f.suffix == '.json']
    if not candidates: return None
    best = max(candidates, key=lambda f: f.stat().st_size)
    with open(best) as fh:
        d = json.load(fh)
    return d if 'trial_details' in d else None


# ─── per-model analysis ────────────────────────────────────────────────────────
def analyze_model(model):
    data = load_trials(model)
    if data is None:
        return None

    td = data['trial_details']

    result = {cond: {'IN_POOL': 0, 'IS_KEY': 0, 'IS_KEY_FRAGMENT': 0,
                     'IS_SUBWORD': 0, 'HALLUCINATION': 0, 'MULTI_TOKEN': 0,
                     'total_garbage': 0, 'examples': []}
              for cond in ['RI', 'PI']}

    for cell_key, cell in td.items():
        for cond in ['RI', 'PI']:
            for t in cell.get(cond, []):
                if t.get('error_type') != 'garbage':
                    continue
                cat = classify_garbage(t['output_raw'], t.get('predicted', ''))
                result[cond][cat] += 1
                result[cond]['total_garbage'] += 1
                if len(result[cond]['examples']) < 6:
                    result[cond]['examples'].append({
                        'cell': cell_key,
                        'all_values': t['all_values'],
                        'output_raw': t['output_raw'],
                        'cat': cat,
                    })

    return result


# ─── SECTION 1: Per-model garbage classification ──────────────────────────────
print("=" * 90)
print("SECTION 1: GARBAGE CLASSIFICATION PER MODEL (arbitrary_single)")
print("=" * 90)
print("""
Categories:
  IN_POOL      = output IS a valid word from the 2300-word pool (wrong slot, real word)
  IS_KEY       = output is one of the 46 category/key names
  IS_SUBWORD   = partial match to a pool word
  HALLUCINATION= none of the above — pure pretraining output
  MULTI_TOKEN  = model output multiple words / a phrase
""")

models = sorted((BASE / 'arbitrary_single').iterdir())
models = [m.name for m in models if m.is_dir()]

all_results = {}
for model in models:
    r = analyze_model(model)
    if r is None: continue
    all_results[model] = r

# Print summary table
cats_order = ['IN_POOL', 'IS_KEY', 'IS_KEY_FRAGMENT', 'IS_SUBWORD',
              'HALLUCINATION', 'MULTI_TOKEN']

for cond in ['RI', 'PI']:
    print(f"\n  Condition: {cond}")
    print(f"  {'Model':<28}  {'total':>6}  " +
          "  ".join(f"{c[:6]:>8}" for c in cats_order))
    print("  " + "-"*85)
    for model in models:
        if model not in all_results: continue
        r = all_results[model][cond]
        n = r['total_garbage']
        if n == 0: continue
        def pct(k): return f"{100*r[k]/n:5.1f}%"
        tag = " ◄" if "Qwen3.5" in model else ""
        print(f"  {model:<28}  {n:>6}  " +
              "  ".join(f"{pct(c):>8}" for c in cats_order) + tag)


# ─── SECTION 2: IN_POOL examples — what word? ─────────────────────────────────
print("\n" + "=" * 90)
print("SECTION 2: IN_POOL GARBAGE EXAMPLES")
print("=" * 90)
print("  Output is a real word from the 2300-word pool — but not in test category's values.")
print("  Could be: value of a DIFFERENT key in the same stream.")
print("  Could be: a semantically plausible word the model associates with the category.")
print()

# For each model, show a sample of IN_POOL garbage
for model in ['Qwen3.5-0.8B', 'Qwen3.5-2B', 'Qwen2.5-3B', 'pythia-410m']:
    data = load_trials(model)
    if data is None: continue
    td = data['trial_details']

    in_pool_examples = []
    for cell_key, cell in td.items():
        for cond in ['RI', 'PI']:
            for t in cell.get(cond, []):
                if t.get('error_type') != 'garbage': continue
                cat = classify_garbage(t['output_raw'], t.get('predicted', ''))
                if cat == 'IN_POOL':
                    in_pool_examples.append({
                        'cond': cond, 'cell': cell_key,
                        'all_values': t['all_values'],
                        'output': t['output_raw'],
                    })

    print(f"  {model} — {len(in_pool_examples)} IN_POOL garbage samples:")
    for ex in in_pool_examples[:8]:
        print(f"    [{ex['cond']} cell={ex['cell']}]  valid={ex['all_values']}  "
              f"got='{ex['output']}'  ('{ex['output']}' IS in the 2300-word pool)")
    print()


# ─── SECTION 3: HALLUCINATION examples ────────────────────────────────────────
print("=" * 90)
print("SECTION 3: HALLUCINATION EXAMPLES")
print("=" * 90)
print("  Output is NOT in the 2300-word pool — model invented it from pretraining.")
print()

for model in ['Qwen3.5-0.8B', 'Qwen2.5-3B', 'pythia-410m']:
    data = load_trials(model)
    if data is None: continue
    td = data['trial_details']

    hallu = []
    for cell_key, cell in td.items():
        for cond in ['RI', 'PI']:
            for t in cell.get(cond, []):
                if t.get('error_type') != 'garbage': continue
                cat = classify_garbage(t['output_raw'], t.get('predicted', ''))
                if cat == 'HALLUCINATION':
                    hallu.append({'cond': cond, 'cell': cell_key,
                                  'output': t['output_raw'],
                                  'all_values': t['all_values']})

    print(f"  {model} — {len(hallu)} HALLUCINATION samples:")
    for ex in hallu[:8]:
        print(f"    [{ex['cond']} cell={ex['cell']}]  valid={ex['all_values']}  "
              f"got='{ex['output']}'  (NOT in pool)")
    print()


# ─── SECTION 4: IS_KEY examples ────────────────────────────────────────────────
print("=" * 90)
print("SECTION 4: IS_KEY EXAMPLES — model output a category NAME instead of a value")
print("=" * 90)
print()

for model in models:
    data = load_trials(model)
    if data is None: continue
    td = data['trial_details']

    key_examples = []
    for cell_key, cell in td.items():
        for cond in ['RI', 'PI']:
            for t in cell.get(cond, []):
                if t.get('error_type') != 'garbage': continue
                cat = classify_garbage(t['output_raw'], t.get('predicted', ''))
                if cat in ('IS_KEY', 'IS_KEY_FRAGMENT'):
                    key_examples.append({'cond': cond, 'output': t['output_raw'],
                                         'all_values': t['all_values']})

    if key_examples:
        print(f"  {model}: {len(key_examples)} IS_KEY garbage examples")
        for ex in key_examples[:5]:
            print(f"    [{ex['cond']}]  valid={ex['all_values']}  "
                  f"got='{ex['output']}'")
        print()


# ─── SECTION 5: MULTI_TOKEN examples ──────────────────────────────────────────
print("=" * 90)
print("SECTION 5: MULTI_TOKEN EXAMPLES — model output a phrase instead of one word")
print("=" * 90)
print()

for model in models:
    data = load_trials(model)
    if data is None: continue
    td = data['trial_details']

    multi = []
    for cell_key, cell in td.items():
        for cond in ['RI', 'PI']:
            for t in cell.get(cond, []):
                if t.get('error_type') != 'garbage': continue
                cat = classify_garbage(t['output_raw'], t.get('predicted', ''))
                if cat == 'MULTI_TOKEN':
                    multi.append({'cond': cond, 'cell': cell_key,
                                  'output': t['output_raw'],
                                  'all_values': t['all_values']})

    if multi:
        print(f"  {model}: {len(multi)} MULTI_TOKEN examples")
        for ex in multi[:5]:
            print(f"    [{ex['cond']} cell={ex['cell']}]  valid={ex['all_values']}")
            print(f"    → '{ex['output']}'")
        print()


# ─── SECTION 6: RI vs PI breakdown comparison ─────────────────────────────────
print("=" * 90)
print("SECTION 6: RI vs PI GARBAGE TYPE COMPARISON (does asymmetry exist?)")
print("=" * 90)
print()

for model in models:
    if model not in all_results: continue
    ri = all_results[model]['RI']
    pi = all_results[model]['PI']
    ri_n = ri['total_garbage']
    pi_n = pi['total_garbage']
    if ri_n == 0 and pi_n == 0: continue

    print(f"  {model}:  RI_garbage={ri_n}  PI_garbage={pi_n}")
    print(f"  {'type':<16}  {'RI %':>7}  {'PI %':>7}  {'diff (RI-PI)':>13}")
    for c in cats_order:
        ri_pct = 100*ri[c]/ri_n if ri_n > 0 else 0
        pi_pct = 100*pi[c]/pi_n if pi_n > 0 else 0
        diff   = ri_pct - pi_pct
        bar = "▲" if diff > 5 else ("▼" if diff < -5 else "≈")
        print(f"  {c:<16}  {ri_pct:>6.1f}%  {pi_pct:>6.1f}%  {diff:>+12.1f}%  {bar}")
    print()


print("=" * 90)
print("DONE")
print("=" * 90)
