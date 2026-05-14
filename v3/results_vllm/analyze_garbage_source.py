#!/usr/bin/env python3
"""
For every garbage trial, reconstruct the full prompt from seed and classify:
  CROSS_KEY   — output is a value of a DIFFERENT key that was in the same prompt
  PRETRAINING — output is not from the stream at all (hallucination / semantic prior)

This tells us whether garbage = "confused which key" vs "gave up on stream entirely".
"""
import json, sys, random, math
from pathlib import Path
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from mechanistic_probing_v2.core.dataset_configs import get_eligible_categories, generate_values_for_trial

BASE = Path(__file__).parent

def mean(xs): return sum(xs)/len(xs) if xs else float("nan")
def fmt(v):   return f"{v:5.1f}%" if not math.isnan(v) else "  nan%"

# ─── reconstruct full stream for one trial ───────────────────────────────────
def reconstruct_stream_values(num_keys, num_updates, seed):
    """Returns {cat: [values]} for all cats in this trial."""
    rng = random.Random(seed)
    eligible = get_eligible_categories('ARBITRARY_SINGLE', min_values=num_updates)
    cats = rng.sample(eligible, min(num_keys, len(eligible)))
    vals = generate_values_for_trial('ARBITRARY_SINGLE', cats, num_updates, rng)
    test_cat = cats[seed % num_keys]
    return cats, test_cat, vals


def classify_garbage_source(pred, test_cat, vals_per_cat):
    """
    pred: what the model output (lowercased)
    Returns: 'CROSS_KEY', 'PRETRAINING'
    """
    # Check all non-test-cat values
    for cat, vals in vals_per_cat.items():
        if cat == test_cat:
            continue
        if pred in [v.lower() for v in vals]:
            return 'CROSS_KEY'
    return 'PRETRAINING'


# ─── load trials ─────────────────────────────────────────────────────────────
def load_trials(model):
    folder = BASE / 'arbitrary_single' / model
    if not folder.exists(): return None
    candidates = [f for f in folder.iterdir()
                  if 'trials' in f.name and 'checkpoint' not in f.name
                  and f.suffix == '.json']
    if not candidates: return None
    best = max(candidates, key=lambda f: f.stat().st_size)
    with open(best) as fh:
        d = json.load(fh)
    return d if 'trial_details' in d else None


# ─── per-model analysis ───────────────────────────────────────────────────────
def analyze(model):
    data = load_trials(model)
    if data is None: return None

    td = data['trial_details']
    stats = {cond: {'CROSS_KEY': 0, 'PRETRAINING': 0, 'total': 0,
                    'by_nk': {}}
             for cond in ['RI', 'PI']}

    # Count total garbage for tqdm
    total_garb = sum(
        1
        for cell in td.values()
        for cond in ['RI','PI']
        for t in cell.get(cond, [])
        if t.get('error_type') == 'garbage'
    )

    with tqdm(total=total_garb, desc=model[:24], leave=False) as pbar:
        for cell_key, cell in td.items():
            nk, nu = map(int, cell_key.split('_'))
            for cond in ['RI', 'PI']:
                for t in cell.get(cond, []):
                    if t.get('error_type') != 'garbage':
                        continue
                    pred = (t.get('predicted') or '').lower().strip()
                    if not pred:
                        pbar.update(1)
                        continue

                    try:
                        cats, test_cat, vals_per_cat = reconstruct_stream_values(nk, nu, t['seed'])
                        src = classify_garbage_source(pred, test_cat, vals_per_cat)
                    except Exception:
                        pbar.update(1)
                        continue

                    stats[cond][src] += 1
                    stats[cond]['total'] += 1

                    # by num_keys bucket
                    if nk not in stats[cond]['by_nk']:
                        stats[cond]['by_nk'][nk] = {'CROSS_KEY': 0, 'PRETRAINING': 0, 'total': 0}
                    stats[cond]['by_nk'][nk][src] += 1
                    stats[cond]['by_nk'][nk]['total'] += 1

                    pbar.update(1)

    return stats


# ─── SECTION 1: Summary table ─────────────────────────────────────────────────
print("=" * 80)
print("GARBAGE SOURCE ANALYSIS — was the output actually in the same prompt?")
print("=" * 80)
print("""
CROSS_KEY   = model output a value of a DIFFERENT key that was in this exact prompt
PRETRAINING = output was not in the prompt stream at all (semantic/pretraining prior)
""")

models = sorted(m.name for m in (BASE / 'arbitrary_single').iterdir() if m.is_dir())

all_stats = {}
for model in models:
    s = analyze(model)
    if s: all_stats[model] = s

print(f"\n  {'Model':<28}   {'RI total':>8}  {'RI CROSS%':>10}  {'RI PRETRAIN%':>12}  "
      f"{'PI total':>8}  {'PI CROSS%':>10}  {'PI PRETRAIN%':>12}")
print("  " + "-"*95)

for model in models:
    if model not in all_stats: continue
    ri = all_stats[model]['RI']
    pi = all_stats[model]['PI']
    def p(d, k): return 100*d[k]/d['total'] if d['total'] > 0 else float('nan')
    tag = " ◄" if "Qwen3.5" in model else ""
    print(f"  {model:<28}   {ri['total']:>8}  {p(ri,'CROSS_KEY'):>9.1f}%  "
          f"{p(ri,'PRETRAINING'):>11.1f}%  "
          f"{pi['total']:>8}  {p(pi,'CROSS_KEY'):>9.1f}%  "
          f"{p(pi,'PRETRAINING'):>11.1f}%{tag}")


# ─── SECTION 2: RI vs PI CROSS_KEY asymmetry ──────────────────────────────────
print("\n" + "=" * 80)
print("RI vs PI CROSS_KEY RATE COMPARISON")
print("=" * 80)
print("  If RI has more CROSS_KEY: model confuses keys more when asked about FIRST value")
print("  If PI has more CROSS_KEY: model confuses keys more when asked about LAST value\n")

print(f"  {'Model':<28}  {'RI CROSS%':>10}  {'PI CROSS%':>10}  {'diff (RI-PI)':>13}  interpretation")
print("  " + "-"*85)

for model in models:
    if model not in all_stats: continue
    ri = all_stats[model]['RI']
    pi = all_stats[model]['PI']
    def p(d, k): return 100*d[k]/d['total'] if d['total'] > 0 else float('nan')
    ri_c = p(ri, 'CROSS_KEY')
    pi_c = p(pi, 'CROSS_KEY')
    diff = ri_c - pi_c
    if math.isnan(diff):
        interp = "no data"
    elif diff > 10:
        interp = "RI confuses keys MORE"
    elif diff < -10:
        interp = "PI confuses keys MORE"
    else:
        interp = "symmetric key confusion"
    tag = " ◄" if "Qwen3.5" in model else ""
    print(f"  {model:<28}  {ri_c:>9.1f}%  {pi_c:>9.1f}%  {diff:>+12.1f}%  {interp}{tag}")


# ─── SECTION 3: CROSS_KEY rate by num_keys ────────────────────────────────────
print("\n" + "=" * 80)
print("CROSS_KEY RATE BY num_keys — does key confusion grow with more keys?")
print("=" * 80)
print("  If key confusion is the mechanism: CROSS_KEY% should rise with num_keys.\n")

focus = ['Qwen3.5-2B', 'Qwen3.5-9B', 'Qwen2.5-3B', 'pythia-410m', 'gemma-3-1b-it']

for model in focus:
    if model not in all_stats: continue
    print(f"\n  {model}:")
    print(f"  {'nk':>4}   {'RI_CROSS%':>10}  {'RI_PRETRAIN%':>13}  {'RI_n':>6}  "
          f"  {'PI_CROSS%':>10}  {'PI_PRETRAIN%':>13}  {'PI_n':>6}")
    all_nks = sorted(set(list(all_stats[model]['RI']['by_nk'].keys()) +
                         list(all_stats[model]['PI']['by_nk'].keys())))
    for nk in all_nks:
        ri_b = all_stats[model]['RI']['by_nk'].get(nk, {'CROSS_KEY':0,'PRETRAINING':0,'total':0})
        pi_b = all_stats[model]['PI']['by_nk'].get(nk, {'CROSS_KEY':0,'PRETRAINING':0,'total':0})
        def pct(d, k): return 100*d[k]/d['total'] if d['total'] > 0 else float('nan')
        ri_c = pct(ri_b, 'CROSS_KEY')
        ri_p = pct(ri_b, 'PRETRAINING')
        pi_c = pct(pi_b, 'CROSS_KEY')
        pi_p = pct(pi_b, 'PRETRAINING')
        print(f"  {nk:>4}   {ri_c:>9.1f}%  {ri_p:>12.1f}%  {ri_b['total']:>6}  "
              f"  {pi_c:>9.1f}%  {pi_p:>12.1f}%  {pi_b['total']:>6}")


print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
