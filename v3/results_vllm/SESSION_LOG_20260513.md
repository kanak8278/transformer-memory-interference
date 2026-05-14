# Session Log — 2026-05-13

## 1. Qwen3.5 Results Integration & Analysis

- Confirmed Qwen3.5 results (0.8B, 2B, 4B, 9B) were already extracted from `qwen35_results.tar.gz` into `results_vllm/`
- Ran comprehensive analysis (`analyze_qwen35.py`) across all 3 datasets: `arbitrary_single`, `semantic_multi`, `narrative_dota2`
- Key findings:
  - Qwen3.5-2B appeared as outlier (46.8% reversed cells) — later shown to be a garbage artifact
  - Qwen3.5-9B cleanest model: 0/79 reversed cells, gap=+0.214
  - All models ran as `completion_few_shot` (not instruct/chat) — potential format mismatch
  - Context length drives degradation: above ~40% fill (~6.5K tokens), performance collapses

## 2. Garbage Deep Dive

- Classified garbage outputs into: `IN_POOL`, `IS_KEY`, `IS_SUBWORD`, `HALLUCINATION`, `MULTI_TOKEN`
- **Dominant finding**: ~88-98% of garbage is `IN_POOL` — real words from the 2300-word pool, not hallucinations
- Built seed-based prompt reconstruction (`analyze_garbage_source.py`) to check if garbage came from the SAME prompt
- Verified reconstruction correct by visual inspection of 10/10 Qwen3.5-9B PI garbage cases — all CROSS_KEY (word was in the stream, wrong key)
- Key split: `CROSS_KEY` (model read a value from another key's stream) vs `PRETRAINING` (not in stream at all)
- Qwen3.5-9B PI garbage: **100% CROSS_KEY** — model always stays in stream, just attributes to wrong key
- StableLM: 100% MULTI_TOKEN — completely different failure mode, should be excluded

## 3. Garbage-Filtered Reversal Analysis (`analyze_reversals.py`)

- Qwen3.5-2B reversal disappears after garbage filtering: raw gap=+0.072 → clean gap=+0.498
- **Not actually reversed** — garbage was inflating apparent reversals
- Three distinct root causes for real outliers:
  - **Pythia-410M**: true recency default (RI failures predict LAST value, frac_last=0.9 at nk=30) — instruction following failure
  - **Qwen2.5-3B**: key-count capacity limit (reversed at high nk even at <15% context fill) — not recency default
  - **Qwen2.5-3B-Instruct / TinyLlama**: key-count capacity limit at nk≥10-15
- Wilson CI stopping was in vllm code but never triggered: threshold (0.05) requires 153+ trials at 90% accuracy — with 100-trial budget it's mathematically impossible for most cells

## 4. U-Curve Experiment Design

Designed experiment to test lost-in-the-middle / serial position effects.

**4 prompt formats:**
- `block`: `[Update K]` block header, all keys grouped per round (shuffled within block)
- `flat_short`: interleaved shuffled stream with `coffee variety [U3]: rural` inline labels + preamble
- `flat_verbose`: interleaved shuffled stream with `coffee variety (update 3): rural` labels
- `flat_nolabel`: interleaved unlabeled stream, query = "What was the 3rd value of X?"
- `original`: existing first/last format (RI/PI baseline)

**Key design decisions:**
- Paired approach: same stream queried at all positions per trial (prefix caching benefit)
- 21 query positions evenly spaced (0%–100% of nu)
- Wilson CI stopping on ALL positions (not just first/last anchors)
- Seed formula: `hash((nk, nu, trial_idx, "ucurve_v1")) % 2**31` — fully deterministic
- Stream-order vs chronological-order distinction critical for unlabeled formats

## 5. Code: `ucurve_prompts.py` + `ucurve_sweep.py`

**Implemented:**
- `generate_stream(nk, nu, seed)` — deterministic, different seeds = different categories/values/shuffle
- `query_positions(nu, n_points)` — evenly spaced percentiles always including 1 and nu
- 5 prompt builders (block, flat_short, flat_verbose, flat_nolabel, original)
- `ordinal(n)` — proper 1st/2nd/3rd/11th/12th suffix handling
- Wilson CI convergence checking all positions
- CLI with `--formats`, `--nk`, `--nu`, `--trials`, `--workers`, `--n-positions`, `--resume`, `--no-early-stop`
- Atomic checkpoint saving (tmp + rename), resume skips completed cells
- Pre-run expected-value verification — runs before every sweep

## 6. Bugs Found and Fixed

| Bug | Root cause | Fix |
|---|---|---|
| `flat_nolabel` query said "Update K" with no labels | Copy-paste from labeled formats | Changed to ordinal: "What was the 6th value?" |
| `original` format expected value used chronological order | Stream is shuffled — "first" = first stream occurrence, not update 1 | Evaluate against `stream_vals[0]` / `stream_vals[-1]` |
| `flat_nolabel` expected value used chronological order | Same issue as original — not caught in code review | Evaluate against `stream_vals[k-1]` |
| `prompt_flat_verbose` referenced `QUERY` after rename to `QUERY_LABELED` | Incomplete rename | Fixed reference |
| Wilson stopping checked only first/last positions | Middle positions could stop early with wide CIs | Changed to check all positions |

## 7. Runs Completed

| Config | Formats | n_trials | Finding |
|---|---|---|---|
| nk=10, nu=50 | block, flat_short, flat_verbose | 50 (early stop) | 100% everywhere — too easy with labels |
| nk=10, nu=50 | flat_nolabel (broken query) | 72 (early stop) | ~0% everywhere — "Update K" query meaningless without labels |
| nk=10, nu=50 | original (broken expected vals) | 50 (early stop) | 0% — bug: chronological vs stream-order mismatch |
| nk=10, nu=50 | original (fixed) | 200 | RI=0.995, PI=0.290 — baseline confirmed on ARBITRARY_SINGLE |
| nk=15, nu=100 | flat_verbose | 89 (early stop) | ~99.7% everywhere — labels immune to position effects even at 1500 entries / ~12K tokens |
| nk=5, nu=30 | block, flat_short, flat_verbose | 62 (early stop) | 100% everywhere |
| nk=5, nu=30 | flat_nolabel (broken expected vals) | 126 (early stop) | ~3% — bug, same mismatch issue |
| nk=5,7, nu=10,30 | original | 153–200 | RI≈1.0, PI=0.49–0.90 — validates ARBITRARY_SINGLE baseline |
| nk=5,7, nu=10,30 | flat_nolabel (fixed) | 200 | Sharp cliff confirmed |
| nk={5,7,10}, nu={10,15,20,30,50,75,100} | flat_nolabel (fixed, 21 positions) | 62–200 | **WIDE SWEEP — full results below** |

### Wide Sweep Results — flat_nolabel, accuracy at absolute positions 1–6

| config | pos=1 | pos=2 | pos=3 | pos=4 | pos=5 | pos=6 |
|---|---|---|---|---|---|---|
| nk=5, nu=10 | 1.000 | 0.990 | 0.870 | 0.320 | 0.090 | 0.065 |
| nk=5, nu=15 | 1.000 | 0.990 | 0.820 | 0.220 | 0.060 | 0.025 |
| nk=5, nu=20 | 1.000 | 0.975 | 0.750 | 0.235 | 0.030 | 0.005 |
| nk=5, nu=30 | 1.000 | 0.965 | — | 0.240 | 0.045 | — |
| nk=5, nu=50 | 1.000 | — | 0.700 | — | — | 0.010 |
| nk=7, nu=10 | 1.000 | 0.985 | 0.870 | 0.250 | 0.130 | 0.065 |
| nk=7, nu=15 | 1.000 | 0.975 | 0.805 | 0.165 | 0.030 | 0.015 |
| nk=7, nu=20 | 0.995 | 0.965 | 0.780 | 0.200 | 0.010 | 0.005 |
| nk=10, nu=10 | 1.000 | 0.980 | 0.825 | 0.230 | 0.065 | 0.060 |
| nk=10, nu=15 | 0.990 | 0.985 | 0.795 | 0.205 | 0.035 | 0.025 |
| nk=10, nu=20 | 1.000 | 0.990 | 0.870 | 0.175 | 0.020 | 0.005 |

(nu=75,100 cells: pos=1≈100%, pos=5+≈0%, cliff at same location)

### Results saved to
- `experiments_cloud/results/ucurve_wide/checkpoint.json` — main wide sweep (27MB)
- `experiments_cloud/results/ucurve/checkpoint.json` — small configs + original baseline
- `experiments_cloud/results/ucurve_hard/checkpoint.json` — nk=15, nu=100 flat_verbose

## 8. Key Findings (FINAL)

1. **Labels completely eliminate position effects.** block/flat_short/flat_verbose at 100% accuracy even at nk=15, nu=100 (1500 stream entries, ~12K tokens). Lost-in-the-middle does not apply when entries have explicit position labels. The model does pure label lookup.

2. **PI>RI is a query-semantics failure, not positional inaccessibility.** Model can retrieve any position perfectly when labeled. Difficulty is in executing "find the first/last" without structural help — not about the information being inaccessible.

3. **Hard serial counting capacity of ~3.** In flat_nolabel (ordinal counting query):
   - pos=1 ("1st"): 100% — activates primacy heuristic, not counting
   - pos=2: ~97–99%
   - pos=3: ~75–87%  
   - pos=4: **~15–32%** ← the cliff, always here
   - pos=5+: ~1–10% (near-random)

4. **The cliff is absolute, not relative.** Always between pos=3 and pos=4 regardless of nk (5→10) or nu (10→100). Number of competing keys and total stream length don't shift it.

5. **"1st" activates primacy heuristic, "Nth" requires counting.** "1st" ≈ "first" semantically → 100% always. "10th" or "30th" → model must count → fails at ~4. The word "last" also activates a recency heuristic (original format PI=0.50–0.90), but "10th/30th" does not — model can't map an ordinal to "last" without knowing nu.

6. **Only ARBITRARY_SINGLE used** in U-curve experiment. SEMANTIC_MULTI not tested in U-curve.

## 9. Open Questions

- Does the cliff at pos≈4 generalize to other model families (Qwen2.5, Gemma, Mamba)?
- Does semantic coherence (SEMANTIC_MULTI) help counting — are semantically meaningful values easier to track ordinally?
- How does the counting capacity (~3) relate to the main PI>RI paper finding mechanistically?
- Should this counting-capacity result be included in the paper as a mechanistic explanation?
