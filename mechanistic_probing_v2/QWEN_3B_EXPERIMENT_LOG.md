# Qwen2.5-3B-Instruct — Experiment Log

**Model:** Qwen/Qwen2.5-3B-Instruct (3.4B params, 36 layers, 16 heads, d_model=2048)
**Started:** 2026-02-22
**Role:** Size scaling within Qwen family (6x jump from 0.5B)

---

## Phase 1: Behavioral Sweep (Synthetic Values) — IN PROGRESS

Ran on Google Colab T4 with pruned grid.

| Field | Value |
|-------|-------|
| **Script** | `notebooks/behavioral_sweep_colab.ipynb` |
| **Data setup** | Semantic keys + synthetic values (Art375, Herb42) |
| **Grid** | keys=[2,3,5,7,10,15,20,25,30,46], updates=[1,3,5,10,20,50,100,200] |
| **Cells** | 80 (pruned from 384) |
| **Trials** | 30/cell → 4,800 total |
| **Status** | Running (first 52 cells done) |

### Preliminary Results (30 trials, synthetic values)

```
          upd=1  upd=3  upd=5  upd=10  upd=20  upd=50  upd=100  upd=200
k=2  RI    97%    93%    93%   100%    97%     97%     100%     93%
     PI   100%    90%    90%    60%    47%     23%      20%     17%

k=3  RI   100%   100%   100%   100%    93%     97%      93%     73%
     PI   100%    97%    73%    47%    23%     17%      30%     30%

k=5  RI    97%   100%   100%    93%    87%     73%      77%     63%
     PI   100%    93%    77%    63%    33%     27%      17%     23%

k=7  RI   100%   100%    93%    90%    90%     63%      67%      0%
     PI    97%    90%    67%    30%    27%     17%      33%      0%

k=10 RI   100%   100%    87%    97%    57%     70%      33%      0%
     PI   100%    70%    47%    30%    27%     27%      13%      0%

k=15 RI   100%    97%    83%    73%    73%     37%       0%
     PI   100%    80%    43%    40%    27%     33%       0%

k=20 RI    90%    97%    87%    70%
     PI    93%    50%    50%    40%
```

### Key observations vs 0.5B

| Metric | 0.5B | 3B | Interpretation |
|--------|------|----|---------------|
| RI at 2k,100u | 73% | **100%** | RI scales strongly with size |
| PI at 2k,10u | 40% | **60%** | PI also improves but less |
| PI collapse (keys=2) | updates=5 | **updates=20** | 3B resists PI longer |
| PI collapse (keys=10) | updates=3 | **updates=5** | Same pattern, shifted right |
| Regime A cells | 12.6% | ~40% (est) | 3B has much more "both work" region |
| PI floor | ~17% | ~17-23% | Similar floor once collapsed |

**3B is much stronger overall but the PI > RI asymmetry is preserved.** PI still collapses before RI at every key level. The 3B model just has more headroom before collapse.

---

## Phase 1.5: Single-Token Recalibration Sweep — COMPLETE

| Field | Value |
|-------|-------|
| **Script** | `experiments/11c_recalibrate_sweep.py --model Qwen/Qwen2.5-3B-Instruct` |
| **Data setup** | Semantic keys + single-token English words (2300 pool) |
| **Grid** | keys=[1,2,3,5], updates=[1,2,3,5,10,15,20] = 28 cells |
| **Trials** | 30/cell |
| **Result** | `results/Qwen2.5-3B-Instruct/0k_0u/recalibrate_sweep.json` |
| **Runtime** | 678s (~11 min) on MPS |

### Single-token results (30 trials)

```
          upd=1  upd=2  upd=3  upd=5  upd=10  upd=15  upd=20
k=1  RI    40%    50%    60%    43%     57%     63%     53%
     PI    43%    43%    57%    27%     20%     17%     13%
    gap   -3%   +7%   +3%  +17%   +37%   +47%   +40%

k=2  RI    47%    53%    47%    43%     53%     57%     43%
     PI    53%    57%    37%    23%      3%     13%     13%
    gap   -7%   -3%  +10%  +20%   +50%   +43%   +30%

k=3  RI    57%    37%    47%    53%     53%     43%     27%
     PI    53%    53%    40%    10%     13%      7%      3%
    gap   +3%  -17%   +7%  +43%   +40%   +37%   +23%

k=5  RI    63%    47%    40%    43%     30%     30%     53%
     PI    57%    60%    30%    13%     10%      7%      3%
    gap   +7%  -13%  +10%  +30%   +20%   +23%   +50%
```

### Comparison: 0.5B vs 1.5B vs 3B (single-token, same grid)

| Config | 0.5B RI | 1.5B RI | 3B RI | 0.5B PI | 1.5B PI | 3B PI |
|--------|---------|---------|-------|---------|---------|-------|
| 1k,5u  | 60%     | 70%     | 43%   | 17%     | 13%     | 27%   |
| 1k,15u | 47%     | 57%     | 63%   | 17%     | 13%     | 17%   |
| 2k,5u  | 43%     | 60%     | 43%   | 20%     | 13%     | 23%   |
| 2k,10u | 23%     | 47%     | 53%   | 7%      | 3%      | 3%    |
| 3k,5u  | 30%     | 47%     | 53%   | 23%     | 7%      | 10%   |

**Observation:** 3B is NOT dramatically better than 0.5B/1.5B with single-token values. RI peaks around 53-63%, same ballpark. The advantage appears mainly with synthetic values.

---

## Operating Points for Mechanistic Experiments — FINAL

| Point | Keys | Updates | Single-Token RI | Single-Token PI | Gap | Role |
|-------|------|---------|----------------|----------------|-----|------|
| **A** | 1 | 3 | 60% | 57% | +3% | Baseline — minimal interference |
| **B** | 1 | 15 | 63% | 17% | +47% | **Primary target — biggest RI, strong gap** |
| **C** | 2 | 10 | 53% | 3% | +50% | Deep asymmetry — PI collapsed |
| **D** | 3 | 5 | 53% | 10% | +43% | Multi-key validation |

### Why these configurations

- **Point A (1k,3u):** Both conditions near 60%. No real interference — baseline.
- **Point B (1k,15u):** Best RI (63%) with strong gap (+47%). Primary target for mechanistic analysis.
- **Point C (2k,10u):** Biggest gap (+50%), PI at floor (3%). Deep asymmetry regime.
- **Point D (3k,5u):** Multi-key with decent RI (53%). Tests whether mechanism changes with more categories.

### Comparison with 0.5B operating points

| Point | 0.5B config | 3B config | Note |
|-------|------------|-----------|------|
| A | 1k, 3u | 1k, 3u | Same — both need minimal interference for baseline |
| B | 1k, 5u | 1k, 15u | 3B needs more updates before PI collapses |
| C | 1k, 15u | 2k, 10u | 3B can handle 2 keys where 0.5B couldn't |
| D | 2k, 5u | 3k, 5u | 3B handles 3 keys |

---

## Phase 2: Mechanistic Analysis — Run Matrix

**Trial counts:** 100 per condition for all experiments. 200 for ablation experiments (15, 16 ablation phase, 22, 23).

**Result path:** `results/Qwen2.5-3B-Instruct/{keys}k_{updates}u/{experiment}.json`

**Script interface:** `--keys K --updates U --trials N --model Qwen/Qwen2.5-3B-Instruct`

### Experiment sequence (per operating point)

| Exp | Name | Trials | Status A | Status B | Status C | Status D |
|-----|------|--------|----------|----------|----------|----------|
| 12 | Logit lens + attention + DLA | 100 | TODO | DONE | TODO | TODO |
| 13 | Positional gradient | 100 | TODO | DONE | TODO | TODO |
| 14 | PI mass distribution | 100 | TODO | DONE | TODO | TODO |
| 15 | Activation patching | 100 | TODO | DONE | TODO | TODO |
| 16 | Head identification + ablation | 100+20 | TODO | DONE | TODO | TODO |
| 17 | Instruction sensitivity | 100 | TODO | DONE | TODO | TODO |
| 18 | Forced attention | 100 | TODO | DONE | TODO | TODO |
| 19 | Positional bias sweep | 100 | TODO | DONE | TODO | TODO |
| 19b | Bias attention proof | 100 | TODO | DONE | TODO | TODO |
| 20 | Minority override | 100 | TODO | DONE | TODO | TODO |
| 21a | Logit lens under ablation | 100 | TODO | DONE | TODO | TODO |
| 21b | DLA split by outcome | analysis | TODO | DONE | TODO | TODO |
| 21c | Failure output classification | analysis | TODO | DONE | TODO | TODO |
| 22 | Query patching granular | 100 | TODO | DONE | TODO | TODO |
| 23 | Ablation + patching interaction | 100 | TODO | DONE | TODO | TODO |

### Point B (1k, 15u) Results Summary

| Exp | Key Finding | Matches 0.5B? |
|-----|-------------|---------------|
| 12 | RI P(init)=1.00, PI P(final)=0.28. RI representation perfect. | YES — same pattern, stronger RI |
| 13 | v1/v0 = 0.000. Literally zero probability for v1. | YES — even sharper cliff |
| 14 | PI peaks at v11.7/14 (rel=0.84). Only 24% at last. | YES — same diffuse peak |
| 15 | Recovery at L32-33 (~100%) for both RI and PI. | YES — last ~4 layers |
| 16 | KO all retrieval → 0%. KO primacy → no change. | YES — same pattern |
| 17 | Heads ignore query word (attn ~0.01). | YES — identical |
| 18 | **Force correct → PI 10%→57% (+47pp).** | YES — even stronger than 0.5B (+18pp) |
| 19 | No sweet spot for positional bias. | YES |
| 19b | Blind bias → 92% to non-value tokens. Oracle works. | YES |
| 20 | Individual head knockouts ≈ 0% impact. | YES |
| 21a | Ablation minimal effect on accuracy and representation. | YES |
| 21b | PI failure from weaker recency, not stronger primacy. | YES — same finding |
| 21c | Primacy intrusion rate: 3%. Mostly garbage/wrong value. | YES — 3% vs 0-5% on 0.5B |
| 22 | Late-layer patching gives NEGATIVE recovery (-68%). | **DIFFERENT** — 0.5B showed +50% |
| 23 | Ablation effect weak. Normal PI=4%, Ablated PI=5%. | YES — weak on both |

### Key questions for cross-model comparison

1. **Same circuit?** Does L16H7 (0.5B's top retrieval head) have an equivalent in 3B's layer ~22-24?
2. **More retrieval heads?** 3B has more heads — does it recruit more for retrieval?
3. **Sharper primacy cliff?** Is v1/v0 ratio smaller (sharper cliff) or larger (more gradual) at 3B?
4. **Same failure mode?** Does 21c show the same pattern (garbage, not primacy intrusion)?
5. **Stronger recency signal?** Does 21b show higher recency DLA on correct PI trials?

---

## Execution Plan

1. ~~Behavioral sweep (Colab, synthetic values)~~ — IN PROGRESS
2. Single-token recalibration sweep (local) — after behavioral sweep completes
3. Validate operating points — confirm regime classification matches
4. Run exp 12 at all 4 points — primary mechanistic data
5. Run exp 13-23 at Point B first — most informative operating point
6. Remaining points if time allows
