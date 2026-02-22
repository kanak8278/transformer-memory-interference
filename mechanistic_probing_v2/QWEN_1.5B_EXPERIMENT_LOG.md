# Qwen2.5-1.5B-Instruct Experiment Log

## Model Info
- **Model**: Qwen/Qwen2.5-1.5B-Instruct
- **Params**: 1.54B
- **Architecture**: 28 layers, 12 heads (GQA 12Q/2KV), d_model=1536
- **Data setup**: Single-token English values (2300 words) + 46 semantic categories
- **Purpose**: Cross-model validation of 0.5B findings

## Operating Points (Regime-Matched to 0.5B)

The 1.5B model reaches the same interference regimes at fewer updates than 0.5B.
PI collapse boundary shifts left — less interference needed to break PI on a larger model.
(Consistent with ACL paper: RI scales with size R²=0.49, PI does not R²=0.06)

| Regime | 0.5B config | 0.5B RI/PI | **1.5B config** | **1.5B RI/PI** | Description |
|--------|-------------|------------|-----------------|----------------|-------------|
| A (baseline) | 1k, 2u | 53%/47% | **1k, 2u** | 47%/47% | Both work, minimal gap |
| B (PI cracking) | 1k, 5u | 60%/17% | **1k, 3u** | 60%/17% | Clear asymmetry |
| C (PI collapsed) | 1k, 15u | 57%/10% | **1k, 5u** | 70%/13% | Deep PI failure |
| D (multi-key) | 2k, 5u | 43%/20% | **2k, 3u** | 47%/23% | Multi-key validation |

Source: recalibration sweep (`11c_recalibrate_sweep.py --model Qwen/Qwen2.5-1.5B-Instruct`)
Results: `results/Qwen2.5-1.5B-Instruct/0k_0u/recalibrate_sweep.json`

## Trial Strategy
- 100 trials per condition per experiment
- Same experiment suite as 0.5B (exp 12-23)

## Experiment Status

| Exp | Name | A (1k,2u) | B (1k,3u) | C (1k,5u) | D (2k,3u) |
|-----|------|-----------|-----------|-----------|-----------|
| 12 | Logit lens + DLA | DONE | DONE | DONE | DONE |
| 13 | Positional gradient | DONE | DONE | DONE | DONE |
| 14 | PI mass distribution | DONE | DONE | DONE | DONE |
| 15 | Activation patching | DONE | DONE | DONE | DONE |
| 16 | Head identification | DONE | DONE | DONE | DONE |
| 17 | Instruction sensitivity | DONE | DONE | DONE | DONE |
| 18 | Forced attention | DONE | DONE | DONE | DONE |
| 19 | Positional bias sweep | DONE | DONE | DONE | DONE |
| 19b | Bias attention proof | DONE | DONE | DONE | DONE |
| 20 | Minority override | DONE | DONE | DONE | DONE |
| 22 | Query patching granular | DONE | DONE | DONE | DONE |
| 23 | Ablation patching interaction | DONE | DONE | DONE | DONE |

## Key Results (Headline Numbers)

### Exp 12: Logit Lens
| Point | RI acc | PI acc | RI P(init) | PI P(final) |
|-------|--------|--------|------------|-------------|
| A (1k,2u) | 61% | 26% | 0.995 | 0.511 |
| B (1k,3u) | 49% | 20% | 0.989 | 0.345 |
| C (1k,5u) | 55% | 21% | 0.984 | 0.402 |
| D (2k,3u) | 61% | 6% | 0.978 | 0.091 |

### Exp 18: Forced Attention
- Points A, B, C: Forcing attention does NOT help PI (unlike 0.5B!)
- Point D: Marginal improvement
- Interpretation: On 1.5B, the OV circuit is ALSO part of the problem, not just QK

### Exp 22: Query Patching
- Point C (1k,5u): Late-layer patching works (+96%) — consistent with 0.5B
- Points A, B, D: Mixed/weak signal — needs investigation

### Notable Differences from 0.5B
1. **Forced attention doesn't help PI on 1.5B** — suggests OV circuit involvement at scale
2. **P(init) is much higher on 1.5B** (0.98-0.99 vs 0.56-0.87) — stronger RI representation
3. **Query patching shows negative recovery at L18** — different layer dynamics (28 vs 24 layers)
4. **1.5B reaches same regimes with fewer updates** — PI collapse boundary shifts left

## Key Questions for Cross-Model Comparison
1. Does the same core circuit appear? (L16H0, L16H7, L23H12 equivalents)
2. Is the primacy cliff equally sharp?
3. Does forced attention fix PI by the same amount? → **NO, key difference found**
4. Does late-layer query patching recover PI? → **Yes at Point C, mixed elsewhere**
5. Are the DLA top heads condition-flipping in the same way?
