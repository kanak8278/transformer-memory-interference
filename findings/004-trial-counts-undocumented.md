# Finding 004 — Trial counts small and undocumented; one mismatch (50 vs 200)

- **Severity:** 🟡 Medium (reproducibility; one count feeds Finding 001)
- **Status:** Open — caption fixes + provenance decisions
- **Related task:** B3
- **Verified:** 2026-07-11

## Issue

Several results use small trial counts that the paper does not state, and in one
case the local data's n disagrees with the paper's claimed n.

| Result | Trials (local data) | Paper says | Source |
|---|---|---|---|
| Main LoRA held-out eval (ARB) | 40–80/cell, **early-stopped** (`stopped_early` field) | not stated (reader assumes ~100–200) | `main_eval_20260523_055309.json` |
| SEM OOD LoRA eval | baseline 20, LoRA 20 then 10 | not stated | `lora_sem_validation_results.json` |
| Dense-IVQ (per position) | 20/position | not stated | `dense_ivq_ivq_20260523_101418.json` |
| Stage-3C promoter ablation | **50** (local JSONs) | **200** (table comment + §7.3) | see Finding 001 |

## Why it matters

- Wilson/bootstrap CIs a reviewer recomputes will not match if the n is
  unstated or wrong. At n=10–20, CIs are wide (e.g. 10/10 → Wilson lower bound
  ~0.72), so "100%" claims are softer than they read.
- The 50-vs-200 ablation gap is not just documentation — it is the crux of
  Finding 001 (the n=200 data is not in the repo).
- The SEM eval was reconstructed from a killed run's log (Finding: see
  `lora_sem_validation_results.json` `_note`).

## Resolution

1. Add explicit trial counts to every table caption.
2. For the main LoRA eval, document the adaptive/early-stop scheme (why 40–80).
3. Decide whether to re-run SEM + dense-IVQ at n≥100 for the camera-ready
   (cheap; removes the small-n and provenance concerns at once).
4. Resolve the ablation n (Finding 001).
