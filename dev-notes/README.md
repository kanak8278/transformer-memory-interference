# dev-notes

Working notes from preparing the paper — session logs, an asset audit, an
experiment-status tracker, and a verified-claims ledger. Kept for provenance
(so a later session, or a reviewer who wants to check our working, can see
exactly what was checked and when), not because they're required reading.

If you're here to understand the project, start at the root
[README.md](../README.md) and [EXPERIMENTS.md](../EXPERIMENTS.md) instead.
Come back to this folder if you want the audit trail behind a specific claim.

| File | What it is |
|---|---|
| `AAAI_PREP_PLAN.md` | Task list from the ACL→AAAI submission port: asset inventory, what was re-verified against raw data, grouped to-dos. |
| `NEW_RESULTS_AND_NEXT_STEPS.md` | Session summary for four extended experiments run after the initial draft (extrapolation frontier, block-vs-LoRA readout, behavioral 3-way sweep, ablation re-run). |
| `EXPERIMENT_STATUS.md` | Older per-experiment status tracker (which models/grids were done, what's missing) from mid-project. |
| `REPO_MAP.md` | A 2026-07-29 audit of which folders are live vs. archived, and why — useful if a path looks suspiciously dead and you want to confirm before touching it. |
| `findings/` | A numbered ledger of specific claims that were checked against raw data during the audit, including one open discrepancy (`001`, see below) and four smaller open text/labeling fixes. |

## One open item worth knowing about

`findings/001-ablation-l32-contradiction.md`: a later re-run (n=200, paired
seeds) of the Stage-3C causal-ablation experiment **supports the direction**
of the paper's amplification claim but **does not reproduce the exact
magnitudes** reported in the submitted paper (`aaai_submission/main.tex`,
§"Two LoRA changes"). The submitted PDF still reports the original numbers.
This doesn't affect the qualitative finding (LoRA amplifies pre-existing
promoter heads — confirmed both ways), but the exact effect sizes in that one
paragraph should be treated as provisional pending a reconciliation pass.
