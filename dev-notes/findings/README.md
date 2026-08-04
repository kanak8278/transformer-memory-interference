# Paper Findings Log

Discrepancies and confirmations found while verifying "Tracked but Suppressed"
against the repo's raw data, ahead of the AAAI port. One file per finding.

**Purpose:** capture *what differs and how* now, so we can fix everything in a
single batched pass later (during the AAAI edit). Do NOT edit `paper/main.tex`
from these yet — this is the record, not the fix.

**Convention:** each finding cites the paper by section + exact quoted text
(line numbers drift and the file will be reformatted for AAAI, so quotes are the
durable anchor). Each lists: paper claim → actual data (with source file) →
how they differ → root-cause hypothesis → resolution.

## Index (severity-ordered)

| ID | Severity | Title | Status | Related task |
|----|----------|-------|--------|--------------|
| [001](001-ablation-l32-contradiction.md) | 🔴 Critical | Stage-3C L32 ablation: paper claim contradicted by local n=50 data (ordering flipped) | Open — needs n=200 data or re-run | C3.1 / C3.4 |
| [002](002-dense-ivq-prose-vs-table.md) | 🟠 High | Dense-IVQ prose says "near-ceiling at every position"; the paper's own table shows 0% | Open — prose fix | C1.2 / B |
| [003](003-lora-cell-count-display.md) | 🟡 Medium | LoRA held-out table shows 16 of 28 cells; captions say "16", cross-family says "16/16" | Open — script + caption fix | B1 |
| [004](004-trial-counts-undocumented.md) | 🟡 Medium | Trial counts small and undocumented (SEM 10–20, dense-IVQ 20, ablation 50-local vs 200-claimed) | Open — caption fix + provenance | B3 |
| [005](005-ri-pi-terminology-leftover.md) | 🟢 Low | Leftover "RI" (old naming) in 4 appendix spots; code/JSON keep RI/PI keys | Open — text fix + release note | B2 |
| [000](000-verified-claims.md) | ✅ Verified | Claims that reproduced cleanly from local data (do not re-verify) | Done | — |

Legend: 🔴 affects a headline claim / could change the narrative · 🟠 internal
contradiction a reviewer catches on sight · 🟡 sloppiness / reproducibility ·
🟢 cosmetic.
