# Finding 005 — Leftover "RI" terminology (old naming scheme)

- **Severity:** 🟢 Low (cosmetic; looks like a find-and-replace miss)
- **Status:** Open — text fix + release-code naming note
- **Related task:** B2
- **Verified:** 2026-07-11

## Issue

The paper standardizes on **FVQ** (first-value) / **CVQ** (current-value), but
four spots in the appendices still use the old **RI** (retroactive interference)
label from the earlier PI/RI-era framing:

- main.tex ~L1447: "mean **RI** 99.4%"
- main.tex ~L1632: condition discrimination "(**RI** vs CVQ query)"
- main.tex ~L1648: "**RI**-correctness is broadly unchanged"
- main.tex ~L1658: "$\Pr(v_{first})$ for **RI**"

(Line numbers approximate; grep `\bRI\b` in main.tex.)

## Deeper note for the code/data release

The **entire codebase and all result JSONs use `RI`/`PI` keys** (e.g.
`stats.RI.accuracy`, `PI correctness`). Renaming these would break every
generator script in `paper/scripts/` and `lora_intervention/`. So:

- Fix the 4 leftover "RI" mentions in `main.tex` → "FVQ".
- Do **not** rename keys in code/JSON. Instead add a one-line mapping note to
  the release README: `RI = FVQ (first-value)`, `PI = CVQ (current/last-value)`.

## Resolution

`grep -n '\bRI\b' paper/main.tex` → replace the 4 with FVQ; add the naming-map
note to the release README.
