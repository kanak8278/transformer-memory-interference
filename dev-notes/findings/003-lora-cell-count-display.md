# Finding 003 — LoRA held-out table shows 16 of 28 cells; captions inconsistent

- **Severity:** 🟡 Medium (headline number inconsistency; data is correct)
- **Status:** Open — table-generator + caption fix
- **Related task:** B1
- **Verified:** 2026-07-11, against `main_eval_20260523_055309.json`

## Paper claim

- §5.2 (main.tex ~L389): "On the **28** held-out Arbitrary-Single cells, both
  FVQ and CVQ accuracy reach 93–100%."
- Negative-control para (main.tex ~L1377): "reaches it on **28/28** cells."
- App L cell table lists **28** held-out cells.
- BUT `tab_lora_arb.tex` / `tab_lora_arb_full.tex` render only **16** rows, and:
  - Table caption (main.tex ~L441): "16 held-out cells."
  - Cross-family summary `tab_lora_cross_family` (main.tex ~L1501 region):
    Qwen "ARB held-out fixed **16/16**" (Gemma correctly 28/28).
  - (The "16/16 SEM OOD" mentions at ~L407 and ~L1363 are **correct** — they
    are Gemma context, and Gemma has 16 SEM OOD cells. Not an error.)

## Actual data

`main_eval_20260523_055309.json` → `results.ARBITRARY_SINGLE` has **28 cells**.
Minimum post-LoRA accuracy across all 28 = **92.5%**; all 28 are 92.5–100%. So
the "28 held-out cells, 93–100%" claim is **true** (92.5 rounds into the band).

The 16 comes from a hard-coded `DISPLAY_CELLS` list in
`paper/scripts/generate_lora_tables.py` that renders only 16 of the 28.

## How they differ

Data has 28; the compact/full tables display 16; captions variously say "16",
"16/16", and "28". The narrative number (28) is right; the tables and their
captions under-report and mislabel.

## Resolution

1. Extend `DISPLAY_CELLS` to all 28 (at least in the appendix full table), or
   re-caption compact tables "16 representative of 28 held-out cells."
2. Fix `tab_lora_cross_family`: Qwen ARB "16/16" → "28/28" (data supports it;
   min 92.5%). Qwen SEM "4/4" and Gemma "28/28"/"16/16" are correct.
3. Regenerate tables and re-grep for every "16" to catch any stragglers.
