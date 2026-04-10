# Canonical Claims Audit (Artifacts vs Draft Text)

Date: 2026-04-09
Scope: `v3/paper/drafts/*.md`
Method: Compare numeric claims in drafts against saved artifacts in `v3/results/*` and explicit model logs.

---

## Canonical Numbers (Artifact-Backed)

### Narrative Transfer (directly backed by `v3/results/narrative/*`)

- **Wildlife, Qwen 1.5B** (`narrative_wildlife_Qwen2.5-1.5B-Instruct.json`)
  - `cells_with_pi_gt_ri = 10`
  - `total_cells = 11`
  - `pct_pi_gt_ri = 0.909`
  - Cell gaps include `+0.58` max, one negative cell `-0.088`

- **ICU, Qwen 1.5B** (`narrative_icu_Qwen2.5-1.5B-Instruct.json`)
  - `cells_with_pi_gt_ri = 5`
  - `total_cells = 15`
  - `pct_pi_gt_ri = 0.333`

- **ATC, Qwen 1.5B** (`narrative_atc_Qwen2.5-1.5B-Instruct.json`)
  - `cells_with_pi_gt_ri = 10`
  - `total_cells = 15`
  - `pct_pi_gt_ri = 0.667`

- **ATC, Claude Haiku API** (`narrative_api_atc_claude-haiku.json`)
  - `cells_with_pi_gt_ri = 0`
  - `total_cells = 18`
  - `mean_gap = -0.339`

- **Wildlife, Claude Haiku API** (`narrative_api_wildlife_claude-haiku.json`)
  - `cells_with_pi_gt_ri = 4`
  - `total_cells = 18`
  - `mean_gap = -0.037`

- **ICU, Claude Haiku API** (`narrative_api_icu_claude-haiku.json`)
  - `cells_with_pi_gt_ri = 5`
  - `total_cells = 18`
  - `mean_gap = -0.016`

### Probing (directly backed by `v3/results/probing/*`)

- **Qwen 1.5B** (`probing_Qwen2.5-1.5B-Instruct_2k_5u.json`)
  - Condition probe reaches `1.0` in late layers
  - RI-correct probe reaches `0.8792`
  - PI-correct probe is mostly `~0.60-0.67`

- **Qwen 3B** (`probing_Qwen2.5-3B-Instruct_5k_3u.json`)
  - Condition probe reaches `1.0` in many layers
  - RI-correct probe reaches `0.6308`
  - PI-correct probe values around `0.60-0.63`

- **Gemma 1B** (`probing_gemma-3-1b-it_2k_5u.json`)
  - Condition probe peaks around `0.9975`
  - RI-correct probe reaches `0.6475`
  - PI-correct probe is empty in this artifact (`{}`)

### Jacobian (directly backed by `v3/results/jacobian/*`)

- **Qwen 1.5B untrained** (`jacobian_Qwen2.5-1.5B-Instruct_untrained.json`)
  - `first_influence = 0.04036`
  - `last_influence = 0.06229`
  - `ratio_first_last = 0.648`

- **Mamba 1.4B untrained** (`jacobian_mamba-1.4b-hf_untrained.json`)
  - `first_influence = 0.04886`
  - `last_influence = 0.000165`
  - `ratio_first_last = 295.296`

### Logit Lens / Causal (directly backed by `v3/results/*stage2*` and `*stage3*`)

- **Qwen 1.5B stage3 summary** (`stage3_causal_20260316_231131.json`)
  - `peak_p_last = 0.1395`
  - `final_p_last = 0.0142`
  - Example large patching head effect: `mean_delta_p_last = +0.0383` (L19H6)

- **Gemma 1B stage3 summary** (`stage3_causal_20260317_000808.json`)
  - `peak_p_last = 0.02469`
  - `final_p_last = 0.0`
  - Example large patching head effect: `mean_delta_p_last = +0.0705` (L15H2)

---

## Claim Status by Draft File

Legend:
- **MATCH**: consistent with artifact-backed numbers
- **NEEDS UPDATE**: directionally right, numbers stale or mixed-scope
- **UNSUPPORTED**: no direct artifact support found in current `v3/results/*`

### `ABSTRACT_DRAFT.md`

- **MATCH**
  - `9 models`, `91%`, `~50pp` framing is consistent with the active v3 narrative across drafts and logs.
  - `87% vs 61%` probing asymmetry is broadly aligned with Qwen 1.5B probing artifact.
- **NEEDS UPDATE**
  - “Mamba gap = +57%” should be tied to a specific operating point and source artifact.
  - “eliminating causal attention/softmax/positional encoding as sole causes” is a strong interpretation claim; keep but explicitly scope as component elimination, not proof of necessity.

### `INTRODUCTION_DRAFT.md`

- **MATCH**
  - `87%` vs `61%` probing wording is consistent with Qwen 1.5B artifact.
  - `295x` Mamba untrained endpoint ratio is directly backed.
- **NEEDS UPDATE**
  - `1.47x` transformer quarter-vs-middle value is not directly stored in current jacobian artifact fields; keep only if tied to the exact script output/log that computed quarter stats.
  - Flan-T5 bidirectional claim should remain labeled preliminary.

### `PAPER_OUTLINE.md`

- **NEEDS UPDATE**
  - “11 models / 7 architecture families” conflicts with other drafts using `9 models` and narrower family grouping.
  - “probing shows 100% RI/PI discrimination” mixes condition discrimination with correctness probes; should be rewritten as condition probe near-ceiling, correctness probes much lower.
  - Several table entries use numbers that appear sourced from older summaries rather than directly from current artifacts.

### `PAPER_RESULTS_DRAFT.md`

- **MATCH**
  - Narrative rows for Qwen wildlife/ICU/ATC align with current narrative artifacts.
  - Probing `87%`, `~61%` and Jacobian `295x` are broadly consistent.
  - Logit lens suppression table for Qwen 1.5B and Gemma is consistent with stage3 stage2-info.
- **NEEDS UPDATE**
  - Haiku narrative rows should be reconciled to API files that use `18` cells in current artifacts (not `8`).
  - Claims about Qwen 3B base should cite a present artifact path in `v3/results` (current visible stage1 artifacts are for Qwen 3B instruct).

### `ONE_PAGER_FULL_PROJECT.md`

- **MATCH**
  - High-level narrative direction is consistent: PI>RI broad trend, Mamba endpoint ratio, probing asymmetry.
- **NEEDS UPDATE**
  - “50,000+ trials” needs a reproducible count source.
  - Haiku narrative claims should use `18`-cell API totals where applicable.

### `ONE_PAGER_NARRATIVE_TRANSFER.md`

- **MATCH**
  - Qwen wildlife (`10/11`, `+27%`), ICU (`5/15`, negative mean), and ATC (`10/15`, small positive) are aligned.
- **NEEDS UPDATE**
  - Haiku counts `3/8`, `0/8` are stale vs current API artifacts (`4/18`, `5/18`, `0/18` for wildlife/ICU/ATC).
  - Keep directionality (wildlife near-null, ICU near-null, ATC strong negative) but update absolute counts.

### `NARRATIVE_TRANSFER_RESULTS.md`

- **MATCH**
  - Qwen domain summaries are aligned with narrative artifacts.
  - Haiku ATC strong reversal is aligned (`mean_gap ~ -0.339`).
- **NEEDS UPDATE**
  - Haiku table uses `8`-cell totals while current API files show `18` cells.
  - Significance language should be tied to concrete CI fields if reported as significant.

### `MI_CONSOLIDATED.md`

- **MATCH**
  - Core mechanistic arc and many headline values are consistent with artifacts.
  - Stage2/3 qualitative conclusions align with available outputs.
- **NEEDS UPDATE**
  - Some hard numbers rely on prior aggregate calculations not stored as one canonical JSON in `v3/results`; add explicit provenance references.
  - Keep caution where this file itself marks partial/broken items (remedy logit lens, SAE behavior coupling).

### `SUBMISSION_CHECKLIST.md`

- **NEEDS UPDATE**
  - Mostly process checklist; numeric conflict appears where “done” implies stronger evidence than currently artifact-backed in a few areas.
  - Add one line requiring all tables to cite exact artifact paths.

---

## Highest-Priority Fixes Before Submission

1. **Unify model/family counts** across all drafts (`9` vs `11`, and architecture-family wording).
2. **Unify Haiku narrative cell counts** to current API artifact totals (`18` where applicable) or explicitly state subset selection criteria.
3. **Separate probe metrics clearly**:
   - condition probe (near-ceiling),
   - RI-correct probe (moderate/high),
   - PI-correct probe (weak/near chance).
4. **Add provenance footnotes** for every table with direct artifact paths.
5. **Downgrade unsupported absolutes** (for example “50,000+ trials”) unless backed by a reproducible counter script output.

---

## Recommendation

Use this file as the lock point for numbers, then do a targeted pass over:

- `ABSTRACT_DRAFT.md`
- `PAPER_RESULTS_DRAFT.md`
- `ONE_PAGER_NARRATIVE_TRANSFER.md`
- `NARRATIVE_TRANSFER_RESULTS.md`
- `PAPER_OUTLINE.md`

to remove count conflicts and align all numeric claims to explicit artifact references.
