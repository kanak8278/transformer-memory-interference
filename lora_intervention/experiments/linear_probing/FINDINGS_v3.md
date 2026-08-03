# Linear probing v3 — findings (Qwen2.5-3B, base vs main LoRA)

Run: `probe_Qwen2.5-3B-Instruct-{base,lora}-v3_*.json` (6 EVAL_CELLS × 7
conditions, both states). Analysis: `analyze_isoaccuracy.py` →
`isoaccuracy_v3.png`, `isoaccuracy_v3_summary.json`.

Supersedes the pilot (`*-full_*.json`), which was confounded — see below.

## What changed vs the pilot

Four fixes to `run_probing.py`, all motivated by the pilot's confound (its
"base FVQ decodable, LoRA not" headline turned out to be an artifact of metric
choice + class balance, not a representational difference):

1. **Metric → ROC-AUC + balanced-accuracy** (was plain accuracy vs majority
   baseline). Chance is 0.5 regardless of class skew, so base and LoRA become
   comparable at every testable cell. The pilot's accuracy metric structurally
   could not beat the majority baseline under skew even when the probe
   separated the classes — it hid signal at ~44/84 cells.
2. **Class-balanced stopping** (was Wilson-CI-on-accuracy). Draw until both
   classes have ≥40 examples, cap 400, bail on genuine saturation. The Wilson
   rule stopped skewed cells at n≈30–60 with 1–4 minority samples
   (statistically dead).
3. **Correctness-INDEPENDENT retrieval probe.** Per layer, a diagonal bilinear
   probe `LR(r_L ⊙ e(v))` (e = unembedding row of value's first token) trained
   to pick the ground-truth answer value out of the in-context candidates.
   Label never depends on the model's output → defined at *every* cell,
   including the ~28 saturated ones the correctness probe can't touch. Reported
   for all / wrong-answer / correct-answer trial subsets.
4. **Iso-accuracy analysis.** Compare probe AUC as a function of *behavioral
   accuracy* (base and LoRA are class-balanced at different loads, so per-cell
   comparison conflates representation with balance).

Numbers below are the **full 31-cell grid** (3 in-training + 28 canonical
held-out cells, K∈{2,5,7,10,15,20,25,30}×N∈{5,10,15,20,30,50,75}); the earlier
6-cell pilot showed the same pattern with smaller n.

## Result 1 — the pilot asymmetry was an artifact (Fix 4a)

Correctness-probe AUC, binned by behavioral accuracy, is **roughly comparable
base vs LoRA** at matched accuracy (base marginally higher in mid bands, but no
large systematic gap):

| beh-acc band | base AUC (n) | lora AUC (n) |
|---|---|---|
| 0.0–0.2 | 0.68 (104) | 0.69 (35) |
| 0.2–0.4 | 0.72 (39)  | 0.63 (32) |
| 0.4–0.6 | 0.74 (17)  | 0.65 (12) |
| 0.6–0.8 | 0.82 (3)   | 0.65 (6)  |
| 0.8–1.0 | 0.82 (2)   | 0.76 (48) |

"Correctness is linearly decodable" is weakly true for both and does not
cleanly distinguish base from LoRA once you control for accuracy. The pilot's
"base FVQ decodable, LoRA not" gap was which cells happened to be class-balanced.

## Result 2 — LoRA represents the answer far more strongly (Fix 4b)

Retrieval-probe AUC (correctness-independent), binned by behavioral accuracy,
is **systematically and substantially higher for LoRA** at every matched band,
now with solid n:

| beh-acc band | base AUC (n) | lora AUC (n) |
|---|---|---|
| 0.0–0.2 | 0.56 (156) | 0.65 (73) |
| 0.2–0.4 | 0.66 (39)  | 0.88 (32) |
| 0.4–0.6 | 0.73 (17)  | 0.95 (12) |
| 0.6–0.8 | 0.85 (3)   | 0.96 (6)  |
| 0.8–1.0 | 0.94 (2)   | 1.00 (94) |

This survives the confound: at the *same* behavioral accuracy, the correct
answer value is far more linearly present in LoRA's residual stream than base's.

## Result 3 — "tracked but suppressed" is LoRA-specific (Fix 3)

Retrieval AUC on **wrong-answer trials only** (correct value decodable despite
the model emitting a wrong answer), cells with n_wrong ≥ 30. Across the 31-cell
grid the ranking is stark: the **top ~60 rows are all LoRA** (0.78–0.99, mostly
IVQ), then base appears clustered near chance:

| state | cell | cond | beh | n_wrong | retr[wrong] |
|---|---|---|---|---|---|
| lora | 15k15u | IVQ_d50 | 48% | 47 | **0.99** |
| lora | 25k15u | IVQ_d25 | 85% | 41 | **0.99** |
| lora | 15k75u | CVQ     | 89% | 40 | **0.97** |
| lora | 15k30u | IVQ_d10 | 89% | 41 | **0.96** |
| … (≈60 LoRA rows, 0.78–0.99) | | | | | |
| base | 30k50u | CVQ     | 37% | 69 | 0.79 |
| base | 5k10u  | FVQ     | 81% | 40 | 0.78 |
| base | 20k75u | CVQ     | 46% | 49 | 0.74 |
| base | IVQ (nearly all depths/cells) | | | large | **0.50–0.63** (≈ chance) |

Base's only above-chance wrong-trial retrieval is on FVQ/CVQ (first/last, the
easy endpoints); its **interior (IVQ)** wrong-trial retrieval is at chance
almost everywhere. LoRA is high across FVQ/CVQ **and** IVQ.

**Interpretation.** On the trials where LoRA outputs the wrong interior value,
the *correct* value is still near-perfectly decodable from the late-layer
residual stream (AUC 0.83–0.98, best layer L33–35) — the correct answer is
tracked internally and suppressed at output. Base does **not** show this for
IVQ: its wrong-trial retrieval sits at chance (~0.55), i.e. base genuinely
fails to track the interior value rather than tracking-then-suppressing.

This is the confound-free version of the paper's "tracked but suppressed"
mechanism, and it is specifically a property of the LoRA adapter.

## Result 4 — cross-model replication: gemma-3-4b-it (± gemma adapter)

Same 31-cell grid, same v3 method, second model family. **Loader note:** gemma-3-4b-it
is a multimodal checkpoint; loading it as `Gemma3ForCausalLM` (or attaching the
adapter to the `Gemma3ForConditionalGeneration` structure) silently leaves the LM
adapter at init → base≡lora. Fixed by transplanting the text tower into a
standalone `Gemma3ForCausalLM` matching the adapter's training structure
(`run_probing._load_gemma_text_causal_lm`). Files: `probe_gemma-3-4b-it-*-v3_*.json`,
`isoaccuracy_v3_gemma-3-4b-it.png`.

**"Tracked but suppressed" REPLICATES (the headline).** Retrieval AUC on
wrong-answer IVQ trials (n_wrong ≥ 30):

| model family | base IVQ-wrong (mean, %cells≥0.75) | lora IVQ-wrong (mean, %cells≥0.75) |
|---|---|---|
| Qwen2.5-3B | ≈0.55 (≈chance) | 0.78–0.99 (dominant) |
| gemma-3-4b-it | **0.60 (3/155 cells ≥0.75)** | **0.73 (55/127 cells ≥0.75)** |

In both families, on interior-value trials the model answers wrong, LoRA still
linearly encodes the correct value while base sits near chance. The
tracked-but-suppressed mechanism is LoRA-specific and cross-model.

**What does NOT fully replicate: Result 2.** The "LoRA represents the answer more
strongly on *all* trials" gap (large in Qwen) is weak/absent in gemma — base and
lora retrieval-AUC largely overlap (right panel of the plot; e.g. 0.2–0.4 band
base 0.78 vs lora 0.86; 0.8–1.0 band 0.99 vs 1.00). gemma's base already
represents the answer fairly strongly, so the adapter adds less on already-correct
trials. The adapter's distinctive effect shows up specifically on the *wrong*
trials (suppression), which is where it matters for the mechanism claim.

## Caveats / next

- The retrieval probe is a trained (diagonal) logit-lens; it discriminates the
  correct value from *in-context distractors*, so it is not merely detecting
  "the value is present in the prompt." A full (non-diagonal / low-rank
  bilinear) probe could be stronger; diagonal is a conservative first pass.
- 6-cell grid, single model. Next: full 31-cell grid, then gemma-3-4b-it ±
  gemma adapter (`--base google/gemma-3-4b-it`), same script.
- A few individual cells show high correctness-AUC on tiny minority counts
  (e.g. base 25k50u IVQ_d50 corr_auc 0.95, n_pos≈2) — do not read per-cell;
  the binned/aggregate view is the robust one.
