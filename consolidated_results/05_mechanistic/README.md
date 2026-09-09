# 05 — Mechanistic interpretability: how LoRA closes the gap

**Question.** Themes 1–4 are what the model *outputs*. This is what happens
*inside*: by what mechanism does LoRA turn a suppressed current-value into a
retrieved one? Five methods — plus a sixth,
[`value_identity_probe/`](value_identity_probe/), which asks the prior question
the LoRA framing takes for granted: **is the answer value in the residual stream
at all on trials the model gets wrong?** That one is base-only by design, so the
theme's "how LoRA does it" title does not cover it.

| file | rows | unit | question |
|---|---|---|---|
| `logit_lens.csv` | 456 | (model, variant, cell, condition, **layer**) | is v_last decodable in late layers, and does LoRA raise P(v_last)? |
| `probing.csv` | 354 | (model, variant, probe, **layer**) | is the info linearly present, regardless of whether the model emits it? |
| `attention_routing.csv` | 45 | (model, regime, **layer / head**) | does LoRA re-point attention at the v_last round? |
| `causal_ablation.csv` | 51 | (model, variant, cell, **layer**) | are those heads *causally* responsible? |
| **`entropy_lens/`** | 34,164 + 465 | (arm, cell, condition, subset, **layer**) | how does *uncertainty* evolve with depth, and is the model's confidence calibrated to its correctness? |
| **`value_identity_probe/`** | 6,967 + 254 | (model, cell, condition, subset, label_set, **layer**) | is the ground-truth *value* linearly decodable — 50-way, chance 2% — even on trials the model answers wrongly? |

Primary model **Qwen2.5-3B-Instruct**; **gemma-3-4b-it** as cross-family
replication. base vs LoRA throughout. Naming: RI→FVQ, PI→CVQ; `v_first`/`v_last`
kept because they are the mechanistic quantity.

**Gemma coverage is not uniform.** `logit_lens`, `probing`, `causal_ablation`
and `value_identity_probe` have both models; **`attention_routing` and
`entropy_lens` are Qwen-only.** State
that if the Qwen-primary / Gemma-replicates framing is presented as covering the
whole mechanistic story. For `entropy_lens` a Gemma arm is a flag change costing
~30 min of GPU (see its README).

`entropy_lens/` is a **subfolder, not a single CSV** — it is the only method here
with more than one unit of measurement per arm (per-layer profiles, a per-cell
confusion view, and behavioural labels), so it splits one directory per arm:
`qwen_base/`, `qwen_lora/`, `gpt2_scratch/`, plus a cross-arm `summary.csv`. It
is also the only mechanistic method that touches the **from-scratch** model of
theme 06, and its scratch arm independently replicates that theme (150 cells,
mean |diff| 0.0156). Full detail: `entropy_lens/README.md`.

`value_identity_probe/` is a subfolder for the same reason, and splits one
directory per **arm = model x (K, N) cell** — `Qwen2.5-3B-Instruct_5k_10u/`,
`Qwen2.5-3B-Instruct_10k_5u/`, `gemma-3-4b-it_5k_10u/`,
`gemma-3-4b-it_10k_5u/`. Inside each, the three subsets are three files:
`probe_all.csv`, `probe_correct.csv`, `probe_wrong.csv`, alongside that arm's
`summary.csv`, `behavioral.csv` and `pool.csv`. `probe_wrong.csv` is the
headline and `probe_correct.csv` the output-head confound check. Two things to
carry over before using it. It is the **only base-only** method here: LoRA
scores 1.000 wherever the closed-pool design is valid, so its wrong-answer
subset is empty at any sample size, and a LoRA arm needs a different design
rather than more trials. And its `cv_first` control **beats** the result at
every interior slot, so the residual stream carries the *first* value more
strongly than the queried one — the "tracked but not surfaced" reading survives
for the first value and weakens for interior ones. Full detail:
`value_identity_probe/README.md`.

---

## The chain of evidence (all at K=2, the clean operating point)

**1. Logit lens — the base model tracks v_last, then suppresses it.**
Qwen, CVQ, P(v_last) by layer:

| layer | base | +LoRA |
|---|---|---|
| L32 | 0.34 | 0.99 |
| L33 | 0.51 | 1.00 |
| L34 | 0.40 | 1.00 |
| L35 (readout) | **0.25** | **1.00** |

The base model *builds* v_last mid-stack (peaks ~0.5 at L33) then **suppresses it
back down to 0.25 by the readout layer** — the "tracked but suppressed" result.
LoRA holds it at ceiling. Gemma replicates (base + LoRA, 34 layers, in the file).

**2. Probing — v_last is linearly decodable even in the base model.**
A linear probe recovers "is v_last correct" from base-model late layers well
above its behavioural accuracy — the information is present; the model just
doesn't route it to the output. LoRA raises the probe too, most on the
CVQ_correct probe (base ~0.58 → LoRA ~0.94 at late layers, CIs exclude zero).

**3. Attention routing — LoRA re-points attention at the last update.**
Head-averaged P(gen-token attends to the v_last round), Qwen primacy regime,
peaks at **L33: base 0.07 → LoRA 0.53 (+0.46)**. 15 "promoter" heads (all L30–33)
each jump +0.53 to +0.68. The same 8 heads are the top movers in the reversal
regime too (see `regime=reversal` rows).

**4. Causal ablation — those heads are responsible, and LoRA leans on them more.**
Ablating the 8 promoter heads hurts LoRA more than base (paired, n=200):

| cell | Δ P(v_last) lora | Δ base | **paired (lora−base)** |
|---|---|---|---|
| K2/N5 | −0.147 | −0.065 | **−0.082** |
| K2/N50 | −0.213 | +0.017 | **−0.230** |

Both CIs exclude zero (`ci_analysis.txt`). K2/N50 is the clean proof: base has
no v_last to lose, LoRA's is head-driven. These reproduce the paper's headline
numbers exactly.

**Together:** the base model *computes* v_last and *represents* it linearly, but
attention doesn't route it to the readout, so it's suppressed. LoRA's fix is
concentrated in ~8 late-layer heads that re-point attention at the last update;
ablating them selectively removes the LoRA gain.

---

## Provenance & tiers

`raw` = from a results JSON on disk. `derived` = back-extracted from a
precomputed `.txt`/`.tex` because the raw run is GPU-box-only.

| file | raw | derived |
|---|---|---|
| logit_lens | Gemma base+LoRA (`v3/results_vllm/logit_lens/`) | Qwen base+LoRA late layers, with CIs (`ci_analysis.txt`) — raw Qwen logit-lens JSON not local |
| probing | Gemma base+LoRA, Qwen-LoRA (`v3/results_vllm/probing/`) | Qwen base+LoRA late layers, with CIs (`ci_analysis.txt`) |
| attention_routing | — | Qwen primacy + reversal, per-layer + top heads (`*_comparison.txt`) — raw per-head JSON not local |
| causal_ablation | Qwen stage-3 trajectories (`v3/scripts/experiments/results/`), Qwen+Gemma n=200 rerun (`experiments/*/ablation*`) | — |

### Two caveats before aggregating

- **Qwen-LoRA probing appears twice**: `raw` (all 36 layers, no CI) and `derived`
  (7 late layers, with CI). They overlap on late layers — don't naively average
  the file. Filter by `provenance_tier`. The `derived` rows are the only source
  of **Qwen base** probing (raw not local).
- **Layer indexing differs by source**: raw Gemma probing uses relative index
  0–7 (the 8 late layers); the `ci_analysis`-derived rows and logit-lens use
  absolute layer numbers. `notes` flags which.

### Deliberately not extracted here

The `stage3` JSONs and block-readout files also carry full per-trial logit
distributions and 36-layer P(v_last) curves under block formatting; only the
paper-relevant summaries are lifted. The block-readout per-layer curves live in
theme 3's source note. Raw attention per-head matrices and raw Qwen logit-lens
were never copied off the GPU box (see `AAAI_PREP_PLAN.md` §1.3).

## Sources

Per-file lineage in `../PROVENANCE.md`; extractors in
`../build/build_05_mechanistic.py`.
