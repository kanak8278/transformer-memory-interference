# Experiments

Every experiment implemented in this repo: what it measures, on which
models/data, what it found, and where the code and results live. Organized
to follow the paper's argument (behavioral failure → two recoveries → does
training ever fix it → mechanism → from-scratch control), then an "extended"
section for work done after the submitted draft that sharpens or qualifies
specific claims.

Naming: **FVQ** = first-value query, **CVQ** = current/last-value query,
**IVQ** = intermediate (k-th-value) query. Datasets: **ARB**
(Arbitrary-Single, single-token values, 2,300-word pool / 46 categories —
used for all mechanistic work) and **SEM** (Semantic-Multi, real multi-token
category members, 2,403 entries — the primary behavioral dataset). Full
definitions: [README.md](README.md#task-formulation).

Shared library code (imported by everything below):
`mechanistic_probing_v2/core/` — `dataset_configs.py` (ARB/SEM construction,
the shared word pool, category definitions), `model_loader.py`,
`inference.py`, `evaluation.py` (matching rule, Wilson CIs, bootstrap).

---

## 1. Behavioral characterization (does the gap exist, and where)

**What it measures:** FVQ/CVQ/IVQ accuracy across the full $(K,N)$ grid, per
model — the paper's central empirical claim (§4).

### 1a. Open-weight models (local, vLLM/HF)

- **Models (11 main + 4 appendix-only):** Qwen2.5 {0.5B, 1.5B, 3B, 3B-Instruct},
  Qwen3.5 {0.8B, 2B, 4B, 9B}, Gemma-3 {270m, 1b, 4b}-it; appendix-only
  architecture controls TinyLlama-1.1B, StableLM-2-1.6B, Pythia-410M (kept
  only where they clear a minimum-coverage bar — see `dev-notes/EXPERIMENT_STATUS.md`).
- **Grid:** $K\in\{2,3,5,7,10,15,20,25,30\}\times N\in\{5,7,10,15,20,30,50,75,100\}$
  = 81 cells, 100–200 trials/cell, both ARB and SEM.
- **Code:** `v3/scripts/experiments/stage1_sweep.py`.
- **Results:** `v3/results_vllm/arbitrary_single/<model>/`,
  `v3/results_vllm/local_models_behavioral.csv`.
- **Finding:** every model shows a substantial FVQ−CVQ gap somewhere on the
  grid; smaller/weaker models cross into a **reversal** regime (CVQ > FVQ,
  both well below ceiling) at high load instead of the more common
  **primacy** shape (FVQ > CVQ).

### 1b. Proprietary frontier models (API)

- **Models:** GPT-4.1, GPT-4.1-mini, Claude-4.5-Haiku, Claude-4.5-Sonnet,
  Gemini-2.5-Flash, Gemini-2.5-Pro.
- **Grid:** $K\in\{2,5,10,15,20,25,30,40,45\}\times N\in\{1,5,10,15,20,30,50\}$
  = 63 cells, adaptive Wilson-CI early stopping (min ~40, cap 200), SEM only
  (ARB's single-token pool doesn't stress frontier models the same way).
- **Code:** `experiments_cloud/sweep_semantic.py` (`models/model_factory.py`
  routes to the right provider client).
- **Results:** `experiments_cloud/results/proprietary_semantic_multi.csv`.
- **Finding:** same qualitative failure, later operating point — e.g.
  Gemini-2.5-Pro stays near ceiling until $K{=}20,N{=}50$ (gap +0.13) while
  Claude-4.5-Haiku shows a +0.47 gap already at $K{=}20,N{=}50$. No model
  uses more than 3.3% of its advertised context window at the cell it fails
  on — ruling out a length/context-limit explanation.

---

## 2. Format intervention (training-free recovery)

**What it measures:** whether making stream positions explicit at the
prompt surface (no weight change) closes the gap (paper §5.1). Also serves
as the **per-position analysis** — same sweep records accuracy at every
position, not just FVQ/CVQ.

- **Formats:** Plain (default), Labeled (per-entry index), Block (round
  headers), Landmark (round separators).
- **Models:** 10 (4 open-weight: Qwen2.5-3B-Instruct, Qwen3.5-2B/4B,
  Gemma-3-4b-it; 6 proprietary).
- **Grid:** local models — $K\in\{5,10\}\times N\in\{10,20,50\}$, 4 formats,
  16 position-points/cell, up to 200 trials. Proprietary — same grid,
  matched trial counts.
- **Code:** `experiments_cloud/ucurve_vllm.py` (local),
  `experiments_cloud/ucurve_sweep.py` (proprietary).
- **Results:** `experiments_cloud/results/ucurve_vllm/<model>/`,
  `experiments_cloud/results/ucurve_proprietary/<model>/`.
- **Finding:** Block closes almost all of the gap for every open-weight
  model at the hardest cell (K=10, N=50) — e.g. Qwen2.5-3B-Instruct goes from
  0.49/0.23 (FVQ/CVQ, Plain) to 0.84/0.92 (Block); Gemma-3-4b-it from
  0.92/0.01 to 0.99/0.89. Labeled and Landmark give partial recoveries —
  round-grouping and explicit numbering each help, neither alone is
  sufficient.

---

## 3. LoRA intervention (weight-based recovery)

**What it measures:** whether a small, task-matched fine-tune surfaces the
capability, and whether the fix is specific to this task (paper §5.2).

- **Base model / adapter:** Qwen2.5-3B-Instruct + LoRA, r=16, α=32, dropout
  0.05, attention-only (Q/K/V/O), lr 2e-4, bf16, 2 epochs. 7.4M trainable
  params (0.24% of weights).
- **Training data:** 18,000 examples, grid $K\in\{2,3,5,10\}\times
  N\in\{5,10,15,20\}$, query mix FVQ 40% / CVQ 40% / IVQ 20%. ARB only,
  600 test-ID / 2,200 test-OOD held out.
- **Code:** `lora_intervention/data_gen.py` (generation, deterministic
  seeds), `lora_intervention/train.py`, `lora_intervention/evaluate.py`,
  `lora_intervention/evaluate_ivq.py`.
- **Checkpoint:** `lora_intervention/checkpoints/adapter/` (43MB, committed).
- **Main result:** on 28 held-out ARB cells extending to $K{=}30,N{=}75$
  (~10× the training grid in stream length), both FVQ and CVQ reach
  93–100% post-adapter. Result file:
  `lora_intervention/results/main_eval_*.json`; table generator
  `paper/scripts/generate_lora_tables.py`.
- **Cross-distribution transfer:** the same adapter (never trained on SEM)
  reaches 100%/100% on 4 held-out SEM cells, including cells in the deep
  reversal regime (baseline gap −45%). Result:
  `lora_sem_finish_results.json`, `lora_sem_validation_results.json`.
- **Family control:** identical recipe on **google/gemma-3-4b-it**
  (`lora_intervention/checkpoints/gemma_adapter/`, 71MB) — a family with the
  opposite baseline regime (primacy up to +87pp, never reversal). Closes
  28/28 held-out ARB cells (mean FVQ 99.4%, CVQ 97.0%) and 16/16 SEM OOD
  cells. Result: `lora_intervention/results/gemma_main_eval_*.json`.
- **Negative control:** the same LoRA recipe trained on GSM8K instead
  (`lora_intervention/checkpoints/qwen_arith_adapter/`, data via
  `data_gen_arithmetic.py`) learns arithmetic (16%→70% GSM8K accuracy) but
  leaves the FVQ−CVQ gap unchanged on every held-out interference cell —
  the recovery is task-specific, not a generic "fine-tuning helps" effect.
  Result: `lora_intervention/results/gsm8k_task_eval_*.json`.
- **Dense intermediate-position sweep:** every interior position, not just
  FVQ/CVQ, across 6 cells. Base collapses within 2 positions of the stream
  start; LoRA recovers near-ceiling at (almost) every position (see
  [§8](#8-extended-analyses-post-submission) for where this breaks down at
  scale). Result: `lora_intervention/results/dense_ivq_*.json`.

---

## 4. Training-dynamics sweep (does pretraining ever fix it unaided?)

**What it measures:** whether the gap closes at any pretraining or
post-training stage, using public checkpoint releases (paper §6).

- **Models:** SmolLM3-3B — 32 published pretraining checkpoints across all
  3 documented stages, plus 5 post-training stages (mid-training, SFT, APO,
  LC-expert, final). SmolLM2-1.7B — 42 checkpoints, second family. 79
  checkpoints total.
- **Grid:** 32-cell ARB grid partitioned by load: low ($K{\le}3$), medium
  ($K{\ge}5,N{<}20$), high ($K{\ge}5,N{\ge}20$).
- **Code:** `v3/scripts/experiments/training_dynamics.py`.
- **Results:** `v3/results_vllm/training_dynamics/` (SmolLM2),
  `v3/results_vllm/training_dynamics_smollm3/` (SmolLM3).
- **Finding:** the gap is directionally biased at **every** checkpoint in
  **every** load regime, for both families — no regime ever stabilizes at
  zero, through pretraining or any post-training stage. Re-running
  SmolLM3's post-training stages under its released chat template
  (instead of completion format, used for cross-checkpoint comparability)
  preserves the direction/ordering of the gap — not a formatting artifact.
  SmolLM3 is the cleaner signal (SmolLM2 is noisier; use SmolLM3 as primary
  if reproducing only one).

---

## 5. Naturalistic-language generalization checks

**What it measures:** whether the synthetic key-value framing is doing the
work, by testing on prose narratives instead. Two attempts, reported
honestly because the first one didn't work cleanly:

- **Dota2 match-commentary domain** (`narrative_generator/dota2/`,
  `experiments_cloud/` narrative scripts): 11 local models, mean gaps
  0.01–0.21 — much weaker than the synthetic task (0.26+), and several
  models even show CVQ > FVQ. The paper's Limitations section notes this
  variant "could not be cleanly constructed" — position-indexed retrieval
  is confounded with natural event ordering in match commentary, so a
  clean signal never emerged. Not used as evidence in the paper; kept here
  for transparency. Results: `v3/results_vllm/narrative_dota2/`.
- **Museum-tour domain** (`narrative_generator/museum/`,
  `experiments_cloud/museum_ivq_haiku.py`, `museum_endpoint_sweep.py`,
  `plot_museum_ivq.py`): a more recent, more carefully constructed
  naturalistic variant (exhibit descriptions with state changes rather than
  match events). Exploratory — not yet in the paper. Results:
  `experiments_cloud/results/museum_endpoint/`.

---

## 6. Mechanistic analysis (what LoRA actually changes)

**What it measures:** the circuit-level story behind the LoRA recovery
(paper §7). All four methods use Qwen2.5-3B-Instruct, ARB, mostly at
$K{=}2,N{=}5$ (plus $K{=}2,N{=}50$ for a genuine-failure comparison point);
cross-family replication on gemma-3-4b-it.

| Method | Question | Code | Result |
|---|---|---|---|
| **Linear probing** | Is CVQ-correctness linearly decodable from the residual stream, per layer? | `v3/scripts/experiments/probing_classifier.py`, `lora_intervention/run_probing_lora_hf.py` | `v3/results_vllm/probing/`, `lora_intervention/results/ci_analysis*.txt` |
| **Logit lens** | At which layer does Pr(v_last) become visible via final-norm→lm_head? | `v3/scripts/experiments/stage2_logit_lens.py`, `lora_intervention/run_logit_lens_lora_hf.py` | `v3/results_vllm/logit_lens/`, `lora_intervention/results/logit_lens_comparison.txt` |
| **Attention routing** | Which heads attend to v_last vs. v_first, and how does that change post-LoRA? | `lora_intervention/run_attention_routing_lora.py` | `lora_intervention/results/attention_routing_*.txt`, `v3/results_vllm/attention_routing/` |
| **Causal ablation** | Do the identified heads *cause* the v_last signal (zero them, measure ΔPr(v_last))? | `lora_intervention/run_stage3_lora.py`, `run_stage3_baseline_promoter_ablation.py` | `v3/scripts/experiments/results/`, `lora_intervention/results/stage3_*.txt` |

**Findings, chained:**

1. **The substrate already exists in baseline.** A condition-discrimination
   probe (FVQ vs. CVQ prompt) hits 100% from layer 6 in both models — LoRA
   adds nothing there. A CVQ-correctness probe sits near chance (57–65%) at
   L27–L33 in baseline while the matching FVQ-correctness probe is at
   77–81% — the baseline doesn't know internally whether it'll get CVQ
   right. Logit lens shows the same pattern at the token level: L32 baseline
   Pr(v_last) = 0.34, briefly surfacing, then decaying to 0.25 by L35 —
   *found, then suppressed*.
2. **LoRA amplifies a partial promoter circuit in L30–L33.** 15 heads gain
   substantial v_last attention post-LoRA (0.05–0.23 → 0.60–0.86, a
   4–12× increase); these heads already sit above the 95th percentile of
   all 576 baseline heads pre-LoRA (median 0.001) — they're not random,
   they're partial carriers already.
3. **Causal ablation confirms it both ways.** Zeroing the 8
   highest-attribution promoter heads drops L32 Pr(v_last) by −0.28 in
   baseline (CI excludes zero) *and* by −0.40 in LoRA (paired difference
   +0.12, CI excludes zero) — the heads carry causal signal before any
   fine-tuning, and LoRA makes them carry more.
4. **LoRA also installs new downstream redundancy.** Ablating the same
   heads leaves a small residue at the final layer in baseline (propagates
   to the readout) but is fully absorbed by L35 in LoRA (paired difference
   −0.05, CI excludes zero) — LoRA adds a compensatory pathway that doesn't
   pre-exist.
5. **Replicates on Gemma-3-4b-it** (different family, opposite baseline
   regime): CVQ-correctness probe Δ +0.11 to +0.19 across 8 layers (all CIs
   exclude zero); logit lens at L33 rises from 0.50–0.80 to ≈1.00. Paired
   causal ablation on Gemma was left to future work in the submitted paper
   — see [§8](#8-extended-analyses-post-submission) for the run that filled
   this in.

One open item on the exact ablation magnitudes: see
`dev-notes/findings/001-ablation-l32-contradiction.md` and
`dev-notes/README.md`.

---

## 7. From-scratch training control (does pretraining matter, and why)

**What it measures:** whether the failure is specific to *this*
pretraining, or would show up in any transformer trained on a similar task
— by removing the pretrained prior entirely (`synthetic_scratch_training/`,
self-contained, own `uv` environment, no imports from anything else in this
repo).

| Exp | Setup | iid_test | heldout_test |
|---|---|---|---|
| 01 | Tiny 2–3 layer transformer, from scratch | 22.6–23.3% | 17.7–18.1% |
| 02 | GPT-2-small architecture, from scratch (random init) | ~15.4% | — |
| 03 | GPT-2-small, **real pretrained weights** | **66.5%** | **52.7%** |
| 04 | GPT-2-small from scratch, fixed data loader (DDP+fp16) | 20.9% | 15.7% |

- **Finding:** pretrained weights transfer massive, fast improvement (exp03:
  66.5% by step 3,000) that architecture-matched from-scratch training never
  reaches, even with a fixed data pipeline and longer training (exp04:
  plateaus ~17–19%). Not memorization — training data is streamed fresh
  every batch, so literal example memorization is combinatorially
  impossible; training *loss itself* plateaus.
- **Per-role breakdown (exp04, final step):** first=28.5%, last=21.5%,
  intermediate=15.8% — even trained from scratch on nothing *but* this
  task, the model solves the *edges* of the problem before the general
  k-th-occurrence case — the same edges-over-interior asymmetry seen
  throughout the pretrained-model experiments.
- **Leading explanation** (see `synthetic_scratch_training/FINDINGS.md` for
  full literature discussion): induction/in-context circuits reliably
  emerge from scratch only under specific distributional properties
  (burstiness, large/rare-class vocabularies, label-count ≫
  exemplar-count — Chan et al. 2022, Reddy 2023/2024); this task's vocab
  (10 possible values, i.i.d., up to 144 exemplars in context) sits close
  to the opposite of all three. A near-identical associative-recall task
  (Zoology/MQAR, Arora et al. 2023) *is* solved from scratch, but at
  40–160× larger vocabularies. Pretrained GPT-2 already has induction heads
  formed on real bursty text; fine-tuning redirects an existing circuit
  rather than bootstrapping one from a distribution that doesn't reward it.
- **Ruled out:** unused position-embedding capacity, hardware/compute
  budget (H100 wouldn't change the plateau, only iteration speed).

---

## 8. Extended analyses (post-submission)

Four experiments run after the submitted draft, to stress-test or sharpen
specific claims. Qwen2.5-3B-Instruct + gemma-3-4b-it (two-family), main
adapters, ARB single-token. Not yet incorporated into the paper text.

### 8a. LoRA extrapolation frontier

**Question:** did the LoRA surface general position-indexed retrieval, or
just push the operating point out? — `lora_intervention/experiments/e1_results/`
(`E1_FINDINGS.md`, `results_multiples.jsonl`).

- Endpoints (FVQ/CVQ) generalize **fully to 5×** training load in both K
  and N: e.g. at N=100 (5× training N), LoRA holds 1.00/1.00 vs. base's
  0.04/0.52.
- Intermediate positions do **not** extrapolate — mean IVQ accuracy decays
  0.56→0.07 as N grows from 1× to 5× training; K barely matters. **N is the
  bottleneck, not K.**
- **Verdict:** the adapter learned robust *endpoint* extraction, not
  general positional indexing — qualifies the paper's §5.2 framing.

### 8b. Block-vs-LoRA readout comparison

**Question:** do the format fix and the LoRA fix converge on the same
internal readout, or just the same output? —
`lora_intervention/experiments/block_readout_results/` (`FINDINGS.md`).

- Per-layer Pr(v_last) at the final layers: base_plain stays suppressed
  (0.35 at L35); base_block and lora_plain both build to **0.98** —
  tracking each other ~8× more closely than either tracks base_plain.
- **"Two roads, one readout":** Block and LoRA reach the fix through
  *different* attention routing (Block barely touches the L30–L33 promoter
  heads LoRA amplifies — pooled attention 0.16 vs. 0.68) but *converge* on
  the same downstream v_last propagation.

### 8c. Behavioral 3-way per-position sweep

**Question:** what does the interior look like, position by position, for
base vs. Block vs. LoRA? —
`lora_intervention/experiments/behavioral_block_lora_results/` (`FINDINGS.md`).

- Endpoints converge across 8 cells and both primacy/reversal regimes
  (base 0.41–0.58 → Block 0.89–1.00, LoRA 0.98–1.00).
- Interior diverges sharply — three distinct per-position shapes:
  **base = U** (only the ends have signal), **Block = flat-high**
  (genuine general position-indexing, holds as N grows), **LoRA =
  boundary-anchored with interior decay that worsens with N**.
- Directly explains 8a: LoRA never learned interior addressing — it
  learned to grab the ends robustly.

### 8d. Stage-3C causal ablation re-run (n=200, paired)

**Question:** resolve a discrepancy in the local n=50 ablation data
(baseline effect appeared larger than LoRA's, the opposite of the paper's
claim). —
`lora_intervention/experiments/ablation_rerun_results/` (`FINDINGS.md`).

- At n=200 with pinned paired seeds, at two cells (K2/N5 — a near-success
  cell, base CVQ 0.77; K2/N50 — a genuine failure, base CVQ 0.40): paired
  (LoRA−base) Δ at L32 is **−0.082** [−0.108,−0.056] and **−0.230**
  [−0.267,−0.196] — both CIs exclude zero, confirming the paper's
  *direction* (ablation hurts LoRA more than base).
- Does **not** reproduce the paper's exact −0.28/−0.40 magnitudes (HF-hook
  ablation is gentler than the TransformerLens ablation used originally;
  the un-ablated baselines match, so it's the ablation method, not the
  measurement). K2/N50 is the cleanest evidence since base has essentially
  no v_last to remove.
- Also ran the **Gemma cross-family paired ablation** left open in the
  submitted paper: promoter heads (cluster at L23) show paired Δ at L33 =
  −0.53 (K2/N30) and −0.74 (K10/N50), both larger than Qwen's — confirms
  amplification generalizes across architecture. See
  `lora_intervention/experiments/gemma_results/GEMMA_CROSS_FAMILY.md`.

---

## Reference: models used across all experiments

| Family | Models | Where |
|---|---|---|
| Open-weight | Qwen2.5 (0.5B, 1.5B, 3B, 3B-Instruct), Qwen3.5 (0.8B, 2B, 4B, 9B), Gemma-3 (270m, 1b, 4b-it) | §1a, §2, §4 (SmolLM only), §6 |
| Proprietary | GPT-4.1, GPT-4.1-mini, Claude-4.5-Haiku, Claude-4.5-Sonnet, Gemini-2.5-Flash, Gemini-2.5-Pro | §1b, §2 |
| Training-dynamics checkpoints | SmolLM3-3B (32+5 stages), SmolLM2-1.7B (42) | §4 |
| Architecture controls (appendix) | TinyLlama-1.1B, StableLM-2-1.6B, Pythia-410M | §1a |
| From-scratch | Tiny 2–3 layer transformer, GPT-2-small (scratch + pretrained) | §7 |

## Reference: where results feed the paper

Every committed table/figure in `paper/figures/` (older ACL draft) and
`aaai_submission/figures/` (submitted version) has a generator script in
`paper/scripts/` with its input paths documented in the script header —
that's the authoritative map from raw result file → table/figure. For a
narrative walkthrough of which raw-data locations are canonical vs. legacy,
see `dev-notes/AAAI_PREP_PLAN.md` §1 and `dev-notes/REPO_MAP.md`.
