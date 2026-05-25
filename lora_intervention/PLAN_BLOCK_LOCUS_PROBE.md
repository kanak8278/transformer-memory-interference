# Plan — Block-format ↔ L30-L33 promoter heads probe (§5 "common locus" claim)

**Status**: planned, not started. Implementation depends on the
attention-routing pipeline outlined in
`PLAN_ATTENTION_ROUTING_GEMMA.md` (the HF-direct attention-from-gen
extractor); this plan reuses that code with a different prompt
builder.

---

## 1. The claim under test

`paper/main.tex` line 536-543 states:

> "The format and LoRA interventions converge: both close the FVQ-CVQ
> gap and restore intermediate-position retrieval. The first does so
> at inference time without gradients; the second with a fraction of a
> percent of the model's parameters. **The capability is therefore
> present in the pretrained architecture but not dispatched by
> default.**"

The Devil's Advocate critique (verbatim from your message):
> "Two interventions that solve different deployment problems with
> no shared mechanism is not 'convergence.'"

The critique is precise: the paper shows two interventions produce the
same *behavioral* outcome (gap closes) but never tests whether they
share an *internal* mechanism. The "common locus" / "dispatching"
framing implies a shared mechanism.

## 2. Testable hypothesis

`H1` (paper's implicit claim): **Block-format input on the base
(no-LoRA) model activates the same L30-L33 promoter heads that
post-LoRA activates on plain-format input.**

`H0` (devil's-advocate position): **Block format and LoRA produce
the same behavior via different internal routes; their attention
patterns differ.**

We can decide between these with a focused 50-trial attention-routing
run.

## 3. Experimental design

Three conditions on Qwen2.5-3B-Instruct, K=2 / N=30, 50 trials each
(seeds 1000-1049 paired with the existing Qwen attention-routing
files for maximal comparability):

| Condition | Model | Prompt format |
|---|---|---|
| **A: Base + Plain** | Qwen2.5-3B-Instruct | normal interleaved (existing file `Qwen2.5-3B-Instruct__normal.json`) |
| **B: Base + Block** | Qwen2.5-3B-Instruct | block-grouped + round headers (**NEW RUN**) |
| **C: LoRA + Plain** | Qwen2.5-3B-Instruct + main LoRA | normal interleaved (existing file `Qwen2.5-3B-Instruct-LoRA__normal.json`) |

We compare conditions on the **15 LoRA-promoter heads** identified
in §5.1 (top heads by Δ between C and A, all in L30-L33):
L32H3, L31H12, L32H7, L31H15, L30H11, L31H8, L32H10, L32H0, L33H8,
L32H14, L32H9, L32H12, L33H11, L31H1, L33H14.

For each of these 15 heads, the focal metric is
`P(attend to v_last round under CVQ)` at the generation position.
Across the 50 trials per condition:

- `p_A = mean P(...)` under base + plain
- `p_B = mean P(...)` under base + block
- `p_C = mean P(...)` under LoRA + plain

If **`p_B ≈ p_C`** on these 15 heads → format intervention activates
the same circuit LoRA activates → common locus supported. Paper's
convergence framing is mechanistically justified.

If **`p_B ≈ p_A`** (block format leaves these heads at baseline) →
format must be working through a different mechanism → common-locus
claim is unsupported and we soften §5 to "two operationally distinct
fixes."

If **`p_B` is intermediate** (partial activation) → write a more
nuanced claim: "format intervention partially recruits the same
heads but does not fully match LoRA's reweighting."

## 4. Why this isn't a 10-line probe

The original critique called it "a 10-line probe." It isn't, because:

1. **Prompt structure differs.** Block format groups the stream by
   round with explicit `[Round j]` headers. The token-position →
   (round, key) attribution is different from plain format; the
   attention-extraction code needs to know where each round's tokens
   start and end. Same problem as
   `PLAN_ATTENTION_ROUTING_GEMMA.md`, just with a different prompt
   builder.

2. **Behavioral validation is needed.** We need to confirm Block
   format actually closes the gap on Qwen2.5-3B-Instruct at the
   chosen operating point (K=2, N=30). The existing remedy data is
   at a different cell grid (`remedy/Qwen2.5-3B-Instruct/...` uses
   K up to 30, N up to 100, but the styles are
   `control/numbered/landmark/recency_cue/combined` — these are
   different format definitions than the paper's
   `Plain/Labeled/Block/Landmark`). The cloud `block` builder in
   `experiments_cloud/remedy_sweep.py` is the one that matches the
   paper's Block. **We need to confirm Block-on-Qwen-closes-gap
   before testing the mechanism.**

3. **Statistical comparison.** Need bootstrap 95% CIs on the
   three-way comparison (p_A, p_B, p_C) per head, then a clear
   joint criterion for "p_B matches p_C" vs "p_B matches p_A."

So the realistic scope is more like a 200-line probe across two
scripts + analysis.

## 5. Implementation

### 5.1 Block-format prompt builder

Lift `build_prompt_block` from `experiments_cloud/remedy_sweep.py`
(already exists, no need to reinvent):

```python
def build_prompt_block(block_items, test_category, condition):
    """[Round j] header + grouped per-round entries (shuffled within round)."""
    lines = []
    for block in block_items:
        lines.append(f"[Round {block[0]['update_idx']}]")
        for item in block:
            lines.append(f"  {item['category']}: {item['value']}")
    stream = "\n".join(lines)
    return (
        f"Read the following key-value stream. Each key is updated multiple times, "
        f"grouped by update round.\n\n"
        f"{stream}\n\n"
        f"{_query(test_category, condition)}"
    )
```

### 5.2 Behavioral pre-check (cheap, runs first)

Before the mechanism probe, confirm Block format actually fixes
Qwen2.5-3B-Instruct at K=2, N=30 (the attention-routing operating
point):

```
python lora_intervention/eval_block_behavior.py \\
    --model Qwen/Qwen2.5-3B-Instruct \\
    --K 2 --N 30 --trials 100 \\
    --formats plain,block
```

Decision rule: only proceed to the mechanism probe if Block lifts
PI accuracy by at least +30 points over Plain on Qwen2.5-3B-Instruct
at K=2, N=30. If Block doesn't close the gap on this cell, the probe
question is moot.

### 5.3 Attention routing under Block prompts

Run `run_attention_routing_hf.py` (the one to be written per
`PLAN_ATTENTION_ROUTING_GEMMA.md`) with two additions:

- `--prompt-format {plain,block}` flag, defaults to `plain`
- When `block`: use the round-grouped prompt; the round → token map
  is *easier* (each round is a contiguous block of tokens between
  two `[Round j]` headers)

CLI for the new condition (B):

```
python lora_intervention/run_attention_routing_hf.py \\
    --merged_path Qwen/Qwen2.5-3B-Instruct \\
    --base Qwen/Qwen2.5-3B-Instruct \\
    --out_name Qwen2.5-3B-Instruct-baseline-block \\
    --K 2 --N 30 --trials 50 \\
    --prompt-format block \\
    --device cuda --dtype bfloat16
```

Existing files A (`Qwen2.5-3B-Instruct__normal.json`) and C
(`Qwen2.5-3B-Instruct-LoRA__normal.json`) are reused as-is.

### 5.4 Three-way comparison

Add to `compute_cis.py` (or a sibling `compute_cis_block_locus.py`):

```python
def block_locus_comparison():
    A = load("Qwen2.5-3B-Instruct__normal.json")           # base + plain
    B = load("Qwen2.5-3B-Instruct-baseline-block__block.json")  # base + block
    C = load("Qwen2.5-3B-Instruct-LoRA__normal.json")      # LoRA + plain

    PROMOTER_HEADS = [
        (32, 3), (31, 12), (32, 7), (31, 15), (30, 11),
        (31, 8), (32, 10), (32, 0), (33, 8), (32, 14),
        (32, 9), (32, 12), (33, 11), (31, 1), (33, 14),
    ]
    for L, H in PROMOTER_HEADS:
        a = per_trial_p_last(A, "expA_CVQ")[:, L, H]
        b = per_trial_p_last(B, "expA_CVQ")[:, L, H]
        c = per_trial_p_last(C, "expA_CVQ")[:, L, H]
        # report point + 95% CI for each, plus deltas:
        # (B - A): does block move this head?
        # (C - A): how much does LoRA move it (sanity)
        # (B - C): does block reach LoRA's activation level?
        # We want B - C close to zero with tight CI for "common locus."
```

Output: `lora_intervention/results/block_locus_comparison.txt`
with the three-column table above and a one-line verdict
("supports common locus" / "supports distinct mechanisms" /
"partial activation").

## 6. Decision criteria (pre-registered)

After computing the table:

- **Common locus supported (paper claim unchanged)**: for ≥ 11 of 15
  promoter heads, the 95% CI of `(p_B - p_C)` contains zero AND
  `p_B > 0.4` (head is meaningfully active).
- **Common locus rejected (soften paper claim)**: for ≥ 11 of 15
  heads, the 95% CI of `(p_B - p_A)` contains zero (block format
  leaves baseline unchanged on these heads).
- **Partial activation (intermediate finding)**: anywhere in between.
  Write a softer claim: "Format activates the LoRA-target heads to
  a substantial but not complete extent, suggesting partial
  overlap of mechanism."

## 7. Effort estimate

| Step | Time |
|---|---|
| Confirm behavioral effect of Block on Qwen at K=2/N=30 | 1 h (script + 100 prompts on vLLM) |
| Extend `run_attention_routing_hf.py` with `--prompt-format` | 1 h |
| Run condition B (base + block) — 50 trials | ~5 min on L4 |
| Three-way comparison script + bootstrap CIs | 1 h |
| Write up + commit | 30 min |
| **Total** | **~3.5 h** + the prereq attention-routing pipeline (~3-4 h, see other plan) |

## 8. Risks & assumptions

1. **Assumes the existing Qwen attention-routing files are usable
   as-is.** They are (already in repo at
   `v3/results_vllm/attention_routing/Qwen2.5-3B-Instruct{,-LoRA}__normal.json`),
   so we don't need to regenerate A or C.

2. **Assumes Block format closes the gap at K=2/N=30 on
   Qwen2.5-3B-Instruct.** The paper claims Block closes the gap
   across the open-weight set at some cell; we need to verify it does
   so at *this specific cell*. If it doesn't, we either (a) pick a
   different cell where it does and re-do attention routing on
   LoRA+plain at that cell for a clean comparison, or (b) report this
   as a finding ("Block doesn't close the gap at K=2/N=30 — so the
   comparison question doesn't apply").

3. **Token-position-to-round mapping must be reliable** under Block
   format. Easier than plain format (rounds are explicit blocks), but
   needs unit-test in the smoke run.

4. **Worst case for the paper**: H0 holds (Block doesn't activate
   L30-L33 promoters). The paper claim has to soften but the
   *finding* is publishable — "format-based and weight-based
   interventions converge on behavior but diverge on
   mechanism" is a real, interesting, defensible result. It just
   isn't what the abstract currently implies.

## 9. Sequencing relative to other in-flight work

This probe is logically downstream of:

1. `PLAN_ATTENTION_ROUTING_GEMMA.md`'s `run_attention_routing_hf.py`
   (the HF-direct attention-from-gen extractor) — that script is the
   prerequisite tool. Once written for the Gemma replication, the
   Block probe is a 1-2 hour extension.

2. The current MLP-LoRA work (testing the attention-vs-MLP critique)
   — independent; can run in parallel once that compute finishes.

Recommended order:
1. Finish MLP-LoRA training + eval (in progress)
2. Write `run_attention_routing_hf.py` (covers both Gemma replication
   and Block probe)
3. Run Block-format behavioral check on Qwen at K=2/N=30
4. Run condition B (base Qwen + block prompts) — attention routing
5. Three-way comparison + verdict
6. Update paper §5 wording based on verdict
