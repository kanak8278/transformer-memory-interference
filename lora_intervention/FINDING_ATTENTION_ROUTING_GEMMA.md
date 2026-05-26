# Finding — Attention routing replicates on Gemma (#3)

**Status**: complete. Replicates the Qwen §5.1 attention-routing
analysis on Gemma-3-4b-it base + Gemma+LoRA, using a new HF-direct
extractor (`run_attention_routing_hf.py`) that avoids
transformer_lens's OOM on Gemma at 23 GB.

**Outputs**:
- `v3/results_vllm/attention_routing/gemma-3-4b-it-baseline-hf__normal.json` — 50 trials, base Gemma
- `v3/results_vllm/attention_routing/gemma-3-4b-it-LoRA__normal.json` — 50 trials, Gemma+LoRA
- `lora_intervention/results/attention_routing_comparison_gemma.txt` — comparison table
- `lora_intervention/results/attention_routing_comparison_gemma.json` — structured data

---

## The critique we tested

> "Mechanism = N=1 model + N=1 family with only 2 of 4 methods.
> Abstract promises 'mechanism analyses localise to the same
> attention-head clusters' but Gemma replication only shows probing
> + logit lens. The 'amplification not installation' claim rests on
> a single-model story. Fix: either (a) run attention routing +
> ablation on Gemma, or (b) demote the cross-family claim."

We do (a) for attention routing. Stage 3 ablation (the other missing
method) is deferred to future work — see GEMMA_RUN_LOG.md §6.

---

## Experimental design

Same operating point and protocol as the Qwen §5.1 baseline:

| | |
|---|---|
| Operating point | K=2, N=30 |
| Mode | normal (plain interleaved stream with explicit rounds) |
| Trials | 50 per condition (FVQ + CVQ) |
| Trial seeds | 1000-1049 |
| Metric | P(attend to v_last round) at gen position, per (layer, head), under CVQ |

---

## Top promoter heads — Qwen (§5.1) vs Gemma (this work)

### Qwen2.5-3B-Instruct top 15 heads (post-LoRA P − base P)

All 15 heads in **L30-L33** (last 11% of network depth):
L32H3, L31H12, L32H7, L31H15, L30H11, L31H8, L32H10, L32H0,
L33H8, L32H14, L32H9, L32H12, L33H11, L31H1, L33H14. Δ range
+0.49 to +0.68. **Tight cluster at the network output.**

### Gemma-3-4b-it top 15 heads (post-LoRA P − base P)

| Head | base | LoRA | Δ |
|---|---|---|---|
| L25H0 | 0.504 | 0.974 | +0.469 |
| L14H4 | 0.098 | 0.545 | +0.448 |
| L23H6 | 0.257 | 0.659 | +0.402 |
| L19H5 | 0.145 | 0.538 | +0.393 |
| L26H5 | 0.297 | 0.688 | +0.391 |
| L25H1 | 0.122 | 0.504 | +0.382 |
| L32H1 | 0.299 | 0.661 | +0.362 |
| L27H6 | 0.381 | 0.737 | +0.356 |
| L20H1 | 0.250 | 0.606 | +0.356 |
| L25H4 | 0.195 | 0.550 | +0.356 |
| L24H6 | 0.220 | 0.572 | +0.352 |
| L18H0 | 0.272 | 0.609 | +0.337 |
| L14H0 | 0.297 | 0.627 | +0.330 |

Layers represented: **L14, L18, L19, L20, L23, L24, L25, L26, L27, L32**.

**Range: L14 to L32 across 19 layers** — distributed across mid-to-late
depth, not concentrated at the very end like Qwen.

### Per-layer mean change

| Layer | base | LoRA | Δ | flag |
|---|---|---|---|---|
| L14 | 0.150 | 0.357 | +0.207 | wave 1 |
| L18 | 0.254 | 0.449 | +0.195 |  |
| L19 | 0.150 | 0.351 | +0.201 |  |
| L20 | 0.268 | 0.529 | +0.261 |  |
| L21 | 0.335 | 0.555 | +0.220 |  |
| L23 | 0.220 | 0.428 | +0.207 |  |
| L24 | 0.165 | 0.344 | +0.178 |  |
| L25 | 0.235 | 0.540 | **+0.305** | **peak** |
| L26 | 0.221 | 0.415 | +0.194 |  |
| L30 | 0.221 | 0.456 | +0.235 | wave 2 |

Per-layer Δ peaks at L25 (+0.305), with secondary peaks at L20
(+0.261) and L30 (+0.235) — three waves of activation.

---

## Architectural contrast

| | Qwen2.5-3B-Instruct | Gemma-3-4b-it |
|---|---|---|
| Total layers | 36 | 34 |
| Attention heads (Q) | 16 (dense) | 8 (grouped-query) |
| Baseline suppressor cluster (§5.1 / Gemma Stage 3) | L26-L30 (late) | L17-L19 + L30H6 (mid + late) |
| Post-LoRA promoter cluster topology | **Concentrated** at L30-L33 (last 4 layers) | **Distributed** L14-L32, peaks at L20/L25/L30 |
| Max single-head Δ | +0.68 (L32H3) | +0.47 (L25H0) |
| Range of layers with top-15 heads | 4 layers | 19 layers |

**The two architectures' baseline suppressor topology predicts their
recovery topology.** Qwen's suppressors are concentrated at the end
of the network; LoRA's promoters are right behind them. Gemma's
suppressors are mid-depth (L17-L19) + one late (L30H6); LoRA's
promoters spread across mid-depth (L20-L26) and have a secondary
late peak at L30. Both architectures use the same *kind* of
mechanism (late-layer attention promotion) at different *positions*.

---

## What this changes for the paper

The abstract's claim "mechanism analyses localise to the same
attention-head clusters" is **partially supported**:
- ✓ Both architectures recruit late-layer attention heads to attend
  to v_last
- ✓ The functional role (promotion of v_last) is the same
- ✗ The *layer indices* are different (concentrated vs distributed)
- ✓ The *relative depth* (mid-to-late) is preserved

A more accurate version of the claim:

> "Both architectures recover the FVQ-CVQ task via late-layer
> attention reweighting of the generation-position query toward the
> v_last round. On Qwen2.5-3B-Instruct the promoter cluster
> concentrates in the last four layers (L30-L33, Δ +0.49 to +0.68
> across 15 heads); on Gemma-3-4b-it the same functional pattern
> spans mid-to-late depth (L14-L32, Δ +0.33 to +0.47) with
> per-layer peaks at L20, L25, and L30. The two clusters
> structurally mirror each architecture's baseline suppressor
> topology — Qwen's late-only, Gemma's mid + late."

This is honest about the differences AND substantiates the
architecture-agnostic claim at a more nuanced level.

---

## Limitations

1. **No paired stage-3 ablation on Gemma.** We have not tested
   whether ablating the Gemma promoter heads disrupts CVQ accuracy
   the way ablating Qwen's L26-L30 cluster does. Future work — needs
   HF-direct attention head zeroing.

2. **Behavioral context not validated at this exact cell.** Gemma+LoRA
   was shown to close the gap on §5.2's 28-cell grid; this attention
   routing was done at a single cell (K=2/N=30) chosen to match Qwen's
   protocol. The behavioral context is "Gemma+LoRA reaches 90%+
   accuracy on this kind of cell" rather than "98% accuracy at this
   specific cell"; the mechanism finding should generalise but we
   haven't tested it.

3. **N=2 architectures still N=2.** This is two-family attention-routing
   evidence (Qwen + Gemma); the abstract's "mechanism analyses
   localise" implies a stronger universality. Adding MLP-LoRA's
   different mechanism (see FINDING_MLP_LORA.md) and the Block
   format's *different* internal route (see FINDING_BLOCK_LOCUS.md)
   complicates the picture: there are *multiple* internal pathways
   that can express FVQ-CVQ. Attention reweighting is one; MLP
   transformation is another; Block-format prompting recruits yet a
   third (currently uncharacterized). The honest framing for the
   paper is "the capacity is present and expressible through
   multiple pathways, of which attention reweighting is the one we
   most fully characterise."

---

## Reproduce

```
# Gemma base:
python lora_intervention/run_attention_routing_hf.py \
    --merged_path google/gemma-3-4b-it \
    --base google/gemma-3-4b-it \
    --out_name gemma-3-4b-it-baseline-hf \
    --K 2 --N 30 --trials 50

# Gemma + LoRA (requires gemma_merged/ — regen via merge_lora.py):
python lora_intervention/run_attention_routing_hf.py \
    --merged_path lora_intervention/checkpoints/gemma_merged \
    --base google/gemma-3-4b-it \
    --out_name gemma-3-4b-it-LoRA \
    --K 2 --N 30 --trials 50

# Comparison:
python lora_intervention/compute_attention_routing_comparison.py \
    --base v3/results_vllm/attention_routing/gemma-3-4b-it-baseline-hf__normal.json \
    --lora v3/results_vllm/attention_routing/gemma-3-4b-it-LoRA__normal.json \
    --condition expA_CVQ --top-k 15 \
    --out lora_intervention/results/attention_routing_comparison_gemma.txt \
    --out-json lora_intervention/results/attention_routing_comparison_gemma.json
```

Each Gemma run is ~10s wall time on L4 + ~1 min model load.
