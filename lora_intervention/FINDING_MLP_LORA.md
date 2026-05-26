# Finding — MLP-only LoRA also closes the FVQ/CVQ gap (#7)

**Status**: complete. 5-cell preview run on Qwen2.5-3B-Instruct;
full 28-cell ARB grid deferred.

**Data**: `lora_intervention/results/qwen_mlp_control_eval_*.json`
+ `qwen_mlp_control_comparison.txt`. Durable adapter at
`lora_intervention/checkpoints/qwen_mlp_adapter/` (102 MB,
adapter_model.safetensors 90 MB — larger than attention-only's 29 MB
because MLP modules have more parameters).

**Training command:**

```
python lora_intervention/train.py \
  --run-name qwen_mlp_lora \
  --model Qwen/Qwen2.5-3B-Instruct \
  --target-modules "gate_proj,up_proj,down_proj" \
  --max-steps 100 \
  --eval-steps 20 \
  --save-steps 20 \
  --batch-size 2 --grad-accum 32 --max-seq-len 1280
```

---

## The critique we tested

> "LoRA = attention-only {Q,K,V,O} → 'amplification not installation'
> is structurally circular. Attention-only LoRA can ONLY reweight; it
> cannot install MLP-style new features. So 'we observed amplification,
> not installation' is partly true by construction. Fix: add an
> MLP-LoRA comparison run."

The pre-registered outcomes:

| Outcome | Interpretation |
|---|---|
| MLP-LoRA also recovers CVQ | "Amplification not installation" framing collapses — need to reframe as multi-pathway |
| MLP-LoRA fails | Strong positive — confirms attention-circuit story |
| Partial recovery | Mixed; some cells fix, others don't |

**Result**: MLP-LoRA fully recovers CVQ.

---

## Headline numbers (5-cell preview, 100 trials/cell, vLLM, Wilson 95% CIs)

| Cell      | Qwen base RI/PI    | Qwen + MLP-LoRA RI/PI | Qwen + attention-LoRA RI/PI (§5.1) | Verdict |
|-----------|--------------------|-----------------------|-------------------------------------|---------|
| K=10, N=50 | 42% / 50%        | **98% / 96%**         | 100% / 100% | ✓ FIXED |
| K=15, N=20 | 32% / 58%        | **100% / 98%**        | 100% / 98%  | ✓ FIXED |
| K=20, N=30 | 21% / 51%        | **96% / 96%**         | 100% / 98%  | ✓ FIXED |
| K=25, N=75 | 6% / 60%         | **100% / 94%**        | 100% / 93%  | ✓ FIXED |
| K=30, N=75 | 1% / 56%         | **96% / 96%**         | 99% / 92%   | ✓ FIXED |

**5 of 5 cells "fixed"** (both RI ≥ 65% AND PI ≥ 65%). All cells RI ≥ 96%, all PI ≥ 94%.

For comparison: attention-only LoRA fixed 28/28 in §5.1 with RI ≥ 99%
and PI ≥ 92% across the full grid. MLP-LoRA on this 5-cell preview is
**indistinguishable** from attention-only LoRA within trial-level
noise. The slight numerical differences are well within the Wilson
95% CIs of n=100 binomial samples.

---

## Training stats

| Step | Train loss | Train acc | Eval loss | Eval acc |
|---|---|---|---|---|
| 10  | 0.924 | 85.8% | – | – |
| 20  | 0.558 | 89.7% | 0.414 | 89.8% |
| 30  | 0.305 | 91.3% | – | – |
| 40  | 0.177 | 94.8% | 0.152 | 95.5% |
| 50  | 0.131 | 95.8% | – | – |
| 60  | 0.140 | 95.8% | 0.115 | 96.1% |
| 70  | 0.113 | 96.4% | – | – |
| 80  | 0.112 | 96.5% | 0.094 | 96.8% |
| 90  | 0.107 | 96.7% | – | – |
| 100 | 0.076 | 97.5% | 0.091 | **97.1%** |

Fit much faster than attention-only LoRA on the same data (the §5.1
LoRA took 2 epochs to reach similar territory; this MLP-LoRA hit
similar eval loss in ~40 steps and improved through step 100).

**Trainable parameters**:
- Attention-only LoRA (§5.1): 14.94 M (0.48% of model)
- **MLP-only LoRA (this run): 22.56 M (0.73% of model)**

MLP has ~1.5× more trainable parameters than attention-only at the
same rank because the intermediate_size > hidden_size in Qwen2.5-3B.
This is one possible explanation for the faster fit, but the
end-of-training accuracy is comparable to attention-only — both
saturate the task.

---

## What this means for the §7 mechanism story

The §7 narrative in the paper currently leans on:
1. Attention routing identifying 15 L30-L33 promoter heads
2. Stage 3 ablation showing those heads carry the gap-closure signal
3. Probing + logit lens converging on attention activations as the locus

The MLP-LoRA result doesn't *invalidate* that mechanism for the
attention-only setting — those 15 heads are real, identified
correctly, and ablating them does what §5.1 says. **But the
"amplification not installation" framing is too strong**: it implies
that the gap-closure mechanism *must* go through attention reweighting
(because LoRA "only reweights"). MLP-LoRA shows this is structural to
the attention-only target choice, not a property of the underlying
recovery mechanism.

### Two compatible reframings the paper can adopt

**Option A — Multi-pathway (recommended):**

> "We show that attention-only LoRA on {Q,K,V,O} closes the gap by
> reweighting a small cluster of late-layer attention heads
> (§5.1, §7). A second LoRA run targeting only MLP modules
> ({gate_proj, up_proj, down_proj}, attention frozen) also closes
> the gap with comparable accuracy at 100 training steps
> (Table~\ref{tab:lora_mlp}). The capability is therefore present in
> the pretrained architecture and expressible through either
> pathway — attention reweighting or MLP transformation. Our
> mechanistic analysis identifies the attention pathway because that
> is what the main LoRA modulates; the MLP-only run is a
> sanity check that the capacity is not exclusively attention-bound."

**Option B — Sufficient-not-necessary:**

> "The L30-L33 attention promoter cluster identified by §7 is a
> *sufficient* circuit for the task: amplifying these heads via LoRA
> closes the gap. It is not *necessary* — MLP-only LoRA also closes
> the gap (App~\ref{app:mlp_lora}) without modifying any attention
> weights. The attention story explains how the attention-trained
> LoRA recovers the task; the broader claim 'this is the task's
> mechanism in the base model' would require comparing the
> attention activations under MLP-only LoRA, which we leave to
> future work."

Either framing is defensible. Option A is honest about multiple
pathways without abandoning the §7 mechanism finding. Option B
narrows the §7 claim to what we measured.

---

## Limitations of this 5-cell preview

1. **Only 5 cells of the 28-cell ARB grid.** The five chosen are
   representative of the hardest part of the grid (high-N reversal
   regime), where attention-only LoRA had to do the most work to
   recover. If MLP-LoRA also fixes these, it's plausibly fixing
   the easier cells too — but not proven. **Full 28-cell run is
   a natural follow-up**.

2. **SEM OOD not tested.** Attention-only LoRA generalized to
   SEM OOD in §5.2; we haven't tested whether MLP-LoRA does the
   same. If MLP-LoRA fails on SEM OOD (e.g., overfits to ARB
   single-token format), that would be an asymmetry worth
   reporting.

3. **100 training steps vs 2 epochs in §5.1.** The MLP-LoRA was
   trained for fewer total optimization steps but reached similar
   fit. The eval loss curves suggest 100 steps was sufficient at
   this scale.

4. **Bootstrap CIs not computed yet for the 5 cells.** Trial
   counts (n=100) give Wilson half-widths ≈ ±5-7 points, and the
   point estimates (94-100%) are all well above the 65% threshold,
   so the verdict is robust to CI variation.

---

## Files

- `lora_intervention/checkpoints/qwen_mlp_adapter/` — durable
  adapter (102 MB, mirrors the convention of `qwen_arith_adapter/`
  and `gemma_adapter/`)
- `lora_intervention/results/qwen_mlp_control_eval_*.json`
- `lora_intervention/results/qwen_mlp_control_comparison.txt`
- `lora_intervention/logs/qwen_mlp_train.log` — full training log

## Reproduce

```
# Train:
python lora_intervention/train.py --run-name qwen_mlp_lora \
  --model Qwen/Qwen2.5-3B-Instruct \
  --target-modules "gate_proj,up_proj,down_proj" \
  --max-steps 100 --eval-steps 20 --save-steps 20 \
  --batch-size 2 --grad-accum 32 --max-seq-len 1280

# Eval (5-cell preview):
python lora_intervention/evaluate.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --adapter lora_intervention/checkpoints/qwen_mlp_adapter \
  --baseline-path v3/results_vllm/arbitrary_single/Qwen2.5-3B-Instruct/stage1_sweep_20260409_000134.json \
  --run-name qwen_mlp_control \
  --cells "10_50,15_20,20_30,25_75,30_75" \
  --skip-ood --trials 100
```
