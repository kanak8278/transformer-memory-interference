# Experiment 07 — First/Last-only training (PRETRAINED GPT-2)

Status: built + validated (pretrained load + forward on CPU), not yet launched.

## Why this experiment

Completes a clean 2x2: {from-scratch, pretrained} x {all-steps, first/last-only}.

|              | from-scratch    | pretrained                |
|--------------|-----------------|---------------------------|
| all steps    | exp05 (running) | exp03 (done: 66.5%/52.7%) |
| first/last   | exp06           | **exp07 (this)**          |

exp03 (pretrained, all steps) per-role: iid first=100%, last=65%, intermediate=58%;
**held-out intermediate=52.7%** — pretrained GPT-2 already generalizes to unseen
interior steps far above the ~16% from-scratch (exp04) manages. BUT exp03 trained on
*some* interior cells. exp07 removes interior from training **entirely**, giving the
**purest zero-shot test**: does the pretrained k-th-counting circuit survive with zero
interior supervision, purely from edge-training + pretrained priors?

The money comparison is **exp06 vs exp07**: train on edges only, then probe unseen
interior. If pretrained >> scratch, that is direct evidence pretraining supplies the
general retrieval/counting circuit rather than it being learned from the task data.

## Design

Same first/last training + eval-split as exp06 (`--first-last-only`), but the model is
**off-the-shelf pretrained GPT-2** (`--pretrained`): `GPT2LMHeadModel.from_pretrained
("gpt2")`, our token ids 0-50 fed straight into the real 50257-row embedding (exactly
exp03's approach; reuses that idea via `model_gpt2.GPT2Pretrained`). Differences from
the from-scratch arms:

- **NO residual scaled re-init** — would destroy the pretrained weights.
- **Fine-tune LR** — exp03 used peak_lr=3e-5 (10x below from-scratch 6e-4) to avoid
  wrecking pretrained representations. exp07 must launch with `--peak-lr 3e-5` (and a
  short warmup, e.g. --warmup 150).
- dropout stays at GPT-2's shipped config value.
- Real 50257-wide logits — `logits_to_keep=2` keeps eval memory sane.

## Launch (when a GPU frees; needs SSL_CERT_FILE for the weight download — in env.sh)

    source ../env.sh && cd src
    python train.py --gpu <g> --schedule cosine --tag fl_pretrained --pretrained \
      --first-last-only --peak-lr 3e-5 --warmup 150 --total-steps 15000 \
      --batch-size 128 --accum 4   # eff batch 512 (faster-updates policy; micro 128 keeps GPU full)

(batch 128: real GPT-2 attention over ~290-token seqs is heavier than the tiny-vocab
from-scratch model — exp03 noted batch 512 blew past memory; re-profile on H100.)

## Changelog
- 2026-07-22: built from exp06 + a GPT2Pretrained wrapper; pretrained load and a
  forward validated on CPU (50257 vocab). Downloading gpt2 weights required pointing
  Python at the system CA bundle (see env.sh / [[h100-env-cache-setup]]).
