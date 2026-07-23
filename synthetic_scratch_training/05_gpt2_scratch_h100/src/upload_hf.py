"""Package the grokked from-scratch checkpoint into HF format + model card and upload.

Usage: python upload_hf.py <ckpt.pt> <repo_id> [--public]
Requires HF_TOKEN in the environment.
"""

import argparse
import json
import os
import tempfile

import torch
from huggingface_hub import HfApi

from model_gpt2 import GPT2FromScratch

CARD = """---
license: mit
library_name: transformers
pipeline_tag: text-generation
tags:
- gpt2
- from-scratch
- grokking
- mechanistic-interpretability
- synthetic-task
- key-value-recall
---

# GPT-2-small (from scratch) — synthetic key-value recall under interference

A GPT-2-small **architecture trained entirely from scratch** (random init, no
pretrained weights) on a synthetic key-value recall task built to study memory
interference. It reaches **80.0%** accuracy on the training-step distribution and
**63.2%** on held-out (never-trained) step positions — reached via a **grokking**
transition late in training, during the learning-rate decay phase.

## The task
The context interleaves updates to `K` keys, each updated `N` times with random
values drawn from 10 possibilities. A query `(key, step k)` asks for that key's
value at its *k*-th update. Closed 51-token vocabulary (26 keys, 10 values, 12
step tokens, BOS/QUERY/EOS) — **no natural-language tokenizer**; `input_ids` are
the raw synthetic ids 0-50. Sequence format:

```
<BOS> <Key><Value> ... <QUERY> <Key> <Step>  ->  <Value> <EOS>
```
The model is supervised only on the answer VALUE token (position -2).

## Key finding
Accuracy is near-perfect across most of the `(K, N)` grid, is U-shaped over step
position (first/last easiest, middle hardest), and **collapses to near-chance once
both the number of keys and the history length are large** (K>=10 and N>=10) — a
capacity/interference threshold rather than gradual degradation. The ability
generalizes to step positions never seen in training (the 63.2% held-out number).

## Training (summary)
GPT-2-small (12 layers, 768 d; ~86M params with the 51-token vocab), bf16, TF32,
`torch.compile`, AdamW (betas 0.9/0.95, decoupled weight decay 0.1 on 2-D params),
grad-clip 1.0, dropout 0, cosine LR 6e-4 -> 6e-5, ~30k steps on one H100. Includes
the GPT-2 residual-scaled init (`0.02/sqrt(2*n_layer)` on `c_proj`). See
`training_metadata.json` for the exact checkpoint step / val accuracy / config.

## Loading
```python
from transformers import GPT2LMHeadModel
import torch
model = GPT2LMHeadModel.from_pretrained("{repo_id}")
# input_ids are raw synthetic ids 0-50 (no tokenizer); read logits[:, -2] for the answer
```

## Code, full results & figures
Training code, evaluation, and the analysis figures live in the project repo:
**https://github.com/kanak8278/transformer-memory-interference**, under
`synthetic_scratch_training/05_gpt2_scratch_h100/` (this run's tag: `h100_cosine`).
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("repo_id")
    ap.add_argument("--public", action="store_true")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = GPT2FromScratch(dropout=0.0)
    model.load_state_dict(ck["model"])

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=not args.public, exist_ok=True)

    with tempfile.TemporaryDirectory() as d:
        model.gpt2.save_pretrained(d)  # config.json + model.safetensors (tied weights handled)
        with open(os.path.join(d, "README.md"), "w") as f:
            f.write(CARD.replace("{repo_id}", args.repo_id))
        meta = {"checkpoint_step": ck.get("step"), "val_acc": ck.get("val_acc"),
                "final_iid_test": 0.8005, "final_heldout_test": 0.6325,
                "train_args": ck.get("args")}
        with open(os.path.join(d, "training_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        api.upload_folder(folder_path=d, repo_id=args.repo_id, repo_type="model",
                          commit_message="Add grokked from-scratch GPT-2 KV-recall model (80.0%/63.2%)")
    print(f"uploaded to https://huggingface.co/{args.repo_id} (private={not args.public})")


if __name__ == "__main__":
    main()
