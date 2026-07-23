# Trained weights — where they live

Checkpoints are **not** in this git repo (each is ~329 MB, over GitHub's 100 MB
per-file limit; `*.pt` under `checkpoints/` is gitignored).

## Hosted (the grokked model — recommended)
The best `h100_cosine` checkpoint (from-scratch, **iid 80.0% / held-out 63.2%**) is on
the Hugging Face Hub, in `save_pretrained` format (directly loadable), with a model
card and `training_metadata.json`:

**https://huggingface.co/kanak8278/gpt2-small-synthetic-kv-interference**  *(currently private)*

```python
from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("kanak8278/gpt2-small-synthetic-kv-interference")
# input_ids = raw synthetic ids 0-50 (no tokenizer); the answer is argmax of logits[:, -2]
```
Uploaded via `src/upload_hf.py` (needs `HF_API_KEY` in the repo-root `.env`).

## Local (on the training box)
Under `05_gpt2_scratch_h100/checkpoints/` (best + last per arm, model weights only):
- `h100_cosine/best.pt`   — the grokked model (step 28000, val 0.797) ← the good one
- `h100_cosine/last.pt`, `h100_plateau/{best,last}.pt`

Each is a dict: `{"model": state_dict, "step", "val_acc", "args"}`.

## Regenerate from scratch
Deterministic (seed=42): `python src/train.py --gpu 0 --schedule cosine --tag h100_cosine
--total-steps 30000 --warmup 600 --batch-size 256 --accum 8` (~5 h on one H100).
