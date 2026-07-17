"""Decode a sample of val predictions vs ground truth into human-readable text, so we
can visually confirm the model is producing valid single VALUE-token answers (not
garbage / out-of-family tokens), not just look at an accuracy number.

Adapted from experiment 01's dump_val_samples.py. One real difference: GPT-2's output
head is the full 50257-token vocab (never resized), so unlike exp01/02's models —
which are structurally incapable of predicting outside our 51-token vocab — this model
genuinely can predict a real GPT-2 BPE token id >= 51. vocab.token_repr() has no entry
for those, so we wrap it with a fallback here rather than let it KeyError.
"""

import argparse
import os
import random
import sys

import torch

SRC_DIR = os.path.dirname(__file__)
EXP01_SRC = os.path.join(SRC_DIR, "..", "..", "01_baseline_step_ood", "src")
sys.path.insert(0, os.path.abspath(EXP01_SRC))
sys.path.insert(0, SRC_DIR)

import vocab
from model_gpt2_pretrained import GPT2Pretrained

EXP01_DATA_DIR = os.path.join(EXP01_SRC, "..", "data")


def safe_token_repr(token_id):
    if 0 <= token_id < vocab.VOCAB_SIZE:
        return vocab.token_repr(token_id)
    return f"<OOV:{token_id}>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--split", type=str, default="iid_val", choices=["iid_val", "iid_test", "heldout_test"])
    ap.add_argument("--n_samples", type=int, default=40)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = GPT2Pretrained().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    packed = torch.load(os.path.join(EXP01_DATA_DIR, f"{args.split}.pt"), weights_only=False)

    flat = []
    for (k, n), group in packed.items():
        for i in range(group["input_ids"].shape[0]):
            flat.append((group["input_ids"][i], group["meta"][i]))
    rng = random.Random(0)
    sample = rng.sample(flat, min(args.n_samples, len(flat)))

    lines = []
    n_correct = 0
    n_oov_pred = 0
    with torch.no_grad():
        for full_ids, meta in sample:
            full_ids = full_ids.to(device)
            inp = full_ids[:-1].unsqueeze(0)
            logits, _ = model(inp)
            pred_value_id = logits[0, -2, :].argmax().item()
            true_value_id = full_ids[-2].item()
            pred_eos_id = logits[0, -1, :].argmax().item()
            correct = pred_value_id == true_value_id
            n_correct += int(correct)

            pred_is_valid_value = vocab.VALUE_BASE <= pred_value_id < vocab.STEP_BASE
            pred_is_valid_eos = pred_eos_id == vocab.EOS
            if pred_value_id >= vocab.VOCAB_SIZE:
                n_oov_pred += 1

            context_str = vocab.decode(full_ids[:-2].tolist())
            lines.append(
                f"K={meta['k']} N={meta['n']} step={meta['query_step']} "
                f"role={'first' if meta['is_first'] else 'last' if meta['is_last'] else 'intermediate'} "
                f"dup={meta['duplicate']}\n"
                f"  ctx: {context_str}\n"
                f"  true={vocab.token_repr(true_value_id)}"
                f"  pred={safe_token_repr(pred_value_id)} (valid_value_token={pred_is_valid_value})"
                f"  pred_eos={safe_token_repr(pred_eos_id)} (valid={pred_is_valid_eos})"
                f"  {'OK' if correct else 'WRONG'}\n"
            )

    header = (
        f"split={args.split} ckpt={args.ckpt} sample_acc={n_correct}/{len(sample)}={n_correct/len(sample):.3f} "
        f"out_of_vocab_preds={n_oov_pred}/{len(sample)}\n\n"
    )
    out_text = header + "\n".join(lines)

    out_path = args.out or os.path.join(os.path.dirname(args.ckpt), "..", "..", "results", "val_samples_readable.txt")
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(out_text)
    print(f"wrote {out_path}")
    print(header)


if __name__ == "__main__":
    main()
