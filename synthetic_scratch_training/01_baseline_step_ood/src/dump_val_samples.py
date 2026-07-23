"""Decode a sample of val predictions vs ground truth into human-readable text, so we
can visually confirm the model is producing valid single VALUE-token answers (not
garbage / out-of-family tokens), not just look at an accuracy number.
"""

import argparse
import os
import random

import torch

import vocab
from model import TinyTransformer

SRC_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(SRC_DIR, "..", "data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--split", type=str, default="iid_val", choices=["iid_val", "iid_test", "heldout_test"])
    ap.add_argument("--n_samples", type=int, default=40)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model_args = ckpt["args"]
    model = TinyTransformer(
        n_layers=model_args["n_layers"], d_model=model_args["d_model"],
        n_heads=model_args["n_heads"], d_ff=model_args["d_ff"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    packed = torch.load(os.path.join(DATA_DIR, f"{args.split}.pt"), weights_only=False)

    # flatten across (k,n) groups, pick a random sample
    flat = []
    for (k, n), group in packed.items():
        for i in range(group["input_ids"].shape[0]):
            flat.append((group["input_ids"][i], group["meta"][i]))
    rng = random.Random(0)
    sample = rng.sample(flat, min(args.n_samples, len(flat)))

    lines = []
    n_correct = 0
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

            context_str = vocab.decode(full_ids[:-2].tolist())
            lines.append(
                f"K={meta['k']} N={meta['n']} step={meta['query_step']} "
                f"role={'first' if meta['is_first'] else 'last' if meta['is_last'] else 'intermediate'} "
                f"dup={meta['duplicate']}\n"
                f"  ctx: {context_str}\n"
                f"  true={vocab.token_repr(true_value_id)}"
                f"  pred={vocab.token_repr(pred_value_id)} (valid_value_token={pred_is_valid_value})"
                f"  pred_eos={vocab.token_repr(pred_eos_id)} (valid={pred_is_valid_eos})"
                f"  {'OK' if correct else 'WRONG'}\n"
            )

    header = f"split={args.split} ckpt={args.ckpt} sample_acc={n_correct}/{len(sample)}={n_correct/len(sample):.3f}\n\n"
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
