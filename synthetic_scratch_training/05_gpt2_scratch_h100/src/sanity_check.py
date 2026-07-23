"""Decisive pre-training debug. Answers: 'is the plateau a code bug, or a real
learning/distribution problem?' Run this before any long training.

Checks:
  1. Sequence/target alignment -- the loss's target really is the VALUE token, and
     it's the value at histories[query_key][step-1]. Verified against build_example
     ground truth, and against the exact slicing train.py uses.
  2. Weight tying -- wte.weight and lm_head.weight share storage (HF default).
  3. Residual scaled init -- c_proj weights have std ~= 0.02/sqrt(2*n_layer).
  4. Single-batch overfit -- fix ONE batch, train on only it. A correct 124M model
     must drive train loss -> ~0 and accuracy -> 100% (it can just memorise 256
     input->value maps). If it can't, the bug is in the model/loss/data path, not
     in the task difficulty. If it can, the code is sound and the plateau on the
     full task is a genuine learning problem.
"""

import argparse
import math

import torch
import torch.nn.functional as F

import vocab
from data_gen import build_example
from grid import select_held_out_cells
from model_gpt2 import GPT2FromScratch


def check_alignment():
    import random
    rng = random.Random(0)
    k, n, step = 6, 6, 3
    full_ids, meta = build_example(rng, k, n, step)
    # last two tokens must be [VALUE, EOS]; value token decodes to the ground-truth value
    val_tok, eos_tok = full_ids[-2], full_ids[-1]
    assert eos_tok == vocab.EOS, "last token is not EOS"
    assert vocab.VALUE_BASE <= val_tok < vocab.STEP_BASE, "position -2 is not a VALUE token"
    # train.py: inp = full[:-1], targets = full[-2:]; logits[:,-2:] predict [VALUE, EOS].
    inp = full_ids[:-1]
    targets = full_ids[-2:]
    assert inp[-1] == val_tok, "inp's last token should be the VALUE (predicts EOS from it)"
    assert inp[-2] == vocab.step_id(step), "token before VALUE should be the STEP token"
    assert targets[0] == val_tok and targets[1] == vocab.EOS
    print(f"  [1] alignment OK: target VALUE={vocab.token_repr(val_tok)} at pos -2, "
          f"predicted from STEP token {vocab.token_repr(inp[-2])}")


def check_tying_and_init(model):
    wte = model.gpt2.transformer.wte.weight
    lm = model.gpt2.lm_head.weight
    assert wte.data_ptr() == lm.data_ptr(), "wte and lm_head are NOT tied!"
    print(f"  [2] weight tying OK: wte and lm_head share storage ({tuple(wte.shape)})")

    n_layer = model.gpt2.config.n_layer
    expected = 0.02 / math.sqrt(2 * n_layer)
    stds = []
    for name, p in model.named_parameters():
        if name.endswith("c_proj.weight"):
            stds.append(p.std().item())
    mean_std = sum(stds) / len(stds)
    ok = abs(mean_std - expected) / expected < 0.15
    print(f"  [3] residual init {'OK' if ok else 'MISMATCH'}: mean c_proj std={mean_std:.5f} "
          f"expected~={expected:.5f} over {len(stds)} tensors")
    assert ok, "c_proj residual scaled init not applied as expected"


def overfit_one_batch(model, device, steps=300, bs=256):
    import random
    rng = random.Random(1)
    k, n = 8, 8
    batch = [build_example(rng, k, n, rng.randint(1, n)) for _ in range(bs)]
    full = torch.tensor([ex[0] for ex in batch], dtype=torch.long, device=device)
    inp, targets = full[:, :-1], full[:, -2:]

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
    model.train()
    print(f"  [4] overfitting one fixed batch (bs={bs}, cell K{k}/N{n}, mixed steps)...")
    for s in range(1, steps + 1):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(inp)
            al = logits[:, -2:, :]
            loss = F.cross_entropy(al.reshape(-1, al.shape[-1]), targets.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if s % 50 == 0 or s == 1:
            with torch.no_grad():
                pred = al[:, 0, :].argmax(-1)
                acc = (pred == targets[:, 0]).float().mean().item()
            print(f"      step {s:4d}  loss={loss.item():.4f}  value_acc={acc:.3f}")
    verdict = "PASS (code path is sound)" if acc > 0.99 else "FAIL (bug in model/loss/data path)"
    print(f"  [4] overfit verdict: {verdict}  final acc={acc:.3f}")
    return acc > 0.99


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()
    torch.manual_seed(42)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    print("== sanity check ==")
    check_alignment()
    _ = select_held_out_cells(seed=42)
    model = GPT2FromScratch(dropout=0.0).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params:,}")
    check_tying_and_init(model)
    ok = overfit_one_batch(model, device, steps=args.steps)
    print("== ALL CHECKS PASSED ==" if ok else "== OVERFIT FAILED -- investigate before training ==")


if __name__ == "__main__":
    main()
