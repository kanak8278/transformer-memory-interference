"""Streaming training loop with gradient accumulation and a warmup-stable-decay LR
schedule.

See ../setup.md "Training budget" for the WSD rationale (chosen so an extended run,
in case of grokking-style late convergence, doesn't fight a schedule pre-committed to
decay-to-zero at a fixed step count).

Gradient accumulation: each micro-batch is single-cell (same (K,N,step) triple, so no
padding is needed), but several differently-sampled cells are accumulated before each
optimizer step — otherwise every update would come from one homogeneous cell, and
consecutive cells can differ hugely in difficulty (K=2,N=2 vs K=12,N=12), making
per-step gradients noisier than necessary.
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import data_gen
import eval_utils
from model import TinyTransformer

SRC_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(SRC_DIR, "..", "data")
CKPT_DIR = os.path.join(SRC_DIR, "..", "checkpoints")
RESULTS_DIR = os.path.join(SRC_DIR, "..", "results")


def lr_at(step, total_steps, peak_lr, warmup_steps, decay_frac=0.1, min_lr_frac=0.1):
    if step < warmup_steps:
        return peak_lr * step / max(1, warmup_steps)
    decay_start = total_steps * (1 - decay_frac)
    if step < decay_start:
        return peak_lr
    if step >= total_steps:
        return peak_lr * min_lr_frac
    progress = (step - decay_start) / max(1, total_steps - decay_start)
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return peak_lr * (min_lr_frac + (1 - min_lr_frac) * cos)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_micro_batch(model, stream, device, loss_scale):
    """One forward+backward on a single-cell micro-batch. Returns (loss_item, k, n)."""
    batch, k, n, q_step = next(stream)
    full_ids = torch.tensor([ex[0] for ex in batch], dtype=torch.long, device=device)
    inp = full_ids[:, :-1]
    targets = full_ids[:, -2:]

    logits, _ = model(inp)
    answer_logits = logits[:, -2:, :]
    loss = F.cross_entropy(answer_logits.reshape(-1, answer_logits.shape[-1]), targets.reshape(-1))
    (loss * loss_scale).backward()
    return loss.item(), k, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--d_ff", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--accum_steps", type=int, default=4, help="micro-batches (different cells) accumulated per optimizer step")
    ap.add_argument("--total_steps", type=int, default=50000, help="optimizer steps, not micro-batches")
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--peak_lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--ckpt_every", type=int, default=2000)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", type=str, default=None, help="checkpoint/results subfolder name, default derived from n_layers")
    ap.add_argument("--resume", type=str, default=None, help="path to checkpoint to resume from")
    args = ap.parse_args()

    tag = args.tag or f"L{args.n_layers}_d{args.d_model}"
    ckpt_dir = os.path.join(CKPT_DIR, tag)
    results_dir = os.path.join(RESULTS_DIR, tag)
    tb_dir = os.path.join(RESULTS_DIR, "tb", tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    device = get_device()
    print(f"device: {device}, tag: {tag}, accum_steps: {args.accum_steps} (effective batch {args.batch_size * args.accum_steps})")

    held_out_info = torch.load(os.path.join(DATA_DIR, "held_out_cells.pt"), weights_only=False)
    held_out_cells = held_out_info["held_out_cells"]
    iid_val = torch.load(os.path.join(DATA_DIR, "iid_val.pt"), weights_only=False)

    torch.manual_seed(args.seed)  # model init was previously unseeded despite setup.md's determinism claim
    model = TinyTransformer(
        n_layers=args.n_layers, d_model=args.d_model, n_heads=args.n_heads, d_ff=args.d_ff,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)

    start_step = 1
    best_val_acc = -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        best_val_acc = ckpt.get("best_val_acc", -1.0)
        print(f"resumed from {args.resume} at step {ckpt['step']}")

    stream = data_gen.stream_training_batches(args.seed, held_out_cells, args.batch_size)
    if args.resume:
        # stream_training_batches is a deterministic random.Random(seed) generator with no
        # way to skip ahead analytically (each draw's cost depends on the randomly-chosen
        # K,N). Fast-forward by replaying (and discarding) the exact micro-batches already
        # consumed before the checkpoint, so training resumes on genuinely new examples
        # instead of silently replaying the same data from the start.
        n_skip = (start_step - 1) * args.accum_steps
        print(f"fast-forwarding data stream by {n_skip} micro-batches to match resume point...")
        t_ff = time.time()
        for _ in range(n_skip):
            next(stream)
        print(f"  done in {time.time() - t_ff:.1f}s")

    train_log = []
    per_cell_losses = {}
    t0 = time.time()

    for step in range(start_step, args.total_steps + 1):
        lr = lr_at(step, args.total_steps, args.peak_lr, args.warmup_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.accum_steps):
            loss_item, k, n = run_micro_batch(model, stream, device, loss_scale=1.0 / args.accum_steps)
            per_cell_losses.setdefault((k, n), []).append(loss_item)
        optimizer.step()

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            recent = {f"K{c[0]}_N{c[1]}": sum(v) / len(v) for c, v in per_cell_losses.items()}
            mean_recent = sum(sum(v) for v in per_cell_losses.values()) / sum(len(v) for v in per_cell_losses.values())
            print(f"step {step}/{args.total_steps} lr={lr:.2e} loss={mean_recent:.4f} elapsed={elapsed:.1f}s")
            train_log.append({"step": step, "lr": lr, "loss": mean_recent, "per_cell": recent})
            writer.add_scalar("train/loss", mean_recent, step)
            writer.add_scalar("train/lr", lr, step)
            for cell_name, cell_loss in recent.items():
                writer.add_scalar(f"train_per_cell/{cell_name}", cell_loss, step)
            per_cell_losses = {}

        if step % args.ckpt_every == 0 or step == args.total_steps:
            results = eval_utils.evaluate_split(model, iid_val, device)
            agg = eval_utils.aggregate_results(results)
            val_acc = agg["overall"]
            print(f"  [val@{step}] overall_acc={val_acc:.4f} by_role={ {k: v['acc'] for k, v in agg['by_role'].items()} }")

            writer.add_scalar("val/overall_acc", val_acc, step)
            for role, d in agg["by_role"].items():
                writer.add_scalar(f"val/acc_{role}", d["acc"], step)
            for cell_name, d in agg["by_cell"].items():
                writer.add_scalar(f"val_per_cell/{cell_name}", d["acc"], step)

            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "val_acc": val_acc,
                "best_val_acc": max(best_val_acc, val_acc),
                "args": vars(args),
            }
            torch.save(ckpt, os.path.join(ckpt_dir, f"step_{step}.pt"))
            with open(os.path.join(results_dir, f"val_agg_step{step}.json"), "w") as f:
                json.dump(agg, f, indent=2)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(ckpt, os.path.join(ckpt_dir, "best.pt"))
                print(f"  new best val_acc={best_val_acc:.4f}")

    with open(os.path.join(results_dir, "train_log.json"), "w") as f:
        json.dump(train_log, f, indent=2)
    writer.close()
    print("training complete.")


if __name__ == "__main__":
    main()
