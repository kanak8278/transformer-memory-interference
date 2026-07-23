"""Training loop for the real-pretrained-GPT-2-small reference experiment.

Reuses experiment 01's data pipeline (data_gen, grid, vocab, eval_utils) directly via
path import — nothing about the task changes, only the model (see
model_gpt2_pretrained.py, loaded from pretrained weights, no vocab remapping) and
whatever batch/step sizing this scale needs (see setup.md — must be re-profiled).
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

SRC_DIR = os.path.dirname(__file__)
EXP01_SRC = os.path.join(SRC_DIR, "..", "..", "01_baseline_step_ood", "src")
sys.path.insert(0, os.path.abspath(EXP01_SRC))
sys.path.insert(0, SRC_DIR)

import data_gen
import eval_utils
from model_gpt2_pretrained import GPT2Pretrained

DATA_DIR = os.path.join(SRC_DIR, "..", "data")
EXP01_DATA_DIR = os.path.join(EXP01_SRC, "..", "data")
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
    ap.add_argument("--batch_size", type=int, default=32, help="micro-batch; needs re-profiling for this scale")
    ap.add_argument("--accum_steps", type=int, default=4)
    ap.add_argument("--total_steps", type=int, default=12500)
    ap.add_argument("--warmup_steps", type=int, default=125)
    ap.add_argument("--peak_lr", type=float, default=3e-5, help="fine-tuning pretrained weights wants a much smaller lr than training from scratch (exp02 uses 3e-4) — avoid destroying pretrained representations before the model adapts them")
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", type=str, default="gpt2_small_pretrained")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--keep_last_n_ckpts", type=int, default=3,
                     help="prune step_*.pt checkpoints beyond the last N (plus best.pt) — at 124M "
                          "params each is ~1GB+, unlike exp01's ~1-2MB")
    ap.add_argument("--eval_chunk_size", type=int, default=64,
                     help="eval_utils' default (512) OOMs on MPS at this scale — attention memory "
                          "scales as batch*heads*seq^2, and GPT-2-small's 12 heads over ~290-token "
                          "sequences blew past 25GB at batch=512. Empirically 128 was fine, 64 is a "
                          "safety margin under real-training memory conditions.")
    args = ap.parse_args()

    ckpt_dir = os.path.join(CKPT_DIR, args.tag)
    results_dir = os.path.join(RESULTS_DIR, args.tag)
    tb_dir = os.path.join(RESULTS_DIR, "tb", args.tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    device = get_device()
    print(f"device: {device}, tag: {args.tag}, accum_steps: {args.accum_steps} (effective batch {args.batch_size * args.accum_steps})")

    held_out_info = torch.load(os.path.join(EXP01_DATA_DIR, "held_out_cells.pt"), weights_only=False)
    held_out_cells = held_out_info["held_out_cells"]
    iid_val = torch.load(os.path.join(EXP01_DATA_DIR, "iid_val.pt"), weights_only=False)

    torch.manual_seed(args.seed)
    model = GPT2Pretrained().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")
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
            results = eval_utils.evaluate_split(model, iid_val, device, chunk_size=args.eval_chunk_size)
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

            step_ckpts = sorted(
                (f for f in os.listdir(ckpt_dir) if f.startswith("step_") and f.endswith(".pt")),
                key=lambda f: int(f[len("step_"):-len(".pt")]),
            )
            for old in step_ckpts[:-args.keep_last_n_ckpts]:
                os.remove(os.path.join(ckpt_dir, old))

    with open(os.path.join(results_dir, "train_log.json"), "w") as f:
        json.dump(train_log, f, indent=2)
    writer.close()
    print("training complete.")


if __name__ == "__main__":
    main()
