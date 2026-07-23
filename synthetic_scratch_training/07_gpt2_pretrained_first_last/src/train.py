"""Experiment 05 -- GPT-2-small from scratch on a single H100, best-practice recipe.

Single-GPU by design (GPT-2-small is tiny; one 80GB H100 holds a large batch, and
running one arm per GPU beats DDP complexity here). Launch one process per arm:

  python train.py --gpu 0 --schedule cosine  --tag cosine
  python train.py --gpu 1 --schedule sgdr    --tag sgdr
  python train.py --gpu 1 --schedule plateau --tag plateau

Best practices incorporated (see setup.md for citations):
  bf16 autocast (no GradScaler) | TF32 matmul high | torch.compile
  AdamW betas (0.9, 0.95), eps 1e-8, fused, weight decay 0.1 on dim>=2 params only
  grad clip 1.0 | dropout 0.0 | peak lr 6e-4 -> min 6e-5 | warmup scaled to budget
  background-prefetched data loader so the H100 never starves on Python data-gen
  weight tying + residual scaled init (asserted in sanity_check.py, HF-provided)
"""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

import schedulers
from data_gen import build_frozen_eval_set, pack_split
from eval_utils import aggregate_results, evaluate_split
from grid import K_VALUES, N_VALUES, all_triples, select_held_out_cells
from model_gpt2 import GPT2FromScratch, GPT2Pretrained
from prefetch import PrefetchTrainLoader

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(HERE)


def build_schedule(name, args):
    if name == "cosine":
        fn = lambda step: schedulers.cosine_lr(
            step, peak_lr=args.peak_lr, min_lr=args.min_lr,
            warmup_steps=args.warmup, total_steps=args.total_steps)
        return ("cosine", fn, None)
    if name == "sgdr":
        fn = lambda step: schedulers.sgdr_lr(
            step, peak_lr=args.peak_lr, min_lr=args.min_lr,
            warmup_steps=args.warmup, t_0=args.sgdr_t0, t_mult=args.sgdr_tmult)
        return ("sgdr", fn, None)
    if name == "plateau":
        obj = schedulers.PlateauRestartLR(
            peak_lr=args.peak_lr, min_lr=args.min_lr, warmup_steps=args.warmup,
            patience=args.plateau_patience, decay_evals=args.plateau_decay_evals)
        return ("plateau", obj.lr, obj)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--schedule", choices=["cosine", "sgdr", "plateau"], default="cosine")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--total-steps", dest="total_steps", type=int, default=30000)
    ap.add_argument("--warmup", type=int, default=600)
    ap.add_argument("--batch-size", dest="batch_size", type=int, default=256)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--peak-lr", dest="peak_lr", type=float, default=6e-4)
    ap.add_argument("--min-lr", dest="min_lr", type=float, default=6e-5)
    ap.add_argument("--weight-decay", dest="weight_decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", dest="grad_clip", type=float, default=1.0)
    ap.add_argument("--eval-every", dest="eval_every", type=int, default=1000)
    ap.add_argument("--log-every", dest="log_every", type=int, default=50)
    ap.add_argument("--eval-chunk", dest="eval_chunk", type=int, default=256)
    ap.add_argument("--num-workers", dest="num_workers", type=int, default=1)
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--first-last-only", dest="first_last_only", action="store_true",
                    help="exp06: train ONLY on step in {1, n}; hold out the whole interior as a zero-shot probe")
    ap.add_argument("--pretrained", action="store_true",
                    help="exp07: load off-the-shelf pretrained GPT-2 (real 50257 vocab) instead of from-scratch init. Use a fine-tune LR (e.g. --peak-lr 3e-5).")
    # sgdr / plateau knobs
    ap.add_argument("--sgdr-t0", dest="sgdr_t0", type=int, default=4000)
    ap.add_argument("--sgdr-tmult", dest="sgdr_tmult", type=int, default=2)
    ap.add_argument("--plateau-patience", dest="plateau_patience", type=int, default=2)
    ap.add_argument("--plateau-decay-evals", dest="plateau_decay_evals", type=int, default=3)
    args = ap.parse_args()
    tag = args.tag or f"h100_{args.schedule}"

    ckpt_dir = os.path.join(EXP_DIR, "checkpoints", tag)
    results_dir = os.path.join(EXP_DIR, "results", tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.cache_size_limit = 64  # ~36 (K,N) input shapes -> raise from default 8

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    torch.manual_seed(SEED)

    held_out_cells = select_held_out_cells(seed=SEED)
    eff_batch = args.batch_size * args.accum
    print(f"[{tag}] gpu={args.gpu} schedule={args.schedule} total_steps={args.total_steps} "
          f"micro_bs={args.batch_size} accum={args.accum} eff_batch={eff_batch}", flush=True)
    print(f"[{tag}] held-out cells ({len(held_out_cells)}): {sorted(held_out_cells)}", flush=True)

    if args.pretrained:
        print(f"[{tag}] loading pretrained GPT-2 (real 50257 vocab, fine-tune)", flush=True)
        raw_model = GPT2Pretrained().to(device)
    else:
        raw_model = GPT2FromScratch(dropout=0.0).to(device)
    opt, opt_info = raw_model.configure_optimizers(
        weight_decay=args.weight_decay, lr=args.peak_lr, betas=(0.9, 0.95),
        eps=1e-8, device_type="cuda")
    n_params = sum(p.numel() for p in raw_model.parameters())
    print(f"[{tag}] params={n_params:,} optim={opt_info}", flush=True)

    model = torch.compile(raw_model) if args.compile else raw_model

    sched_name, lr_fn, plateau_obj = build_schedule(args.schedule, args)

    # triple sets. first/last edges are always step in {1, n}; interior is 2..n-1.
    first_last_triples = [(k, n, s) for k in K_VALUES for n in N_VALUES for s in (1, n)]
    interior_triples = [(k, n, s) for k in K_VALUES for n in N_VALUES for s in range(2, n)]
    # step tokens the model actually sees in first/last-only training: 1 (first) and every
    # n in N_VALUES (last). An interior probe example is "seen-token" if its step is one of
    # those, else "novel-token" -- reported separately so the zero-shot result is readable.
    seen_step_tokens = {1} | set(N_VALUES)

    # frozen iid-val for monitoring (seed=42). In first/last mode we monitor the edges only.
    mon_triples = first_last_triples if args.first_last_only else all_triples(held_out_cells, only_held_out=False)
    iid_val = pack_split(build_frozen_eval_set(SEED + 1, mon_triples, 50))

    loader = PrefetchTrainLoader(SEED, held_out_cells, args.batch_size,
                                 num_workers=args.num_workers, first_last_only=args.first_last_only)
    stream = iter(loader)

    train_log = []
    per_cell_losses = {}
    best_val = -1.0
    t0 = time.time()

    try:
        for step in range(1, args.total_steps + 1):
            lr = lr_fn(step)
            for g in opt.param_groups:
                g["lr"] = lr

            opt.zero_grad(set_to_none=True)
            for _ in range(args.accum):
                input_ids, k, n = next(stream)
                input_ids = input_ids.to(device, non_blocking=True)
                inp, targets = input_ids[:, :-1], input_ids[:, -2:]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, _ = model(inp)
                    al = logits[:, -2:, :]
                    loss = F.cross_entropy(al.reshape(-1, al.shape[-1]), targets.reshape(-1)) / args.accum
                loss.backward()
                per_cell_losses.setdefault((k, n), []).append(loss.item() * args.accum)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
            opt.step()

            if step % args.log_every == 0:
                elapsed = time.time() - t0
                mean_loss = sum(sum(v) for v in per_cell_losses.values()) / sum(len(v) for v in per_cell_losses.values())
                sps = step / elapsed
                print(f"[{tag}] step {step}/{args.total_steps} lr={lr:.2e} loss={mean_loss:.4f} "
                      f"{sps:.2f} steps/s elapsed={elapsed:.0f}s", flush=True)
                train_log.append({"step": step, "lr": lr, "loss": mean_loss, "elapsed": elapsed})
                per_cell_losses = {}

            if step % args.eval_every == 0 or step == args.total_steps:
                results = evaluate_split(raw_model, iid_val, device, args.eval_chunk)
                agg = aggregate_results(results)
                val_acc = agg["overall"]
                by_role = {r: round(v["acc"], 4) for r, v in agg["by_role"].items()}
                print(f"[{tag}]   val@{step} acc={val_acc:.4f} by_role={by_role}", flush=True)
                agg["step"] = step
                with open(os.path.join(results_dir, f"val_agg_step{step}.json"), "w") as f:
                    json.dump(agg, f, indent=2)
                with open(os.path.join(results_dir, "train_log.json"), "w") as f:
                    json.dump(train_log, f, indent=2)
                if plateau_obj is not None:
                    plateau_obj.on_eval(val_acc)
                    with open(os.path.join(results_dir, "plateau_history.json"), "w") as f:
                        json.dump(plateau_obj.history, f, indent=2)

                ckpt = {"model": raw_model.state_dict(), "step": step, "val_acc": val_acc,
                        "args": vars(args)}
                torch.save(ckpt, os.path.join(ckpt_dir, "last.pt"))
                if val_acc > best_val:
                    best_val = val_acc
                    torch.save(ckpt, os.path.join(ckpt_dir, "best.pt"))
                    print(f"[{tag}]   new best val_acc={best_val:.4f}", flush=True)
    finally:
        loader.close()

    # final held-out eval on best checkpoint
    print(f"[{tag}] training done. final eval on best.pt...", flush=True)
    best = torch.load(os.path.join(ckpt_dir, "best.pt"), map_location=device, weights_only=False)
    raw_model.load_state_dict(best["model"])

    if args.first_last_only:
        # primary: first/last edges (IID -- these ARE what we trained on)
        fl_test = pack_split(build_frozen_eval_set(SEED + 2, first_last_triples, 200))
        fl_agg = aggregate_results(evaluate_split(raw_model, fl_test, device, args.eval_chunk))
        # zero-shot probe: the entire interior, never trained on. Split by whether the
        # query's step token was seen in training (as some n's "last") or is fully novel.
        probe = evaluate_split(raw_model, pack_split(build_frozen_eval_set(SEED + 3, interior_triples, 200)),
                               device, args.eval_chunk)
        seen = [r for r in probe if r["query_step"] in seen_step_tokens]
        novel = [r for r in probe if r["query_step"] not in seen_step_tokens]
        acc = lambda rs: (sum(r["correct"] for r in rs) / len(rs)) if rs else None
        probe_agg = aggregate_results(probe)
        final = {"tag": tag, "schedule": sched_name, "mode": "first_last_only",
                 "trained_to_step": best["step"], "best_val_acc": best["val_acc"],
                 "first_last_iid": fl_agg,
                 "intermediate_zeroshot": probe_agg,
                 "intermediate_zeroshot_seen_token_acc": acc(seen),
                 "intermediate_zeroshot_novel_token_acc": acc(novel)}
        with open(os.path.join(results_dir, "final_eval.json"), "w") as f:
            json.dump(final, f, indent=2)
        fl_role = {r: round((v["acc"] or 0), 4) for r, v in fl_agg["by_role"].items()}
        print(f"[{tag}] FINAL first/last IID acc={fl_agg['overall']:.4f} by_role={fl_role}", flush=True)
        print(f"[{tag}] intermediate ZERO-SHOT acc={probe_agg['overall']:.4f} "
              f"(seen-token={acc(seen)}, novel-token={acc(novel)})", flush=True)
        print(f"[{tag}] ALL DONE", flush=True)
        return

    held_triples = all_triples(held_out_cells, only_held_out=True)
    iid_test = pack_split(build_frozen_eval_set(SEED + 2, mon_triples, 200))
    heldout_test = pack_split(build_frozen_eval_set(SEED + 3, held_triples, 500))
    iid_agg = aggregate_results(evaluate_split(raw_model, iid_test, device, args.eval_chunk))
    heldout_agg = aggregate_results(evaluate_split(raw_model, heldout_test, device, args.eval_chunk))
    final = {"tag": tag, "schedule": sched_name, "trained_to_step": best["step"],
             "best_val_acc": best["val_acc"], "iid_test": iid_agg, "heldout_test": heldout_agg}
    with open(os.path.join(results_dir, "final_eval.json"), "w") as f:
        json.dump(final, f, indent=2)
    print(f"[{tag}] FINAL iid_test={iid_agg['overall']:.4f} heldout_test={heldout_agg['overall']:.4f}", flush=True)
    print(f"[{tag}] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
