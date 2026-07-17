"""Self-contained Kaggle kernel: experiment 04 (GPT-2-small from scratch, DDP on T4 x2).

Bundles vocab / grid / the fixed per-example-step data_gen / eval_utils / model /
train logic into one script, since Kaggle kernels run as a single push-and-execute
script. Regenerates eval data deterministically (seed=42) instead of uploading it.
Writes checkpoints/results to /kaggle/working/ so `kaggle kernels output` can pull
them back, and so a checkpoint can be packaged as a Kaggle Dataset to resume from in
a later session (see RESUME_CKPT_PATH below).

Config (see ../setup.md and ../LOG.md for the full reasoning):
  - batch_size=32/GPU, accum_steps=4, world_size=2 -> effective batch 256
  - fp16 autocast + GradScaler (not bf16 -- T4 predates bf16 tensor core support)
  - dropout=0.1 (GPT-2's real published default -- OpenAI's original GPT-2
    implementation/paper, faithfully mirrored by HF's GPT2Config defaults; matches
    exp03's pretrained run too)
  - data loader samples query_step independently per example within a batch (not
    once per batch like experiments 01/02/03), giving natural role diversity inside
    every micro-batch -- see data_gen.py's docstring
  - eval every 750 steps (not 250) -- early steps already show the trend, no need
    to pay full eval cost that often
  - total_steps=15000 for this first push; resume support is real (not
    aspirational) since Kaggle sessions don't span this run's full wall-clock budget
"""

import contextlib
import json
import math
import os
import random
import subprocess
import sys
import time

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "transformers"], check=True)

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2Config, GPT2LMHeadModel

# ============================== vocab.py ==============================

NUM_KEYS = 26
NUM_VALUES = 10
NUM_STEPS = 12

KEY_BASE = 0
VALUE_BASE = KEY_BASE + NUM_KEYS
STEP_BASE = VALUE_BASE + NUM_VALUES
BOS = STEP_BASE + NUM_STEPS
QUERY = BOS + 1
EOS = QUERY + 1
VOCAB_SIZE = EOS + 1


def key_id(key_idx):
    return KEY_BASE + key_idx


def value_id(value):
    return VALUE_BASE + value


def step_id(step):
    return STEP_BASE + (step - 1)


# ============================== grid.py ==============================

K_VALUES = [2, 4, 6, 8, 10, 12]
N_VALUES = [2, 4, 6, 8, 10, 12]
HELD_OUT_FRACTION = 0.25


def interior_steps(n):
    return list(range(2, n))


def all_interior_cells():
    cells = []
    for n in N_VALUES:
        for s in interior_steps(n):
            cells.append((n, s))
    return cells


def select_held_out_cells(seed, fraction=HELD_OUT_FRACTION):
    cells = all_interior_cells()
    rng = random.Random(seed)
    order = cells[:]
    rng.shuffle(order)

    exposure = {}
    for n, s in cells:
        exposure.setdefault(s, 1 if s in N_VALUES else 0)
    for n, s in cells:
        exposure[s] += 1

    target = round(len(cells) * fraction)
    held_out = []
    for n, s in order:
        if len(held_out) >= target:
            break
        if exposure[s] - 1 >= 1:
            held_out.append((n, s))
            exposure[s] -= 1

    return set(held_out)


def allowed_steps_for_training(n, held_out_cells):
    return [s for s in range(1, n + 1) if (n, s) not in held_out_cells]


def all_triples(held_out_cells=None, only_held_out=False):
    triples = []
    for k in K_VALUES:
        for n in N_VALUES:
            for s in range(1, n + 1):
                is_held_out = held_out_cells is not None and (n, s) in held_out_cells
                if only_held_out and not is_held_out:
                    continue
                if not only_held_out and held_out_cells is not None and is_held_out:
                    continue
                triples.append((k, n, s))
    return triples


# ============================== data_gen.py (fixed: per-example step) ==============================

def build_example(rng, k, n, query_step):
    active_keys = rng.sample(range(NUM_KEYS), k)
    query_key = rng.choice(active_keys)

    histories = {key: [rng.randrange(NUM_VALUES) for _ in range(n)] for key in active_keys}

    remaining = {key: n for key in active_keys}
    context_ids = []
    while remaining:
        key = rng.choice(list(remaining.keys()))
        step_idx = n - remaining[key]
        value = histories[key][step_idx]
        context_ids.append(key_id(key))
        context_ids.append(value_id(value))
        remaining[key] -= 1
        if remaining[key] == 0:
            del remaining[key]

    target_value = histories[query_key][query_step - 1]
    duplicate_flag = any(
        v == target_value for i, v in enumerate(histories[query_key]) if i != query_step - 1
    )

    full_ids = (
        [BOS]
        + context_ids
        + [QUERY, key_id(query_key), step_id(query_step)]
        + [value_id(target_value), EOS]
    )
    meta = {
        "k": k, "n": n, "query_key": query_key, "query_step": query_step,
        "is_first": query_step == 1, "is_last": query_step == n, "duplicate": duplicate_flag,
    }
    return full_ids, meta


def build_frozen_eval_set(seed, triples, per_triple):
    rng = random.Random(seed)
    examples = []
    for k, n, step in triples:
        for _ in range(per_triple):
            examples.append(build_example(rng, k, n, step))
    return examples


def stream_training_batches(seed, held_out_cells, batch_size):
    """Per-example step sampling: K,N fixed per micro-batch (required for padding-free
    batching), but query_step sampled independently per example -- every micro-batch
    gets natural first/last/intermediate role mixing internally."""
    rng = random.Random(seed)
    while True:
        k = rng.choice(K_VALUES)
        n = rng.choice(N_VALUES)
        allowed_steps = allowed_steps_for_training(n, held_out_cells)
        batch = [build_example(rng, k, n, rng.choice(allowed_steps)) for _ in range(batch_size)]
        yield batch, k, n


def pack_split(examples):
    groups = {}
    for full_ids, meta in examples:
        key = (meta["k"], meta["n"])
        groups.setdefault(key, {"input_ids": [], "meta": []})
        groups[key]["input_ids"].append(full_ids)
        groups[key]["meta"].append(meta)
    packed = {}
    for (k, n), g in groups.items():
        packed[(k, n)] = {"input_ids": torch.tensor(g["input_ids"], dtype=torch.long), "meta": g["meta"]}
    return packed


# ============================== eval_utils.py ==============================

@torch.no_grad()
def evaluate_split(model, packed, device, chunk_size):
    model.eval()
    results = []
    for (k, n), group in packed.items():
        input_ids_full = group["input_ids"].to(device)
        meta_list = group["meta"]
        num = input_ids_full.shape[0]
        for start in range(0, num, chunk_size):
            chunk = input_ids_full[start:start + chunk_size]
            inp = chunk[:, :-1]
            targets = chunk[:, -2:]
            logits, _ = model(inp)
            answer_logits = logits[:, -2:, :]
            loss = F.cross_entropy(
                answer_logits.reshape(-1, answer_logits.shape[-1]), targets.reshape(-1), reduction="none",
            ).reshape(chunk.shape[0], 2)
            pred_value = answer_logits[:, 0, :].argmax(dim=-1)
            correct = (pred_value == targets[:, 0])
            for i in range(chunk.shape[0]):
                m = meta_list[start + i]
                results.append({**m, "correct": bool(correct[i].item()), "value_loss": float(loss[i, 0].item())})
    model.train()
    return results


def aggregate_results(results):
    def acc(rs):
        return sum(r["correct"] for r in rs) / len(rs) if rs else None

    agg = {"overall": acc(results), "n": len(results)}

    by_cell = {}
    for r in results:
        by_cell.setdefault((r["k"], r["n"]), []).append(r)
    agg["by_cell"] = {f"K{k}_N{n}": {"acc": acc(rs), "n": len(rs)} for (k, n), rs in by_cell.items()}

    by_role = {"first": [], "last": [], "intermediate": []}
    for r in results:
        if r["is_first"]:
            by_role["first"].append(r)
        elif r["is_last"]:
            by_role["last"].append(r)
        else:
            by_role["intermediate"].append(r)
    agg["by_role"] = {role: {"acc": acc(rs), "n": len(rs)} for role, rs in by_role.items()}

    by_step = {}
    for r in results:
        if r["is_first"] or r["is_last"]:
            continue
        by_step.setdefault(r["query_step"], []).append(r)
    agg["by_step"] = {str(s): {"acc": acc(rs), "n": len(rs)} for s, rs in sorted(by_step.items())}

    by_dup = {True: [], False: []}
    for r in results:
        by_dup[r["duplicate"]].append(r)
    agg["by_duplicate"] = {str(k): {"acc": acc(rs), "n": len(rs)} for k, rs in by_dup.items()}

    return agg


# ============================== model_gpt2.py ==============================

GPT2_SMALL_DEFAULTS = dict(n_positions=1024, n_embd=768, n_layer=12, n_head=12)


class GPT2FromScratch(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE):
        super().__init__()
        config = GPT2Config(
            vocab_size=vocab_size, bos_token_id=BOS, eos_token_id=EOS,
            resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1,  # GPT-2's real default, matches exp03
            **GPT2_SMALL_DEFAULTS,
        )
        self.gpt2 = GPT2LMHeadModel(config)

    def forward(self, input_ids, return_attn=False):
        out = self.gpt2(input_ids=input_ids, output_attentions=return_attn, logits_to_keep=2)
        attn = list(out.attentions) if return_attn else None
        return out.logits, attn


# ============================== train.py (DDP + fp16 + resume) ==============================

SEED = 42
BATCH_SIZE = 32          # per-GPU micro-batch; profiled safe at the real worst-case seq_len=294
ACCUM_STEPS = 4          # -> effective_batch = 32 * 4 * world_size(2) = 256
TOTAL_STEPS = 15000
WARMUP_STEPS = 300
CKPT_EVERY = 750
LOG_EVERY = 50
PEAK_LR = 3e-4
WEIGHT_DECAY = 0.05
EVAL_CHUNK_SIZE = 64
KEEP_LAST_N_CKPTS = 2
TAG = "gpt2_scratch_exp04_ddp"

# Set to a checkpoint path (e.g. "/kaggle/input/<dataset-slug>/best.pt") to resume a
# later session from a checkpoint packaged as a Kaggle Dataset. None for a fresh run.
RESUME_CKPT_PATH = None

OUT_DIR = "/kaggle/working"
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints", TAG)
RESULTS_DIR = os.path.join(OUT_DIR, "results", TAG)

IID_VAL_PER_TRIPLE = 50
IID_TEST_PER_TRIPLE = 200
HELDOUT_TEST_PER_TRIPLE = 500


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


def main_worker(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29502"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    is_main = rank == 0

    if is_main:
        os.makedirs(CKPT_DIR, exist_ok=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        print(f"world_size={world_size}, per-gpu batch={BATCH_SIZE}, accum_steps={ACCUM_STEPS}, "
              f"effective_batch={BATCH_SIZE * ACCUM_STEPS * world_size}, total_steps={TOTAL_STEPS}", flush=True)
        print(f"GPU{rank}: {torch.cuda.get_device_name(rank)}", flush=True)

    held_out_cells = select_held_out_cells(seed=SEED)  # same on every rank, no rank-dependence
    if is_main:
        print(f"held-out cells ({len(held_out_cells)}): {sorted(held_out_cells)}", flush=True)

    torch.manual_seed(SEED)  # same init on every rank; DDP also broadcasts rank0's weights at construction
    model = GPT2FromScratch().to(device)

    start_step = 1
    best_val_acc = -1.0
    if RESUME_CKPT_PATH:
        ckpt = torch.load(RESUME_CKPT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_step = ckpt["step"] + 1
        best_val_acc = ckpt.get("best_val_acc", -1.0)
        if is_main:
            print(f"resumed from {RESUME_CKPT_PATH} at step {ckpt['step']}", flush=True)

    model = DDP(model, device_ids=[rank])
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda")
    if RESUME_CKPT_PATH:
        opt.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])

    if is_main:
        n_params = sum(p.numel() for p in model.module.parameters())
        print(f"model params: {n_params:,}", flush=True)

    # each rank draws different data via a rank-offset seed
    stream = stream_training_batches(SEED + rank, held_out_cells, BATCH_SIZE)
    if RESUME_CKPT_PATH:
        n_skip = (start_step - 1) * ACCUM_STEPS
        if is_main:
            print(f"fast-forwarding data stream by {n_skip} micro-batches/rank to match resume point...", flush=True)
        for _ in range(n_skip):
            next(stream)

    # only rank 0 needs eval data
    iid_val = None
    if is_main:
        print("generating eval data (seed=42, deterministic)...", flush=True)
        allowed_triples = all_triples(held_out_cells, only_held_out=False)
        held_out_triples = all_triples(held_out_cells, only_held_out=True)
        iid_val = pack_split(build_frozen_eval_set(SEED + 1, allowed_triples, IID_VAL_PER_TRIPLE))

    train_log = []
    per_cell_losses = {}
    t0 = time.time()

    for step in range(start_step, TOTAL_STEPS + 1):
        lr = lr_at(step, TOTAL_STEPS, PEAK_LR, WARMUP_STEPS)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        for micro in range(ACCUM_STEPS):
            batch, k, n = next(stream)
            full_ids = torch.tensor([ex[0] for ex in batch], dtype=torch.long, device=device)
            inp = full_ids[:, :-1]
            targets = full_ids[:, -2:]

            sync_ctx = model.no_sync() if micro < ACCUM_STEPS - 1 else contextlib.nullcontext()
            with sync_ctx:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits, _ = model(inp)
                    answer_logits = logits[:, -2:, :]
                    loss = F.cross_entropy(
                        answer_logits.reshape(-1, answer_logits.shape[-1]), targets.reshape(-1)
                    ) / ACCUM_STEPS
                scaler.scale(loss).backward()
            per_cell_losses.setdefault((k, n), []).append(loss.item() * ACCUM_STEPS)
        scaler.step(opt)
        scaler.update()

        if is_main and step % LOG_EVERY == 0:
            elapsed = time.time() - t0
            mean_recent = sum(sum(v) for v in per_cell_losses.values()) / sum(len(v) for v in per_cell_losses.values())
            print(f"step {step}/{TOTAL_STEPS} lr={lr:.2e} loss={mean_recent:.4f} elapsed={elapsed:.1f}s", flush=True)
            train_log.append({"step": step, "lr": lr, "loss": mean_recent, "elapsed": elapsed})
            per_cell_losses = {}

        if step % CKPT_EVERY == 0 or step == TOTAL_STEPS:
            dist.barrier()
            if is_main:
                results = evaluate_split(model.module, iid_val, device, EVAL_CHUNK_SIZE)
                agg = aggregate_results(results)
                val_acc = agg["overall"]
                print(f"  [val@{step}] overall_acc={val_acc:.4f} by_role={ {r: v['acc'] for r, v in agg['by_role'].items()} }", flush=True)

                with open(os.path.join(RESULTS_DIR, f"val_agg_step{step}.json"), "w") as f:
                    json.dump(agg, f, indent=2)
                with open(os.path.join(RESULTS_DIR, "train_log.json"), "w") as f:
                    json.dump(train_log, f, indent=2)

                ckpt_out = {
                    "model": model.module.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                    "step": step, "val_acc": val_acc, "best_val_acc": max(best_val_acc, val_acc),
                }
                torch.save(ckpt_out, os.path.join(CKPT_DIR, f"step_{step}.pt"))
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(ckpt_out, os.path.join(CKPT_DIR, "best.pt"))
                    print(f"  new best val_acc={best_val_acc:.4f}", flush=True)

                step_ckpts = sorted(
                    (f for f in os.listdir(CKPT_DIR) if f.startswith("step_") and f.endswith(".pt")),
                    key=lambda f: int(f[len("step_"):-len(".pt")]),
                )
                for old in step_ckpts[:-KEEP_LAST_N_CKPTS]:
                    os.remove(os.path.join(CKPT_DIR, old))
            dist.barrier()

    if is_main:
        print("training complete. running final held-out eval...", flush=True)
        best_ckpt = torch.load(os.path.join(CKPT_DIR, "best.pt"), map_location=device, weights_only=False)
        model.module.load_state_dict(best_ckpt["model"])

        allowed_triples = all_triples(held_out_cells, only_held_out=False)
        held_out_triples = all_triples(held_out_cells, only_held_out=True)
        iid_test = pack_split(build_frozen_eval_set(SEED + 2, allowed_triples, IID_TEST_PER_TRIPLE))
        heldout_test = pack_split(build_frozen_eval_set(SEED + 3, held_out_triples, HELDOUT_TEST_PER_TRIPLE))

        iid_results = evaluate_split(model.module, iid_test, device, EVAL_CHUNK_SIZE)
        heldout_results = evaluate_split(model.module, heldout_test, device, EVAL_CHUNK_SIZE)
        iid_agg = aggregate_results(iid_results)
        heldout_agg = aggregate_results(heldout_results)

        steps_held_out = sorted({s for (_, s) in held_out_cells})
        gap_report = {}
        for s in steps_held_out:
            gap_report[str(s)] = {
                "trained_elsewhere_acc": iid_agg["by_step"].get(str(s), {}).get("acc"),
                "held_out_acc": heldout_agg["by_step"].get(str(s), {}).get("acc"),
            }

        final_report = {
            "trained_to_step": best_ckpt["step"], "best_val_acc": best_ckpt["val_acc"],
            "iid_test": iid_agg, "heldout_test": heldout_agg,
            "held_out_steps": steps_held_out, "generalization_gap_by_step": gap_report,
        }
        with open(os.path.join(RESULTS_DIR, "final_eval.json"), "w") as f:
            json.dump(final_report, f, indent=2)

        print(f"iid_test overall: {iid_agg['overall']:.4f} (n={iid_agg['n']})", flush=True)
        print(f"heldout_test overall: {heldout_agg['overall']:.4f} (n={heldout_agg['n']})", flush=True)
        print(f"iid by_duplicate: { {k: v['acc'] for k, v in iid_agg['by_duplicate'].items()} }", flush=True)
        print(f"heldout by_duplicate: { {k: v['acc'] for k, v in heldout_agg['by_duplicate'].items()} }", flush=True)
        print("ALL DONE", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}", flush=True)
    n_gpus = torch.cuda.device_count()
    print(f"visible GPUs: {n_gpus}", flush=True)
    import transformers as _tf
    print(f"transformers={_tf.__version__}", flush=True)

    if n_gpus < 2:
        print("WARNING: fewer than 2 GPUs visible, falling back to world_size=1", flush=True)
        mp.spawn(main_worker, args=(1,), nprocs=1, join=True)
    else:
        mp.spawn(main_worker, args=(n_gpus,), nprocs=n_gpus, join=True)
