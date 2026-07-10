"""
E1 — LoRA extrapolation frontier (AAAI task C1.7).

GOAL
----
Decide whether the LoRA-surfaced capability is a GENERAL position-indexed
retrieval skill or just a pushed-out operating point. We evaluate the EXISTING
main adapter (no retraining) far beyond its training grid (K<=10, N<=20) and
locate where it breaks and how the decay is shaped (does it fail near-last like
the base model?).

DESIGN (Scan A runs first; B/C optional follow-ups)
  Scan A  N-axis, K=2 : N in {30,50,75,100,150,200,350,500,750,1000}  (<=50x train N)
  Scan B  K-axis, N=20: K in {10,15,20,30,46}   (46 = max category-keys)
  Scan C  diag  K=N   : K=N in {15,20,30,40,46} (46^2=2116 <= 2300 pool)
Constraint: ARB uses a SHARED 2,300-word pool -> K*N <= 2300; only 46 keys exist.

Per cell we measure base vs LoRA on:
  FVQ (pos 1), CVQ (pos N), and IVQ at ~5 relative depths (0.1/0.25/0.5/0.75/0.9).

STATISTICS
  100 trials/condition MAX, with Wilson-CI early stopping identical to
  lora_intervention/evaluate.py: batches of 10, floor 30 trials, stop a
  condition once its Wilson 95% half-width <= 0.07. Greedy decoding (temp 0).
  Seeds depend only on (dataset,K,N,condition,pos,trial) so base and LoRA see
  IDENTICAL streams (paired comparison).

ROBUSTNESS (Colab-flaky-CLI safe)
  - Results appended to a Drive-mounted JSONL, one line per finished
    (model,scan,K,N,condition,pos) -> RESUMABLE (skip completed on restart).
  - Progress mirrored to a log file for `colab download` monitoring.
  - Honours a STOP sentinel file (upload to halt cleanly).
  - Processes base fully, tears down, then merges+runs LoRA, so a mid-LoRA
    crash never loses the base results.

USAGE (on the VM, launched detached exactly once)
  python lora_intervention/experiments/e1_extrapolation_frontier.py \
      --out-dir /content/drive/MyDrive/transformer-memory-interference/e1_extrapolation_frontier \
      --scans A            # A | B | C | A,B,C
"""

import os
import gc
import sys
import math
import json
import time
import random
import argparse
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Faithful reuse of the paper's exact logic (data_gen is torch-free; safe to import):
from lora_intervention.data_gen import (
    shuffle_no_consecutive, ordinal, stable_seed, SYSTEM_PROMPT,
)
from mechanistic_probing_v2.core.dataset_configs import generate_values_for_trial


# Inlined verbatim from lora_intervention/evaluate.py (avoids importing that
# module, which pulls in torch at module load). Keep in sync with evaluate.py.
def wilson_hw(n, k, z=1.96):
    if n == 0:
        return 1.0
    n_t = n + z ** 2
    p_t = (k + z ** 2 / 2) / n_t
    return z * math.sqrt(p_t * (1 - p_t) / n_t)


def is_correct(predicted, expected):
    pred = predicted.lower().strip()
    exp = expected.lower().strip()
    return pred == exp or exp in pred or pred.startswith(exp)

DATASET = "ARBITRARY_SINGLE"

# Statistics (match evaluate.py)
MAX_TRIALS   = 100
MIN_TRIALS   = 30
CI_THRESHOLD = 0.07
BATCH_TRIALS = 10

IVQ_DEPTHS = [0.10, 0.25, 0.50, 0.75, 0.90]   # relative positions for intermediate queries

SCANS = {
    "A": [(2, n) for n in (30, 50, 75, 100, 150, 200, 350, 500, 750, 1000)],
    "B": [(k, 20) for k in (10, 15, 20, 30, 46)],
    "C": [(k, k) for k in (15, 20, 30, 40, 46)],
}

# Category-key names for ARB (values come from the shared pool; keys are labels).
# Uses the full 46-category namespace via dataset_configs eligibility.
from mechanistic_probing_v2.core.dataset_configs import get_eligible_categories
ALL_KEYS = get_eligible_categories(DATASET, min_values=1)


# ─── condition set for a cell ─────────────────────────────────────────────────

def conditions_for(n):
    """Return list of (name, position_k_1indexed). FVQ=1, CVQ=N, IVQ=depths."""
    conds = [("FVQ", 1), ("CVQ", n)]
    seen = {1, n}
    for d in IVQ_DEPTHS:
        k = max(2, min(n - 1, round(d * n)))
        if k not in seen:
            conds.append((f"IVQ@{d:.2f}", k))
            seen.add(k)
    return conds


# ─── one prompt (identical construction to data_gen.build_example) ────────────

def build_prompt_and_label(tokenizer, k_keys, n_updates, position_k, seed):
    """position_k is 1-indexed within the target category's value list."""
    rng = random.Random(seed)
    cats = rng.sample(ALL_KEYS, k_keys)
    try:
        values_per_cat = generate_values_for_trial(DATASET, cats, n_updates, rng)
    except ValueError:
        return None, None
    test_category = cats[seed % len(cats)]
    items = []
    for cat in cats:
        for val in values_per_cat[cat]:
            items.append({"category": cat, "value": val})
    items = shuffle_no_consecutive(items, rng)
    stream = "\n".join(f"{it['category']}: {it['value']}" for it in items)
    cat_values = [it["value"] for it in items if it["category"] == test_category]

    if position_k == 1:
        query_word = "first"
    elif position_k == len(cat_values):
        query_word = "last"
    else:
        query_word = ordinal(position_k)
    expected = cat_values[position_k - 1]

    user_text = (
        "Read the following key-value stream. Each key gets updated multiple times.\n\n"
        f"{stream}\n\n"
        f"What was the {query_word} value of {test_category}?"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt, expected


# ─── result IO (resumable) ────────────────────────────────────────────────────

def load_done(results_path):
    done = set()
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            done.add((r["model"], r["scan"], r["num_keys"], r["num_updates"],
                      r["condition"]))
    return done


def append_result(results_path, rec):
    with open(results_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def log(logf, msg):
    stamp = time.strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    with open(logf, "a") as f:
        f.write(line + "\n")
        f.flush()


# ─── per-condition evaluation with Wilson early stopping ──────────────────────

def eval_condition(engine, tokenizer, scan, model_tag, k_keys, n_updates,
                   cond_name, position_k):
    results = []
    trial = 0
    while trial < MAX_TRIALS:
        # build one batch of prompts (skip trials whose stream can't be built)
        prompts, expecteds = [], []
        for _ in range(BATCH_TRIALS):
            seed = stable_seed(DATASET, k_keys, n_updates, cond_name, position_k, trial)
            trial += 1
            p, e = build_prompt_and_label(tokenizer, k_keys, n_updates, position_k, seed)
            if p is not None:
                prompts.append(p); expecteds.append(e)
            if trial >= MAX_TRIALS:
                break
        if prompts:
            outs = engine.generate(prompts)
            for pred, exp in zip(outs, expecteds):
                results.append(1 if is_correct(pred, exp) else 0)
        # convergence check (floor 30, Wilson HW <= 0.07)
        n = len(results)
        if n >= MIN_TRIALS and wilson_hw(n, sum(results)) <= CI_THRESHOLD:
            break
    n = len(results); k = sum(results)
    acc = k / n if n else 0.0
    return {
        "model": model_tag, "scan": scan, "dataset": DATASET,
        "num_keys": k_keys, "num_updates": n_updates,
        "condition": cond_name, "position_k": position_k,
        "accuracy": acc, "n_trials": n, "n_correct": k,
        "wilson_hw": wilson_hw(n, k), "stopped_early": n < MAX_TRIALS,
    }


# ─── vLLM engine wrapper (matches evaluate.py) ────────────────────────────────

class VLLMEngine:
    def __init__(self, model_path, max_len=16384):
        from vllm import LLM
        self.llm = LLM(model=model_path, dtype="bfloat16", max_model_len=max_len,
                       gpu_memory_utilization=0.85, enforce_eager=False)

    def generate(self, prompts, max_new_tokens=8):
        from vllm import SamplingParams
        sp = SamplingParams(temperature=0, max_tokens=max_new_tokens)
        return [r.outputs[0].text.strip() for r in self.llm.generate(prompts, sp)]

    def close(self):
        del self.llm
        gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass


def merge_lora(base_id, adapter_path, tmp_dir):
    """Merge adapter into base and save; return merged path."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True)
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model = model.merge_and_unload()
    tok = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(tmp_dir); tok.save_pretrained(tmp_dir)
    del model; gc.collect()
    return tmp_dir


# ─── main ─────────────────────────────────────────────────────────────────────

def run_model(model_tag, model_path, cells_by_scan, out_dir, tokenizer, stop_file):
    results_path = out_dir / "results.jsonl"
    logf = out_dir / "run.log"
    done = load_done(results_path)
    engine = VLLMEngine(model_path)
    log(logf, f"=== model={model_tag} path={model_path} ===")
    try:
        for scan, cells in cells_by_scan.items():
            for (k_keys, n_updates) in cells:
                for cond_name, position_k in conditions_for(n_updates):
                    if stop_file.exists():
                        log(logf, "STOP sentinel found — halting cleanly.")
                        return
                    key = (model_tag, scan, k_keys, n_updates, cond_name)
                    if key in done:
                        continue
                    t0 = time.time()
                    rec = eval_condition(engine, tokenizer, scan, model_tag,
                                         k_keys, n_updates, cond_name, position_k)
                    append_result(results_path, rec)
                    log(logf, f"{scan} K={k_keys} N={n_updates} {cond_name}: "
                              f"acc={rec['accuracy']:.3f} n={rec['n_trials']} "
                              f"hw={rec['wilson_hw']:.3f} ({time.time()-t0:.0f}s)")
    finally:
        engine.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scans", default="A", help="comma list of A,B,C")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default=str(_PROJECT_ROOT / "lora_intervention" / "checkpoints" / "adapter"))
    ap.add_argument("--merged-dir", default="/content/merged_lora")
    ap.add_argument("--only", choices=["base", "lora", "both"], default="both")
    ap.add_argument("--smoke", action="store_true",
                    help="Quick end-to-end check: 8 trials, only the smallest + "
                         "largest cell of each scan (exercises fast + long-context paths).")
    args = ap.parse_args()

    global MAX_TRIALS, MIN_TRIALS
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    stop_file = out_dir / "STOP"
    logf = out_dir / "run.log"
    cells_by_scan = {s: SCANS[s] for s in args.scans.split(",") if s in SCANS}
    if args.smoke:
        MAX_TRIALS = MIN_TRIALS = 8
        cells_by_scan = {s: sorted({cells[0], cells[-1]})
                         for s, cells in cells_by_scan.items()}
        log(logf, "SMOKE MODE: 8 trials, first+last cell per scan.")
    log(logf, f"E1 start. scans={list(cells_by_scan)} cells={cells_by_scan} out={out_dir}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    if args.only in ("base", "both"):
        run_model("base", args.base_model, cells_by_scan, out_dir, tokenizer, stop_file)
    if args.only in ("lora", "both") and not stop_file.exists():
        log(logf, "Merging LoRA adapter...")
        merged = merge_lora(args.base_model, args.adapter, args.merged_dir)
        run_model("lora", merged, cells_by_scan, out_dir, tokenizer, stop_file)

    log(logf, "E1 done.")


if __name__ == "__main__":
    main()
