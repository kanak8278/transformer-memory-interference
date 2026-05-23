"""
Per-position IVQ accuracy evaluation.

Tests accuracy at each position k (1 to N) for both:
  - The base model (Qwen2.5-3B-Instruct, no adapter)
  - The dense-IVQ LoRA adapter

Outputs:
  - Per-position accuracy table (k=1 to k=N) for each (K,N) cell
  - JSON results file for plotting
  - Summary comparison: base vs LoRA at each position

Key question: does the LoRA model learn a smooth positional indexing function
(high accuracy at all k) or just improve endpoints (k=1, k=N)?

Usage:
    python lora_intervention/evaluate_ivq.py --run-name dense_ivq
    python lora_intervention/evaluate_ivq.py --run-name dense_ivq --smoke
    python lora_intervention/evaluate_ivq.py --run-name dense_ivq --base-only
"""

import os, sys, json, math, random, argparse, tempfile
from pathlib import Path
from datetime import datetime, timezone

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER",  "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("TOKENIZERS_PARALLELISM",       "false")

import importlib.metadata as _m
_o = _m.version
def _s(p):
    try: v=_o(p); return v if v else "0.0.0"
    except: return "0.0.0"
_m.version = _s

import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from mechanistic_probing_v2.core.dataset_configs import (
    get_eligible_categories, generate_values_for_trial,
)

MODEL_ID    = "Qwen/Qwen2.5-3B-Instruct"
CKPT_DIR    = Path(__file__).parent / "checkpoints"
RESULTS_DIR = Path(__file__).parent / "results"

SYSTEM_PROMPT = (
    "You are a precise data extraction tool. "
    "Output ONLY a single word - the exact value requested. "
    "No other text, no explanation, no punctuation."
)

# Evaluation grid: representative cells spanning training and held-out
EVAL_CELLS = {
    "train": [
        (5,  10),
        (10, 20),
    ],
    "held_moderate": [
        (10, 30),
        (15, 30),
    ],
    "held_hard": [
        (20, 30),
        (25, 50),
    ],
}

TRIALS_PER_POSITION = 20   # trials per (k, condition) per cell
BATCH_SIZE = 40


# ─── ordinal ──────────────────────────────────────────────────────────────────

_ORDINALS = {1:"first",2:"2nd",3:"3rd",4:"4th",5:"5th",6:"6th",7:"7th",
             8:"8th",9:"9th",10:"10th"}

def ordinal(k):
    if k == 1: return "first"
    if k in _ORDINALS: return _ORDINALS[k]
    if 11 <= k%100 <= 13: return f"{k}th"
    return f"{k}{['th','st','nd','rd','th'][min(k%10,4)]}"


# ─── stream + prompt ──────────────────────────────────────────────────────────

def shuffle_nc(items, rng):
    for _ in range(100):
        c=items.copy(); rng.shuffle(c)
        if all(c[i]["category"]!=c[i-1]["category"] for i in range(1,len(c))):
            return c
    result, remaining = [], items.copy(); rng.shuffle(remaining); last=None
    while remaining:
        valid=[i for i,x in enumerate(remaining) if x["category"]!=last]
        if not valid: result.extend(remaining); break
        idx=rng.choice(valid); result.append(remaining.pop(idx)); last=result[-1]["category"]
    return result


def make_prompt(tokenizer, nk, nu, k, seed):
    """Build prompt for position k (1-indexed). Returns (prompt, expected) or (None, None)."""
    rng = random.Random(seed)
    elig = get_eligible_categories("ARBITRARY_SINGLE", min_values=nu)
    cats = rng.sample(elig, nk)
    try:
        vals = generate_values_for_trial("ARBITRARY_SINGLE", cats, nu, rng)
    except ValueError:
        return None, None

    test_cat = cats[seed % nk]
    items = [{"category": c, "value": v} for c in cats for v in vals[c]]
    items = shuffle_nc(items, rng)
    cv = [it["value"] for it in items if it["category"] == test_cat]
    N = len(cv)

    if k > N: return None, None
    expected   = cv[k - 1]
    query_word = "last" if k == N else ordinal(k)

    stream   = "\n".join(f"{it['category']}: {it['value']}" for it in items)
    user_txt = (
        f"Read the following key-value stream. Each key gets updated multiple times.\n\n"
        f"{stream}\n\nWhat was the {query_word} value of {test_cat}?"
    )
    msgs   = [{"role":"system","content":SYSTEM_PROMPT},
              {"role":"user","content":user_txt}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return prompt, expected


def is_correct(pred, exp):
    p,e = pred.lower().strip(), exp.lower().strip()
    return p==e or e in p or p.startswith(e)


# ─── vLLM engine ──────────────────────────────────────────────────────────────

class VLLMEngine:
    def __init__(self, model_path, max_len=16384):
        from vllm import LLM
        print(f"  Loading {model_path}...", flush=True)
        self.llm = LLM(model=model_path, dtype="bfloat16", max_model_len=max_len,
                       trust_remote_code=True, gpu_memory_utilization=0.92,
                       enable_prefix_caching=True)
        print("  ✓ ready", flush=True)

    def generate(self, prompts, max_new_tokens=8):
        from vllm import SamplingParams
        sp = SamplingParams(temperature=0, max_tokens=max_new_tokens)
        return [r.outputs[0].text.strip() for r in self.llm.generate(prompts, sp)]


# ─── cell evaluation ──────────────────────────────────────────────────────────

def eval_cell_by_position(engine, tokenizer, nk, nu, trials_per_pos, max_k=None):
    """
    Returns dict: {k: {"correct": int, "total": int, "accuracy": float}}
    max_k caps the highest position tested (for speed).
    """
    # Determine N from a sample trial
    seed0 = hash(("ivq_eval", nk, nu, 0)) % (2**31)
    rng0  = random.Random(seed0)
    elig  = get_eligible_categories("ARBITRARY_SINGLE", min_values=nu)
    cats0 = rng0.sample(elig, nk)
    try:
        vals0 = generate_values_for_trial("ARBITRARY_SINGLE", cats0, nu, rng0)
    except ValueError:
        return {}

    test_cat0 = cats0[seed0 % nk]
    items0    = [{"category":c,"value":v} for c in cats0 for v in vals0[c]]
    items0    = shuffle_nc(items0, rng0)
    N = len([it for it in items0 if it["category"] == test_cat0])

    if max_k:
        N = min(N, max_k)

    results_by_k = {k: {"correct": 0, "total": 0} for k in range(1, N+1)}

    # Build all prompts: for each k, trials_per_pos seeds
    all_prompts, all_meta = [], []
    for k in range(1, N+1):
        for t in range(trials_per_pos):
            seed = hash(("ivq_eval", nk, nu, k, t)) % (2**31)
            p, exp = make_prompt(tokenizer, nk, nu, k, seed)
            if p is None: continue
            all_prompts.append(p)
            all_meta.append((k, exp))

    if not all_prompts:
        return {}

    # Batch inference
    responses = engine.generate(all_prompts)

    for (k, exp), pred in zip(all_meta, responses):
        correct = is_correct(pred, exp)
        results_by_k[k]["correct"] += int(correct)
        results_by_k[k]["total"]   += 1

    for k in results_by_k:
        r = results_by_k[k]
        r["accuracy"] = round(r["correct"]/r["total"], 4) if r["total"] else 0.0

    return results_by_k


# ─── merge + load ─────────────────────────────────────────────────────────────

def load_model(adapter_path, tmp_dir):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading base {MODEL_ID}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True,
    )
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    if adapter_path:
        print(f"  Loading adapter {adapter_path}...", flush=True)
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model = model.merge_and_unload()
        print("  Merged.", flush=True)

    model.save_pretrained(tmp_dir)
    tok.save_pretrained(tmp_dir)
    return tok


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name",  default="dense_ivq")
    p.add_argument("--adapter",   default=None)
    p.add_argument("--trials",    type=int, default=TRIALS_PER_POSITION)
    p.add_argument("--max-k",     type=int, default=None,
                   help="Cap the max position evaluated per cell (for speed)")
    p.add_argument("--base-only", action="store_true",
                   help="Only evaluate base model (no adapter)")
    p.add_argument("--smoke",     action="store_true")
    return p.parse_args()


def run_eval(engine, tokenizer, trials, max_k, smoke, label):
    all_results = {}
    for split, cells in EVAL_CELLS.items():
        all_results[split] = {}
        print(f"\n--- {split} ---")
        for nk, nu in cells:
            mk = min(nu, max_k) if max_k else (10 if smoke else None)
            print(f"  K={nk}, N={nu}  (testing k=1..{mk or nu})...", flush=True)
            res = eval_cell_by_position(engine, tokenizer, nk, nu,
                                        trials_per_pos=3 if smoke else trials,
                                        max_k=mk)
            all_results[split][f"{nk}_{nu}"] = res

            # Print compact table
            ks      = sorted(res.keys())
            accs    = [f"{res[k]['accuracy']:.0%}" for k in ks]
            k_str   = " ".join(f"k{k}:{a}" for k,a in zip(ks, accs))
            print(f"    {k_str}")
    return all_results


def main():
    args    = parse_args()
    ts      = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    adapter_path = None
    if not args.base_only:
        adapter_path = Path(args.adapter) if args.adapter else \
                       CKPT_DIR / args.run_name / "final"
        if not adapter_path.exists():
            print(f"ERROR: adapter not found at {adapter_path}")
            sys.exit(1)

    print("=" * 60)
    print(f"IVQ Per-Position Evaluation — {args.run_name}")
    print(f"  Trials per position: {args.trials}")
    print(f"  Base-only:           {args.base_only}")
    print("=" * 60)

    all_output = {}

    # ── Base model ────────────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmp:
        print("\n[BASE MODEL]")
        tok = load_model(None, tmp)
        engine = VLLMEngine(tmp)
        all_output["base"] = run_eval(engine, tok, args.trials, args.max_k,
                                      args.smoke, "base")
        del engine  # release GPU before loading adapter

    # ── LoRA adapter ──────────────────────────────────────────────────────────
    if not args.base_only:
        with tempfile.TemporaryDirectory() as tmp:
            print("\n[LORA ADAPTER]")
            tok = load_model(adapter_path, tmp)
            engine = VLLMEngine(tmp)
            all_output["lora"] = run_eval(engine, tok, args.trials, args.max_k,
                                          args.smoke, "lora")

    # ── Summary: base vs LoRA at each position ────────────────────────────────
    if "base" in all_output and "lora" in all_output:
        print("\n" + "=" * 70)
        print("COMPARISON — base vs LoRA  (accuracy at each position k)")
        for split, cells in EVAL_CELLS.items():
            print(f"\n  [{split}]")
            for nk, nu in cells:
                ck = f"{nk}_{nu}"
                b  = all_output["base"].get(split, {}).get(ck, {})
                l  = all_output["lora"].get(split, {}).get(ck, {})
                ks = sorted(set(b) | set(l))
                if not ks: continue
                print(f"  K={nk}, N={nu}")
                print(f"    k:    " + "  ".join(f"{k:>4}" for k in ks))
                print(f"    base: " + "  ".join(f"{b.get(k,{}).get('accuracy',0):>4.0%}" for k in ks))
                print(f"    lora: " + "  ".join(f"{l.get(k,{}).get('accuracy',0):>4.0%}" for k in ks))

    # ── Save ──────────────────────────────────────────────────────────────────
    out = {"run_name": args.run_name, "timestamp": ts,
           "trials_per_position": args.trials, "results": all_output}
    out_path = RESULTS_DIR / f"{args.run_name}_ivq_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
