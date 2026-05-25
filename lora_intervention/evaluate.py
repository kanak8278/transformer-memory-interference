"""
Post-LoRA behavioral evaluation on the test grid.

Loads the LoRA adapter, merges it with the base model, then runs vLLM
inference on held-out test cells (ARBITRARY_SINGLE + SEMANTIC_MULTI OOD).
Compares RI/PI accuracy to the baseline stage1 results.

Usage:
    cd /home/sagemaker-user/transformer-memory-interference

    # Evaluate main run
    python lora_intervention/evaluate.py --run-name main

    # Smoke test
    python lora_intervention/evaluate.py --run-name main --smoke

    # Endpoint-only comparison
    python lora_intervention/evaluate.py --run-name endpoint_only

Output:
    lora_intervention/results/<run_name>_eval_<timestamp>.json
    lora_intervention/results/<run_name>_comparison.txt   (human-readable table)
"""

import os
import sys
import json
import math
import random
import argparse
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER",   "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD",  "spawn")
os.environ.setdefault("TOKENIZERS_PARALLELISM",        "false")

import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from mechanistic_probing_v2.core.dataset_configs import (
    get_eligible_categories,
    generate_values_for_trial,
)

# ─── constants ────────────────────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen2.5-3B-Instruct"
CKPT_DIR    = Path(__file__).parent / "checkpoints"
RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_BASELINE = _ROOT / "v3/results_vllm/arbitrary_single/Qwen2.5-3B-Instruct/stage1_sweep_20260409_000134.json"

SYSTEM_PROMPT = (
    "You are a precise data extraction tool. "
    "Output ONLY a single word - the exact value requested. "
    "No other text, no explanation, no punctuation."
)

# Held-out test grid (no overlap with training cells)
TEST_GRID_ARBI = {
    7:  [30, 50, 75],
    10: [30, 50, 75],
    15: [15, 20, 30, 50, 75],
    20: [15, 20, 30, 50, 75],
    25: [10, 15, 20, 30, 50, 75],
    30: [10, 15, 20, 30, 50, 75],
}

TEST_GRID_SEM = {
    7:  [10, 20, 30, 50],
    10: [10, 20, 30, 50],
    15: [10, 20, 30, 50],
    20: [10, 20, 30, 50],
}

DEFAULT_TRIALS = 100   # per cell per condition
MIN_TRIALS     = 30
CI_THRESHOLD   = 0.07

BATCH_TRIALS   = 10


# ─── helpers ──────────────────────────────────────────────────────────────────

def wilson_hw(n, k, z=1.96):
    if n == 0: return 1.0
    n_t = n + z**2
    p_t = (k + z**2 / 2) / n_t
    return z * math.sqrt(p_t * (1 - p_t) / n_t)


def converged(ri_results, pi_results, threshold=CI_THRESHOLD, min_n=MIN_TRIALS):
    for results in (ri_results, pi_results):
        n = len(results)
        if n < min_n: return False
        k = sum(results)
        if wilson_hw(n, k) > threshold: return False
    return True


def is_correct(predicted, expected):
    pred = predicted.lower().strip()
    exp  = expected.lower().strip()
    return pred == exp or exp in pred or pred.startswith(exp)


def shuffle_no_consecutive(items, rng):
    for _ in range(100):
        c = items.copy()
        rng.shuffle(c)
        if all(c[i]["category"] != c[i-1]["category"] for i in range(1, len(c))):
            return c
    result, remaining = [], items.copy()
    rng.shuffle(remaining)
    last_cat = None
    while remaining:
        valid = [i for i, x in enumerate(remaining) if x["category"] != last_cat]
        if not valid:
            result.extend(remaining)
            break
        idx = rng.choice(valid)
        result.append(remaining.pop(idx))
        last_cat = result[-1]["category"]
    return result


def build_prompt(tokenizer, dataset_type, categories, num_updates, condition, seed):
    rng = random.Random(seed)
    try:
        values_per_cat = generate_values_for_trial(dataset_type, categories, num_updates, rng)
    except ValueError:
        return None, None

    test_category = categories[seed % len(categories)]
    items = []
    for cat in categories:
        for val in values_per_cat[cat]:
            items.append({"category": cat, "value": val})
    items = shuffle_no_consecutive(items, rng)

    stream = "\n".join(f"{it['category']}: {it['value']}" for it in items)
    cat_values = [it["value"] for it in items if it["category"] == test_category]

    query_word = "first" if condition == "RI" else "last"
    expected   = cat_values[0] if condition == "RI" else cat_values[-1]

    user_text = (
        f"Read the following key-value stream. Each key gets updated multiple times.\n\n"
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


# ─── merge LoRA + save to temp dir ────────────────────────────────────────────

def _model_class_for(model_id: str):
    """Gemma-3 ships as VLM via Auto*; pick text-only causal head explicitly."""
    from transformers import AutoModelForCausalLM
    if "gemma-3" in model_id.lower():
        try:
            from transformers import Gemma3ForCausalLM
            return Gemma3ForCausalLM
        except ImportError:
            pass
    return AutoModelForCausalLM


def merge_and_save(adapter_path: Path, tmp_dir: str, base_model_id: str):
    """Merge LoRA adapter into base model and save for vLLM loading."""
    from peft import PeftModel
    from transformers import AutoTokenizer

    print(f"  Loading base model {base_model_id}...", flush=True)
    model_cls = _model_class_for(base_model_id)
    model = model_cls.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16, device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

    print(f"  Loading adapter from {adapter_path}...", flush=True)
    model = PeftModel.from_pretrained(model, str(adapter_path))

    print("  Merging weights...", flush=True)
    model = model.merge_and_unload()

    print(f"  Saving merged model to {tmp_dir}...", flush=True)
    model.save_pretrained(tmp_dir)
    tokenizer.save_pretrained(tmp_dir)
    print("  Merge complete.", flush=True)


# ─── vLLM engine ──────────────────────────────────────────────────────────────

class VLLMEngine:
    def __init__(self, model_path: str, max_len: int = 8192):
        from vllm import LLM
        print(f"  Loading vLLM engine from {model_path}...", flush=True)
        self.llm = LLM(
            model=model_path,
            dtype="bfloat16",
            max_model_len=max_len,
            trust_remote_code=True,
            gpu_memory_utilization=0.92,
            enable_prefix_caching=True,
        )
        print("  ✓ vLLM ready", flush=True)

    def generate(self, prompts, max_new_tokens=8):
        from vllm import SamplingParams
        sp = SamplingParams(temperature=0, max_tokens=max_new_tokens)
        results = self.llm.generate(prompts, sp)
        return [r.outputs[0].text.strip() for r in results]


# ─── cell evaluation ──────────────────────────────────────────────────────────

def eval_cell(engine, tokenizer, dataset_type, nk, nu, max_trials):
    ri_correct, pi_correct = [], []
    trial_idx = 0
    stopped_early = False

    eligible = get_eligible_categories(dataset_type, min_values=nu)
    if len(eligible) < nk:
        return None

    while trial_idx < max_trials:
        batch_end = min(trial_idx + BATCH_TRIALS, max_trials)

        prompts, meta = [], []
        for t in range(trial_idx, batch_end):
            for condition in ["RI", "PI"]:
                seed = hash((dataset_type, "eval", nk, nu, condition, t)) % (2**31)
                rng  = random.Random(seed)
                cats = rng.sample(eligible, nk)
                prompt, expected = build_prompt(tokenizer, dataset_type, cats, nu, condition, seed)
                if prompt is None:
                    continue
                prompts.append(prompt)
                meta.append((condition, expected))

        responses = engine.generate(prompts)

        for (condition, expected), predicted in zip(meta, responses):
            correct = is_correct(predicted, expected)
            if condition == "RI":
                ri_correct.append(correct)
            else:
                pi_correct.append(correct)

        trial_idx = batch_end
        if converged(ri_correct, pi_correct):
            stopped_early = True
            break

    if not ri_correct or not pi_correct:
        return None

    def stats(corrects):
        n = len(corrects); k = sum(corrects)
        acc = k / n if n else 0.0
        hw  = wilson_hw(n, k)
        return {"accuracy": round(acc, 4),
                "ci_lower": round(max(0, acc - hw), 4),
                "ci_upper": round(min(1, acc + hw), 4),
                "n": n}

    ri_s = stats(ri_correct)
    pi_s = stats(pi_correct)
    gap  = round(ri_s["accuracy"] - pi_s["accuracy"], 4)

    if pi_s["accuracy"] > ri_s["accuracy"]:   regime = "D"
    elif gap >= 0.25:                          regime = "C"
    elif gap >= 0.15:                          regime = "B"
    elif gap < 0.05:                           regime = "A"
    else:                                      regime = "AB"

    return {
        "num_keys": nk, "num_updates": nu, "dataset": dataset_type,
        "stats": {"RI": ri_s, "PI": pi_s},
        "gap": gap, "regime": regime,
        "n_trials": len(ri_correct), "stopped_early": stopped_early,
    }


# ─── comparison table ─────────────────────────────────────────────────────────

def load_baseline(baseline_path: Path):
    if not baseline_path or not baseline_path.exists():
        return {}
    with open(baseline_path) as f:
        d = json.load(f)
    # Support both eval-format (results.ARBITRARY_SINGLE) and stage1-format (cells)
    if "results" in d and "ARBITRARY_SINGLE" in d.get("results", {}):
        return d["results"]["ARBITRARY_SINGLE"]
    return d.get("cells", {})


def print_comparison(eval_results, baseline_cells, run_name):
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"COMPARISON: baseline vs {run_name}")
    lines.append(f"{'Cell':>8}  {'BSE RI':>6} {'BSE PI':>6} {'BSE gap':>7}  "
                 f"{'FT  RI':>6} {'FT  PI':>6} {'FT  gap':>7}  {'verdict':>12}")
    lines.append("-" * 80)

    n_fixed = n_worse = n_same = 0
    for cell_key, cell in sorted(eval_results.items(),
                                  key=lambda x: (int(x[0].split("_")[0]),
                                                  int(x[0].split("_")[1]))):
        nk, nu = cell_key.split("_")
        b = baseline_cells.get(cell_key)

        ri_ft  = cell["stats"]["RI"]["accuracy"]
        pi_ft  = cell["stats"]["PI"]["accuracy"]
        gap_ft = cell["gap"]

        if b:
            ri_b   = b["stats"]["RI"]["accuracy"]
            pi_b   = b["stats"]["PI"]["accuracy"]
            gap_b  = ri_b - pi_b
            b_reg  = b["regime"]
        else:
            ri_b = pi_b = gap_b = None
            b_reg = "N/A"

        # Success: both RI≥0.65 AND PI≥0.65
        fixed = ri_ft >= 0.65 and pi_ft >= 0.65
        # Shortcut: only PI rose while RI dropped
        shortcut = (b and pi_ft > pi_b + 0.10 and ri_ft < ri_b - 0.10)
        # Worse: gap widened
        worse = (b and gap_ft < gap_b - 0.10)

        if fixed:
            verdict = "✓ FIXED"
            n_fixed += 1
        elif shortcut:
            verdict = "⚠ SHORTCUT"
            n_worse += 1
        elif worse:
            verdict = "↓ WORSE"
            n_worse += 1
        elif b and (gap_ft > gap_b + 0.05):
            verdict = "↑ improved"
        else:
            verdict = "~ same"
            n_same += 1

        b_str = f"{ri_b:.0%}  {pi_b:.0%}  {gap_b:+.0%}" if b else "  ---    ---    ---"
        lines.append(f"{nk:>2}k_{nu:>3}u  {b_str}   {ri_ft:.0%}   {pi_ft:.0%}  {gap_ft:+.0%}   {verdict}")

    lines.append(f"\nFixed: {n_fixed}  Worse: {n_worse}  Same/Improved: {n_same}")
    lines.append("=" * 80)
    text = "\n".join(lines)
    print(text)
    return text


# ─── CLI + main ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="main")
    p.add_argument("--model",    default=MODEL_ID,
                   help=f"Base HF model id (default: {MODEL_ID})")
    p.add_argument("--adapter",  default=None,
                   help="Path to adapter (default: checkpoints/<run_name>/final). "
                        "Use 'none' to evaluate the base model with no adapter.")
    p.add_argument("--baseline-path", default=str(DEFAULT_BASELINE),
                   help="Path to baseline JSON for comparison table. Empty to skip.")
    p.add_argument("--trials",   type=int, default=DEFAULT_TRIALS)
    p.add_argument("--skip-ood", action="store_true",
                   help="Skip SEMANTIC_MULTI OOD evaluation")
    p.add_argument("--cells",    default=None,
                   help="Override ARB grid with comma-separated K_N cells, e.g. "
                        "'10_50,15_20,20_30,25_75,30_75' for a minimal control run.")
    p.add_argument("--smoke",    action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    use_adapter = args.adapter != "none"
    adapter_path = None
    if use_adapter:
        adapter_path = Path(args.adapter) if args.adapter else CKPT_DIR / args.run_name / "final"
        if not adapter_path.exists():
            print(f"ERROR: adapter not found at {adapter_path}")
            sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    max_trials = 5 if args.smoke else args.trials

    print("=" * 60)
    print(f"LoRA Evaluation — {args.run_name}")
    print(f"  Base model: {args.model}")
    print(f"  Adapter:    {adapter_path if use_adapter else 'NONE (baseline eval)'}")
    print(f"  Trials:     up to {max_trials} per cell per condition")
    print(f"  Smoke:      {args.smoke}")
    print("=" * 60)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # ── Load engine (merged if adapter, base otherwise) ───────────────────────
    tmp = None
    if use_adapter:
        tmp = tempfile.TemporaryDirectory()
        merge_and_save(adapter_path, tmp.name, args.model)
        engine = VLLMEngine(tmp.name, max_len=16384)
    else:
        engine = VLLMEngine(args.model, max_len=16384)

    all_results = {}

    # Optionally narrow the ARB grid
    arb_grid = dict(TEST_GRID_ARBI)
    if args.cells:
        from collections import defaultdict
        narrowed = defaultdict(list)
        for spec in args.cells.split(","):
            k, n = spec.strip().split("_")
            narrowed[int(k)].append(int(n))
        arb_grid = {k: sorted(v) for k, v in narrowed.items()}
        print(f"\nUsing custom grid: {arb_grid}")

    # ── ARBITRARY_SINGLE (in-distribution, held-out cells) ────────────────────
    print("\n--- ARBITRARY_SINGLE (held-out cells) ---")
    arbi_results = {}
    for nk, nu_list in arb_grid.items():
        for nu in nu_list:
            cell_key = f"{nk}_{nu}"
            print(f"  K={nk:>2}, N={nu:>3}...", end=" ", flush=True)
            cell = eval_cell(engine, tokenizer, "ARBITRARY_SINGLE", nk, nu, max_trials)
            if cell:
                arbi_results[cell_key] = cell
                ri = cell["stats"]["RI"]["accuracy"]
                pi = cell["stats"]["PI"]["accuracy"]
                print(f"RI={ri:.0%}  PI={pi:.0%}  gap={cell['gap']:+.0%}  {cell['regime']}")
            else:
                print("SKIPPED")
    all_results["ARBITRARY_SINGLE"] = arbi_results

    # ── SEMANTIC_MULTI (OOD) ──────────────────────────────────────────────────
    if not args.skip_ood:
        print("\n--- SEMANTIC_MULTI (OOD) ---")
        sem_results = {}
        for nk, nu_list in TEST_GRID_SEM.items():
            for nu in nu_list:
                cell_key = f"{nk}_{nu}"
                print(f"  K={nk:>2}, N={nu:>3}...", end=" ", flush=True)
                cell = eval_cell(engine, tokenizer, "SEMANTIC_MULTI", nk, nu, max_trials)
                if cell:
                    sem_results[cell_key] = cell
                    ri = cell["stats"]["RI"]["accuracy"]
                    pi = cell["stats"]["PI"]["accuracy"]
                    print(f"RI={ri:.0%}  PI={pi:.0%}  gap={cell['gap']:+.0%}  {cell['regime']}")
                else:
                    print("SKIPPED")
        all_results["SEMANTIC_MULTI"] = sem_results

    if tmp is not None:
        tmp.cleanup()

    # ── Comparison vs baseline ────────────────────────────────────────────────
    baseline_path = Path(args.baseline_path) if args.baseline_path else None
    baseline = load_baseline(baseline_path)
    comparison_text = print_comparison(arbi_results, baseline, args.run_name)

    # ── Save ──────────────────────────────────────────────────────────────────
    out = {
        "run_name":    args.run_name,
        "model":       args.model,
        "adapter":     str(adapter_path) if adapter_path else None,
        "timestamp":   ts,
        "max_trials":  max_trials,
        "results":     all_results,
    }
    out_path = RESULTS_DIR / f"{args.run_name}_eval_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    cmp_path = RESULTS_DIR / f"{args.run_name}_comparison.txt"
    with open(cmp_path, "w") as f:
        f.write(comparison_text)

    print(f"\nSaved → {out_path}")
    print(f"Saved → {cmp_path}")


if __name__ == "__main__":
    main()
