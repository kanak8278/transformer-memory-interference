"""
Quick intermediate probe — load a LoRA checkpoint and test on ~6 key cells
without running full vLLM (uses HF inference for speed).

Cells chosen to span the interesting space:
  - K=10, N=15 (regime D in training range — should be fixed if anything is)
  - K=10, N=30 (held-out, moderate reversal)
  - K=15, N=20 (held-out, strong reversal)
  - K=15, N=50 (held-out, very strong reversal)
  - K=20, N=30 (held-out, extreme reversal)
  - K=2,  N=30 (regime C control — should still be RI > PI)

Usage:
    python lora_intervention/quick_probe.py --checkpoint checkpoint-100
    python lora_intervention/quick_probe.py --checkpoint final
"""
import sys, os, json, random, argparse, math
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# patch broken tensorflow version string
import importlib.metadata as _imeta
_orig = _imeta.version
def _safe(pkg):
    try:
        v = _orig(pkg); return v if v is not None else "0.0.0"
    except: return "0.0.0"
_imeta.version = _safe

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from mechanistic_probing_v2.core.dataset_configs import (
    get_eligible_categories, generate_values_for_trial,
)

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
CKPT_DIR = Path(__file__).parent / "checkpoints" / "main"
SYSTEM_PROMPT = (
    "You are a precise data extraction tool. "
    "Output ONLY a single word - the exact value requested. "
    "No other text, no explanation, no punctuation."
)
BASELINE_PATH = _ROOT / "v3/results_vllm/arbitrary_single/Qwen2.5-3B-Instruct/stage1_sweep_20260409_000134.json"

PROBE_CELLS = [
    # (K, N, label)
    (10, 15, "train-D"),    # regime D, in training range
    (10, 30, "held-D"),     # held-out, moderate reversal
    (15, 20, "held-D"),     # held-out, strong reversal
    (15, 50, "held-D"),     # held-out, very strong reversal
    (20, 30, "held-D"),     # held-out, extreme reversal
    (2,  30, "ctrl-C"),     # control: regime C, should stay RI>PI
]
TRIALS_PER_CELL = 30   # enough for a signal, fast to run


def shuffle_no_consecutive(items, rng):
    for _ in range(100):
        c = items.copy(); rng.shuffle(c)
        if all(c[i]["category"] != c[i-1]["category"] for i in range(1, len(c))):
            return c
    result, remaining = [], items.copy(); rng.shuffle(remaining); last = None
    while remaining:
        valid = [i for i,x in enumerate(remaining) if x["category"] != last]
        if not valid: result.extend(remaining); break
        idx = rng.choice(valid); result.append(remaining.pop(idx)); last = result[-1]["category"]
    return result


def make_prompt(tokenizer, nk, nu, condition, seed):
    rng = random.Random(seed)
    eligible = get_eligible_categories("ARBITRARY_SINGLE", min_values=nu)
    cats = rng.sample(eligible, nk)
    try:
        vals = generate_values_for_trial("ARBITRARY_SINGLE", cats, nu, rng)
    except ValueError:
        return None, None
    test_cat = cats[seed % nk]
    items = []
    for cat in cats:
        for v in vals[cat]:
            items.append({"category": cat, "value": v})
    items = shuffle_no_consecutive(items, rng)
    stream = "\n".join(f"{it['category']}: {it['value']}" for it in items)
    cat_vals = [it["value"] for it in items if it["category"] == test_cat]
    qword = "first" if condition == "RI" else "last"
    expected = cat_vals[0] if condition == "RI" else cat_vals[-1]
    user_text = (
        f"Read the following key-value stream. Each key gets updated multiple times.\n\n"
        f"{stream}\n\nWhat was the {qword} value of {test_cat}?"
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_text}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt, expected


def is_correct(pred, exp):
    p, e = pred.lower().strip(), exp.lower().strip()
    return p == e or e in p or p.startswith(e)


def run_cell(model, tokenizer, device, nk, nu, n_trials):
    ri_correct, pi_correct = [], []
    for t in range(n_trials):
        for condition in ["RI", "PI"]:
            seed = hash(("probe", nk, nu, condition, t)) % (2**31)
            prompt, expected = make_prompt(tokenizer, nk, nu, condition, seed)
            if prompt is None:
                continue
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=8, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = out[0][inputs["input_ids"].shape[1]:]
            pred = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            correct = is_correct(pred, expected)
            (ri_correct if condition == "RI" else pi_correct).append(correct)
    ri = sum(ri_correct)/len(ri_correct) if ri_correct else 0
    pi = sum(pi_correct)/len(pi_correct) if pi_correct else 0
    return ri, pi


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoint-100",
                   help="Checkpoint name under checkpoints/main/ (default: checkpoint-100)")
    p.add_argument("--trials", type=int, default=TRIALS_PER_CELL)
    return p.parse_args()


def main():
    args = parse_args()
    ckpt_path = CKPT_DIR / args.checkpoint

    print(f"Loading baseline...")
    with open(BASELINE_PATH) as f:
        baseline = json.load(f)["cells"]

    print(f"Loading base model + adapter from {ckpt_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    # Force CPU if GPU is occupied by training
    if device == "cuda":
        free = torch.cuda.mem_get_info()[0]
        if free < 8 * 1024**3:  # less than 8GB free
            print(f"  GPU has only {free/1024**3:.1f}GB free — falling back to CPU")
            device = "cpu"

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
        device_map=device, trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, str(ckpt_path), device_map=device)
    model.eval()
    print(f"  Device: {device}  |  Trials: {args.trials} per cell\n")

    print(f"{'Cell':>10}  {'label':>8}  "
          f"{'BASE RI':>7} {'BASE PI':>7} {'BASE gap':>8}  "
          f"{'FT  RI':>7} {'FT  PI':>7} {'FT  gap':>8}  {'verdict':>12}")
    print("-" * 80)

    for nk, nu, label in PROBE_CELLS:
        ck = f"{nk}_{nu}"
        b = baseline.get(ck)
        b_ri = b["stats"]["RI"]["accuracy"] if b else None
        b_pi = b["stats"]["PI"]["accuracy"] if b else None
        b_gap = (b_ri - b_pi) if b else None

        ri, pi = run_cell(model, tokenizer, device, nk, nu, args.trials)
        gap = ri - pi

        fixed    = ri >= 0.65 and pi >= 0.65
        shortcut = (b and pi > b_pi + 0.10 and ri < b_ri - 0.10)
        if fixed:            verdict = "✓ FIXED"
        elif shortcut:       verdict = "⚠ SHORTCUT"
        elif b and gap > b_gap + 0.10: verdict = "↑ improved"
        elif b and gap < b_gap - 0.10: verdict = "↓ worse"
        else:                verdict = "~ same"

        b_str = f"{b_ri:.0%}   {b_pi:.0%}   {b_gap:+.0%}" if b else "  ---     ---     ---"
        print(f"{nk:>2}k_{nu:>3}u  {label:>8}  "
              f"{b_str}   {ri:.0%}   {pi:.0%}  {gap:+.0%}   {verdict}")

    print("-" * 80)
    print(f"\nCheckpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
