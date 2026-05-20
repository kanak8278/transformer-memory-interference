#!/usr/bin/env python3
"""
U-curve positional + format sweep — local models via HuggingFace / vLLM.

Same logic as ucurve_sweep.py but uses batch inference instead of threaded API calls.
Backend auto-selects:
  • vLLM  — if CUDA is available (NVIDIA GPU)
  • HF    — if MPS or CPU (Apple Silicon / CPU fallback)

Usage:
  # Quick smoke test (5 trials, 2 positions)
  python ucurve_vllm.py --model Qwen2.5-3B-Instruct --nk 5 --nu 10 --trials 5 --smoke

  # Run specific formats on a model
  python ucurve_vllm.py --model Qwen2.5-3B-Instruct \\
      --formats block flat_nolabel flat_verbose flat_nolabel_last \\
      --nk 5 10 --nu 10 20 50 --trials 50

  # Resume from checkpoint
  python ucurve_vllm.py --model Qwen2.5-3B-Instruct --resume

  # Force HF backend (even on CUDA)
  python ucurve_vllm.py --model Qwen2.5-3B-Instruct --backend hf

MODEL REGISTRY (add new models here):
  Key = short alias used with --model
  hf_id     = HuggingFace model id
  instruct  = True if chat-template model; False = completion/base
  dtype     = torch dtype for loading (bfloat16 recommended for modern models)
"""

import sys, os, json, math, time, argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments_cloud.ucurve_prompts import (
    generate_stream, make_seed, query_positions,
    prompt_block, prompt_flat_short, prompt_flat_verbose,
    prompt_flat_nolabel, prompt_original, prompt_landmark,
    LASTQUERY_BUILDERS,
    SYSTEM_PROMPT,
)

# ─── model registry ───────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    # Qwen2.5 family
    "Qwen2.5-0.5B-Instruct": {
        "hf_id":    "Qwen/Qwen2.5-0.5B-Instruct",
        "instruct": True,
        "dtype":    "bfloat16",
        "max_len":  8192,
    },
    "Qwen2.5-1.5B-Instruct": {
        "hf_id":    "Qwen/Qwen2.5-1.5B-Instruct",
        "instruct": True,
        "dtype":    "bfloat16",
        "max_len":  8192,
    },
    "Qwen2.5-3B": {
        "hf_id":    "Qwen/Qwen2.5-3B",
        "instruct": False,
        "dtype":    "bfloat16",
        "max_len":  16384,
    },
    "Qwen2.5-3B-Instruct": {
        "hf_id":    "Qwen/Qwen2.5-3B-Instruct",
        "instruct": True,
        "dtype":    "bfloat16",
        "max_len":  16384,
    },
    # Qwen3.5 family
    "Qwen3.5-0.8B": {
        "hf_id":    "Qwen/Qwen3.5-0.8B",
        "instruct": False,
        "dtype":    "bfloat16",
        "max_len":  32768,
    },
    "Qwen3.5-2B": {
        "hf_id":    "Qwen/Qwen3.5-2B",
        "instruct": False,
        "dtype":    "bfloat16",
        "max_len":  32768,
    },
    "Qwen3.5-4B": {
        "hf_id":    "Qwen/Qwen3.5-4B",
        "instruct": False,
        "dtype":    "bfloat16",
        "max_len":  32768,
    },
    "Qwen3.5-9B": {
        "hf_id":    "Qwen/Qwen3.5-9B",
        "instruct": False,
        "dtype":    "bfloat16",
        "max_len":  32768,
    },
    # Gemma-3 family
    "gemma-3-270m-it": {
        "hf_id":    "google/gemma-3-270m-it",
        "instruct": True,
        "dtype":    "bfloat16",
        "max_len":  8192,
    },
    "gemma-3-1b-it": {
        "hf_id":    "google/gemma-3-1b-it",
        "instruct": True,
        "dtype":    "bfloat16",
        "max_len":  8192,
    },
    "gemma-3-4b-it": {
        "hf_id":    "google/gemma-3-4b-it",
        "instruct": True,
        "dtype":    "bfloat16",
        "max_len":  8192,
    },
}

# ─── format definitions ───────────────────────────────────────────────────────
_LAST_FMTS = {
    "block_last", "flat_short_last", "flat_verbose_last",
    "flat_nolabel_last", "landmark_last",
}
_STREAM_COUNT_FMTS = {"flat_nolabel", "flat_nolabel_last"}

ALL_FORMATS = [
    "block", "flat_short", "flat_verbose", "flat_nolabel", "landmark",
    "block_last", "flat_short_last", "flat_verbose_last",
    "flat_nolabel_last", "landmark_last",
]
DEFAULT_FORMATS = [
    "block", "flat_nolabel", "flat_verbose", "landmark",
    "block_last", "flat_nolabel_last",
]

PROMPT_BUILDERS = {
    "block":             lambda flat, block, test, k: prompt_block(block, test, k),
    "flat_short":        lambda flat, block, test, k: prompt_flat_short(flat, test, k),
    "flat_verbose":      lambda flat, block, test, k: prompt_flat_verbose(flat, test, k),
    "flat_nolabel":      lambda flat, block, test, k: prompt_flat_nolabel(flat, test, k),
    "original":          lambda flat, block, test, k: prompt_original(flat, test, k),
    "landmark":          lambda flat, block, test, k: prompt_landmark(block, test, k),
    "block_last":        lambda flat, block, test, k: prompt_block(block, test, k),
    "flat_short_last":   lambda flat, block, test, k: prompt_flat_short(flat, test, k),
    "flat_verbose_last": lambda flat, block, test, k: prompt_flat_verbose(flat, test, k),
    "flat_nolabel_last": lambda flat, block, test, k: prompt_flat_nolabel(flat, test, k),
    "landmark_last":     lambda flat, block, test, k: prompt_landmark(block, test, k),
}

DEFAULT_NK     = [5, 10]
DEFAULT_NU     = [10, 20, 50]
DEFAULT_TRIALS = 50
SAVE_BASE      = "experiments_cloud/results/ucurve_vllm"


# ─── inference backends ───────────────────────────────────────────────────────
class HFEngine:
    """HuggingFace transformers batch inference — works on MPS and CPU."""

    def __init__(self, hf_id: str, dtype_str: str, max_len: int, batch_size: int):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}.get(dtype_str, torch.bfloat16)

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"  Loading {hf_id} on {self.device} ({dtype_str})...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # for causal LM batch generation

        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=dtype,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()
        self.max_len   = max_len
        self.batch_size = batch_size
        print(f"  ✓ Loaded ({sum(p.numel() for p in self.model.parameters())/1e6:.0f}M params)")

    def generate_batch(self, prompts: list[str], max_new_tokens: int = 8) -> list[str]:
        """Generate one response per prompt. Returns list of raw strings."""
        outputs = []
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i : i + self.batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_len - max_new_tokens,
            ).to(self.device)

            with torch.no_grad():
                gen = self.model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,          # greedy
                    temperature=1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Decode only the newly generated tokens
            input_len = enc["input_ids"].shape[1]
            for seq in gen:
                new_tokens = seq[input_len:]
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                outputs.append(text.strip())

        return outputs


class VLLMEngine:
    """vLLM offline inference — NVIDIA GPU only."""

    def __init__(self, hf_id: str, dtype_str: str, max_len: int, batch_size: int):
        from vllm import LLM, SamplingParams

        print(f"  Loading {hf_id} via vLLM ({dtype_str})...", flush=True)
        self.llm = LLM(
            model=hf_id,
            dtype=dtype_str,
            max_model_len=max_len,
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
            enforce_eager=False,
            enable_prefix_caching=True,
        )
        self.sampling = SamplingParams(
            temperature=0,
            max_tokens=8,
            stop=None,
        )
        self.batch_size = batch_size  # unused for vLLM (it handles internally)
        print("  ✓ vLLM engine ready")

    def generate_batch(self, prompts: list[str], max_new_tokens: int = 8) -> list[str]:
        from vllm import SamplingParams
        sp = SamplingParams(temperature=0, max_tokens=max_new_tokens)
        results = self.llm.generate(prompts, sp)
        return [r.outputs[0].text.strip() for r in results]


def build_engine(model_alias: str, backend: str, batch_size: int) -> tuple:
    """Returns (engine, cfg, is_instruct)."""
    cfg = MODEL_REGISTRY[model_alias]
    hf_id    = cfg["hf_id"]
    dtype    = cfg["dtype"]
    max_len  = cfg["max_len"]
    instruct = cfg["instruct"]

    if backend == "auto":
        backend = "vllm" if torch.cuda.is_available() else "hf"

    print(f"\nBackend: {backend.upper()}")
    if backend == "vllm":
        engine = VLLMEngine(hf_id, dtype, max_len, batch_size)
    else:
        engine = HFEngine(hf_id, dtype, max_len, batch_size)

    return engine, cfg, instruct


# ─── prompt formatting (instruct vs base) ────────────────────────────────────
def apply_chat_template(tokenizer, user_text: str) -> str:
    """Wrap user_text in the model's chat template. Returns full formatted string."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_text},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def format_prompt_for_model(raw_prompt: str, instruct: bool,
                             tokenizer=None) -> str:
    """For instruct models: apply chat template. For base models: use as-is."""
    if instruct and tokenizer is not None:
        return apply_chat_template(tokenizer, raw_prompt)
    return raw_prompt


# ─── error classification (same as ucurve_sweep.py) ──────────────────────────
def classify(predicted: str, expected: str, all_values: list[str]) -> dict:
    pred = predicted.lower().strip()
    exp  = expected.lower().strip()
    if pred == exp or exp in pred or pred.startswith(exp):
        return {"error_type": "correct",
                "predicted_idx": all_values.index(expected),
                "predicted_relative_pos": round(
                    all_values.index(expected) / max(len(all_values) - 1, 1), 4)}
    for idx, val in enumerate(all_values):
        v = val.lower()
        if v == pred or v in pred or pred.startswith(v):
            return {"error_type": "in_sequence",
                    "predicted_idx": idx,
                    "predicted_relative_pos": round(idx / max(len(all_values) - 1, 1), 4)}
    return {"error_type": "garbage", "predicted_idx": None, "predicted_relative_pos": None}


def wilson_hw(n: int, k: int, z: float = 1.96) -> float:
    if n == 0: return 1.0
    n_t = n + z**2
    p_t = (k + z**2 / 2) / n_t
    return z * math.sqrt(p_t * (1 - p_t) / n_t)


# ─── cell runner ──────────────────────────────────────────────────────────────
def run_cell(engine, fmt_name: str, nk: int, nu: int,
             max_trials: int, positions: list[int],
             instruct: bool, tokenizer, dataset: str) -> tuple[dict, list]:
    """
    Builds all prompts for the cell, fires them in one batched generate call.
    Returns (cell_summary, trial_details).
    """
    is_last_fmt    = fmt_name in _LAST_FMTS
    is_stream_cnt  = fmt_name in _STREAM_COUNT_FMTS
    builder        = PROMPT_BUILDERS[fmt_name]
    lq_builder     = LASTQUERY_BUILDERS.get(fmt_name)
    all_pos_keys   = list(positions) + (["last"] if is_last_fmt else [])

    # ── build all prompts upfront ─────────────────────────────────────────────
    # Layout: for each trial → for each position (+ optional last query)
    trial_meta   = []   # (trial_idx, seed, test_cat, stream_vals, all_values)
    prompt_list  = []   # flat list of formatted prompts
    pos_key_list = []   # parallel to prompt_list: which (trial_idx, pos_key)

    for trial_idx in range(max_trials):
        seed = make_seed(nk, nu, trial_idx)
        cats, test, vals, flat, block = generate_stream(nk, nu, seed, dataset=dataset)
        all_values   = vals[test]
        stream_vals  = [i["value"] for i in flat if i["category"] == test]

        trial_meta.append({
            "trial_idx":     trial_idx,
            "seed":          seed,
            "test_category": test,
            "categories":    cats,
            "all_values":    all_values,
            "stream_vals":   stream_vals,
        })

        # Positional queries
        for k in positions:
            raw = builder(flat, block, test, k)
            prompt_list.append(format_prompt_for_model(raw, instruct, tokenizer))
            pos_key_list.append((trial_idx, k))

        # Optional semantic "last" query
        if is_last_fmt and lq_builder:
            items = block if ("block" in fmt_name or "landmark" in fmt_name) else flat
            raw_last = lq_builder(items, test)
            prompt_list.append(format_prompt_for_model(raw_last, instruct, tokenizer))
            pos_key_list.append((trial_idx, "last"))

    # ── batch inference ───────────────────────────────────────────────────────
    responses = engine.generate_batch(prompt_list, max_new_tokens=8)

    # ── parse responses back into trial structure ─────────────────────────────
    trial_details       = []
    correct_by_pos      = {k: [] for k in all_pos_keys}

    # index responses by (trial_idx, pos_key)
    resp_map = {}
    for (ti, pk), resp in zip(pos_key_list, responses):
        resp_map[(ti, pk)] = resp

    for meta in trial_meta:
        ti          = meta["trial_idx"]
        all_values  = meta["all_values"]
        stream_vals = meta["stream_vals"]
        test        = meta["test_category"]

        pos_results = {}
        for pk in all_pos_keys:
            raw = resp_map.get((ti, pk), "")
            # Determine expected
            if pk == "last":
                expected = stream_vals[-1] if is_stream_cnt else all_values[-1]
                ref_list = stream_vals    if is_stream_cnt else all_values
            elif fmt_name == "original":
                expected = stream_vals[0] if pk == 1 else stream_vals[-1]
                ref_list = stream_vals
            elif is_stream_cnt:
                expected = stream_vals[pk - 1]
                ref_list = stream_vals
            else:
                expected = all_values[pk - 1]
                ref_list = all_values

            cls = classify(raw, expected, ref_list)
            correct = cls["error_type"] == "correct"
            correct_by_pos[pk].append(correct)

            pos_results[pk] = {
                "expected":               expected,
                "predicted":              raw,
                "correct":                correct,
                "error_type":             cls["error_type"],
                "predicted_idx":          cls["predicted_idx"],
                "predicted_relative_pos": cls["predicted_relative_pos"],
            }

        trial_details.append({
            "trial_idx":     ti,
            "seed":          meta["seed"],
            "test_category": test,
            "categories":    meta["categories"],
            "positions":     pos_results,
        })

    # ── build position stats ──────────────────────────────────────────────────
    position_stats = {}
    for pk in all_pos_keys:
        corrects = correct_by_pos[pk]
        n = len(corrects)
        k = sum(corrects)
        acc = k / n if n else 0.0
        hw  = wilson_hw(n, k)
        position_stats[str(pk)] = {
            "accuracy":  round(acc, 4),
            "ci_lower":  round(max(0, acc - hw), 4),
            "ci_upper":  round(min(1, acc + hw), 4),
            "wilson_hw": round(hw, 4),
            "n":         n,
        }

    pos_first = str(positions[0])
    pos_last  = str(positions[-1])
    cell_summary = {
        "num_keys":    nk,
        "num_updates": nu,
        "format":      fmt_name,
        "n_trials":    max_trials,
        "positions":   position_stats,
        "first_acc":   position_stats[pos_first]["accuracy"],
        "last_acc":    position_stats[pos_last]["accuracy"],
        "gap":         round(
            position_stats[pos_first]["accuracy"] -
            position_stats[pos_last]["accuracy"], 4),
    }
    return cell_summary, trial_details


# ─── checkpoint helpers ───────────────────────────────────────────────────────
def save_checkpoint(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, str(path))


def cell_done(results: dict, fmt: str, nk: int, nu: int) -> bool:
    return fmt in results.get("cells", {}).get(f"{nk}_{nu}", {})


# ─── CLI ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="U-curve positional+format sweep — local models (HF/vLLM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model",   required=True,
                   choices=list(MODEL_REGISTRY.keys()),
                   help="Model alias from MODEL_REGISTRY")
    p.add_argument("--formats", nargs="+", default=DEFAULT_FORMATS,
                   choices=ALL_FORMATS, metavar="FMT",
                   help=f"Formats to run (default: {DEFAULT_FORMATS})")
    p.add_argument("--nk",      nargs="+", type=int, default=DEFAULT_NK,
                   help=f"Key levels (default: {DEFAULT_NK})")
    p.add_argument("--nu",      nargs="+", type=int, default=DEFAULT_NU,
                   help=f"Update levels (default: {DEFAULT_NU})")
    p.add_argument("--trials",  type=int, default=DEFAULT_TRIALS,
                   help=f"Trials per cell (default: {DEFAULT_TRIALS})")
    p.add_argument("--n-positions", type=int, default=16,
                   help="Number of query positions per cell (default: 16)")
    p.add_argument("--dataset", default="ARBITRARY_SINGLE",
                   choices=["ARBITRARY_SINGLE", "SEMANTIC_MULTI"],
                   help="Dataset (default: ARBITRARY_SINGLE)")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Inference batch size for HF backend (default: 8)")
    p.add_argument("--backend", default="auto", choices=["auto", "hf", "vllm"],
                   help="Inference backend: auto=vLLM on CUDA else HF (default: auto)")
    p.add_argument("--save-dir", default=SAVE_BASE,
                   help=f"Output directory base (default: {SAVE_BASE})")
    p.add_argument("--resume",  action="store_true",
                   help="Resume from checkpoint if it exists")
    p.add_argument("--smoke",   action="store_true",
                   help="Smoke test: 5 trials, first 2 positions only")
    return p.parse_args()


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    if args.smoke:
        args.trials     = 5
        args.n_positions = 2
        args.nk         = [args.nk[0]]
        args.nu         = [args.nu[0]]
        args.formats    = [args.formats[0]]
        print("── SMOKE TEST MODE ──")

    save_dir  = Path(args.save_dir) / args.model
    ckpt_path = save_dir / "checkpoint.json"

    print(f"\nU-curve vLLM sweep")
    print(f"  model      : {args.model}  ({MODEL_REGISTRY[args.model]['hf_id']})")
    print(f"  dataset    : {args.dataset}")
    print(f"  formats    : {args.formats}")
    print(f"  nk grid    : {args.nk}")
    print(f"  nu grid    : {args.nu}")
    print(f"  trials     : {args.trials} per cell")
    print(f"  n_positions: {args.n_positions}")
    print(f"  batch_size : {args.batch_size}")
    print(f"  save to    : {save_dir}")

    # ── build engine ──────────────────────────────────────────────────────────
    engine, cfg, instruct = build_engine(args.model, args.backend, args.batch_size)
    tokenizer = None
    if instruct:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["hf_id"], trust_remote_code=True)
        print(f"  Chat template: {tokenizer.chat_template is not None}")

    # ── load or init results ──────────────────────────────────────────────────
    if args.resume and ckpt_path.exists():
        with open(ckpt_path) as f:
            results = json.load(f)
        print(f"\nResuming from {ckpt_path}")
    else:
        results = {
            "model":    args.model,
            "hf_id":    cfg["hf_id"],
            "instruct": instruct,
            "dataset":  args.dataset,
            "backend":  args.backend,
            "config": {
                "formats":      args.formats,
                "nk_levels":    args.nk,
                "nu_levels":    args.nu,
                "max_trials":   args.trials,
                "n_positions":  args.n_positions,
                "batch_size":   args.batch_size,
            },
            "start_time":   datetime.now(timezone.utc).isoformat(),
            "cells":        {},
            "trial_details":{},
        }

    # ── main loop ─────────────────────────────────────────────────────────────
    total_cells = len(args.nk) * len(args.nu) * len(args.formats)
    skipped     = 0
    done_cells  = 0

    pbar = tqdm(total=total_cells, desc="Cells", unit="cell")

    for nk in args.nk:
        for nu in args.nu:
            cell_key  = f"{nk}_{nu}"
            positions = query_positions(nu, n_points=args.n_positions)

            for fmt in args.formats:
                if args.resume and cell_done(results, fmt, nk, nu):
                    skipped += 1
                    done_cells += 1
                    pbar.update(1)
                    continue

                done_cells += 1
                pbar.set_postfix(nk=nk, nu=nu, fmt=fmt[:12])

                t0 = time.time()
                cell_sum, trial_det = run_cell(
                    engine    = engine,
                    fmt_name  = fmt,
                    nk        = nk,
                    nu        = nu,
                    max_trials= args.trials,
                    positions = positions,
                    instruct  = instruct,
                    tokenizer = tokenizer,
                    dataset   = args.dataset,
                )
                elapsed = time.time() - t0

                if cell_key not in results["cells"]:
                    results["cells"][cell_key] = {}
                results["cells"][cell_key][fmt] = cell_sum

                if cell_key not in results["trial_details"]:
                    results["trial_details"][cell_key] = {}
                results["trial_details"][cell_key][fmt] = trial_det

                tqdm.write(
                    f"  [{done_cells}/{total_cells}] nk={nk} nu={nu} fmt={fmt:20s} | "
                    f"first={cell_sum['first_acc']:.3f}  "
                    f"last={cell_sum['last_acc']:.3f}  "
                    f"gap={cell_sum['gap']:+.3f}  "
                    f"({elapsed:.0f}s)"
                )

                results["last_updated"] = datetime.now(timezone.utc).isoformat()
                save_checkpoint(results, ckpt_path)
                pbar.update(1)

    pbar.close()

    results["end_time"] = datetime.now(timezone.utc).isoformat()
    save_checkpoint(results, ckpt_path)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = save_dir / f"ucurve_{ts}.json"
    save_checkpoint(results, final_path)

    print(f"\n✓ Done. {done_cells - skipped} cells computed, {skipped} skipped.")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Final file : {final_path}")


if __name__ == "__main__":
    main()
