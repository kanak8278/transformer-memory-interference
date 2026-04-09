"""
Run Stage 2 (Logit Lens) and Stage 3 (Causal) for all 11 models.

Loads each model via TransformerLens, runs Stage 2 (logit lens at 3 operating points),
then Stage 3 (attribution patching + targeted patching + ablation).

Operating points selected from Stage 1 vLLM results (ARBITRARY_SINGLE).

Usage:
    export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
    export HF_TOKEN=${HF_TOKEN}
    python v3/scripts/experiments/run_stage2_stage3_all.py --gpu 0
    python v3/scripts/experiments/run_stage2_stage3_all.py --gpu 0 --stage2-only
    python v3/scripts/experiments/run_stage2_stage3_all.py --gpu 0 --stage3-only
"""

import os
import sys
import gc
import json
import time
import argparse
import torch
from pathlib import Path

# Fix GLIBCXX + HF token
os.environ.setdefault("LD_LIBRARY_PATH", "")
if "/opt/conda/lib" not in os.environ["LD_LIBRARY_PATH"]:
    os.environ["LD_LIBRARY_PATH"] = f"/opt/conda/lib:{os.environ['LD_LIBRARY_PATH']}"
if "HF_TOKEN" not in os.environ:
    pass  # Set HF_TOKEN env var before running

# Path setup
_SCRIPT_DIR = Path(__file__).resolve().parent
_V3_SCRIPTS = _SCRIPT_DIR.parent
_REPO_ROOT = _V3_SCRIPTS.parent.parent
for p in [str(_V3_SCRIPTS), str(_REPO_ROOT), str(_SCRIPT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

RESULTS_BASE = _V3_SCRIPTS.parent / "results_vllm"

# Operating points per model — selected from Stage 1 ARBITRARY_SINGLE data.
# Format: [(keys, updates), ...] — 3 points spanning low/medium/high difficulty.
# keys=2 used for all (universally available, even for 2K-context models).
# Operating points: (keys, updates) triples — selected from Stage 1 ARBITRARY_SINGLE.
# All use keys=2 for cross-model comparability and small-context model compatibility.
# 3 points per model: low (early failure), medium (clear failure), high (saturation).
MODEL_OPERATING_POINTS = {
    # Qwen series — clean data, low garbage
    "Qwen/Qwen2.5-0.5B-Instruct":          [(2, 5), (2, 15), (2, 50)],   # +40% → +70% → +62%
    "Qwen/Qwen2.5-1.5B-Instruct":          [(2, 5), (2, 20), (2, 75)],   # +42% → +69% → +80%
    "Qwen/Qwen2.5-3B-Instruct":            [(2, 5), (2, 10), (2, 50)],   # +13% → +35%? → +70%
    "Qwen/Qwen2.5-3B":                     [(2, 5), (2, 10), (2, 50)],   # +6% → ? → +71%
    # Gemma — 270m is very weak (high garbage), 1b all regime C, 4b has nice transition
    "google/gemma-3-270m-it":               [(2, 5), (2, 7), (2, 10)],    # weak model, ~0-5% acc
    "google/gemma-3-1b-it":                 [(2, 5), (2, 15), (2, 50)],   # +59% → +68% → +66%
    "google/gemma-3-4b-it":                 [(2, 5), (2, 10), (2, 50)],   # +13% → +35% → +61%
    # Others
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0":  [(2, 5), (2, 15), (2, 50)],   # +92% → +87% → +53%
    "stabilityai/stablelm-2-1_6b-chat":     [(2, 10), (2, 15), (2, 30)],  # +15% → +58% → +84%
    "EleutherAI/pythia-410m":               [(2, 5), (2, 7), (2, 15)],    # +47% → ? → +34%
    "state-spaces/mamba-1.4b-hf":           [(2, 5), (2, 7), (2, 10)],    # +65% → ? → +59%
}

# Models where TransformerLens may not work — try anyway, skip on failure
POSSIBLY_UNSUPPORTED = {
    "google/gemma-3-270m-it",   # gemma3_text arch — TL may not support
    "google/gemma-3-1b-it",     # same
    "google/gemma-3-4b-it",     # same
    "state-spaces/mamba-1.4b-hf",  # SSM — TL has some Mamba support
}


def get_stage2_save_dir(model_name):
    from mechanistic_probing_v2.core.model_loader import model_short_name
    return RESULTS_BASE / "logit_lens" / model_short_name(model_name)


def get_stage3_save_dir(model_name):
    from mechanistic_probing_v2.core.model_loader import model_short_name
    return RESULTS_BASE / "causal" / model_short_name(model_name)


def has_stage2_results(model_name):
    d = get_stage2_save_dir(model_name)
    return d.exists() and list(d.glob("stage2_logit_lens_*.json"))


def has_stage3_results(model_name):
    d = get_stage3_save_dir(model_name)
    return d.exists() and list(d.glob("stage3_causal_*.json"))


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_stage2_for_model(model_name, gpu_idx, trials=100):
    """Run Stage 2 logit lens for a model."""
    from stage2_logit_lens import run_stage2, save_results as s2_save
    from mechanistic_probing_v2.core.model_loader import load_model, model_short_name

    save_dir = get_stage2_save_dir(model_name)
    if save_dir.exists() and list(save_dir.glob("stage2_logit_lens_*.json")):
        print(f"  SKIP Stage 2: {model_name} — already done")
        return None, None, None, None

    points = MODEL_OPERATING_POINTS.get(model_name, [(2, 5), (2, 10), (2, 20)])

    config = {
        "model": model_name,
        "points": points,
        "trials": trials,
        "gpu": gpu_idx,
        "n_ctx": 4096,  # conservative for TL
    }

    print(f"\n{'='*70}")
    print(f"STAGE 2: {model_name}")
    print(f"  Points: {points}")
    print(f"  Trials: {trials}")
    print(f"{'='*70}")

    # Load model
    model, tokenizer, info = load_model(model_name, gpu_idx=gpu_idx, n_ctx=4096)

    # Monkey-patch save_dir to use our results_vllm path
    import stage2_logit_lens
    orig_get_save_dir = stage2_logit_lens.get_save_dir
    stage2_logit_lens.get_save_dir = lambda mn: get_stage2_save_dir(mn)

    try:
        results = run_stage2(config, model=model, tokenizer=tokenizer, info=info)
    finally:
        stage2_logit_lens.get_save_dir = orig_get_save_dir

    return model, tokenizer, info, results


def run_stage3_for_model(model_name, gpu_idx, model=None, tokenizer=None, info=None, trials=50):
    """Run Stage 3 causal analysis for a model (requires Stage 2 results)."""
    from stage3_causal import run_stage3, save_results as s3_save
    from mechanistic_probing_v2.core.model_loader import load_model

    save_dir = get_stage3_save_dir(model_name)
    if save_dir.exists() and list(save_dir.glob("stage3_causal_*.json")):
        print(f"  SKIP Stage 3: {model_name} — already done")
        return

    # Find Stage 2 results
    s2_dir = get_stage2_save_dir(model_name)
    s2_files = sorted(s2_dir.glob("stage2_logit_lens_*.json")) if s2_dir.exists() else []
    if not s2_files:
        print(f"  SKIP Stage 3: {model_name} — no Stage 2 results")
        return

    stage2_path = str(s2_files[-1])

    config = {
        "model": model_name,
        "stage2_path": stage2_path,
        "point": None,  # auto-select best from Stage 2
        "trials": trials,
        "top_k": 20,
        "gpu": gpu_idx,
        "experiments": ["3A", "3B", "3C"],
    }

    print(f"\n{'='*70}")
    print(f"STAGE 3: {model_name}")
    print(f"  Stage 2: {stage2_path}")
    print(f"  Trials: {trials}")
    print(f"{'='*70}")

    if model is None:
        model, tokenizer, info = load_model(model_name, gpu_idx=gpu_idx, n_ctx=4096)

    # Monkey-patch save_dir
    import stage3_causal
    orig_get_save_dir = stage3_causal.get_save_dir
    stage3_causal.get_save_dir = lambda mn: get_stage3_save_dir(mn)

    try:
        run_stage3(config, model=model, tokenizer=tokenizer, info=info)
    finally:
        stage3_causal.get_save_dir = orig_get_save_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--trials-s2", type=int, default=100, help="Trials for Stage 2")
    parser.add_argument("--trials-s3", type=int, default=50, help="Trials for Stage 3")
    parser.add_argument("--stage2-only", action="store_true")
    parser.add_argument("--stage3-only", action="store_true")
    parser.add_argument("--model", default=None, help="Run single model only")
    args = parser.parse_args()

    run_s2 = not args.stage3_only
    run_s3 = not args.stage2_only

    models = [args.model] if args.model else list(MODEL_OPERATING_POINTS.keys())

    print("=" * 70)
    print("STAGE 2 + STAGE 3 RUNNER — ALL MODELS")
    print(f"  Models: {len(models)} | GPU: {args.gpu}")
    print(f"  Stage 2: {'YES' if run_s2 else 'SKIP'} ({args.trials_s2} trials)")
    print(f"  Stage 3: {'YES' if run_s3 else 'SKIP'} ({args.trials_s3} trials)")
    print("=" * 70)

    total_start = time.time()
    status = {}

    for i, model_name in enumerate(models, 1):
        print(f"\n{'#' * 70}")
        print(f"# [{i}/{len(models)}] {model_name}")
        print(f"{'#' * 70}")

        model_status = {}
        model = tokenizer = info = None

        # Stage 2
        if run_s2:
            try:
                model, tokenizer, info, s2_results = run_stage2_for_model(
                    model_name, args.gpu, trials=args.trials_s2)
                model_status["stage2"] = "OK" if s2_results or has_stage2_results(model_name) else "SKIP"
            except Exception as e:
                print(f"  Stage 2 FAILED: {e}")
                model_status["stage2"] = f"FAILED: {str(e)[:80]}"

        # Stage 3
        if run_s3:
            try:
                run_stage3_for_model(
                    model_name, args.gpu, model=model, tokenizer=tokenizer,
                    info=info, trials=args.trials_s3)
                model_status["stage3"] = "OK" if has_stage3_results(model_name) else "SKIP"
            except Exception as e:
                print(f"  Stage 3 FAILED: {e}")
                model_status["stage3"] = f"FAILED: {str(e)[:80]}"

        # Free GPU
        del model, tokenizer, info
        free_gpu()

        status[model_name] = model_status

    elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"ALL DONE — {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"{'=' * 70}")
    for m, s in status.items():
        print(f"  {m}")
        for stage, st in s.items():
            print(f"    {stage}: {st}")


if __name__ == "__main__":
    main()
