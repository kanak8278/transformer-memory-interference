"""
Kaggle kernel entry-point: big-cell IVQ eval (base vs LoRA) on ARBITRARY_SINGLE.

Runs the committed E1 runner (scan C = 20 cells, K in {15,20,25,30} x
N in {10,15,20,30,50}) measuring FVQ + CVQ + 3 interior depths, base vs the
main LoRA adapter. Self-contained: clones the repo on the VM (Kaggle ships only
this one file), installs peft, runs, writes results to /kaggle/working/out.

HOW TO RUN
  1. Set MODEL below to "qwen" or "gemma".
  2. Gemma is a GATED model -> add an HF token as a Kaggle secret named
     HF_TOKEN (Notebook editor: Add-ons > Secrets). Qwen needs no token.
  3. Push forcing a T4 (P100 crashes: PyTorch on Kaggle drops sm_60):
        kaggle kernels push -p <this_dir> --accelerator NvidiaTeslaT4
     (the exact string NvidiaTeslaT4 is required; "T4" is silently ignored.)
  4. Poll:  kaggle kernels status <owner>/<slug>
     Pull:  kaggle kernels output <owner>/<slug> -p ./out   (repo is in /tmp,
            so the output is just the small results file + log.)

NOTES / lessons baked in
  - micro-batch is small (T4 16GB OOMs at 6 on the N>=30 long streams).
  - gemma runs a base-FVQ SMOKE first and aborts if ~0 (a known gemma-3
    generation bug on some transformers versions — fail in minutes, not hours).
  - Never hardcode the token; it comes from the Kaggle secret / env only.
"""
import os, subprocess, sys, json

MODEL = "qwen"          # <-- "qwen" or "gemma"

CFG = {
    "qwen":  dict(base="Qwen/Qwen2.5-3B-Instruct",
                  adapter="lora_intervention/checkpoints/adapter",
                  micro_batch=4, gated=False, smoke=False),
    "gemma": dict(base="google/gemma-3-4b-it",
                  adapter="lora_intervention/checkpoints/gemma_adapter",
                  micro_batch=2, gated=True,  smoke=True),
}[MODEL]

if CFG["gated"]:
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        try:
            from kaggle_secrets import UserSecretsClient
            tok = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            tok = None
    if tok:
        os.environ["HF_TOKEN"] = tok
        os.environ["HUGGING_FACE_HUB_TOKEN"] = tok
    else:
        print("WARN: no HF_TOKEN secret found; gated model download will fail", flush=True)

REPO = "/tmp/repo"          # scratch, NOT /kaggle/working -> keeps output small
OUT  = "/kaggle/working/out"
def sh(c): print("::", c, flush=True); subprocess.run(c, shell=True, check=False)

sh(f"git clone --depth 1 -b aaai-prep "
   f"https://github.com/kanak8278/transformer-memory-interference.git {REPO}")
sh(f"{sys.executable} -m pip install -q peft accelerate")
os.makedirs(OUT, exist_ok=True)

runner = f"{REPO}/lora_intervention/experiments/e1_extrapolation_frontier.py"
base_args = ["--base-model", CFG["base"],
             "--adapter", f"{REPO}/{CFG['adapter']}",
             "--backend", "hf", "--micro-batch", str(CFG["micro_batch"])]

if CFG["smoke"]:
    print("=== SMOKE (base FVQ sanity) ===", flush=True)
    subprocess.run([sys.executable, runner, "--out-dir", OUT, "--scans", "C",
                    "--only", "base", "--smoke"] + base_args, cwd=REPO, check=False)
    sm = os.path.join(OUT, "smoke", "results.jsonl")
    accs = [json.loads(l)["accuracy"] for l in open(sm)] if os.path.exists(sm) else []
    fvq = [json.loads(l)["accuracy"] for l in open(sm)
           if json.loads(l)["condition"] == "FVQ"] if os.path.exists(sm) else []
    print(f"SMOKE FVQ accs: {fvq}", flush=True)
    if fvq and max(fvq) < 0.2:
        print("!!! SMOKE FAILED: base FVQ ~0 -> generation broken. Aborting.", flush=True)
        sys.exit(2)

print("=== FULL (base + LoRA) ===", flush=True)
subprocess.run([sys.executable, runner, "--out-dir", OUT, "--scans", "C",
                "--only", "both"] + base_args, cwd=REPO, check=False)
print("=== KERNEL DONE ===", flush=True)
