"""
Stage-3C promoter-head ablation, HF-hooks version (no TransformerLens).

Resolves Finding 001: the paper's §7.3 claim (ablating the 8 promoter heads drops
L32 P(v_last) MORE in LoRA than base -> paired +0.12) is contradicted by the only
local data (n=50, opposite ordering) and the n=200 files aren't in the repo. This
re-runs it properly: n=200, deterministic PAIRED seeds (same stream in base and
LoRA per trial), per-trial export -> bootstrap CIs.

Method (HF forward-hooks, equivalent to the TL ablation, avoids TL+LoRA+Xet pain
on Colab):
  - Ablate a head by zeroing its slice of the o_proj INPUT (concatenated head
    outputs) via a forward_pre_hook -> removes that head's contribution.
  - Logit lens: per-layer residual -> final_norm -> lm_head -> P(v_last)
    (identical to run_logit_lens_lora_hf.py). Delta = ablated - normal.
  - The 8 promoter heads (paper's, from baseline attention-routing top-8):
    L32H3, L32H7, L31H12, L31H15, L32H10, L30H11, L32H0, L32H14.
  - Same 8 heads ablated in BOTH base and LoRA, at BOTH cells (paired).

Cells: K2/N5 (paper's ablation cell, mild-failure) and K2/N50 (clear-failure).
Also records a behavioral FVQ/CVQ pre-check per cell (confirms base fails CVQ).

Greedy, ARB single-token, Qwen2.5-3B +/- main LoRA (merged). Writes per-trial
arrays to results.json; CIs computed post-hoc.
"""
import os, gc, sys, json, time, argparse
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from lora_intervention.data_gen import stable_seed  # deterministic, PYTHONHASHSEED-independent
from lora_intervention.evaluate_ivq import make_prompt, is_correct

# Gemma-3-4b promoter heads: top-8 FVQ-CVQ from
# v3/results_vllm/attention_routing/head_analysis/head_analysis_report.md
# (gemma-3-4b-it__normal). Same discovery metric as Qwen's 8; Gemma's cluster
# at L23 (mid-depth) rather than Qwen's L30-33 (late).
PROMOTER_HEADS = [(23, 3), (23, 1), (23, 0), (23, 6), (23, 7), (17, 5), (17, 0), (29, 4)]
CELLS = [(2, 30), (10, 50)]   # mild (base CVQ~0.61) + genuine failure (base CVQ~0.32)
TRIALS = 200
PRECHECK_TRIALS = 60


def first_tok(tok, value):
    ids = set()
    for s in (f" {value}", value):
        e = tok.encode(s, add_special_tokens=False)
        if e:
            ids.add(e[0])
    return list(ids)


def _model_class_for(base_id):
    """Gemma-3 ships as a VLM via Auto*; pick the text-only causal head."""
    from transformers import AutoModelForCausalLM
    if "gemma-3" in base_id.lower():
        try:
            from transformers import Gemma3ForCausalLM
            return Gemma3ForCausalLM
        except ImportError:
            pass
    return AutoModelForCausalLM


def load(base_id, adapter=None):
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    m = _model_class_for(base_id).from_pretrained(base_id, torch_dtype=torch.bfloat16,
                                             device_map="cuda", attn_implementation="sdpa",
                                             trust_remote_code=True)
    if adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
    m.eval()
    return m, tok


def final_norm_of(model):
    mm = getattr(model, "model", None)
    for attr in ("norm", "final_layer_norm"):
        c = getattr(mm, attr, None) if mm is not None else None
        if c is not None:
            return c
    return None


def head_dim_of(model):
    cfg = model.config
    return getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)


def install_ablation_hooks(model, heads, head_dim):
    """Zero each (layer,head)'s slice of the o_proj input. Returns handles."""
    by_layer = {}
    for L, h in heads:
        by_layer.setdefault(L, []).append(h)
    handles = []
    for L, hs in by_layer.items():
        oproj = model.model.layers[L].self_attn.o_proj
        def make_hook(hs_local):
            def pre_hook(module, args):
                x = args[0].clone()
                for h in hs_local:
                    x[..., h * head_dim:(h + 1) * head_dim] = 0
                return (x,) + args[1:]
            return pre_hook
        handles.append(oproj.register_forward_pre_hook(make_hook(hs)))
    return handles


def layer_p_last(model, tok, prompt, v_last_ids, final_norm):
    """Per-layer P(v_last) via logit lens. Returns list over layers."""
    import torch
    n_layers = model.config.num_hidden_layers
    lm_head = model.get_output_embeddings()
    enc = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        r = model(**enc, output_hidden_states=True, use_cache=False, return_dict=True)
    out = []
    for L in range(n_layers):
        resid = r.hidden_states[L + 1][0, -1, :]
        if final_norm is not None and L < n_layers - 1:
            resid = final_norm(resid.unsqueeze(0)).squeeze(0)
        probs = torch.softmax(lm_head(resid).float(), dim=-1)
        out.append(max(probs[i].item() for i in v_last_ids) if v_last_ids else 0.0)
    del r, enc
    return out


def precheck(model, tok, nk, nu):
    """Behavioral FVQ (k=1) and CVQ (k=N) accuracy, base model, quick."""
    import torch
    res = {}
    for cond, k in [("FVQ", 1), ("CVQ", nu)]:
        correct = 0; n = 0
        for t in range(PRECHECK_TRIALS):
            seed = stable_seed("ablate_precheck", nk, nu, cond, t)
            p, exp = make_prompt(tok, nk, nu, k, seed)
            if p is None:
                continue
            enc = tok(p, return_tensors="pt").to(model.device)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
            pred = tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            correct += int(is_correct(pred, exp)); n += 1
        res[cond] = {"acc": correct / n if n else 0.0, "n": n}
    return res


def run_ablation(model, tok, nk, nu, head_dim, final_norm, target_layers=(30, 31, 32, 33)):
    per_trial = []  # each: {L: [normal, ablated]} for target layers, at position N (CVQ)
    n = 0
    for t in range(TRIALS):
        seed = stable_seed("ablate", nk, nu, "CVQ", t)  # paired across models
        prompt, expected = make_prompt(tok, nk, nu, nu, seed)  # k=N -> CVQ "last"
        if prompt is None:
            continue
        vids = first_tok(tok, expected)
        normal = layer_p_last(model, tok, prompt, vids, final_norm)
        handles = install_ablation_hooks(model, PROMOTER_HEADS, head_dim)
        try:
            ablated = layer_p_last(model, tok, prompt, vids, final_norm)
        finally:
            for h in handles:
                h.remove()
        per_trial.append({str(L): [normal[L], ablated[L]] for L in target_layers})
        n += 1
        if (n) % 25 == 0:
            import torch; torch.cuda.empty_cache()
    return per_trial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default=str(_ROOT / "lora_intervention" / "checkpoints" / "adapter"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "results.json"; logf = out_dir / "run.log"

    def log(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        with open(logf, "a") as f:
            f.write(line + "\n"); f.flush()

    results = json.loads(res_path.read_text()) if res_path.exists() else {"promoter_heads": [f"L{L}H{h}" for L, h in PROMOTER_HEADS], "trials": TRIALS, "cells": {}}

    def done(model_tag, nk, nu):
        ck = f"K{nk}N{nu}"
        return ck in results["cells"] and model_tag in results["cells"][ck]

    def save():
        res_path.write_text(json.dumps(results))

    log(f"Stage-3C ablation re-run (HF hooks). heads={results['promoter_heads']} trials={TRIALS} cells={CELLS}")

    for tag, adapter in [("base", None), ("lora", args.adapter)]:
        if all(done(tag, nk, nu) for nk, nu in CELLS):
            continue
        log(f"loading {tag}...")
        model, tok = load(args.base_model, adapter)
        head_dim = head_dim_of(model); fn = final_norm_of(model)
        log(f"{tag} loaded. head_dim={head_dim}, final_norm={type(fn).__name__ if fn else None}")
        try:
            for nk, nu in CELLS:
                ck = f"K{nk}N{nu}"
                if done(tag, nk, nu):
                    continue
                cell = results["cells"].setdefault(ck, {})
                t0 = time.time()
                pc = precheck(model, tok, nk, nu) if tag == "base" else None
                per_trial = run_ablation(model, tok, nk, nu, head_dim, fn)
                # summary at L32
                import statistics
                d32 = [pt["32"][1] - pt["32"][0] for pt in per_trial]
                entry = {"per_trial": per_trial, "n": len(per_trial),
                         "mean_delta_L32": statistics.mean(d32) if d32 else None,
                         "mean_normal_L32": statistics.mean(pt["32"][0] for pt in per_trial) if per_trial else None,
                         "mean_ablated_L32": statistics.mean(pt["32"][1] for pt in per_trial) if per_trial else None}
                if pc:
                    entry["precheck_base"] = pc
                cell[tag] = entry
                save()
                extra = f" | base precheck FVQ={pc['FVQ']['acc']:.2f} CVQ={pc['CVQ']['acc']:.2f}" if pc else ""
                log(f"{ck} {tag}: L32 delta={entry['mean_delta_L32']:+.3f} "
                    f"(normal={entry['mean_normal_L32']:.3f} ablated={entry['mean_ablated_L32']:.3f}) "
                    f"n={entry['n']} ({time.time()-t0:.0f}s){extra}")
        finally:
            del model; gc.collect()
            import torch; torch.cuda.empty_cache()
    log("DONE")


if __name__ == "__main__":
    main()
