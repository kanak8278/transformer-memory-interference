"""
Block vs LoRA — readout-level convergence probe.

QUESTION (extends App block_locus, which tested ROUTING and found Block and LoRA
use DIFFERENT heads). block_locus never checked whether the two fixes CONVERGE at
the readout. Hypothesis: "convergent readout, divergent routing" — both Block
(format) and LoRA (weights) release the suppressed v_last so it propagates to the
output, even though they trigger it via different upstream attention.

Three conditions at one cell (default K=2, N=30, ARB single-token, Qwen2.5-3B):
  base_plain : base model, plain interleaved stream   -> FAILS (tracked-but-suppressed)
  base_block : base model, [Round j]-grouped stream   -> format fix
  lora_plain : base+LoRA (merged), plain stream        -> weight fix

Methods (the two the paper has for base-vs-LoRA but never ran under Block):
  1. Logit lens: per-layer P(v_last)/P(v_first) via final-norm -> lm_head
     (exact methodology of run_logit_lens_lora_hf.py). PRIMARY signal.
  2. CVQ-correctness linear probe: per-layer logistic-reg CV accuracy predicting
     whether the model answers CVQ correctly. SECONDARY (degenerate where a
     condition saturates near 0%/100% correct — flagged, not forced).

If base_block's per-layer P(v_last) trajectory + probe recovery track lora_plain's
(build at ~L32 and PROPAGATE to the final layer, vs base_plain's build-then-decay)
while routing stays divergent -> "two roads, one readout".

Greedy/deterministic; seeds depend only on (K,N,cond,trial) so conditions see
matched value assignments. HF backend (Qwen is not Xet-backed -> Colab-safe).
"""
import os, gc, sys, json, time, random, argparse
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from mechanistic_probing_v2.core.dataset_configs import (
    get_eligible_categories, generate_values_for_trial,
)

DATASET = "ARBITRARY_SINGLE"
SYSTEM = ("You are a precise data extraction tool. Output ONLY a single word - "
          "the exact value requested. No other text, no explanation, no punctuation.")


def stream_items(k_keys, n_updates, seed):
    """Return (ordered values_per_cat, test_category, interleaved-shuffled items).
    first/last are defined by STREAM position (post-shuffle), matching the paper."""
    rng = random.Random(seed)
    eligible = get_eligible_categories(DATASET, min_values=n_updates)
    cats = rng.sample(eligible, min(k_keys, len(eligible)))
    tc = cats[seed % len(cats)]
    vpc = generate_values_for_trial(DATASET, cats, n_updates, rng)  # ordered per cat
    items = [{"category": c, "value": v, "round": j + 1}
             for c in cats for j, v in enumerate(vpc[c])]
    rng.shuffle(items)
    return vpc, cats, tc, items


def build_plain(items, tc, condition):
    stream = "\n".join(f"{it['category']}: {it['value']}" for it in items)
    cat_vals = [it["value"] for it in items if it["category"] == tc]  # stream order
    qword = "first" if condition == "FVQ" else "last"
    expected = cat_vals[0] if condition == "FVQ" else cat_vals[-1]
    user = ("Read the following key-value stream. Each key gets updated multiple "
            f"times.\n\n{stream}\n\nWhat was the {qword} value of {tc}?")
    return user, expected


def build_block(vpc, cats, tc, n_updates, condition):
    """[Round j] grouped rendering; query targets round 1 (FVQ) or N (CVQ)."""
    lines = []
    for j in range(1, n_updates + 1):
        lines.append(f"[Update {j}]")
        for c in cats:
            lines.append(f"  {c}: {vpc[c][j - 1]}")
    stream = "\n".join(lines)
    rnd = 1 if condition == "FVQ" else n_updates
    expected = vpc[tc][rnd - 1]
    user = ("Read the following key-value stream. Each key is updated multiple "
            f"times, grouped by update round.\n\n{stream}\n\n"
            f"What was the value of {tc} in Update {rnd}?")
    return user, expected


def make_prompt(tokenizer, k_keys, n_updates, condition, fmt, seed):
    vpc, cats, tc, items = stream_items(k_keys, n_updates, seed)
    if fmt == "block":
        user, expected = build_block(vpc, cats, tc, n_updates, condition)
    else:
        user, expected = build_plain(items, tc, condition)
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    # candidate token ids for first & last correct targets (for logit lens)
    cat_vals_stream = [it["value"] for it in items if it["category"] == tc]
    v_first = vpc[tc][0] if fmt == "block" else cat_vals_stream[0]
    v_last = vpc[tc][-1] if fmt == "block" else cat_vals_stream[-1]
    return prompt, expected, v_first, v_last


def first_tok(tokenizer, value):
    ids = set()
    for s in (f" {value}", value):
        e = tokenizer.encode(s, add_special_tokens=False)
        if e:
            ids.add(e[0])
    return list(ids)


def load(base_id, adapter=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    return model, tok


def final_norm_of(model):
    m = getattr(model, "model", None)
    for attr in ("norm", "final_layer_norm"):
        c = getattr(m, attr, None) if m is not None else None
        if c is not None:
            return c
    return None


def run_condition(model, tok, k_keys, n_updates, trials, fmt):
    import torch, numpy as np
    n_layers = model.config.num_hidden_layers
    lm_head = model.get_output_embeddings()
    fnorm = final_norm_of(model)
    out = {}
    for condition in ["FVQ", "CVQ"]:
        p_last = np.zeros(n_layers); p_first = np.zeros(n_layers)
        resid_by_layer = [[] for _ in range(n_layers)]  # for probe (CVQ only)
        correct = []
        n = 0
        for t in range(trials):
            seed = (hash((DATASET, k_keys, n_updates, condition, t)) & 0x7fffffff)
            prompt, expected, v_first, v_last = make_prompt(tok, k_keys, n_updates, condition, fmt, seed)
            tf = first_tok(tok, v_first); tl = first_tok(tok, v_last)
            enc = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                r = model(**enc, output_hidden_states=True, use_cache=False, return_dict=True)
            for L in range(n_layers):
                resid = r.hidden_states[L + 1][0, -1, :]
                if condition == "CVQ":
                    resid_by_layer[L].append(resid.float().cpu().numpy())
                proj = resid
                if fnorm is not None and L < n_layers - 1:
                    proj = fnorm(proj.unsqueeze(0)).squeeze(0)
                probs = torch.softmax(lm_head(proj).float(), dim=-1)
                if tl:
                    p_last[L] += max(probs[i].item() for i in tl)
                if tf:
                    p_first[L] += max(probs[i].item() for i in tf)
            pred = tok.decode([int(r.logits[0, -1, :].argmax())]).strip().lower()
            correct.append(int(pred == expected.lower().strip()))
            n += 1
            del r, enc
            if (t + 1) % 25 == 0:
                torch.cuda.empty_cache()
        rec = {"condition": condition, "n": n,
               "behavioral_acc": sum(correct) / n if n else 0.0,
               "p_last_per_layer": (p_last / n).tolist(),
               "p_first_per_layer": (p_first / n).tolist()}
        if condition == "CVQ":
            rec["probe"] = fit_probe(resid_by_layer, correct)
        out[condition] = rec
    return out


def fit_probe(resid_by_layer, correct):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    y = np.array(correct)
    pos, neg = int(y.sum()), int((1 - y).sum())
    if pos < 8 or neg < 8:   # need variance in both classes
        return {"degenerate": True, "pos": pos, "neg": neg, "acc_per_layer": None}
    accs = []
    for L in range(len(resid_by_layer)):
        X = np.stack(resid_by_layer[L])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        accs.append(float(cross_val_score(clf, X, y, cv=5).mean()))
    return {"degenerate": False, "pos": pos, "neg": neg, "acc_per_layer": accs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default=str(_ROOT / "lora_intervention" / "checkpoints" / "adapter"))
    ap.add_argument("--keys", type=int, default=2)
    ap.add_argument("--updates", type=int, default=30)
    ap.add_argument("--trials", type=int, default=100)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "results.json"
    logf = out_dir / "run.log"

    def log(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        with open(logf, "a") as f:
            f.write(line + "\n"); f.flush()

    results = {"cell": {"keys": args.keys, "updates": args.updates}, "trials": args.trials,
               "conditions": {}}
    if res_path.exists():
        results = json.loads(res_path.read_text())

    def save():
        res_path.write_text(json.dumps(results, indent=1))

    log(f"Block-readout comparison K={args.keys} N={args.updates} trials={args.trials}")

    # base model: run base_plain and base_block (reuse one load)
    if "base_plain" not in results["conditions"] or "base_block" not in results["conditions"]:
        log("loading base...")
        model, tok = load(args.base_model)
        for fmt, name in [("plain", "base_plain"), ("block", "base_block")]:
            if name in results["conditions"]:
                continue
            log(f"running {name}...")
            results["conditions"][name] = run_condition(model, tok, args.keys, args.updates, args.trials, fmt)
            log(f"{name}: CVQ acc={results['conditions'][name]['CVQ']['behavioral_acc']:.2f}")
            save()
        del model; gc.collect()
        import torch; torch.cuda.empty_cache()

    # LoRA + plain
    if "lora_plain" not in results["conditions"]:
        log("loading LoRA (merged)...")
        model, tok = load(args.base_model, args.adapter)
        log("running lora_plain...")
        results["conditions"]["lora_plain"] = run_condition(model, tok, args.keys, args.updates, args.trials, "plain")
        log(f"lora_plain: CVQ acc={results['conditions']['lora_plain']['CVQ']['behavioral_acc']:.2f}")
        save()
        del model; gc.collect()
        import torch; torch.cuda.empty_cache()

    log("DONE")


if __name__ == "__main__":
    main()
